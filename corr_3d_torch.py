"""
Absorption correction for X-ray scattering data collected in a cylindrical capillary.

Overview
--------
This module performs an absorption correction for measured scattering intensities
I(q) by explicitly integrating the beam intensity distribution and attenuation
through a cylindrical sample + capillary wall using 3D numerical integration
(TorchQuad + PyTorch).

Key steps
---------
1. Load measured scattering data (q, intensity, sigma).
2. Load beam profiles in the vertical (y) and horizontal (z) directions and
   build interpolating functions based on torch.bucketize (GPU-friendly).
3. Compute a beam-weighted volume normalization integral over the illuminated
   capillary volume (no attenuation; just beam weighting and geometry).
4. For a grid of scattering angles θ, compute the transmission / attenuation
   factor A(θ) by Simpson integration over capillary volume:
   - Map unit cube (u, v, w) ∈ [0,1]^3 → (x, y, z) in cylindrical capillary.
   - For each point, compute path length inside sample and wall along the
     scattered ray, and the corresponding attenuation exp(-μ l_sample - μ_w l_wall).
   - Weight by beam intensity profiles.
5. Fit a cubic spline to A(θ) for smooth interpolation.
6. For each experimental q, convert to θ and evaluate A(θ).
7. Divide measured intensity by A(θ) to obtain absorption-corrected intensity.

Dependencies
------------
- numpy
- pandas
- scipy (interpolation and cubic spline)
- torch
- torchquad
- matplotlib (for diagnostic plots)
"""

import os
import inspect  # currently unused; imported for potential debugging/introspection

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import torch
from torchquad import Simpson, set_up_backend

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Torch / TorchQuad backend setup
# ---------------------------------------------------------------------------

# Select GPU if available, otherwise fall back to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tell TorchQuad to use PyTorch backend with single-precision floats.
set_up_backend("torch", data_type="float32")


# ---------------------------------------------------------------------------
# 1D interpolation in torch (GPU-friendly analogue of np.interp)
# ---------------------------------------------------------------------------

def interp1d_torch(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor
) -> torch.Tensor:
    """
    Parameters
    ----------
    x : torch.Tensor
        Query points at which to evaluate the interpolant. Will be flattened.
    xp : torch.Tensor
        Sorted 1D tensor of x-coordinates of the data points (monotonically increasing).
    fp : torch.Tensor
        1D tensor of y-coordinates of the data points, same shape as `xp`.

    Returns
    -------
    torch.Tensor
        Interpolated values at the points in `x`, with the same shape as `x.flatten()`.
        Values outside the range [xp[0], xp[-1]] are set to 0.
    """
    
    x = x.flatten()

 
    idx = torch.bucketize(x, xp)


    idx = idx.clamp(1, len(xp) - 1)

    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]

    slope = (y1 - y0) / (x1 - x0 + 1e-8)  # small epsilon avoids division by zero
    y = y0 + slope * (x - x0)
    y = torch.where(
        (x < xp[0]) | (x > xp[-1]),
        torch.tensor(0.0, device=x.device),
        y
    )
    return y


# ---------------------------------------------------------------------------
# Beam profile loader
# ---------------------------------------------------------------------------

def load_profile(
    filename: str,
    symmetric: bool = False,
    device: str = "cuda"
):
    """

    Parameters
    ----------
    filename : str
        Path to the CSV file containing (coord, intensity).
    symmetric : bool, optional
        If True, the profile is mirrored around 0 to yield a symmetric profile.
        This is useful if the measured profile is only on one side.
    device : str, optional
        Device on which the underlying torch tensors are stored ("cuda" or "cpu").

    Returns
    -------
    callable
        A function `profile_fn(x)` where `x` is a torch tensor of positions, and
        the return value is the interpolated intensity at those positions.
    """

    data = np.loadtxt(filename, delimiter=",")
    coord, intensity = data[:, 0], data[:, 1]

    # Optionally enforce symmetry about 0
    if symmetric:
        # Mirror coordinates and intensities
        coord = np.concatenate((-coord[::-1], coord))
        intensity = np.concatenate((intensity[::-1], intensity))


    coord_tensor = torch.tensor(coord, dtype=torch.float32, device=device)
    intensity_tensor = torch.tensor(intensity, dtype=torch.float32, device=device)

    def profile_fn(x: torch.Tensor) -> torch.Tensor:
        return interp1d_torch(x, coord_tensor, intensity_tensor)
       
    return profile_fn


# ---------------------------------------------------------------------------
# 3D beam volume normalization over capillary
# ---------------------------------------------------------------------------

def compute_beam_volume_normalization(
    I0_y_func,
    I0_z_func,
    R: float,
    t: float,
    a: float,
    b: float,
    n_pts: int,
    device: str = "cuda"
) -> float:
    """
    Compute a normalization factor for the beam intensity over the illuminated
    3D volume of the capillary.

    This computes:
        ∫∫∫ I_y(y) I_z(z) dV

    Geometry / mapping
    ------------------
    - y ∈ [-a, a] mapped from u ∈ [0, 1]
    - z ∈ [-b, b] mapped from v ∈ [0, 1]
    - For a given y, x runs through the circle of radius R:
          x ∈ [-sqrt(R² - y²), +sqrt(R² - y²)]
      mapped from w ∈ [0, 1]
    - The Jacobian J(u, v, w) = 8 a b sqrt(R² - y²)

    Parameters
    ----------
    I0_y_func : callable
        Interpolator for beam intensity as function of y (torch tensor → torch tensor).
    I0_z_func : callable
        Interpolator for beam intensity as function of z (torch tensor → torch tensor).
    R : float
        Inner radius of the cylindrical sample (capillary inner radius).
    t : float
        Wall thickness (not used directly here, but kept for symmetry w.r.t. sample integrals).
    a : float
        Half-width of the illuminated beam in y direction.
    b : float
        Half-length (or half-height) of the illuminated region in z direction.
    n_pts : int
        Number of integration points for the TorchQuad Simpson integrator.
    device : str, optional
        Device for the integrand computations ("cuda" or "cpu").

    Returns
    -------
    float
        Normalization integral value.
    """

    def f(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Beam intensity as a function of position, factorized as I_y(y) * I_z(z).
        `x` is unused here, but included for symmetry with other integrands.
        """
        I_y = I0_y_func(y)
        I_z = I0_z_func(z)
        return I_y * I_z

    def transformed_integrand(uvw: torch.Tensor) -> torch.Tensor:
        """
        Integrand in unit cube coordinates (u, v, w).

        Parameters
        ----------
        uvw : torch.Tensor
            Shape (N, 3), values in [0, 1] for each dimension.

        Returns
        -------
        torch.Tensor
            Integrand values at each (u, v, w) point.
        """
        uvw = uvw.to(device)
        u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]

        # Map to physical y, z
        y = 2 * a * u - a
        z = 2 * b * v - b

        # For each y, x spans the chord inside the circle of radius R
        sqrt_term = torch.sqrt(R**2 - y**2)
        x = 2 * sqrt_term * w - sqrt_term

        # Jacobian of the transformation (u,v,w) → (x,y,z)
        J = 8 * a * b * sqrt_term

        return f(x, y, z) * J

    integrator = Simpson()

    # We do not need gradients during integration
    with torch.no_grad():
        result = integrator.integrate(
            transformed_integrand,
            dim=3,
            N=n_pts,
            integration_domain=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        )

    return result.item()


# ---------------------------------------------------------------------------
# Data reader for scattering file
# ---------------------------------------------------------------------------

def read_scattering_file(file_path: str) -> pd.DataFrame:
    """
    Read a plain-text scattering file with three whitespace-separated columns:
    q_nm^-1, intensity, sigma.

    Lines that do not parse cleanly to three floats are skipped.

    Parameters
    ----------
    file_path : str
        Path to the scattering file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ['q_nm^-1', 'intensity', 'sigma'].
    """
    clean_rows = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                try:
                    row = list(map(float, parts))
                    clean_rows.append(row)
                except ValueError:
                    # Skip lines that don't contain valid floats
                    continue

    return pd.DataFrame(clean_rows, columns=["q_nm^-1", "intensity", "sigma"])


# ---------------------------------------------------------------------------
# Sample attenuation integral for a single q (via θ)
# ---------------------------------------------------------------------------

def compute_correction_sample(
    q: float,
    R: float,
    a: float,
    b: float,
    mu: float,
    t: float,
    mu_w: float,
    I0_y_func,
    I0_z_func,
    norm: float,
    wavelength: float = 0.15406,
    d: float = 267.0,
    n_pts: int = 20,
    device: str = "cuda",
) -> float:
    """
    Parameters
    ----------
    q : float
        Scattering vector magnitude in nm^-1 (or consistent units with wavelength).
    R : float
        Inner radius of the cylindrical sample.
    a, b : float
        Half-width (y) and half-length (z) of the illuminated beam region.
    mu : float
        Linear attenuation coefficient of the sample.
    t : float
        Capillary wall thickness.
    mu_w : float
        Linear attenuation coefficient of the wall.
    I0_y_func : callable
        Beam intensity profile in y: torch.Tensor → torch.Tensor.
    I0_z_func : callable
        Beam intensity profile in z: torch.Tensor → torch.Tensor.
    norm : float
        Normalization factor from `compute_beam_volume_normalization`.
    wavelength : float, optional
        X-ray wavelength.
    d : float, optional
        Sample-to-detector distance.
    n_pts : int, optional
        Number of integration points in each dimension for TorchQuad Simpson.
    device : str, optional
        Device for tensor computations.

    Returns
    -------
    float
        unnormalized transmission integral
    """
    theta = 2 * np.arcsin(np.clip(q * wavelength / (4 * np.pi), -1, 1))
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    def f(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
   
        I_y = I0_y_func(y)
        I_z = I0_z_func(z)

        Dx = d * cos_theta
        Dy = d * sin_theta

        A = (Dx - x) ** 2 + (Dy - y) ** 2
        B = 2 * (x * (Dx - x) + y * (Dy - y))

        C1 = x**2 + y**2 - R**2
        D1 = torch.clamp(B**2 - 4 * A * C1, min=1e-10)  # discriminant
        t1 = (-B + torch.sqrt(D1)) / (2 * A)

        l1 = (
            t1
            * torch.sqrt((Dx - x) ** 2 + (Dy - y) ** 2 + z**2)
            + x
            + torch.sqrt(R**2 - y**2)
        )

        C2 = x**2 + y**2 - (R + t) ** 2
        D2 = torch.clamp(B**2 - 4 * A * C2, min=1e-10)
        t2 = (-B + torch.sqrt(D2)) / (2 * A)

        l2 = (
            t2
            * torch.sqrt((Dx - x) ** 2 + (Dy - y) ** 2 + z**2)
            + x
            + torch.sqrt((R + t) ** 2 - y**2)
        )

        return torch.exp(-mu * l1 - mu_w * (l2 - l1)) * I_y * I_z

    def transformed_integrand(uvw: torch.Tensor) -> torch.Tensor:
        uvw = uvw.to(device)
        u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]
       
        y = 2 * a * u - a
        z = 2 * b * v - b

        sqrt_term = torch.sqrt(R**2 - y**2)
        x = 2 * sqrt_term * w - sqrt_term

        J = 8 * a * b * sqrt_term

        return f(x, y, z) * J

    integrator = Simpson()

    result = integrator.integrate(
        transformed_integrand,
        dim=3,
        N=n_pts,
        integration_domain=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    )

    return result.item()


# ---------------------------------------------------------------------------
# Transmission vs θ: compute table and spline fit
# ---------------------------------------------------------------------------

def _compute_A_for_theta(args):
    """
    Internal helper: given a single θ (and other parameters), compute A(θ).

    Parameters
    ----------
    args : tuple
        (theta, R, mu, a, b, t, mu_w, I0_y_func, I0_z_func, norm, wavelength, d, n_pts)

    Returns
    -------
    tuple
        (theta, A(theta)), where A(theta) is the normalized attenuation factor.
    """
    (
        theta,
        R,
        mu,
        a,
        b,
        t,
        mu_w,
        I0_y_func,
        I0_z_func,
        norm,
        wavelength,
        d,
        n_pts,
    ) = args

    q = (4 * np.pi / wavelength) * np.sin(theta / 2)

    AS = compute_correction_sample(
        q, R, a, b, mu, t, mu_w, I0_y_func, I0_z_func, norm, wavelength, d, n_pts
    )

    A = AS / norm
    return (theta, A)


def compute_transmission_vs_theta(
    R: float,
    mu: float,
    a: float,
    b: float,
    t: float,
    mu_w: float,
    I0_y_func,
    I0_z_func,
    norm: float,
    theta_vals: np.ndarray,
    wavelength: float = 0.15406,
    d: float = 267.0,
    n_pts: int = 20,
):
    """

    Parameters
    ----------
    R, mu, a, b, t, mu_w, I0_y_func, I0_z_func, norm : see above
        Geometry, material properties, beam profiles, and normalization.
    theta_vals : np.ndarray
        Array of scattering angles in radians at which to compute A(θ).
    wavelength : float, optional
        X-ray wavelength.
    d : float, optional
        Sample-to-detector distance.
    n_pts : int, optional
        Integration resolution parameter for TorchQuad Simpson.

    Returns
    -------
    list of tuple
        List of (theta, A(theta)) pairs.
    """
    results = []
    for theta in theta_vals:
        args = (
            theta,
            R,
            mu,
            a,
            b,
            t,
            mu_w,
            I0_y_func,
            I0_z_func,
            norm,
            wavelength,
            d,
            n_pts,
        )

        theta_val, A_val = _compute_A_for_theta(args)
        # Diagnostic printout in degrees
        print(f"theta = {np.degrees(theta_val):.2f}° -> A = {A_val:.6f}")
        results.append((theta_val, A_val))

    return results


def fit_transmission_spline(trans_data):
   
    """
    Parameters
    ----------
    trans_data : list of tuple
        List of (theta, A(theta)) pairs.

    Returns
    -------
    scipy.interpolate.CubicSpline
        Cubic spline object representing A(θ).
    """
   
    theta_vals, A_vals = zip(*trans_data)
    return CubicSpline(theta_vals, A_vals, bc_type="natural")


def evaluate_transmission_spline(theta: float, spline_func: CubicSpline) -> float:
    """
    Parameters
    ----------
    theta : float or np.ndarray
        Scattering angle(s) in radians.
    spline_func : CubicSpline
        Spline obtained from `fit_transmission_spline`.

    Returns
    -------
    float or np.ndarray
        Interpolated A(θ) value(s).
    """
    return spline_func(theta)


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_transmission_spline(trans_data, spline_func):
    """
    Plot the raw transmission data and its cubic spline fit vs θ (in degrees).

    Parameters
    ----------
    trans_data : list of tuple
        (theta, A(theta)) pairs.
    spline_func : CubicSpline
        Fitted spline for A(θ).
    """
    theta_vals, A_vals = zip(*trans_data)
    theta_grid = np.linspace(min(theta_vals), max(theta_vals), 200)
    A_fitted = spline_func(theta_grid)

    plt.figure(figsize=(7, 5))
    plt.plot(np.degrees(theta_vals), A_vals, "o", label="Computed (raw)", markersize=5)
    plt.plot(
        np.degrees(theta_grid), A_fitted, "-", label="Cubic Spline Fit", linewidth=2
    )
    plt.xlabel(r"$\theta$ (degrees)")
    plt.ylabel(r"$A(\theta)$ (Transmission coefficient)")
    plt.title("Transmission Coefficient Fit (Cubic Spline)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_corrected_data(df: pd.DataFrame):
    """
    Plot original and absorption-corrected scattering intensities vs q.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing columns 'q_nm^-1', 'intensity', 'corrected_intensity'.
    """
    q_vals = df["q_nm^-1"].to_numpy()
    original = df["intensity"].to_numpy()
    corrected = df["corrected_intensity"].to_numpy()

    plt.figure(figsize=(7, 5))
    plt.plot(q_vals, original, label="Original", alpha=0.6)
    plt.plot(q_vals, corrected, label="Corrected", linewidth=2)
    plt.xlabel(r"$q$ (nm$^{-1}$)")
    plt.ylabel("Intensity")
    plt.title("Scattering Intensity Before and After Absorption Correction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Saving transmission data (spline + raw)
# ---------------------------------------------------------------------------

def save_transmission_spline(
    spline_func: CubicSpline,
    theta_range,
    output_folder: str = "spline_output",
    filename: str = "transmission_spline_wwater_15d2024.csv",
):
    os.makedirs(output_folder, exist_ok=True)

    theta_grid = np.linspace(theta_range[0], theta_range[1], 500)
    A_interp = spline_func(theta_grid)

    df_spline = pd.DataFrame(
        {
            "theta_deg": np.degrees(theta_grid),
            "A(theta)": A_interp,
        }
    )

    output_path = os.path.join(output_folder, filename)
    df_spline.to_csv(output_path, index=False)
    print(f" Spline data saved to: {output_path}")


def save_transmission_raw_data(
    trans_data,
    output_folder: str = "spline_output",
    filename: str = "transmission_raw_wwater_15d2024.csv",
):
    os.makedirs(output_folder, exist_ok=True)

    theta_vals, A_vals = zip(*trans_data)
    df_raw = pd.DataFrame(
        {
            "theta_deg": np.degrees(theta_vals),
            "A(theta)": A_vals,
        }
    )

    output_path = os.path.join(output_folder, filename)
    df_raw.to_csv(output_path, index=False)
    print(f" Raw transmission data saved to: {output_path}")


# ---------------------------------------------------------------------------
# Absorption correction entry point
# ---------------------------------------------------------------------------

def apply_absorption_correction(
    df: pd.DataFrame,
    I0_y_func,
    I0_z_func,
    R: float = 0.47,
    mu: float = 0.5,
    a: float = 0.1,
    b: float = 12.0,
    t: float = 0.03,
    mu_w: float = 8.0,
    wavelength: float = 0.15406,
    d: float = 300.0,
    n_pts: int = 10,
) -> pd.DataFrame:
    """
    Apply absorption correction to a scattering dataset.
    --------
    1. Compute beam-volume normalization integral.
    2. Compute A(θ) on a grid of θ values (here 0–60 degrees).
    3. Fit a cubic spline A(θ).
    4. For each q in the input DataFrame:
       - q → θ
       - Evaluate A(θ) via spline.
       - Correct intensity: I_corr = I / A(θ).
    5. Store corrected intensities in new column 'corrected_intensity'.

    Parameters
    ----------
    df : pandas.DataFrame
        Input scattering data with columns 'q_nm^-1', 'intensity', 'sigma'.
    I0_y_func : callable
        Beam intensity profile in y (torch-based interpolator).
    I0_z_func : callable
        Beam intensity profile in z (torch-based interpolator).
    R, mu, a, b, t, mu_w, wavelength, d : see above.
    n_pts : int, optional
        Integration resolution for TorchQuad.

    Returns
    -------
    pandas.DataFrame
        Copy of `df` with an additional column 'corrected_intensity'.
    """

    norm = compute_beam_volume_normalization(I0_y_func, I0_z_func, R, t, a, b, n_pts)
    print("normalization factor sample = ", norm)

    theta_vals = np.linspace(0, np.radians(60), 13)
    trans_data = compute_transmission_vs_theta(
        R,
        mu,
        a,
        b,
        t,
        mu_w,
        I0_y_func,
        I0_z_func,
        norm,
        theta_vals,
        wavelength,
        d,
        n_pts,
    )

    spline_func = fit_transmission_spline(trans_data)

    plot_transmission_spline(trans_data, spline_func)
    save_transmission_spline(spline_func, (min(theta_vals), max(theta_vals)))
    save_transmission_raw_data(trans_data)

    corrected_intensities = []
    for _, row in df.iterrows():
        q = row["q_nm^-1"]

        theta = 2 * np.arcsin(np.clip(q * wavelength / (4 * np.pi), -1, 1))

        A_interp = evaluate_transmission_spline(theta, spline_func)
        corrected = row["intensity"] / A_interp if A_interp > 0 else 0.0
        corrected_intensities.append(corrected)

    df = df.copy()
    df["corrected_intensity"] = corrected_intensities
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1) Load raw scattering data
    df = read_scattering_file("HxOH_bn.bin")

    # 2) Define experimental / geometric parameters
    R, mu, a, b, t, mu_w = 0.34, 0.406, 0.12, 12.0, 0.01, 8.099
    wavelength = 0.15406  # nm
    d = 267.0             # sample-to-detector distance (same as used above)
    n_pts = 2**27         # large integration resolution – adjust as needed

    # 3) Load beam profiles along y and z
    I0_y_func = load_profile("width_profile.csv", symmetric=False)
    I0_z_func = load_profile("length_profile.csv", symmetric=True)

    # 4) Apply absorption correction
    df_corrected = apply_absorption_correction(
        df,
        I0_y_func,
        I0_z_func,
        R,
        mu,
        a,
        b,
        t,
        mu_w,
        wavelength,
        d,
        n_pts,
    )

    # 5) Save corrected data to CSV
    df_corrected.to_csv("HxOH_bn.csv", index=False)
    print("Corrected data saved to 'HxOH_bn.csv'")

    # 6) Plot original vs corrected intensity
    plot_corrected_data(df_corrected)
