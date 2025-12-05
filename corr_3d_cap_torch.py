"""
Wall-only absorption correction for X-ray scattering in a cylindrical capillary.

Overview
--------
This module computes an angular transmission/attenuation correction for scattering
measured from an *empty* or partially filled capillary, focusing on the capillary 
wall contribution.

Conceptually, we:

1. Load a 3D beam-weighted normalization over the annular wall volume
   (between radii R and R + t).
2. For each scattering angle θ (or equivalently q), compute two attenuation
   contributions:
   - `compute_correction_wall_1`: wall segment on the x < 0 side
   - `compute_correction_wall_2`: wall segment on the x > 0 side
   These correspond to the two halves of the cylindrical wall.
3. Combine the two contributions and normalize by the beam-volume normalization
   to obtain A(θ) = transmission factor of the capillary.
4. Fit a cubic spline A(θ) vs θ.
5. For each measured q, convert to θ, evaluate A(θ), and correct the intensity.

Integration is performed using TorchQuad's MonteCarlo integrator on top of PyTorch.
"""

import os
import inspect  # currently unused; kept for potential introspection/debugging

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import torch
from torchquad import Simpson, set_up_backend 

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Torch / TorchQuad backend setup
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    Perform 1D linear interpolation in PyTorch (similar to np.interp).

    Uses torch.bucketize to find the bracketing indices and then performs
    standard linear interpolation. Values outside [xp[0], xp[-1]] are set to 0.

    Parameters
    ----------
    x : torch.Tensor
        Query points where the interpolant is evaluated. Arbitrary shape;
        will be flattened internally.
    xp : torch.Tensor
        1D tensor of sorted x-coordinates (monotonically increasing).
    fp : torch.Tensor
        1D tensor of y-values corresponding to `xp`.

    Returns
    -------
    torch.Tensor
        Interpolated values at `x.flatten()`.
    """
    x = x.flatten()

    # idx[i] is s.t. xp[idx-1] <= x[i] < xp[idx]
    idx = torch.bucketize(x, xp)
    idx = idx.clamp(1, len(xp) - 1)

    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]

    # Linear interpolation
    slope = (y1 - y0) / (x1 - x0 + 1e-8)  # epsilon avoids division by zero
    y = y0 + slope * (x - x0)

    # Simple extrapolation rule: outside the xp range -> 0
    y = torch.where(
        (x < xp[0]) | (x > xp[-1]),
        torch.tensor(0.0, device=x.device),
        y,
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
    Load a 1D beam profile from CSV and return a torch-based interpolator.

    The CSV file is expected to have two comma-separated columns:
    (coordinate, intensity).

    Parameters
    ----------
    filename : str
        Path to the CSV file.
    symmetric : bool, optional
        If True, the profile is mirrored about 0 so that you can use a
        one-sided measurement symmetrically.
    device : str, optional
        Device where coordinate and intensity tensors are stored.

    Returns
    -------
    callable
        A function `profile_fn(x)` where `x` is a torch.Tensor of positions
        and the return value is the interpolated intensity.
    """
    data = np.loadtxt(filename, delimiter=",")
    coord, intensity = data[:, 0], data[:, 1]

    if symmetric:
        # Mirror profile about 0 (negatives added in reverse order)
        coord = np.concatenate((-coord[::-1], coord))
        intensity = np.concatenate((intensity[::-1], intensity))

    coord_tensor = torch.tensor(coord, dtype=torch.float32, device=device)
    intensity_tensor = torch.tensor(intensity, dtype=torch.float32, device=device)

    def profile_fn(x: torch.Tensor) -> torch.Tensor:
        """
        Interpolate intensity at positions x using torch-based interpolation.
        """
        return interp1d_torch(x, coord_tensor, intensity_tensor)

    return profile_fn


# ---------------------------------------------------------------------------
# Beam-volume normalization over annular capillary wall
# ---------------------------------------------------------------------------

def compute_beam_volume_normalization(
    I0_y_func,
    I0_z_func,
    R: float,
    t: float,
    a: float,
    b: float,
    n_pts: int = 50,
    device: str = "cuda",
) -> float:
    """
    Compute beam-weighted volume normalization over the capillary wall.

    The integration domain is the annular region between radii R and R + t,
    but only on one side (x < 0). At the end we multiply by 2 to account
    for both halves of the cylinder.

    Mapping
    -------
    - u ∈ [0, 1] → y ∈ [-a, a]
    - v ∈ [0, 1] → z ∈ [-b, b]
    - w ∈ [0, 1] → x traverses *radially* between the outer and inner wall
      at fixed y, via:
          sqrt_outer = sqrt((R + t)^2 - y^2)
          sqrt_inner = sqrt(R^2 - y^2)
          x = -((1 - w) * sqrt_outer + w * sqrt_inner)   (x < 0 side)

      This parameterization covers the wall thickness along x, from outer to inner.

    The Jacobian J accounts for:
        dy/d(u) = 2a
        dz/d(v) = 2b
        dx/d(w) = -(sqrt_inner - sqrt_outer) = sqrt_outer - sqrt_inner

    So:
        J = 4 a b (sqrt_outer - sqrt_inner)

    Parameters
    ----------
    I0_y_func : callable
        Beam profile in y: torch.Tensor → torch.Tensor.
    I0_z_func : callable
        Beam profile in z: torch.Tensor → torch.Tensor.
    R : float
        Inner radius of the capillary.
    t : float
        Wall thickness.
    a : float
        Half-width of the illuminated region in y.
    b : float
        Half-height (or half-length) of the illuminated region in z.
    n_pts : int, optional
        Number of samples for Monte Carlo integration.
    device : str, optional
        Device for torch operations.

    Returns
    -------
    float
        Beam-weighted volume integral over the *entire* wall (both sides).
    """

    def f(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Factorized beam intensity at (x, y, z) = I_y(y) * I_z(z).
        """
        I_y = I0_y_func(y)
        I_z = I0_z_func(z)
        return I_y * I_z

    def transformed_integrand(uvw: torch.Tensor) -> torch.Tensor:
        """
        Integrand expressed in unit cube (u, v, w) coordinates.

        uvw : (N, 3) tensor with each component in [0, 1].
        """
        uvw = uvw.to(device)
        u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]

        # Map u, v to y, z
        y = 2 * a * u - a
        z = 2 * b * v - b

        # x between outer and inner radius at given y (x < 0 side)
        sqrt_outer = torch.sqrt((R + t) ** 2 - y**2)
        sqrt_inner = torch.sqrt(R**2 - y**2)
        x = -((1 - w) * sqrt_outer + w * sqrt_inner)

        # Jacobian determinant for transformation (u, v, w) → (x, y, z)
        jacobian = 4 * a * b * (sqrt_outer - sqrt_inner)

        return f(x, y, z) * jacobian

    integrator = Simpson()

    with torch.no_grad():
        result = integrator.integrate(
            transformed_integrand,
            dim=3,
            N=n_pts,
            integration_domain=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        )

    # Multiply by 2 to account for symmetric x > 0 half of the wall
    return 2 * result.item()


# ---------------------------------------------------------------------------
# Data reader for scattering file
# ---------------------------------------------------------------------------

def read_scattering_file(file_path: str) -> pd.DataFrame:
    """
    Read a plain-text scattering file with three columns:
    q_nm^-1, intensity, sigma.

    Lines that cannot be parsed as three floats are ignored.

    Parameters
    ----------
    file_path : str
        Path to the file.

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
                    continue

    return pd.DataFrame(clean_rows, columns=["q_nm^-1", "intensity", "sigma"])


# ---------------------------------------------------------------------------
# Wall attenuation: two halves of the annular wall
# ---------------------------------------------------------------------------

def compute_correction_wall_1(
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
) -> float:
    """
    Compute the wall attenuation contribution for the x < 0 half of the capillary.

    Geometry
    --------
    - We consider rays from a scattering point (x, y, z) to the detector at
      (d*cosθ, d*sinθ, 0).
    - We compute intersections of the ray with inner and outer radii (R, R + t).
    - Path lengths:
        l1_inc : "incidental" direct-through-sample path along x when the beam 
                 passes through the central region.
        l1_dif : additional in-sample path length inferred from quadratic roots
                 when D1 > 0 and roots are positive.
        l2     : path length to outer surface (wall + sample).

      The total sample and wall path lengths are:
        L_sample = l1_dif + l1_inc
        L_wall   = l2 - L_sample

    Attenuation factor at each point:
        exp(-mu * L_sample - mu_w * L_wall) * I_y(y) * I_z(z)

    Parameters
    ----------
    q, R, a, b, mu, t, mu_w, I0_y_func, I0_z_func, norm : see module docstring.
    wavelength : float, optional
        X-ray wavelength.
    d : float, optional
        Sample-detector distance.
    n_pts : int, optional
        Number of Monte Carlo samples.

    Returns
    -------
    float
        Monte Carlo estimate of the attenuation integral for wall_1.
    """
    # Convert q → scattering angle θ (radians)
    theta = 2 * np.arcsin(np.clip(q * wavelength / (4 * np.pi), -1, 1))
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    def f(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Attenuation integrand at (x, y, z) for the x < 0 wall.
        """
        I_y = I0_y_func(y)
        I_z = I0_z_func(z)
        sqrt = torch.sqrt

        # Ray direction to detector
        Dx = d * cos_theta
        Dy = d * sin_theta

        # Quadratic coefficients for intersection with cylindrical shells
        A = (Dx - x) ** 2 + (Dy - y) ** 2
        B = 2 * (x * (Dx - x) + y * (Dy - y))
        C1 = x**2 + y**2 - R**2
        C2 = x**2 + y**2 - (R + t) ** 2

        D1 = B**2 - 4 * A * C1
        D2 = B**2 - 4 * A * C2

        # Outer radius intersection
        D2_clipped = torch.clamp(D2, min=1e-10)
        t2 = (-B + sqrt(D2_clipped)) / (2 * A)
        length_scale = sqrt((Dx - x) ** 2 + (Dy - y) ** 2 + z**2)
        l2 = t2 * length_scale + x + sqrt((R + t) ** 2 - y**2)

        # Incidental direct-through-sample path in central region:
        # only where |y| < R and x > 0
        mask_l1_inc = (torch.abs(y) < R) & (x > 0)
        l1_inc = torch.zeros_like(x)
        l1_inc[mask_l1_inc] = 2 * sqrt(R**2 - y[mask_l1_inc] ** 2)

        # Inner radius intersections
        D1_clipped = torch.clamp(D1, min=1e-10)
        t1p = (-B + sqrt(D1_clipped)) / (2 * A)
        t1m = (-B - sqrt(D1_clipped)) / (2 * A)

        # Valid diffusive contribution where discriminant is positive
        # and both roots are > 0
        valid_t1 = (D1 > 0) & (t1p > 0) & (t1m > 0)

        l1_dif = torch.zeros_like(x)
        l1_dif[valid_t1] = (t1p[valid_t1] - t1m[valid_t1]) * length_scale[valid_t1]

        # Total sample and wall path lengths
        L_sample = l1_dif + l1_inc
        L_wall = l2 - L_sample

        attenuation = torch.exp(-mu * L_sample - mu_w * L_wall) * I_y * I_z
        return attenuation

    def transformed_integrand(uvw: torch.Tensor) -> torch.Tensor:
        """
        Integrand in unit cube coordinates for x < 0 half of the wall.
        """
        uvw = uvw.to(device)
        u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]

        # Map to physical y, z
        y = 2 * a * u - a
        z = 2 * b * v - b

        # Radii at given y
        sqrt_outer = torch.sqrt((R + t) ** 2 - y**2)
        sqrt_inner = torch.sqrt(R**2 - y**2)

        # x between outer and inner radius, negative side
        x = -((1 - w) * sqrt_outer + w * sqrt_inner)

        # Jacobian determinant
        jacobian = 4 * a * b * (sqrt_outer - sqrt_inner)

        return f(x, y, z) * jacobian

    integrator = Simpson()

    with torch.no_grad():
        result = integrator.integrate(
            transformed_integrand,
            dim=3,
            N=n_pts,
            integration_domain=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        )

    return result.item()


def compute_correction_wall_2(
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
) -> float:
    """
    Compute the wall attenuation contribution for the x > 0 half of the capillary.

    This is analogous to `compute_correction_wall_1`, but with the mapping
    for x set to the positive side:

        x = (1 - w) * sqrt_inner + w * sqrt_outer  (x > 0)

    Parameters and returns are the same as for `compute_correction_wall_1`.
    """
    theta = 2 * np.arcsin(np.clip(q * wavelength / (4 * np.pi), -1, 1))
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    def f(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Attenuation integrand at (x, y, z) for the x > 0 wall.
        """
        I_y = I0_y_func(y)
        I_z = I0_z_func(z)
        sqrt = torch.sqrt

        Dx = d * cos_theta
        Dy = d * sin_theta

        A = (Dx - x) ** 2 + (Dy - y) ** 2
        B = 2 * (x * (Dx - x) + y * (Dy - y))
        C1 = x**2 + y**2 - R**2
        C2 = x**2 + y**2 - (R + t) ** 2

        D1 = B**2 - 4 * A * C1
        D2 = B**2 - 4 * A * C2

        D2_clipped = torch.clamp(D2, min=1e-10)
        t2 = (-B + sqrt(D2_clipped)) / (2 * A)
        length_scale = sqrt((Dx - x) ** 2 + (Dy - y) ** 2 + z**2)
        l2 = t2 * length_scale + x + sqrt((R + t) ** 2 - y**2)

        # Incidental sample path
        mask_l1_inc = (torch.abs(y) < R) & (x > 0)
        l1_inc = torch.zeros_like(x)
        l1_inc[mask_l1_inc] = 2 * sqrt(R**2 - y[mask_l1_inc] ** 2)

        # Inner radius intersection
        D1_clipped = torch.clamp(D1, min=1e-10)
        t1p = (-B + sqrt(D1_clipped)) / (2 * A)
        t1m = (-B - sqrt(D1_clipped)) / (2 * A)

        valid_t1 = (D1 > 0) & (t1p > 0) & (t1m > 0)
        l1_dif = torch.zeros_like(x)
        l1_dif[valid_t1] = (t1p[valid_t1] - t1m[valid_t1]) * length_scale[valid_t1]

        L_sample = l1_dif + l1_inc
        L_wall = l2 - L_sample

        attenuation = torch.exp(-mu * L_sample - mu_w * L_wall) * I_y * I_z
        return attenuation

    def transformed_integrand(uvw: torch.Tensor) -> torch.Tensor:
        """
        Integrand in unit cube coordinates for the x > 0 half of the wall.
        """
        uvw = uvw.to(device)
        u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]

        y = 2 * a * u - a
        z = 2 * b * v - b

        sqrt_inner = torch.sqrt(R**2 - y**2)
        sqrt_outer = torch.sqrt((R + t) ** 2 - y**2)

        # x from inner to outer, positive side
        x = (1 - w) * sqrt_inner + w * sqrt_outer
        jacobian = 4 * a * b * (sqrt_outer - sqrt_inner)

        return f(x, y, z) * jacobian

    integrator = Simpson()

    with torch.no_grad():
        result = integrator.integrate(
            transformed_integrand,
            dim=3,
            N=n_pts,
            integration_domain=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        )

    return result.item()


# ---------------------------------------------------------------------------
# Transmission vs θ: tabulation and spline fit
# ---------------------------------------------------------------------------

def _compute_A_for_theta(args):
    """
    Internal helper: compute normalized wall attenuation A(θ) for a single angle.

    Parameters
    ----------
    args : tuple
        (theta, R, mu, a, b, t, mu_w, I0_y_func, I0_z_func, norm, wavelength, d, n_pts)

    Returns
    -------
    tuple
        (theta, A(theta)), with A(theta) = (A1 + A2) / norm.
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

    # θ → q (used for the correction functions)
    q = (4 * np.pi / wavelength) * np.sin(theta / 2)

    A1 = compute_correction_wall_1(
        q, R, a, b, mu, t, mu_w, I0_y_func, I0_z_func, norm, wavelength, d, n_pts
    )
    A2 = compute_correction_wall_2(
        q, R, a, b, mu, t, mu_w, I0_y_func, I0_z_func, norm, wavelength, d, n_pts
    )

    A = (A1 + A2) / norm
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
    Compute the normalized wall transmission A(θ) on a grid of θ values.

    Parameters
    ----------
    R, mu, a, b, t, mu_w, I0_y_func, I0_z_func, norm : see above.
    theta_vals : np.ndarray
        Array of angles (radians) where A(θ) should be computed.
    wavelength : float, optional
        X-ray wavelength.
    d : float, optional
        Sample-detector distance.
    n_pts : int, optional
        Number of Monte Carlo samples for each θ.

    Returns
    -------
    list of (float, float)
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
        print(f"theta = {np.degrees(theta_val):.2f}° -> A = {A_val:.6f}")
        results.append((theta_val, A_val))

    return results


def fit_transmission_spline(trans_data):
    """
    Fit a cubic spline to transmission data A(θ).

    Parameters
    ----------
    trans_data : list of tuple
        (theta, A(theta)) pairs.

    Returns
    -------
    scipy.interpolate.CubicSpline
        Spline interpolant A(θ).
    """
    theta_vals, A_vals = zip(*trans_data)
    return CubicSpline(theta_vals, A_vals, bc_type="natural")


def evaluate_transmission_spline(theta, spline_func: CubicSpline):
    """
    Evaluate the cubic spline interpolated transmission A(θ).

    Parameters
    ----------
    theta : float or np.ndarray
        Scattering angle(s) in radians.
    spline_func : CubicSpline
        Spline fitted by `fit_transmission_spline`.

    Returns
    -------
    float or np.ndarray
        Interpolated A(θ).
    """
    return spline_func(theta)


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_transmission_fit(trans_data, coeffs):
    """
    Plot raw transmission data and a polynomial fit A(θ) (if available).

    Note
    ----
    This routine expects a helper function `evaluate_transmission(theta, coeffs)`
    to exist in the same namespace, which evaluates the polynomial at θ.
    """
    theta_vals, A_vals = zip(*trans_data)
    theta_grid = np.linspace(min(theta_vals), max(theta_vals), 200)
    A_fitted = [evaluate_transmission(theta, coeffs) for theta in theta_grid]

    plt.figure(figsize=(7, 5))
    plt.plot(
        np.degrees(theta_vals), A_vals, "o", label="Computed (raw)", markersize=5
    )
    plt.plot(
        np.degrees(theta_grid),
        A_fitted,
        "-",
        label="Fitted polynomial",
        linewidth=2,
    )
    plt.xlabel(r"$\theta$ (degrees)")
    plt.ylabel(r"$A(\theta)$ (Total attenuation)")
    plt.title("Transmission Correction Fit")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_transmission_spline(trans_data, spline_func):
    """
    Plot raw A(θ) and cubic spline A_spline(θ) vs θ (degrees).
    """
    theta_vals, A_vals = zip(*trans_data)
    theta_grid = np.linspace(min(theta_vals), max(theta_vals), 200)
    A_fitted = spline_func(theta_grid)

    plt.figure(figsize=(7, 5))
    plt.plot(np.degrees(theta_vals), A_vals, "o", label="Computed (raw)", markersize=5)
    plt.plot(
        np.degrees(theta_grid),
        A_fitted,
        "-",
        label="Cubic Spline Fit",
        linewidth=2,
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
    Plot original and corrected intensities vs q.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'q_nm^-1', 'intensity', 'corrected_intensity'.
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
# Saving transmission data
# ---------------------------------------------------------------------------

def save_transmission_spline(
    spline_func: CubicSpline,
    theta_range,
    output_folder: str = "spline_output",
    filename: str = "transmission_spline_wecap.csv",
):
    """
    Save cubic-spline transmission A(θ) to CSV on a dense θ grid.

    Parameters
    ----------
    spline_func : CubicSpline
        Spline of A(θ).
    theta_range : tuple
        (theta_min, theta_max) in radians.
    output_folder : str, optional
        Destination folder (created if needed).
    filename : str, optional
        Output CSV name.
    """
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
    print(f"📁 Spline data saved to: {output_path}")


def save_transmission_raw_data(
    trans_data,
    output_folder: str = "spline_output",
    filename: str = "transmission_raw_wecap.csv",
):
    """
    Save raw (theta, A(theta)) transmission data to CSV.

    Parameters
    ----------
    trans_data : list of tuple
        (theta, A(theta)) pairs.
    output_folder : str, optional
        Destination folder.
    filename : str, optional
        CSV filename.
    """
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
    print(f"📁 Raw transmission data saved to: {output_path}")


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
    Apply capillary-wall absorption correction to a scattering dataset.

    Steps
    -----
    1. Compute beam-volume normalization over the (full) wall.
    2. Compute A(θ) on θ-grid from 0 to 180 degrees.
    3. Fit cubic spline A(θ).
    4. For each q in df:
        - convert q → θ
        - interpolate A(θ)
        - compute corrected intensity I_corr = I / A(θ).

    Parameters
    ----------
    df : pandas.DataFrame
        Input data with columns 'q_nm^-1', 'intensity', 'sigma'.
    I0_y_func, I0_z_func : callable
        Beam profiles in y and z (torch-based interpolators).
    R, mu, a, b, t, mu_w, wavelength, d : see module docstring.
    n_pts : int, optional
        Monte Carlo sample count for each θ.

    Returns
    -------
    pandas.DataFrame
        Copy of df with new column 'corrected_intensity'.
    """
    # 1) Normalization over wall volume
    norm = compute_beam_volume_normalization(I0_y_func, I0_z_func, R, t, a, b, n_pts)
    print("normalization factor sample = ", norm)

    # 2) θ-grid from 0 to 180 degrees (inclusive)
    theta_vals = np.linspace(0, np.radians(180), 37)

    # 3) Compute A(θ) and fit spline
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

    # Diagnostics: visualize and save A(θ)
    plot_transmission_spline(trans_data, spline_func)
    save_transmission_spline(spline_func, (min(theta_vals), max(theta_vals)))
    save_transmission_raw_data(trans_data)

    # 4) Apply correction per data point
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
# Main script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1) Load scattering data
    df = read_scattering_file("wecap_07f.bin")

    # 2) Parameters for this experiment
    R, mu, a, b, t, mu_w = 0.34, 0.0, 0.1, 12.0, 0.01, 8.099
    wavelength = 0.15406
    d = 267.0

    # Number of Monte Carlo samples per θ
    n_pts = 2**27

    # 3) Load beam profiles (y: width, z: length)
    I0_y_func = load_profile("width_profile.csv", symmetric=False)
    I0_z_func = load_profile("length_profile.csv", symmetric=True)

    # 4) Apply correction
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
    df_corrected.to_csv("cap.csv", index=False)
    print("✅ Corrected data saved to 'cap.csv'")

    # 5) Visualize corrected vs original
    plot_corrected_data(df_corrected)
