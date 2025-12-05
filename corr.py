#!/usr/bin/env python3
"""
Unified CLI for 3D absorption correction:
- mode 'sample'    -> uses corr_3d_torch (sample + capillary wall)
- mode 'capillary' -> uses corr_3d_cap_torch (empty capillary, wall only)

Usage examples
--------------
# Full correction (sample + wall), defaults
python corr.py \
    --mode sample \
    -i HxOH_bn.bin \
    -o HxOH_bn_corr.csv

# Empty capillary, custom geometry and mu_w
python corr.py \
    --mode capillary \
    -i wecap_07f.bin \
    -o cap_corr.csv \
    --R 0.35 --a 0.12 --mu_w 9.0
"""

import argparse
import os

import corr_3d_torch as sample_corr
import corr_3d_cap_torch as cap_corr


def parse_args():
    parser = argparse.ArgumentParser(
        description="3D absorption correction in a cylindrical capillary "
                    "(sample+wall or empty-capillary wall-only)."
    )

    parser.add_argument(
        "-m", "--mode",
        choices=["sample", "capillary"],
        required=True,
        help="Correction mode: 'sample' = full sample+wall, "
             "'capillary' = empty capillary (wall only)."
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input scattering file (text, 3 columns: q, intensity, sigma)."
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output CSV file for corrected data."
    )

    parser.add_argument(
        "--width-profile",
        default="width_profile.csv",
        help="CSV with vertical (y) beam profile [default: width_profile.csv]."
    )

    parser.add_argument(
        "--length-profile",
        default="length_profile.csv",
        help="CSV with horizontal/longitudinal (z) beam profile "
             "[default: length_profile.csv]."
    )

    # Geometry / physical parameters: CLI can override, otherwise mode defaults are used.
    parser.add_argument("--R", type=float, default=None,
                        help="Capillary radius [cm?] (default: mode-specific).")
    parser.add_argument("--mu", type=float, default=None,
                        help="Linear attenuation coefficient of sample (default: mode-specific).")
    parser.add_argument("--a", type=float, default=None,
                        help="Beam half-width in y (default: mode-specific).")
    parser.add_argument("--b", type=float, default=None,
                        help="Beam half-length in z (default: mode-specific).")
    parser.add_argument("--t", type=float, default=None,
                        help="Capillary wall thickness (default: mode-specific).")
    parser.add_argument("--mu_w", type=float, default=None,
                        help="Linear attenuation coefficient of wall (default: mode-specific).")
    parser.add_argument("--wavelength", type=float, default=None,
                        help="X-ray wavelength in nm (default: mode-specific).")
    parser.add_argument("--d", type=float, default=None,
                        help="Sample–detector distance (default: mode-specific).")
    parser.add_argument("--n-pts", type=int, default=None,
                        help="Number of integration points for TorchQuad "
                             "(default: mode-specific).")

    return parser.parse_args()


def get_mode_defaults(mode: str):
    """
    Default parameters copied from the __main__ blocks of:
    - corr_3d_torch.py (sample mode)
    - corr_3d_cap_torch.py (capillary mode)
    """
    if mode == "sample":
        # From corr_3d_torch
        return dict(
            R=0.34,
            mu=0.406,
            a=0.12,
            b=12.0,
            t=0.01,
            mu_w=8.099,
            wavelength=0.15406,  # nm
            d=267.0,
            n_pts=2**27,
        )
    elif mode == "capillary":
        # From corr_3d_cap_torch
        return dict(
            R=0.34,
            mu=0.0,            # no sample in capillary
            a=0.1,
            b=12.0,
            t=0.01,
            mu_w=8.099,
            wavelength=0.15406,
            d=267.0,
            n_pts=2**27,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    args = parse_args()

    # choose which physics to use
    if args.mode == "sample":
        mod = sample_corr
    else:
        mod = cap_corr

    # Get defaults and override with CLI values if provided
    defaults = get_mode_defaults(args.mode)

    R = args.R if args.R is not None else defaults["R"]
    mu = args.mu if args.mu is not None else defaults["mu"]
    a = args.a if args.a is not None else defaults["a"]
    b = args.b if args.b is not None else defaults["b"]
    t = args.t if args.t is not None else defaults["t"]
    mu_w = args.mu_w if args.mu_w is not None else defaults["mu_w"]
    wavelength = args.wavelength if args.wavelength is not None else defaults["wavelength"]
    d = args.d if args.d is not None else defaults["d"]
    n_pts = args.n_pts if args.n_pts is not None else defaults["n_pts"]

    print(f"Mode          : {args.mode}")
    print(f"Input file    : {args.input}")
    print(f"Output file   : {args.output}")
    print(f"Width profile : {args.width_profile}")
    print(f"Length profile: {args.length_profile}")
    print("Parameters:")
    print(f"  R          = {R}")
    print(f"  mu         = {mu}")
    print(f"  a          = {a}")
    print(f"  b          = {b}")
    print(f"  t          = {t}")
    print(f"  mu_w       = {mu_w}")
    print(f"  wavelength = {wavelength}")
    print(f"  d          = {d}")
    print(f"  n_pts      = {n_pts}")

    # 1) Load scattering data (both modules expose read_scattering_file)
    df = mod.read_scattering_file(args.input)

    # 2) Load beam profiles
    I0_y_func = mod.load_profile(args.width_profile, symmetric=False)
    I0_z_func = mod.load_profile(args.length_profile, symmetric=True)

    # 3) Apply the corresponding absorption correction
    if args.mode == "sample":
        df_corr = sample_corr.apply_absorption_correction(
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
    else:
        df_corr = cap_corr.apply_absorption_correction(
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

    # 4) Save to CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df_corr.to_csv(args.output, index=False)
    print(f"✅ Corrected data saved to '{args.output}'")


if __name__ == "__main__":
    main()
