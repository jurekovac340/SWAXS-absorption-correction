
# 3D Absorption Correction for X-ray Scattering in Cylindrical Capillaries

This repository contains the Python code used to perform 3D absorption corrections for
small-angle / wide-angle X-ray scattering measured in cylindrical capillaries. The method
explicitly integrates the beam intensity distribution and attenuation through the
capillary (sample + wall, or wall only) using **PyTorch** and **TorchQuad**.

The code is intended to accompany the research paper:

> **[Paper Title]**  
> [Author list]  
> [Journal / preprint, year]

---

## Repository Structure

- `corr_3d_torch.py`  
  3D absorption correction for a **filled capillary** (sample + glass wall).

- `corr_3d_cap_torch.py`  
  3D absorption correction for an **empty capillary** (wall-only correction).

- `corr.py`  
  Command-line interface that wraps the two modules and exposes a simple
  `--mode sample` / `--mode capillary` switch, plus options to override
  geometric and physical parameters.

- `width_profile.csv`  
  Example vertical (y) beam profile.

- `length_profile.csv`  
  Example horizontal/longitudinal (z) beam profile.

- `requirements.txt`  
  Python dependencies.

---

## Method Overview

For each scattering vector \( q \), the algorithm:

1. Loads measured scattering data \( I(q) \) with uncertainties.
2. Loads 1D beam profiles in the vertical (y) and longitudinal (z) directions,
   and constructs fast interpolators (via `torch.bucketize`).
3. Computes a beam-weighted normalization integral over the illuminated
   volume of the capillary (sample region or wall region).
4. For a grid of scattering angles \( \theta \), evaluates the transmission /
   attenuation factor \( A(\theta) \) using 3D numerical integration with
   **TorchQuad**.
5. Fits a cubic spline to \( A(\theta) \).
6. Converts each \( q \) to \( \theta \) and corrects the intensity via
   \[
   I_{\text{corr}}(q) = \frac{I(q)}{A(\theta(q))}
   \]
   (with appropriate handling of errors/uncertainties).

The output is a corrected intensity curve that can be directly used in
subsequent data analysis.

---

## Installation

The recommended way to set up the environment is with **conda** (Anaconda or Miniconda).

### 1. Create and activate a Conda environment

Create a fresh environment (name it as you like, here: `saxscorr`):

```bash
conda create -n saxscorr python=3.11
conda activate saxscorr
```

You can also use Python 3.10 if preferred:

```bash
conda create -n saxscorr python=3.10
conda activate saxscorr
```

### 2. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 3. Install dependencies

Install the main Python dependencies via `pip` **inside** the conda environment:

```bash
pip install -r dependencies.txt
```

> **Notes**
>
> * `torchquad` is currently distributed via `pip`, so `pip` is used even though we are in a conda environment.
> * PyTorch can also be installed from conda if you need a specific CUDA build; see below.

### Optional: GPU-accelerated PyTorch via conda

If you want a GPU-enabled PyTorch from the official conda channels:

```bash
# Example for Linux, CUDA 12.1 – adjust to your system
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Then install torchquad (and other pure-Python dependencies) via pip:
pip install torchquad
```

In that case, you may remove or adjust the `torch` line in `dependencies.txt`
to avoid version conflicts.

### Quick check

Once the environment is active and dependencies are installed, you can run:

```bash
python corr.py --help
```

to see the available command-line options.

---

## Input File Formats


### 2. Beam profiles

Both `width_profile.csv` and `length_profile.csv` are simple CSV files with
**two columns**:

```text
position, intensity
-1.0,     0.01
-0.9,     0.02
...
```

* `width_profile.csv` — vertical (y) profile.
* `length_profile.csv` — longitudinal (z) profile.

For convenience, the CLI assumes:

* `width_profile.csv` is **not** automatically mirrored.
* `length_profile.csv` is treated as **symmetric** around zero.

You can change this behavior directly in `corr_3d_cli.py` if needed.

---

## Command-Line Usage

The main entry point is `corr_3d_cli.py`. It supports two modes:

* `--mode sample`    → sample + wall correction (uses `corr_3d_torch.py`)
* `--mode capillary` → empty-capillary wall-only correction (uses `corr_3d_cap_torch.py`)

General syntax:

```bash
python corr.py \
    --mode {sample|capillary} \
    -i INPUT_FILE \
    -o OUTPUT_FILE \
    [--width-profile WIDTH_PROFILE.csv] \
    [--length-profile LENGTH_PROFILE.csv] \
    [--R RADIUS] \
    [--mu MU_SAMPLE] \
    [--a A_HALF_WIDTH] \
    [--b B_HALF_LENGTH] \
    [--t WALL_THICKNESS] \
    [--mu_w MU_WALL] \
    [--wavelength WAVELENGTH_NM] \
    [--d SAMPLE_DETECTOR_DISTANCE] \
    [--n-pts N_INTEGRATION_POINTS]
```

If a parameter is not specified on the command line, a **mode-specific default**
is used (the defaults are taken from the scripts).

### Example 1: Sample + wall (default parameters)

```bash
python corr.py \
    --mode sample \
    -i HxOH_bn.bin \
    -o HxOH_bn_corr.csv
```

This will:

* read `HxOH_bn.bin` (3-column text file),
* use `width_profile.csv` and `length_profile.csv` in the current directory,
* perform full sample + wall absorption correction,
* write the corrected data to `HxOH_bn_corr.csv`.

### Example 2: Empty capillary (wall-only correction)

```bash
python corr.py \
    --mode capillary \
    -i wecap_07f.bin \
    -o wecap_07f_corr.csv
```

This applies only the capillary-wall correction, appropriate for an empty or
background capillary measurement.

### Example 3: Override geometry and attenuation parameters

```bash
python corr.py \
    --mode sample \
    -i HxOH_bn.bin \
    -o HxOH_bn_R035_mu045.csv \
    --R 0.35 \
    --mu 0.45 \
    --t 0.012 \
    --mu_w 9.0
```

Here, the capillary radius `R`, sample attenuation `mu`, wall thickness `t`,
and wall attenuation `mu_w` are all set explicitly, overriding the defaults.

### Example 4: Reduce integration cost (for quick tests)

```bash
python corr.py \
    --mode capillary \
    -i wecap_07f.bin \
    -o wecap_07f_fast.csv \
    --n-pts 1000000
```

Reducing `--n-pts` speeds up the calculation at the cost of some accuracy.
For final results, use the higher default `n_pts` value.

---

## Using the Modules Directly in Python

You can also call the modules directly from your own Python scripts or notebooks:

```python
import corr_3d_torch as sample_corr
import corr_3d_cap_torch as cap_corr

# 1) Load scattering data
df = sample_corr.read_scattering_file("HxOH_bn.bin")

# 2) Load beam profiles
I0_y = sample_corr.load_profile("width_profile.csv",  symmetric=False)
I0_z = sample_corr.load_profile("length_profile.csv", symmetric=True)

# 3) Apply correction (sample + wall)
df_corr = sample_corr.apply_absorption_correction(
    df,
    I0_y,
    I0_z,
    R=0.34,
    mu=0.406,
    a=0.12,
    b=12.0,
    t=0.01,
    mu_w=8.099,
    wavelength=0.15406,
    d=267.0,
    n_pts=2**27,
)
```

The `cap_corr.apply_absorption_correction` function has the **same signature**,
but implements the empty-capillary wall-only correction.

---

## License

This code is distributed under the ** GPL-3.0 license** (see `LICENSE`).

Please cite the associated paper if you use this code in your own work.

```
