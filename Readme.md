# RamanSim: A Synthetic Raman Spectrum and Mapping Simulator

`RamanSim` is a Python toolkit for generating **synthetic Raman spectroscopy data** for AI model development, algorithm benchmarking, and physical interpretability testing.  
It supports single-spectrum synthesis, temporal series simulation, and spatial mapping (1D/2D) with physically meaningful noise, baseline, and drift models.

---

## 🌟 Key Features

### 🔹 Spectrum Simulation
- Generate single or multiple Raman spectra from arbitrary peak definitions.
- Gaussian, Lorentzian, and pseudo-Voigt line shapes.
- Support for peak-level and groupwise **random jitter** (for temporal drift simulation).
- Configurable **baseline models** (fluorescence, graphite-like, laser leak, etc.).
- Multiple **noise models**:
  - Gaussian, multiplicative, Poisson, 1/f (pink noise), and random spikes.
- Optional **instrumental resolution** convolution (IRF FWHM).
- Returns all components (`signal`, `baseline`, `noise`, `final`) for detailed inspection.

### 🔹 Raman Mapping Simulation
- Simulate **1D line scans** or **2D Raman maps**.
- Define multiple analytes (each as a group of correlated peaks).
- Control spatial abundance distribution via:
  - `random` — uniform random intensity field.
  - `gaussian_clusters` — clustered hot spots (SERS-like).
  - `smooth_field` — continuous correlated field.
- Adds baseline and noise pixel-wise.
- Returns Raman data cubes and coordinate axes for direct visualization.

### 🔹 Analyte Library
- Easily define analyte species and their associated peaks.
- Supports **mixtures** via simple weight dictionaries.
- Enables group-level correlated jitter and composition variability.

### 🔹 Data Export and Streaming
- **In-memory streaming** for direct use in model training:
  - `iter_maps()` yields `(cube, aux)` samples without saving to disk.
- **Disk export** for dataset creation:
  - `save_maps_to_npz()` for compressed per-sample `.npz` files.
  - `save_maps_to_h5()` for large `.h5` datasets (requires `h5py`).

---

## 🧩 Core Classes and APIs

### `Peak`
Represents a single Raman peak.

| Parameter | Type | Description |
|------------|------|-------------|
| `pos` | float | Center position (cm⁻¹) |
| `height` | float | Peak height |
| `fwhm` | float | Full-width at half maximum |
| `shape` | str | `'gaussian'`, `'lorentzian'`, or `'pvoigt'` |
| `eta` | float | Mixing ratio for pseudo-Voigt |
| `group` | str | Optional group tag (for correlated jitter) |

---

### `SpectrumSimulator`
Generates a single Raman spectrum.

```python
y, parts = SpectrumSimulator(seed=42).simulate(
    x,
    peaks,                          # list of Peak objects
    baseline=BaselinePreset.graphite_like(...),
    noise=NoiseConfig(...),
    irf_fwhm=6.0,                   # instrumental broadening
    drift_ppm=10.0,                 # global axis drift
    peak_jitter_std=1.0,            # per-peak random shift
    groupwise_jitter=True,          # correlated jitter for same group
)
```

**Returns**  
`y`: final noisy spectrum  
`parts`: dictionary with intermediate components (`signal`, `baseline`, `final`, etc.)

---

### `MapSimulator`
Simulates 1D or 2D Raman mapping datasets.

```python
cube, aux = MapSimulator(sim).simulate_map2d(
    x,
    analytes=[A, B],             # list of list of peaks
    analyte_weights=[1.0, 0.8],
    cfg=MapConfig(ny=64, nx=64, cluster_mode='gaussian_clusters'),
    baseline=baseline,
    noise=noise,
    irf_fwhm=6.0
)
```

**Returns**
- `cube`: `(ny, nx, n_points)` array (Raman intensity)
- `aux`: dictionary containing:
  - `'fields'`: (A, ny, nx) abundance fields
  - `'unit_spec'`: (A, n_points) analyte spectra
  - `'coords'`: `{'r','x','y','t'}` coordinate arrays

---

### `MapConfig`
Configures spatial mapping simulation.

| Parameter | Type | Description |
|------------|------|-------------|
| `ny`, `nx` | int | Map dimensions |
| `cluster_mode` | str | `'random'`, `'gaussian_clusters'`, `'smooth_field'` |
| `n_clusters` | int | Number of hot spots for cluster mode |
| `cluster_std` | float | Std. deviation (pixels) of each cluster |
| `smooth_sigma` | float | Gaussian blur sigma for smooth mode |
| `variability` | float | Field intensity variation amplitude |
| `seed` | int | Random seed |
| `x_coords`, `y_coords`, `t_coords` | np.ndarray | Optional coordinate arrays |

---

### `AnalyteLibrary` and `build_mixture_from_library`
Define reusable analyte peak sets.

```python
lib = AnalyteLibrary.from_arrays({
    'A': {'positions':[612, 1000], 'heights':[1.0, 0.6]},
    'B': {'positions':[1330, 1580, 2700], 'shape':'lorentzian'}
})
analytes, weights = build_mixture_from_library(lib, {'A':1.0, 'B':0.8})
```

---

### `iter_maps()` – Stream generator
```python
for cube, aux in iter_maps(sim, x, analytes, weights, cfg,
                           baseline=baseline, noise=noise,
                           n_samples=5, seed0=2025):
    # Training loop or inspection
    ...
```
## 🧪 Example Scripts

All examples are located in the `examples/` directory.  
Each can be run independently by setting the `if __name__ == "__main__":` flag.

| File | Description |
|------|--------------|
| `single_spectrum.py` | Generate and visualize one synthetic Raman spectrum |
| `time_series.py` | Simulate a temporal series with baseline drift and jitter |
| `map2d.py` | Generate and visualize a 2D Raman map (pixel images at specific wavenumbers) |
| `iter_stream.py` | Demonstrate in-memory streaming of multiple maps for AI training |

Example (single spectrum):

```bash
python examples/single_spectrum.py
```

Example (2D map):

```bash
python examples/map2d.py
```

---

## 📦 Installation

### Requirements
- Python ≥ 3.8  
- NumPy  
- Matplotlib (for visualization)  
- h5py (optional, for `.h5` export)

### Install
```bash
pip install numpy matplotlib h5py
```

If you plan to package it:
```bash
pip install -e .
```

---

## 🧠 Typical Workflow

1. **Define peaks** for each analyte (`peaks_from_arrays()`).
2. **Generate single spectra** with desired baseline/noise.
3. **Build analyte library** for complex mixtures.
4. **Simulate 1D/2D maps** using `MapSimulator`.
5. **Stream data** into ML models with `iter_maps()`.
6. **Export datasets** using `.npz` or `.h5` when needed.

---

## 🧩 Physically Inspired Effects

| Effect | Parameter | Description |
|---------|------------|-------------|
| Peak position drift | `drift_ppm`, `peak_jitter_std` | Global axis or per-peak random shifts |
| Hotspot clustering | `cluster_mode='gaussian_clusters'` | Creates spatially correlated high-intensity regions |
| Baseline fluorescence | `BaselinePreset.graphite_like()` | Adds broad background curvature |
| Detector noise | `NoiseConfig` | Gaussian + Poisson + 1/f + spikes |
| Instrumental broadening | `irf_fwhm` | Convolutional blur (Gaussian kernel) |

---

## 🧾 Citation

If you use `RamanSim` in academic or industrial research, please cite:

```
RamanSim: A Synthetic Raman Spectrum Generator for Physics-Grounded AI Evaluation
Version 1.0, 2025.
```

---

## 🔧 Future Work

- Time-resolved mapping (3D cubes with evolving spectra)  
- Calibration artifacts and baseline drifts  
- Experimental matching utilities  
- Integration with PyTorch `Dataset` class for direct ML pipelines  

---

**Author:** JJL and GP 

**License:** MIT (free for academic and commercial use)
