# ramansim — Synthetic Raman Spectrum & Mapping Simulator

`ramansim` is a lightweight Python toolkit to generate **synthetic Raman spectroscopy data** for AI model testing, benchmarking, and physics‑grounded analysis. It supports single‑spectrum synthesis, 1D/2D spatial maps, and time series with realistic baselines, noise, drift/jitter, validity masks, and statistical variability.

---

## 🌟 What’s inside (current version)

- **Peak modeling**
  - `Peak(pos, height, fwhm, shape, eta, group)` with Gaussian / Lorentzian / pseudo‑Voigt (`'pvoigt'`) line shapes.
  - Optional `group` tag to enable **correlated jitter** among peaks of the same analyte.

- **Single‑spectrum simulation** — `SpectrumSimulator.simulate(...)`
  - Global axis drift (`drift_ppm`) and **per‑peak jitter** (`peak_jitter_std`), optionally **group‑wise**.
  - Baseline via user‑supplied callable or presets in **`BaselinePreset`**:
    - `polynomial(coeffs)`, `fluorescence_rise(...)`, `laser_leak_slope(...)`, `graphite_like(...)`.
  - **Noise models** via `NoiseConfig`:
    - Additive Gaussian, Poisson/shot noise, **1/f** noise, multiplicative noise, and random **cosmic‑ray spikes**.
  - Optional **instrumental broadening** using Gaussian IRF (`irf_fwhm`).
  - Returns the final spectrum and (optionally) a dict of components: axis, drifted axis, signal, baseline, convolved, noise, final.

- **Mapping (1D/2D)** — `MapSimulator.simulate_map2d(...)` / `simulate_map1d(...)`
  - Build pixel‑wise maps from **analyte peak sets** and **spatial abundance fields**.
  - Field generators via `MapConfig` with `cluster_mode in {'random','gaussian_clusters','smooth_field'}`.
  - Per‑pixel baseline & noise, optional IRF, and global drift.
  - Returns data cube `(ny, nx, n_points)` plus auxiliaries (fields, unit spectra, coordinates).

- **Validity masks & sparse signals**
  - `simulate_map2d_with_validity(...)`: compose signal+baseline+noise for valid pixels and background‑only for invalid ones.
  - `ValidityMaskConfig` supports `independent`, `gaussian_clusters`, and `smooth_field` modes (also 1D helper).

- **Time series**
  - `simulate_time_series(...)`: generate `N` frames with optional per‑frame **validity**, **height variation**, **baseline schedule**, and **drift schedule**.

- **Statistical generators**
  - **Uncorrelated heights**: `VariationSpec` + `generate_spectra_n(...)` (supports std/range/rel_range/cv, with auto/normal/uniform/lognormal).
  - **Correlated heights**: `CorrelatedHeightSpec` + `generate_spectra_n_correlated(...)` for multivariate normal/lognormal peak heights.

- **Analyte library**
  - `AnalyteLibrary.from_arrays(...)` and `build_mixture_from_library(...)` for reusable analyte definitions; auto‑assigns `group` for correlated jitter.

- **Anomalies & domain effects (optional)**
  - `AnomalyConfig` + helpers: inject **spurious peaks**, apply **axis warp** (quadratic), **saturation**, and **heteroscedastic Gaussian** noise.
  - Map‑level anomalies: `simulate_map2d_with_validity_and_anomalies(...)` (e.g., **bad pixels** → NaNs).
  - Label noise & domain shift helpers: `corrupt_labels(...)`, `make_domain_shift_configs(...)`.

- **I/O & utilities**
  - `save_csv(...)`, `save_maps_to_npz(...)`, `save_maps_to_h5(...)`.
  - Axis/coordinate helpers: `make_axis(...)`, `make_linear_coords(...)`.

---

## 🧩 Quick start

```python
from ramansim import SpectrumSimulator, Peak, BaselinePreset, NoiseConfig
import numpy as np

x = np.linspace(200, 2000, 4000)  # wavenumber axis (cm^-1)

peaks = [
    Peak(pos=612,  height=1.0, fwhm=10, shape='gaussian'),
    Peak(pos=1000, height=0.6, fwhm=14, shape='lorentzian'),
    Peak(pos=1580, height=0.9, fwhm=25, shape='pvoigt', eta=0.3),
]

noise = NoiseConfig(
    gaussian_sigma=0.02,
    poisson_scale=1500,
    one_over_f_strength=0.005,
    spike_rate=0.0005, spike_height=(3, 8),
    multiplicative_sigma=0.01,
)

sim = SpectrumSimulator(seed=42)
y, parts = sim.simulate(
    x,
    peaks=peaks,
    baseline=BaselinePreset.graphite_like(intensity=0.4, center=1580, fwhm=250, offset=0.02, slope=2e-4),
    noise=noise,
    irf_fwhm=6.0,
    drift_ppm=10.0,            # global ppm drift
    peak_jitter_std=1.0,       # per-peak random jitter (cm^-1 scale in axis units)
    groupwise_jitter=True,     # correlated jitter for peaks with same `group`
)
# y is the final spectrum; parts contains components for inspection
```

---

## 🗺️ 2D Raman mapping

```python
from ramansim import MapSimulator, MapConfig, peaks_from_arrays, SpectrumSimulator, BaselinePreset, NoiseConfig

x = np.linspace(200, 2000, 2400)
sim = SpectrumSimulator(seed=123)
mapsim = MapSimulator(sim)

# Two analytes, each as a list of Peak
A = peaks_from_arrays([612, 1000], heights=[1.0, 0.6], fwhms=[10, 14], shape='gaussian')
B = peaks_from_arrays([1330, 1580, 2700], fwhms=[18, 25, 40], shape='lorentzian')
for p in A: p.group = 'A'
for p in B: p.group = 'B'

cfg = MapConfig(ny=64, nx=64, cluster_mode='gaussian_clusters', n_clusters=4, cluster_std=5.0, variability=0.4, seed=7)

cube, aux = mapsim.simulate_map2d(
    x,
    analytes=[A, B],
    analyte_weights=[1.0, 0.8],
    cfg=cfg,
    baseline=BaselinePreset.fluorescence_rise(intensity=0.25, scale=900.0, offset=0.01, slope=1e-4),
    noise=NoiseConfig(gaussian_sigma=0.01, poisson_scale=3000),
    irf_fwhm=6.0,
    drift_ppm=5.0,
)
# cube: (ny, nx, n_points), aux: fields/unit_spec/coords
```

---

## ⏱️ Time series with schedules

```python
from ramansim import simulate_time_series, SpectrumSimulator, Peak, VariationSpec, BaselinePreset, NoiseConfig

x = np.linspace(200, 2000, 2400)
peaks = [Peak(612,1.0,10), Peak(1000,0.6,14), Peak(1580,0.9,25,'pvoigt',0.3)]
sim = SpectrumSimulator(seed=1)

baseline_fn   = lambda xx, i, N: BaselinePreset.laser_leak_slope(offset=0.01, slope=1e-4)(xx)
drift_ppm_fn  = lambda i, N: 5.0*np.sin(2*np.pi*i/max(N-1,1))
height_spec   = VariationSpec(kind='cv', value=0.2)

Y, meta = simulate_time_series(
    sim, x, peaks, N=200,
    validity=None,                 # or ValidityMaskConfig(...)
    height_spec=height_spec,       # per-frame peak heights
    baseline_fn=baseline_fn,       # per-frame baseline
    noise=NoiseConfig(gaussian_sigma=0.01, poisson_scale=2000),
    irf_fwhm=6.0,
    drift_ppm_fn=drift_ppm_fn,     # per-frame drift
    peak_jitter_std=0.6,
    groupwise_jitter=True,
    seed0=2025,
)
# Y: (N, n_points), meta includes coords and validity mask
```

---

## 📚 API reference (selected)

### Peak helpers
- `peaks_from_arrays(positions, heights=None, fwhms=None, shape='gaussian', default_fwhm=12.0, eta=0.5) -> List[Peak]`

### Baselines
- `BaselinePreset.polynomial(coeffs)` — highest power first.
- `BaselinePreset.fluorescence_rise(intensity=0.3, scale=800, offset=0.0, slope=0.0)`
- `BaselinePreset.laser_leak_slope(offset=0.0, slope=1e-4)`
- `BaselinePreset.graphite_like(intensity=0.2, center=1580, fwhm=250, offset=0.0, slope=0.0)`

### Noise
```python
NoiseConfig(
  gaussian_sigma=0.0,
  poisson_scale=None,
  one_over_f_strength=0.0,
  spike_rate=0.0,
  spike_height=(3.0, 8.0),
  multiplicative_sigma=0.0,
)
```

### Spectrum simulator
```python
SpectrumSimulator(seed=None).simulate(
  x,
  peaks,
  baseline=None,
  noise=None,
  irf_fwhm=None,
  drift_ppm=0.0,
  peak_jitter_std=0.0,
  groupwise_jitter=False,
  anomalies=None,              # AnomalyConfig, optional post effects (see below)
  return_components=True,
) -> (y, parts)
```

`parts` contains: `axis`, `axis_eff`, `signal`, `baseline`, `convolved`, `noise`, `final`, `coords`.

### Mapping
```python
MapSimulator(SpectrumSimulator).simulate_map2d(
  x,
  analytes: List[List[Peak]],
  analyte_weights: Optional[List[float]] = None,
  per_analyte_fields: Optional[List[np.ndarray]] = None,
  cfg: MapConfig = MapConfig(...),
  baseline=None,
  noise=None,
  irf_fwhm=None,
  drift_ppm=0.0,
) -> (cube, aux)
```
`MapConfig(ny, nx, cluster_mode='random'|'gaussian_clusters'|'smooth_field', n_clusters=3, cluster_std=4.0, smooth_sigma=6.0, base_intensity=0.0, variability=0.3, seed=None, x_coords=None, y_coords=None, t_coords=None)`

### Validity & anomalies
- `simulate_map2d_with_validity(...)` with `ValidityMaskConfig(proportion_valid=0.5, mode='independent'|'gaussian_clusters'|'smooth_field', ...)`  
- `simulate_map2d_with_validity_and_anomalies(...)` with `AnomalyMapConfig(bad_pixel_rate=..., seed=...)`

### Statistical generators
- `VariationSpec(kind, value, clip_min=0.0, distribution='auto')`
- `generate_spectra_n(...)` → `Y, meta`
- `CorrelatedHeightSpec(distribution='mvnormal'|'mvlognormal', mean, cov)`
- `generate_spectra_n_correlated(...)` → `Y, meta`

### Post‑processing & helpers
- `apply_heteroscedastic_gaussian(y, ref, base_sigma, scale=0.0, power=1.0, seed=None)`
- `inject_spurious_peaks(x, y, AnomalyConfig)` / `apply_axis_warp(x, y, quad)` / `apply_saturation(y, level)`
- `corrupt_labels(labels, prob, num_classes, seed=None)`
- `make_domain_shift_configs(map_cfg, valid_cfg, train_delta, test_delta)`

### I/O
- `save_csv(path, x, y, header='wavenumber,intensity')`
- `save_maps_to_npz(path_template, iterator)`
- `save_maps_to_h5(path, iterator, n_samples, ny=None, nx=None, n_points=None, compression='gzip')`

---

## 📦 Installation

**Requirements**: Python ≥ 3.8, NumPy.  
Optional: Matplotlib (viz), `h5py` (HDF5 export).

```bash
pip install numpy matplotlib h5py
# If used as a local package:
pip install -e .
```

---

## 🧠 Typical workflow

1. Define peaks or use `peaks_from_arrays()`.
2. Generate single spectra with desired baseline/noise/jitter/drift.
3. Build analyte library and mixtures.
4. Simulate 1D/2D maps, optionally with validity masks or anomalies.
5. Generate time series with schedules (baseline/drift/height).
6. Stream or export datasets with `iter_maps()`, `.npz`, or `.h5`.

---

## 🧾 Citation
If you use this toolkit in research, please cite:

```
ramansim: A Synthetic Raman Spectrum & Mapping Simulator for Physics‑Grounded AI Evaluation
Version 1.0, 2025.
```

---

**Authors**: JJL & GPT

**License**: MIT
