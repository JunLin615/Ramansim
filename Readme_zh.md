# ramansim — 合成拉曼光谱与成像模拟器

`ramansim` 是一个轻量级 Python 工具包，用于生成**合成拉曼光谱数据**，便于 AI 模型测试、算法基准与基于物理的可解释性分析。支持单条光谱、1D/2D 空间成像（Raman map）以及时间序列模拟，并提供真实的基线、噪声、漂移/抖动、有效性掩膜与统计变异建模。

---

## 🌟 功能概览（当前版本）

- **峰建模**
  - `Peak(pos, height, fwhm, shape, eta, group)`：支持 Gaussian / Lorentzian / 伪 Voigt（`'pvoigt'`）线型。
  - 可选 `group` 标签：同一组峰可触发**相关抖动**（group-wise jitter）。

- **单谱模拟** — `SpectrumSimulator.simulate(...)`
  - 全局坐标漂移（`drift_ppm`）与**逐峰抖动**（`peak_jitter_std`），可选**按组相关**。
  - 基线由可调用函数或 **`BaselinePreset`** 预设提供：
    - `polynomial(coeffs)`, `fluorescence_rise(...)`, `laser_leak_slope(...)`, `graphite_like(...)`。
  - **噪声模型**（`NoiseConfig`）：
    - 加性高斯、Poisson/散粒噪声、**1/f** 噪声、乘性噪声、随机**宇宙线尖峰**。
  - 可选**仪器展宽**（高斯 IRF，`irf_fwhm`）。
  - 返回最终光谱以及可选组件字典：原轴、漂移后轴、信号、基线、卷积后、噪声、最终结果。

- **成像（1D/2D）** — `MapSimulator.simulate_map2d(...)` / `simulate_map1d(...)`
  - 由**分析物峰集**与**空间丰度场**构成像素级光谱。
  - `MapConfig` 提供场生成模式：`'random' | 'gaussian_clusters' | 'smooth_field'`。
  - 支持像素级基线与噪声、可选 IRF 与全局漂移。
  - 返回数据立方体 `(ny, nx, n_points)` 与辅助信息（丰度场、单位光谱、坐标）。

- **有效性掩膜与稀疏信号**
  - `simulate_map2d_with_validity(...)`：有效像素为信号+基线+噪声；无效像素仅含基线+噪声。
  - `ValidityMaskConfig` 支持 `independent`、`gaussian_clusters`、`smooth_field` 模式（含 1D 版本）。

- **时间序列**
  - `simulate_time_series(...)`：生成 `N` 帧，可指定逐帧**有效性**、**峰高变动**、**基线日程**与**漂移日程**。

- **统计生成器**
  - **非相关峰高**：`VariationSpec` + `generate_spectra_n(...)`（支持 std/range/rel_range/cv，分布 auto/normal/uniform/lognormal）。
  - **相关峰高**：`CorrelatedHeightSpec` + `generate_spectra_n_correlated(...)`，多元正态/对数正态。

- **分析物库**
  - `AnalyteLibrary.from_arrays(...)` 与 `build_mixture_from_library(...)` 复用分析物定义；自动为峰赋 `group` 便于相关抖动。

- **异常与域效应（可选）**
  - `AnomalyConfig` 及辅助函数：注入**伪峰**、应用**坐标轴形变**（二次项）、**饱和**与**异方差高斯**噪声。
  - 图级异常：`simulate_map2d_with_validity_and_anomalies(...)`（如 **坏像素** → NaN）。
  - 标签噪声与域偏移工具：`corrupt_labels(...)`、`make_domain_shift_configs(...)`。

- **I/O 与常用工具**
  - `save_csv(...)`, `save_maps_to_npz(...)`, `save_maps_to_h5(...)`。
  - 坐标轴/坐标系：`make_axis(...)`, `make_linear_coords(...)`。

---

## 🧩 快速上手

```python
from ramansim import SpectrumSimulator, Peak, BaselinePreset, NoiseConfig
import numpy as np

x = np.linspace(200, 2000, 4000)  # 波数轴 (cm^-1)

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
    drift_ppm=10.0,           # 全局 ppm 漂移
    peak_jitter_std=1.0,      # 逐峰抖动（与坐标单位一致）
    groupwise_jitter=True,    # 同组峰相关抖动
)
# y 为最终光谱；parts 含组件便于检查
```

---

## 🗺️ 2D Raman 成像

```python
from ramansim import MapSimulator, MapConfig, peaks_from_arrays, SpectrumSimulator, BaselinePreset, NoiseConfig

x = np.linspace(200, 2000, 2400)
sim = SpectrumSimulator(seed=123)
mapsim = MapSimulator(sim)

# 两个分析物（各自是一组 Peak）
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
# cube: (ny, nx, n_points)；aux: fields / unit_spec / coords
```

---

## ⏱️ 带日程的时间序列

```python
from ramansim import simulate_time_series, SpectrumSimulator, Peak, VariationSpec, BaselinePreset, NoiseConfig
import numpy as np

x = np.linspace(200, 2000, 2400)
peaks = [Peak(612,1.0,10), Peak(1000,0.6,14), Peak(1580,0.9,25,'pvoigt',0.3)]
sim = SpectrumSimulator(seed=1)

baseline_fn  = lambda xx, i, N: BaselinePreset.laser_leak_slope(offset=0.01, slope=1e-4)(xx)
drift_ppm_fn = lambda i, N: 5.0*np.sin(2*np.pi*i/max(N-1,1))
height_spec  = VariationSpec(kind='cv', value=0.2)

Y, meta = simulate_time_series(
    sim, x, peaks, N=200,
    validity=None,                 # 或 ValidityMaskConfig(...)
    height_spec=height_spec,       # 逐帧峰高
    baseline_fn=baseline_fn,       # 逐帧基线
    noise=NoiseConfig(gaussian_sigma=0.01, poisson_scale=2000),
    irf_fwhm=6.0,
    drift_ppm_fn=drift_ppm_fn,     # 逐帧漂移
    peak_jitter_std=0.6,
    groupwise_jitter=True,
    seed0=2025,
)
# Y: (N, n_points)，meta 含坐标与有效性掩膜
```

---

## 📚 主要 API（节选）

### 峰与辅助
- `peaks_from_arrays(positions, heights=None, fwhms=None, shape='gaussian', default_fwhm=12.0, eta=0.5) -> List[Peak]`

### 基线
- `BaselinePreset.polynomial(coeffs)` — 系数按**高次在前**。
- `BaselinePreset.fluorescence_rise(intensity=0.3, scale=800, offset=0.0, slope=0.0)`
- `BaselinePreset.laser_leak_slope(offset=0.0, slope=1e-4)`
- `BaselinePreset.graphite_like(intensity=0.2, center=1580, fwhm=250, offset=0.0, slope=0.0)`

### 噪声
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

### 单谱模拟
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
  anomalies=None,              # 可选：异常后处理（见下）
  return_components=True,
) -> (y, parts)
```
`parts` 包含：`axis`, `axis_eff`, `signal`, `baseline`, `convolved`, `noise`, `final`, `coords`。

### 成像
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

### 有效性与异常
- `simulate_map2d_with_validity(...)`，`ValidityMaskConfig(proportion_valid=0.5, mode='independent'|'gaussian_clusters'|'smooth_field', ...)`  
- `simulate_map2d_with_validity_and_anomalies(...)`，`AnomalyMapConfig(bad_pixel_rate=..., seed=...)`

### 统计生成
- `VariationSpec(kind, value, clip_min=0.0, distribution='auto')`
- `generate_spectra_n(...)` → `Y, meta`
- `CorrelatedHeightSpec(distribution='mvnormal'|'mvlognormal', mean, cov)`
- `generate_spectra_n_correlated(...)` → `Y, meta`

### 后处理与工具
- `apply_heteroscedastic_gaussian(y, ref, base_sigma, scale=0.0, power=1.0, seed=None)`
- `inject_spurious_peaks(x, y, AnomalyConfig)` / `apply_axis_warp(x, y, quad)` / `apply_saturation(y, level)`
- `corrupt_labels(labels, prob, num_classes, seed=None)`
- `make_domain_shift_configs(map_cfg, valid_cfg, train_delta, test_delta)`

### I/O
- `save_csv(path, x, y, header='wavenumber,intensity')`
- `save_maps_to_npz(path_template, iterator)`
- `save_maps_to_h5(path, iterator, n_samples, ny=None, nx=None, n_points=None, compression='gzip')`

---

## 📦 安装

**依赖**：Python ≥ 3.8，NumPy  
可选：Matplotlib（可视化）、`h5py`（HDF5 导出）

```bash
pip install numpy matplotlib h5py
# 若以本地包形式使用：
pip install -e .
```

---

## 🧠 典型流程

1. 定义峰或使用 `peaks_from_arrays()` 构造。
2. 在所需基线/噪声/抖动/漂移条件下生成单谱。
3. 构建分析物库与混合物。
4. 模拟 1D/2D map，可选有效性掩膜或异常。
5. 按日程生成时间序列（基线/漂移/峰高）。
6. 通过 `iter_maps()`、`.npz` 或 `.h5` 流式/导出数据集。

---

## 🧾 引用

如在科研或工程中使用本工具，请引用：

```
ramansim: A Synthetic Raman Spectrum & Mapping Simulator for Physics-Grounded AI Evaluation
Version 1.0, 2025.
```

---

**作者**：JJL & GPT 

**许可证**：MIT
