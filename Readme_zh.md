# RamanSim: A Synthetic Raman Spectrum and Mapping Simulator

`RamanSim` 是一个用于生成 **合成拉曼光谱数据** 的 Python 工具包，适用于 AI 模型开发、算法基准测试和物理可解释性验证。
它支持单光谱合成、时间序列仿真以及具有物理意义的噪声、基线和漂移模型的一维/二维空间映射。

---

## 🌟 主要功能

### 🔹 光谱模拟
- 从任意峰定义生成单个或多个拉曼光谱。
- 支持高斯、洛伦兹和伪 Voigt 线型。
- 支持峰级和组级的 **随机抖动**（用于时间漂移仿真）。
- 可配置的 **基线模型**（荧光、石墨状、激光泄露等）。
- 多种 **噪声模型**：
  - 高斯噪声、乘性噪声、泊松噪声、1/f（粉红噪声）和随机尖峰。
- 可选 **仪器分辨率卷积**（IRF FWHM）。
- 返回所有组成部分（`signal`、`baseline`、`noise`、`final`）以便详细分析。

### 🔹 拉曼映射模拟
- 模拟 **一维线扫描** 或 **二维拉曼图像**。
- 定义多个分析物（每个分析物由一组相关峰组成）。
- 控制空间丰度分布方式：
  - `random` — 均匀随机强度场。
  - `gaussian_clusters` — 聚集的热点（类似 SERS）。
  - `smooth_field` — 连续的相关场。
- 按像素添加基线和噪声。
- 返回拉曼数据立方体和坐标轴，可直接用于可视化。

### 🔹 分析物库
- 轻松定义分析物及其相关峰。
- 通过简单的权重字典支持 **混合物**。
- 实现组级相关抖动与成分变化。

### 🔹 数据导出与流式生成
- **内存流式生成**，可直接用于模型训练：
  - `iter_maps()` 生成 `(cube, aux)` 样本，无需保存到磁盘。
- **磁盘导出**，用于数据集创建：
  - `save_maps_to_npz()` 生成压缩的 `.npz` 文件。
  - `save_maps_to_h5()` 生成大型 `.h5` 数据集（需安装 `h5py`）。

---

## 🧩 核心类与 API

### `Peak`
表示单个拉曼峰。

| 参数 | 类型 | 描述 |
|------|------|------|
| `pos` | float | 峰位中心（cm⁻¹） |
| `height` | float | 峰高 |
| `fwhm` | float | 半高全宽 |
| `shape` | str | `'gaussian'`、`'lorentzian'` 或 `'pvoigt'` |
| `eta` | float | 伪 Voigt 混合比例 |
| `group` | str | 可选组标签（用于相关抖动） |

---

### `SpectrumSimulator`
生成单个拉曼光谱。

```python
y, parts = SpectrumSimulator(seed=42).simulate(
    x,
    peaks,                          # Peak 对象列表
    baseline=BaselinePreset.graphite_like(...),
    noise=NoiseConfig(...),
    irf_fwhm=6.0,                   # 仪器展宽
    drift_ppm=10.0,                 # 全局轴漂移
    peak_jitter_std=1.0,            # 单峰随机偏移
    groupwise_jitter=True,          # 相同组的相关抖动
)
```

**返回值**  
`y`: 含噪光谱  
`parts`: 包含中间结果的字典（如 `signal`、`baseline`、`final` 等）

---

### `MapSimulator`
模拟一维或二维拉曼映射数据集。

```python
cube, aux = MapSimulator(sim).simulate_map2d(
    x,
    analytes=[A, B],             # 峰列表
    analyte_weights=[1.0, 0.8],
    cfg=MapConfig(ny=64, nx=64, cluster_mode='gaussian_clusters'),
    baseline=baseline,
    noise=noise,
    irf_fwhm=6.0
)
```

**返回值**
- `cube`: `(ny, nx, n_points)` 数组（拉曼强度）
- `aux`: 包含以下内容的字典：
  - `'fields'`: (A, ny, nx) 丰度场
  - `'unit_spec'`: (A, n_points) 分析物光谱
  - `'coords'`: `{'r','x','y','t'}` 坐标数组

---

### `MapConfig`
配置空间映射仿真参数。

| 参数 | 类型 | 描述 |
|------|------|------|
| `ny`, `nx` | int | 图像尺寸 |
| `cluster_mode` | str | `'random'`、`'gaussian_clusters'`、`'smooth_field'` |
| `n_clusters` | int | 聚类热点数 |
| `cluster_std` | float | 每个聚类的标准差（像素） |
| `smooth_sigma` | float | 平滑模式下的高斯模糊 σ |
| `variability` | float | 强度场变化幅度 |
| `seed` | int | 随机种子 |
| `x_coords`, `y_coords`, `t_coords` | np.ndarray | 可选坐标数组 |

---

### `AnalyteLibrary` 与 `build_mixture_from_library`
定义可复用的分析物峰集合。

```python
lib = AnalyteLibrary.from_arrays({
    'A': {'positions':[612, 1000], 'heights':[1.0, 0.6]},
    'B': {'positions':[1330, 1580, 2700], 'shape':'lorentzian'}
})
analytes, weights = build_mixture_from_library(lib, {'A':1.0, 'B':0.8})
```

---

### `iter_maps()` – 流式生成器
```python
for cube, aux in iter_maps(sim, x, analytes, weights, cfg,
                           baseline=baseline, noise=noise,
                           n_samples=5, seed0=2025):
    # 训练循环或可视化
    ...
```

## 🧪 示例脚本

所有示例位于 `examples/` 目录中。
每个脚本都可独立运行，需设置 `if __name__ == "__main__":` 标志。

| 文件 | 描述 |
|------|------|
| `single_spectrum.py` | 生成并可视化单个合成拉曼光谱 |
| `time_series.py` | 模拟带有基线漂移和抖动的时间序列 |
| `map2d.py` | 生成并可视化二维拉曼图像（特定波数下的像素强度图） |
| `iter_stream.py` | 演示用于 AI 训练的多图流式生成 |

示例（单光谱）：

```bash
python examples/01_single_spectrum.py
```

示例（二维映射）：

```bash
python examples/03_map2d.py
```

---

## 📦 安装

### 依赖
- Python ≥ 3.8  
- NumPy  
- Matplotlib（用于可视化）  
- h5py（可选，用于 `.h5` 导出）

### 安装命令
```bash
pip install numpy matplotlib h5py
```

如需打包：
```bash
pip install -e .
```

---

## 🧠 典型工作流程

1. **定义峰参数**（使用 `peaks_from_arrays()`）。
2. **生成单光谱**，可加入所需基线和噪声。
3. **构建分析物库**，用于复杂混合体系。
4. **使用 `MapSimulator` 模拟一维/二维映射。**
5. **通过 `iter_maps()` 流式输入数据** 到机器学习模型。
6. **必要时导出数据集** 为 `.npz` 或 `.h5` 文件。

---

## 🧩 物理启发效应

| 效应 | 参数 | 描述 |
|------|------|------|
| 峰位漂移 | `drift_ppm`, `peak_jitter_std` | 全局轴或单峰随机偏移 |
| 热点聚集 | `cluster_mode='gaussian_clusters'` | 生成空间相关的高强度区域 |
| 基线荧光 | `BaselinePreset.graphite_like()` | 添加宽背景曲率 |
| 探测器噪声 | `NoiseConfig` | 高斯 + 泊松 + 1/f + 尖峰噪声 |
| 仪器展宽 | `irf_fwhm` | 卷积模糊（高斯核） |

---

## 🧾 引用

若您在学术或工业研究中使用了 `RamanSim`，请引用：

```
RamanSim: A Synthetic Raman Spectrum Generator for Physics-Grounded AI Evaluation
Version 1.0, 2025.
```

---

## 🔧 未来工作

- 时间分辨映射（随时间演变的 3D 光谱立方体）  
- 校准伪影与基线漂移  
- 实验匹配工具  
- 与 PyTorch `Dataset` 类集成，用于直接 ML 管线  

---

**作者：** JJL and GPT

**许可证：** MIT（学术与商业用途免费）
