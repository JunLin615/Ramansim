import sys
import os

# 获取当前脚本所在文件夹的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父文件夹的路径（向上一级）
parent_dir = os.path.dirname(current_dir)
# 将父文件夹添加到系统路径
sys.path.append(parent_dir)
import numpy as np
from ramansim import (SpectrumSimulator, MapConfig, MapSimulator, peaks_from_arrays,
iter_maps, BaselinePreset, NoiseConfig)


# Define analytes
A = peaks_from_arrays([612, 1000], heights=[1.0, 0.6], default_fwhm=12)
B = peaks_from_arrays([1330, 1580, 2700], heights=[0.7, 1.0, 0.3], default_fwhm=30, shape='lorentzian')


# Axes & configs
x = np.linspace(200, 2000, 1500)
sim = SpectrumSimulator(seed=123)
cfg = MapConfig(ny=32, nx=32, cluster_mode='smooth_field', smooth_sigma=6.0, seed=11)
baseline = BaselinePreset.laser_leak_slope(offset=0.01, slope=8e-5)
noise = NoiseConfig(gaussian_sigma=0.01, multiplicative_sigma=0.02)


# Streaming maps (no disk)
stream = iter_maps(sim, x, [A, B], [1.0, 0.8], cfg, baseline=baseline, noise=noise, irf_fwhm=6.0,
                   n_samples=3, seed0=2025)


for i, (cube, aux) in enumerate(stream):
    print(f"Sample {i}: cube shape = {cube.shape}; r-range = [{aux['coords']['r'][0]:.1f}, {aux['coords']['r'][-1]:.1f}] cm^-1")
    # Here insert your training step using `cube` and `aux` ...

