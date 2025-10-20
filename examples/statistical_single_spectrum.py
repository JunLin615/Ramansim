import sys
import os

# 获取当前脚本所在文件夹的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父文件夹的路径（向上一级）
parent_dir = os.path.dirname(current_dir)
# 将父文件夹添加到系统路径
sys.path.append(parent_dir)
from ramansim import SpectrumSimulator, peaks_from_arrays, VariationSpec, generate_spectra_n, NoiseConfig
import numpy as np

x = np.linspace(200, 2000, 3000)
peaks0 = peaks_from_arrays([612, 1000, 1580], heights=[1.0, 0.6, 0.9], default_fwhm=15)

sim = SpectrumSimulator(seed=123)
spec = VariationSpec(kind='cv', value=0.2, clip_min=0.0)  # 变异系数20%
noise = NoiseConfig(gaussian_sigma=0.01, multiplicative_sigma=0.02)

Y, meta = generate_spectra_n(sim, x, peaks0, n=128, height_spec=spec,
                             peak_jitter_std=0.8, groupwise_jitter=True,
                             noise=noise, irf_fwhm=6.0)
# Y: (128, 3000), meta['heights']: (128, 3)
