import sys
import os

# 获取当前脚本所在文件夹的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父文件夹的路径（向上一级）
parent_dir = os.path.dirname(current_dir)
# 将父文件夹添加到系统路径
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
from ramansim import (SpectrumSimulator, MapSimulator, MapConfig, BaselinePreset, NoiseConfig,
peaks_from_arrays)


x = np.linspace(200, 2000, 2000)
A = peaks_from_arrays([612, 1000], heights=[1.0, 0.6], default_fwhm=12)
B = peaks_from_arrays([1330, 1580, 2700], heights=[0.7, 1.0, 0.3], default_fwhm=30, shape='lorentzian')


sim = SpectrumSimulator(seed=7)
mapsim = MapSimulator(sim)


cfg = MapConfig(ny=64, nx=64, cluster_mode='gaussian_clusters', n_clusters=4, cluster_std=5.0, seed=7)
noise = NoiseConfig(gaussian_sigma=0.01, multiplicative_sigma=0.02)
baseline = BaselinePreset.graphite_like(intensity=0.2, center=1580, fwhm=250, offset=0.02, slope=1e-4)


cube, aux = mapsim.simulate_map2d(x, analytes=[A, B], analyte_weights=[1.0, 0.8], cfg=cfg,
                                  baseline=baseline, noise=noise, irf_fwhm=6.0)


# Visualize: map at 1580 cm^-1 and integrated band 1550-1610 cm^-1
r = aux['coords']['r']
idx = np.argmin(np.abs(r - 1580.0))
img_peak = cube[:, :, idx]
band = (r >= 1550) & (r <= 1610)
img_band = cube[:, :, band].sum(axis=2)


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(img_peak, origin='lower', cmap='viridis')
plt.title('Map at 1580 cm$^{-1}$')
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(img_band, origin='lower', cmap='viridis')
plt.title('Band-integrated 1550–1610 cm$^{-1}$')
plt.colorbar()
plt.tight_layout()
plt.show()