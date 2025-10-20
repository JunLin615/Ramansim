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
from ramansim import SpectrumSimulator, peaks_from_arrays, BaselinePreset, NoiseConfig


x = np.linspace(200, 2000, 4000)
peaks = peaks_from_arrays([612, 1000, 1580], heights=[1.0, 0.6, 0.9], fwhms=[12, 14, 25])


sim = SpectrumSimulator(seed=42)
noise = NoiseConfig(gaussian_sigma=0.02, poisson_scale=1500, one_over_f_strength=0.005,
spike_rate=0.0005, spike_height=(3,8), multiplicative_sigma=0.01)
baseline = BaselinePreset.graphite_like(intensity=0.2, center=1580, fwhm=250, offset=0.02, slope=2e-4)


y, parts = sim.simulate(x, peaks=peaks, baseline=baseline, noise=noise, irf_fwhm=6.0, drift_ppm=10.0)


plt.figure()
plt.plot(x, y, label='final')
plt.plot(x, parts['signal'], alpha=0.6, label='signal')
plt.plot(x, parts['baseline'], alpha=0.6, label='baseline')
plt.xlabel('Raman shift (cm$^{-1}$)')
plt.ylabel('Intensity (a.u.)')
plt.title('Single synthetic Raman spectrum')
plt.legend()
plt.tight_layout()
plt.show()