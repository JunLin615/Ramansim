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


# Time series: simulate N frames with slow baseline drift + small peak jitter
N = 50
x = np.linspace(200, 2000, 2500)
base0 = BaselinePreset.laser_leak_slope(offset=0.02, slope=1.0e-4)
peaks0 = peaks_from_arrays([612, 1000, 1580], heights=[1.0, 0.6, 0.9], fwhms=[12, 14, 25])


sim = SpectrumSimulator(seed=7)
noise = NoiseConfig(gaussian_sigma=0.01, multiplicative_sigma=0.01)


Y = []
for i in range(N):
    # baseline drift: add a slow exponential rise component increasing with time
    frac = i / max(N-1, 1)
    base = (lambda xx, f=frac: base0(xx) + 0.15*f*(1.0 - np.exp(-(xx-xx.min())/800.0)))
    y, parts = sim.simulate(x, peaks=peaks0, baseline=base, noise=noise, irf_fwhm=6.0,
                            drift_ppm=5.0*frac, peak_jitter_std=0.8, groupwise_jitter=False,
                            return_components=True)
    Y.append(y)
Y = np.stack(Y, axis=0) # (N, n_points)


# visualize as an image (time vs Raman shift)
plt.figure()
plt.imshow(Y, aspect='auto', origin='lower', extent=[x.min(), x.max(), 0, N-1])
plt.xlabel('Raman shift (cm$^{-1}$)')
plt.ylabel('Time index')
plt.title('Time-series synthetic spectra')
plt.colorbar(label='Intensity (a.u.)')
plt.tight_layout()
plt.show()