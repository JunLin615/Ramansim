"""
ramansim: simple Raman spectrum simulator for AI model testing

Core goals:
- Specify peak positions/shapes to generate synthetic spectra
- Add realistic noise (Gaussian, Poisson/shot, 1/f, random spikes)
- Add baseline effects (fluorescence rise, unfiltered laser slope, broad substrate bands)
- Optional instrument effects (resolution convolution), wavenumber drift/jitter
- Deterministic via random seed, with component-wise outputs for explainability

Usage example
-------------
from ramansim import SpectrumSimulator, Peak, BaselinePreset, NoiseConfig
import numpy as np

x = np.linspace(200, 2000, 4000)  # wavenumber axis (cm^-1)
peaks = [
    Peak(pos=612, height=1.0, fwhm=10, shape='gaussian'),
    Peak(pos=1000, height=0.6, fwhm=14, shape='lorentzian'),
    Peak(pos=1580, height=0.9, fwhm=25, shape='pvoigt', eta=0.3),
]
noise = NoiseConfig(gaussian_sigma=0.02, poisson_scale=1500, one_over_f_strength=0.005,
                    spike_rate=0.0005, spike_height=(3, 8), multiplicative_sigma=0.01)

sim = SpectrumSimulator(seed=42)
y, parts = sim.simulate(
    x,
    peaks=peaks,
    baseline=BaselinePreset.graphite_like(intensity=0.4, center=1580, fwhm=250, offset=0.02, slope=2e-4),
    noise=noise,
    irf_fwhm=6.0,
    drift_ppm=10.0,  # global wavenumber drift in parts-per-million
)
# y is the final spectrum; parts is a dict with components for inspection

"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np

ArrayLike = Union[np.ndarray, List[float]]

# -------------------- Peak definitions --------------------
@dataclass
class Peak:
    pos: float                   # center position in axis units (e.g., cm^-1)
    height: float = 1.0          # peak height (arbitrary units)
    fwhm: float = 10.0           # full width at half max (same units as axis)
    shape: str = 'gaussian'      # 'gaussian' | 'lorentzian' | 'pvoigt'
    eta: float = 0.5             # mixing for pvoigt (0=Gaussian, 1=Lorentzian)

    def profile(self, x: np.ndarray) -> np.ndarray:
        if self.shape == 'gaussian':
            return gaussian(x, self.pos, self.fwhm, self.height)
        elif self.shape == 'lorentzian':
            return lorentzian(x, self.pos, self.fwhm, self.height)
        elif self.shape == 'pvoigt':
            g = gaussian(x, self.pos, self.fwhm, self.height)
            l = lorentzian(x, self.pos, self.fwhm, self.height)
            return (1 - self.eta) * g + self.eta * l
        else:
            raise ValueError(f"Unknown peak shape: {self.shape}")

# -------------------- Noise configuration --------------------
@dataclass
class NoiseConfig:
    gaussian_sigma: float = 0.0              # additive white noise std (A.U.)
    poisson_scale: Optional[float] = None    # if set, simulate shot noise: scale -> photon count per unit intensity
    one_over_f_strength: float = 0.0         # amplitude for 1/f noise (additive)
    spike_rate: float = 0.0                  # probability per point of a cosmic-like spike
    spike_height: Tuple[float, float] = (3.0, 8.0) # range of spike multipliers relative to local std
    multiplicative_sigma: float = 0.0        # multiplicative noise std (intensity-proportional)

# -------------------- Baseline presets --------------------
class BaselinePreset:
    @staticmethod
    def polynomial(coeffs: Iterable[float]) -> Callable[[np.ndarray], np.ndarray]:
        coeffs = np.array(list(coeffs), dtype=float)
        def f(x: np.ndarray) -> np.ndarray:
            # numpy polyval expects highest power first
            return np.polyval(coeffs, x)
        return f

    @staticmethod
    def fluorescence_rise(intensity: float=0.3, scale: float=800.0, offset: float=0.0, slope: float=0.0) -> Callable[[np.ndarray], np.ndarray]:
        # Exponential rise baseline: offset + intensity * (1 - exp(-(x - x0)/scale)) + slope*x
        # Here x0 set to min(x) when evaluated
        def f(x: np.ndarray) -> np.ndarray:
            x0 = float(np.min(x))
            return offset + intensity * (1.0 - np.exp(-(x - x0)/scale)) + slope * (x - x0)
        return f

    @staticmethod
    def laser_leak_slope(offset: float=0.0, slope: float=1e-4) -> Callable[[np.ndarray], np.ndarray]:
        # Simple linear baseline capturing imperfect laser rejection
        def f(x: np.ndarray) -> np.ndarray:
            x0 = float(np.min(x))
            return offset + slope * (x - x0)
        return f

    @staticmethod
    def graphite_like(intensity: float=0.2, center: float=1580.0, fwhm: float=250.0, offset: float=0.0, slope: float=0.0) -> Callable[[np.ndarray], np.ndarray]:
        # Broad Lorentzian baseline plus optional offset & slope
        def f(x: np.ndarray) -> np.ndarray:
            base = lorentzian(x, center, fwhm, intensity)
            x0 = float(np.min(x))
            return base + offset + slope * (x - x0)
        return f

# -------------------- Core simulator --------------------
@dataclass
class SpectrumSimulator:
    seed: Optional[int] = None

    def simulate(
            self,
            x: ArrayLike,
            peaks: Iterable[Peak],
            baseline: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            noise: Optional[NoiseConfig] = None,
            irf_fwhm: Optional[float] = None,
            drift_ppm: float = 0.0,
            peak_jitter_std: float = 0.0,         # 新增：峰位随机漂移（单位同x）
            groupwise_jitter: bool = False,       # 新增：若True，同组peak共用同一漂移
            return_components: bool = True,
        ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

            x = np.asarray(x, dtype=float)
            rng = np.random.default_rng(self.seed)

            # Apply drift in ppm (axis scaling)
            x_eff = x * (1.0 + drift_ppm * 1e-6)

            # Sum peaks
            signal = np.zeros_like(x_eff)
            if peaks:
                group_shift = {}
                if groupwise_jitter and peak_jitter_std > 0:
                    for pk in peaks:
                        if pk.group not in group_shift:
                            group_shift[pk.group] = rng.normal(0.0, peak_jitter_std)
                for pk in peaks:
                    shift = 0.0
                    if peak_jitter_std > 0:
                        if groupwise_jitter:
                            shift = group_shift.get(pk.group, 0.0)
                        else:
                            shift = rng.normal(0.0, peak_jitter_std)
                    signal += Peak(
                        pos=pk.pos + shift,
                        height=pk.height,
                        fwhm=pk.fwhm,
                        shape=pk.shape,
                        eta=pk.eta
                    ).profile(x_eff)

            # Baseline
            base = baseline(x) if baseline is not None else np.zeros_like(x)

            # IRF convolution
            convolved = signal + base
            if irf_fwhm and irf_fwhm > 0:
                sigma = irf_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
                dx = np.median(np.diff(x))
                half_width = int(np.ceil(4 * sigma / dx))
                grid = np.arange(-half_width, half_width + 1) * dx
                kernel = np.exp(-0.5 * (grid / sigma) ** 2)
                kernel /= kernel.sum()
                convolved = np.convolve(convolved, kernel, mode="same")

            y = np.copy(convolved)

            # Noise
            if noise is not None:
                if noise.multiplicative_sigma and noise.multiplicative_sigma > 0:
                    y *= (1.0 + rng.normal(0.0, noise.multiplicative_sigma, size=y.shape))
                if noise.gaussian_sigma and noise.gaussian_sigma > 0:
                    y += rng.normal(0.0, noise.gaussian_sigma, size=y.shape)
                if noise.poisson_scale is not None and noise.poisson_scale > 0:
                    counts = np.clip(y * noise.poisson_scale, 0, None)
                    y = rng.poisson(counts).astype(float) / noise.poisson_scale

            parts = {
                'axis': x,
                'axis_eff': x_eff,
                'signal': signal,
                'baseline': base,
                'convolved': convolved,
                'noise': y - convolved,
                'final': y,
                'coords': {'r': x},
            } if return_components else {}

            return y, parts


# -------------------- Peak shape functions --------------------
def gaussian(x: np.ndarray, mu: float, fwhm: float, height: float) -> np.ndarray:
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return height * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def lorentzian(x: np.ndarray, x0: float, fwhm: float, height: float) -> np.ndarray:
    gamma = fwhm / 2.0
    return height * (gamma ** 2) / ((x - x0) ** 2 + gamma ** 2)

# -------------------- Colored noise --------------------
def colored_noise_1overf(rng: np.random.Generator, n: int, strength: float) -> np.ndarray:
    # Generate 1/f noise by shaping white noise in frequency domain
    freqs = np.fft.rfftfreq(n)
    white = rng.normal(0.0, 1.0, size=freqs.size) + 1j * rng.normal(0.0, 1.0, size=freqs.size)
    # Avoid division by zero at f=0 by setting weight there to zero
    weights = np.zeros_like(freqs)
    nz = freqs > 0
    weights[nz] = 1.0 / freqs[nz]
    shaped = white * weights
    # IFFT to time domain
    y = np.fft.irfft(shaped, n=n)
    # Normalize to unit std then scale
    if np.std(y) > 0:
        y = y / np.std(y)
    return strength * y

# -------------------- Helpers --------------------
def _robust_std(y: np.ndarray) -> float:
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    return 1.4826 * mad + 1e-12

# -------------------- Convenience factories --------------------
def make_axis(start: float, stop: float, step: float) -> np.ndarray:
    """Create a uniformly spaced axis inclusive of stop if divisibility permits."""
    n = int(np.floor((stop - start) / step)) + 1
    return start + np.arange(n) * step

def make_linear_coords(n: int, start: float = 0.0, step: float = 1.0) -> np.ndarray:
    """Convenience: 1D coordinate array of length n with given start/step."""
    return start + np.arange(n) * step

# -------------------- I/O --------------------
def save_csv(path: str, x: np.ndarray, y: np.ndarray, header: str = 'wavenumber,intensity') -> None:
    arr = np.column_stack([x, y])
    np.savetxt(path, arr, delimiter=',', header=header, comments='')

# -------------------- Convenience factories --------------------
# (existing functions above)

# -------------------- Peak list helpers --------------------
def peaks_from_arrays(positions: ArrayLike,
                      heights: Optional[ArrayLike] = None,
                      fwhms: Optional[ArrayLike] = None,
                      shape: str = 'gaussian',
                      default_fwhm: float = 12.0,
                      eta: float = 0.5) -> List[Peak]:
    """Create a list of Peak objects from arrays/lists.

    Any of heights/fwhms can be None; missing values default to 1.0 and default_fwhm.
    """
    pos = np.asarray(positions, dtype=float)
    if heights is None:
        heights = np.ones_like(pos, dtype=float)
    else:
        heights = np.asarray(heights, dtype=float)
    if fwhms is None:
        fwhms = np.full_like(pos, fill_value=default_fwhm, dtype=float)
    else:
        fwhms = np.asarray(fwhms, dtype=float)
    if not (pos.size == heights.size == fwhms.size):
        raise ValueError("positions, heights, fwhms must all have same length")
    return [Peak(p, h, w, shape=shape, eta=eta) for p, h, w in zip(pos, heights, fwhms)]

# -------------------- Spatial mapping simulation --------------------
@dataclass
class MapConfig:
    ny: int
    nx: int
    cluster_mode: str = 'random'  # 'random' | 'gaussian_clusters' | 'smooth_field'
    n_clusters: int = 3           # for gaussian_clusters
    cluster_std: float = 4.0      # in pixels (std of Gaussian blobs)
    smooth_sigma: float = 6.0     # for smooth_field (Gaussian blur sigma)
    base_intensity: float = 0.0   # additive background per-pixel (before peaks)
    variability: float = 0.3      # multiplicative variability amplitude of analyte abundance field
    seed: Optional[int] = None
    # ---- coordinate systems (optional) ----
    x_coords: Optional[np.ndarray] = None   # length nx; if None -> np.arange(nx)
    y_coords: Optional[np.ndarray] = None   # length ny; if None -> np.arange(ny)
    t_coords: Optional[np.ndarray] = None   # optional time/frame axis for future extensions


class MapSimulator:
    """Simulate 1D/2D Raman mapping datasets (data cubes).

    Core idea:
      - Each analyte corresponds to a set of peaks (same analyte → correlated intensity).
      - A spatial abundance field is generated for each analyte.
      - The spectrum at each pixel is the weighted sum of analyte spectra + baseline + noise.

    Returns
    -------
    cube : ndarray, shape (ny, nx, n_points)
        Simulated Raman map.
    aux : dict
        Includes:
          'fields' : (A, ny, nx) abundance fields per analyte
          'unit_spec' : (A, n_points) unit spectra per analyte
          'coords' : {'r','x','y','t'} coordinate arrays
    """

    def __init__(self, spectrum_sim: SpectrumSimulator):
        self.spectrum_sim = spectrum_sim

    def _make_field(self, cfg: MapConfig, rng: np.random.Generator) -> np.ndarray:
        ny, nx = cfg.ny, cfg.nx
        if cfg.cluster_mode == 'random':
            field = rng.random((ny, nx))
        elif cfg.cluster_mode == 'gaussian_clusters':
            field = np.zeros((ny, nx), dtype=float)
            for _ in range(max(cfg.n_clusters, 1)):
                cy = rng.uniform(0, ny - 1)
                cx = rng.uniform(0, nx - 1)
                y = np.arange(ny)[:, None]
                x = np.arange(nx)[None, :]
                field += np.exp(-0.5 * (((y - cy) ** 2 + (x - cx) ** 2) / (cfg.cluster_std ** 2 + 1e-9)))
            field = (field - field.min()) / (field.max() - field.min() + 1e-12)
        elif cfg.cluster_mode == 'smooth_field':
            raw = rng.random((ny, nx))
            field = _gaussian_blur2d(raw, sigma=cfg.smooth_sigma)
            field = (field - field.min()) / (field.max() - field.min() + 1e-12)
        else:
            raise ValueError(f"Unknown cluster_mode: {cfg.cluster_mode}")
        field = (1.0 - cfg.variability) + cfg.variability * field
        return field

    def simulate_map2d(
        self,
        x: ArrayLike,
        analytes: List[List[Peak]],
        analyte_weights: Optional[List[float]] = None,
        per_analyte_fields: Optional[List[np.ndarray]] = None,
        cfg: Optional[MapConfig] = None,
        baseline: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        noise: Optional[NoiseConfig] = None,
        irf_fwhm: Optional[float] = None,
        drift_ppm: float = 0.0
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

        x = np.asarray(x, dtype=float)
        ny, nx = (cfg.ny, cfg.nx) if cfg else (16, 16)
        cfg = cfg or MapConfig(ny=ny, nx=nx)
        rng = np.random.default_rng(cfg.seed if cfg.seed is not None else self.spectrum_sim.seed)

        # Coordinate axes
        x_coords = cfg.x_coords if cfg.x_coords is not None else np.arange(nx)
        y_coords = cfg.y_coords if cfg.y_coords is not None else np.arange(ny)
        t_coords = cfg.t_coords

        n_pts = x.size
        n_analytes = len(analytes)
        if analyte_weights is None:
            analyte_weights = [1.0] * n_analytes
        if per_analyte_fields is None:
            per_analyte_fields = [self._make_field(cfg, rng) for _ in range(n_analytes)]
        fields = np.stack(per_analyte_fields, axis=0)
        cube = np.zeros((ny, nx, n_pts), dtype=float)

        # Precompute unit spectra
        unit_spec = []
        for pkset in analytes:
            y0, _ = self.spectrum_sim.simulate(x, peaks=pkset, baseline=None, noise=None,
                                               irf_fwhm=irf_fwhm, drift_ppm=drift_ppm,
                                               return_components=False)
            unit_spec.append(y0)
        unit_spec = np.stack(unit_spec, axis=0)

        for iy in range(ny):
            for ix in range(nx):
                mix = fields[:, iy, ix] * np.asarray(analyte_weights)
                y_no_base = (mix[:, None] * unit_spec).sum(axis=0)
                y, parts = self.spectrum_sim.simulate(x, peaks=[], baseline=baseline, noise=noise,
                                                       irf_fwhm=None, drift_ppm=0.0, return_components=True)
                convolved = y_no_base + parts['baseline']
                if irf_fwhm and irf_fwhm > 0:
                    sigma = irf_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
                    dx = np.median(np.diff(x))
                    half_width = int(np.ceil(4 * sigma / dx))
                    grid = np.arange(-half_width, half_width + 1) * dx
                    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
                    kernel /= kernel.sum()
                    convolved = np.convolve(convolved, kernel, mode='same')
                y_final = convolved
                if noise is not None:
                    rng_local = np.random.default_rng(cfg.seed + iy * nx + ix if cfg.seed is not None else None)
                    if noise.multiplicative_sigma and noise.multiplicative_sigma > 0:
                        y_final *= (1.0 + rng_local.normal(0.0, noise.multiplicative_sigma, size=n_pts))
                    if noise.gaussian_sigma and noise.gaussian_sigma > 0:
                        y_final += rng_local.normal(0.0, noise.gaussian_sigma, size=n_pts)
                    if noise.poisson_scale is not None and noise.poisson_scale > 0:
                        counts = np.clip(y_final * noise.poisson_scale, 0, None)
                        y_final = rng_local.poisson(counts).astype(float) / noise.poisson_scale
                cube[iy, ix, :] = y_final

        aux = {
            'fields': fields,
            'unit_spec': unit_spec,
            'coords': {'r': x, 'x': x_coords, 'y': y_coords, 't': t_coords}
        }
        return cube, aux

    def simulate_map1d(self, *args, **kwargs) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        cfg = kwargs.get('cfg', None)
        if cfg is None:
            kwargs['cfg'] = MapConfig(ny=1, nx=kwargs.get('nx', 128))
        cube, aux = self.simulate_map2d(*args, **kwargs)
        line = cube[0, :, :]
        if 'coords' in aux:
            c = aux['coords']
            aux['coords'] = {'r': c['r'], 'x': c['x'], 'y': None, 't': c['t']}
        return line, aux

# -------------------- 2D utilities --------------------
def _gaussian_blur2d(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img.copy()
    # build 1D kernel
    radius = max(1, int(np.ceil(4 * sigma)))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    # separable conv
    tmp = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), axis=0, arr=img)
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), axis=1, arr=tmp)
    return out

# -------------------- Analyte library & mixtures --------------------
@dataclass
class Analyte:
    name: str
    peaks: List[Peak]

class AnalyteLibrary:
    def __init__(self, mapping: Dict[str, List[Peak]]):
        self._map = dict(mapping)
    @classmethod
    def from_arrays(cls, spec: Dict[str, Dict[str, ArrayLike]], default_fwhm: float = 12.0) -> 'AnalyteLibrary':
        mapping = {}
        for name, cfg in spec.items():
            peaks = peaks_from_arrays(cfg['positions'], cfg.get('heights'), cfg.get('fwhms'),
                                      shape=cfg.get('shape','gaussian'), default_fwhm=default_fwhm,
                                      eta=cfg.get('eta',0.5))
            for p in peaks:
                p.group = name
            mapping[name] = peaks
        return cls(mapping)
    def names(self) -> List[str]:
        return list(self._map.keys())
    def analyte(self, name: str) -> List[Peak]:
        return [Peak(**{**p.__dict__}) for p in self._map[name]]

def build_mixture_from_library(lib: AnalyteLibrary, selection: Dict[str, float]) -> Tuple[List[List[Peak]], List[float]]:
    analytes = []
    weights = []
    for name, w in selection.items():
        analytes.append(lib.analyte(name))
        weights.append(float(w))
    return analytes, weights

# -------------------- Dataset generators (in-memory & on-disk) --------------------

def iter_maps(spectrum_sim: SpectrumSimulator,
              x: ArrayLike,
              analytes: List[List[Peak]],
              analyte_weights: List[float],
              cfg: MapConfig,
              baseline: Optional[Callable[[np.ndarray], np.ndarray]] = None,
              noise: Optional[NoiseConfig] = None,
              irf_fwhm: Optional[float] = None,
              drift_ppm: float = 0.0,
              n_samples: int = 1,
              seed0: Optional[int] = None):
    """Yield (cube, aux) n_samples times with varying seed (no disk IO)."""
    base_seed = seed0 if seed0 is not None else cfg.seed
    mapsim = MapSimulator(spectrum_sim)
    for i in range(n_samples):
        cfg_i = MapConfig(**{**cfg.__dict__})
        cfg_i.seed = (base_seed + i) if base_seed is not None else None
        cube, aux = mapsim.simulate_map2d(x, analytes, analyte_weights, None, cfg_i, baseline, noise, irf_fwhm, drift_ppm)
        yield cube, aux

# ---- NPZ saving (per-sample) ----

def save_maps_to_npz(path_template: str, iterator) -> List[str]:
    """Save each (cube, aux) from iterator to 'path_template_{i:04d}.npz'. Return paths."""
    paths = []
    for i, (cube, aux) in enumerate(iterator):
        path = f"{path_template}_{i:04d}.npz"
        np.savez_compressed(path, cube=cube, coords=aux.get('coords', {}), fields=aux.get('fields', None), unit_spec=aux.get('unit_spec', None))
        paths.append(path)
    return paths

# ---- HDF5 saving (single file) ----

def save_maps_to_h5(path: str, iterator, n_samples: int, ny: Optional[int]=None, nx: Optional[int]=None, n_points: Optional[int]=None, compression: Optional[str]='gzip') -> None:
    """Save maps into a single HDF5 file with datasets: /cube (N, ny, nx, n_points) and /coords.*.
    Requires h5py installed.
    """
    try:
        import h5py  # type: ignore
    except Exception as e:
        raise RuntimeError("h5py is required for save_maps_to_h5; please install it.") from e
    first_cube, first_aux = next(iterator)
    if ny is None or nx is None or n_points is None:
        ny, nx, n_points = first_cube.shape
    with h5py.File(path, 'w') as f:
        dset = f.create_dataset('cube', shape=(n_samples, ny, nx, n_points), dtype=first_cube.dtype, compression=compression)
        coords = first_aux.get('coords', {})
        for k, v in coords.items():
            if v is None:
                continue
            f.create_dataset(f'coords/{k}', data=np.asarray(v))
        dset[0] = first_cube
        for i, (cube, aux) in enumerate(iterator, start=1):
            dset[i] = cube

# -------------------- Extra ideas (to implement later) --------------------
# - Baseline drift over time for time-series acquisitions
# - Laser power fluctuations across scans
# - Saturation / clipping effects
# - SERS-specific variability models (log-normal random enhancement factors per-peak)
# - Calibration errors (axis nonlinearity), and pixel binning
# - API to draw peak sets from external spectral libraries (files)
