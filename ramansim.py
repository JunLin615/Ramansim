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
    group: Optional[str] = None  # optional group tag; peaks with same group can share correlated jitter

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
        peak_jitter_std: float = 0.0,
        groupwise_jitter: bool = False,
        anomalies: Optional['AnomalyConfig'] = None,
        return_components: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate a synthetic spectrum.

        Parameters
        ----------
        x : array-like
            Wavenumber axis (cm^-1) or wavelength/time axis; must be sorted.
        peaks : iterable of Peak
            Collection of peak definitions.
        baseline : callable x->y, optional
            Function returning baseline at x. Use BaselinePreset.* helpers.
        noise : NoiseConfig, optional
            Noise parameters. If None, no noise added.
        irf_fwhm : float, optional
            Instrument resolution (FWHM) for Gaussian convolution (applied to signal+baseline before noise).
        drift_ppm : float, default 0.0
            Global axis drift in parts-per-million (positive shifts peaks to higher x).
        return_components : bool
            If True, return dict with components for analysis.

        Returns
        -------
        y : ndarray
            Final simulated spectrum.
        parts : dict
            Components: 'axis', 'signal', 'baseline', 'convolved', 'noise', 'final'.
        """
        rng = np.random.default_rng(self.seed)
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError("x must be 1D array")

        # Apply global drift (ppm)
        x_eff = x * (1.0 + drift_ppm * 1e-6)

        # Sum peaks with optional jitter
        signal = np.zeros_like(x_eff)
        if peaks:
            group_shift: Dict[Optional[str], float] = {}
            if groupwise_jitter and peak_jitter_std > 0:
                for pk in peaks:
                    key = getattr(pk, 'group', None)
                    if key not in group_shift:
                        group_shift[key] = rng.normal(0.0, peak_jitter_std)
            for pk in peaks:
                shift = 0.0
                if peak_jitter_std > 0:
                    if groupwise_jitter:
                        shift = group_shift.get(getattr(pk, 'group', None), 0.0)
                    else:
                        shift = rng.normal(0.0, peak_jitter_std)
                signal += Peak(pos=pk.pos + shift, height=pk.height, fwhm=pk.fwhm, shape=pk.shape, eta=pk.eta).profile(x_eff)

        # Baseline
        x_base = x_eff  # use drifted axis for baseline evaluation
        base = baseline(x_base) if baseline is not None else np.zeros_like(x_base)
        raw = signal + base

        # Convolve with Gaussian IRF if provided
        if irf_fwhm and irf_fwhm > 0:
            sigma = irf_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            dx = np.median(np.diff(x_eff))
            # kernel length ~ +/- 4 sigma
            half_width = int(np.ceil(4 * sigma / dx))
            grid = np.arange(-half_width, half_width + 1) * dx
            kernel = np.exp(-0.5 * (grid / sigma) ** 2)
            kernel /= kernel.sum()
            convolved = np.convolve(raw, kernel, mode='same')
        else:
            convolved = raw.copy()

        y = convolved.copy()
        noise_vec = np.zeros_like(y)

        if noise is not None:
            # multiplicative noise (proportional)
            if noise.multiplicative_sigma and noise.multiplicative_sigma > 0:
                y *= (1.0 + rng.normal(0.0, noise.multiplicative_sigma, size=y.size))

            # additive Gaussian noise
            if noise.gaussian_sigma and noise.gaussian_sigma > 0:
                noise_vec += rng.normal(0.0, noise.gaussian_sigma, size=y.size)

            # Poisson / shot noise: convert to counts with scaling, then back
            if noise.poisson_scale is not None and noise.poisson_scale > 0:
                counts = np.clip(y * noise.poisson_scale, 0, None)
                # Use normal approximations for large counts to keep speed; otherwise rng.poisson
                # Here, directly sample Poisson for generality
                counts_noisy = rng.poisson(counts)
                y = counts_noisy.astype(float) / noise.poisson_scale

            # 1/f noise via colored noise in frequency domain
            if noise.one_over_f_strength and noise.one_over_f_strength > 0:
                one_over_f = colored_noise_1overf(rng, y.size, noise.one_over_f_strength)
                noise_vec += one_over_f

            # Random spikes (cosmic rays)
            if noise.spike_rate and noise.spike_rate > 0:
                n_spikes = rng.binomial(y.size, min(max(noise.spike_rate, 0.0), 1.0))
                if n_spikes > 0:
                    idx = rng.choice(y.size, size=n_spikes, replace=False)
                    # local std estimate using robust MAD
                    local_std = _robust_std(y)
                    heights = rng.uniform(noise.spike_height[0], noise.spike_height[1], size=n_spikes)
                    y[idx] += heights * local_std

            # finally add additive components accumulated in noise_vec
            y = y + noise_vec

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

# -------------------- Statistical generators & validity masks --------------------

@dataclass
class VariationSpec:
    """Statistical variation spec for peak heights across samples.

    kind: 'std' | 'range' | 'rel_range' | 'cv'
    value: float (interpretation depends on kind)
    clip_min: float = 0.0  # minimum allowed sampled height
    distribution: str = 'auto'  # 'auto' | 'normal' | 'uniform' | 'lognormal'
    """
    kind: str
    value: float
    clip_min: float = 0.0
    distribution: str = 'auto'


def _sample_heights(rng: np.random.Generator, base: np.ndarray, spec: VariationSpec) -> np.ndarray:
    b = np.asarray(base, dtype=float)
    if spec.kind == 'std':
        std = float(spec.value)
        if spec.distribution in ('auto','normal'):
            h = b + rng.normal(0.0, std, size=b.shape)
        else:
            # match std for uniform: std = w/sqrt(12) => w = std*sqrt(12)
            w = std * np.sqrt(12.0)
            h = b + rng.uniform(-w/2, w/2, size=b.shape)
    elif spec.kind == 'range':
        w = float(spec.value)
        h = b + rng.uniform(-w/2, w/2, size=b.shape)
    elif spec.kind == 'rel_range':
        rr = float(spec.value)
        h = b * (1.0 + rng.uniform(-rr/2, rr/2, size=b.shape))
    elif spec.kind == 'cv':
        # lognormal with desired coefficient of variation: cv = sqrt(exp(s^2)-1)
        cv = max(float(spec.value), 1e-12)
        sigma = np.sqrt(np.log(cv*cv + 1.0))
        mu = np.log(np.maximum(b, 1e-12)) - 0.5 * sigma * sigma
        h = rng.lognormal(mean=mu, sigma=sigma, size=b.shape)
    else:
        raise ValueError(f"Unknown VariationSpec.kind: {spec.kind}")
    if spec.clip_min is not None:
        h = np.clip(h, spec.clip_min, None)
    return h


def generate_spectra_n(sim: SpectrumSimulator,
                        x: ArrayLike,
                        peaks_template: List[Peak],
                        n: int,
                        height_spec: Optional[VariationSpec] = None,
                        peak_jitter_std: float = 0.0,
                        groupwise_jitter: bool = False,
                        seed0: Optional[int] = None,
                        **simulate_kwargs) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Generate N single spectra with statistical variation in peak heights.

    Returns Y of shape (n, len(x)) and dict with 'heights' (n, n_peaks) used.
    Additional simulate() kwargs (baseline, noise, irf_fwhm, drift_ppm, ...) are supported.
    """
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed0 if seed0 is not None else sim.seed)
    Y = []
    H = []
    base_heights = np.array([pk.height for pk in peaks_template], dtype=float)
    for i in range(n):
        if height_spec is not None:
            h_i = _sample_heights(rng, base_heights, height_spec)
        else:
            h_i = base_heights.copy()
        H.append(h_i)
        peaks_i = [
            Peak(
                pos=pk.pos,
                height=float(h_i[j]),
                fwhm=pk.fwhm,
                shape=pk.shape,
                eta=pk.eta,
                group=getattr(pk, 'group', None)
            )
            for j, pk in enumerate(peaks_template)
        ]
        y, _ = sim.simulate(x, peaks=peaks_i, peak_jitter_std=peak_jitter_std, groupwise_jitter=groupwise_jitter,
                            return_components=False, **simulate_kwargs)
        Y.append(y)
    Y = np.stack(Y, axis=0)
    H = np.stack(H, axis=0)
    return Y, {'heights': H, 'coords': {'r': x}}

@dataclass
class ValidityMaskConfig:
    proportion_valid: float = 0.5   # fraction of pixels/frames with valid analyte signal
    mode: str = 'independent'       # 'independent' | 'gaussian_clusters' | 'smooth_field'
    n_clusters: int = 3             # for cluster mode
    cluster_std: float = 4.0        # pixels
    smooth_sigma: float = 6.0       # for smooth_field
    seed: Optional[int] = None


def _threshold_to_proportion(rng: np.random.Generator, field: np.ndarray, p: float) -> np.ndarray:
    # choose threshold so that proportion of field>th equals ~p
    flat = field.ravel()
    k = max(1, int(np.floor((1.0 - p) * flat.size)))
    thr = np.partition(flat, k)[k]
    return (field > thr).astype(bool)


def make_validity_mask_2d(cfg: ValidityMaskConfig, ny: int, nx: int) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    p = np.clip(cfg.proportion_valid, 0.0, 1.0)
    if cfg.mode == 'independent':
        mask = rng.random((ny, nx)) < p
    elif cfg.mode == 'gaussian_clusters':
        field = np.zeros((ny, nx), dtype=float)
        for _ in range(max(cfg.n_clusters,1)):
            cy = rng.uniform(0, ny-1)
            cx = rng.uniform(0, nx-1)
            y = np.arange(ny)[:, None]
            x = np.arange(nx)[None, :]
            field += np.exp(-0.5 * (((y - cy)**2 + (x - cx)**2) / (cfg.cluster_std**2 + 1e-9)))
        field = (field - field.min()) / (field.max() - field.min() + 1e-12)
        mask = _threshold_to_proportion(rng, field, p)
    elif cfg.mode == 'smooth_field':
        raw = rng.random((ny, nx))
        field = _gaussian_blur2d(raw, sigma=cfg.smooth_sigma)
        field = (field - field.min()) / (field.max() - field.min() + 1e-12)
        mask = _threshold_to_proportion(rng, field, p)
    else:
        raise ValueError(f"Unknown validity mode: {cfg.mode}")
    return mask


def make_validity_mask_1d(cfg: ValidityMaskConfig, n: int) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    p = np.clip(cfg.proportion_valid, 0.0, 1.0)
    if cfg.mode == 'independent':
        return rng.random(n) < p
    elif cfg.mode == 'smooth_field':
        raw = rng.random(n)
        # 1D Gaussian blur via convolution
        sigma = max(cfg.smooth_sigma, 1.0)
        radius = int(np.ceil(4 * sigma))
        x = np.arange(-radius, radius+1)
        k = np.exp(-0.5 * (x/sigma)**2)
        k /= k.sum()
        field = np.convolve(raw, k, mode='same')
        field = (field - field.min()) / (field.max() - field.min() + 1e-12)
        thr = np.partition(field, max(1, int(np.floor((1.0 - p) * n))))[max(1, int(np.floor((1.0 - p) * n)))]
        return field > thr
    else:
        # treat gaussian_clusters same as smooth_field in 1D
        return make_validity_mask_1d(ValidityMaskConfig(proportion_valid=p, mode='smooth_field', smooth_sigma=cfg.smooth_sigma, seed=cfg.seed), n)


def simulate_map2d_with_validity(spectrum_sim: SpectrumSimulator,
                                 x: ArrayLike,
                                 analytes: List[List[Peak]],
                                 analyte_weights: List[float],
                                 cfg: MapConfig,
                                 validity: ValidityMaskConfig,
                                 baseline: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                                 noise: Optional[NoiseConfig] = None,
                                 irf_fwhm: Optional[float] = None,
                                 drift_ppm: float = 0.0) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Simulate a map where only a given proportion of pixels carry valid analyte signal.
    Invalid pixels contain baseline+noise only.
    """
    mapsim = MapSimulator(spectrum_sim)
    # full (signal+baseline+noise)
    cube_sig, aux = mapsim.simulate_map2d(x, analytes, analyte_weights, None, cfg, baseline, noise, irf_fwhm, drift_ppm)
    # baseline+noise only
    cube_bg, _ = mapsim.simulate_map2d(x, analytes=[], analyte_weights=[], per_analyte_fields=None, cfg=cfg, baseline=baseline, noise=noise, irf_fwhm=irf_fwhm, drift_ppm=drift_ppm)
    ny, nx, _ = cube_sig.shape
    mask = make_validity_mask_2d(validity, ny, nx)
    mask3 = mask[:, :, None].astype(float)
    cube = mask3 * cube_sig + (1.0 - mask3) * cube_bg
    aux['valid_mask'] = mask
    return cube, aux


def simulate_time_series(sim: SpectrumSimulator,
                         x: ArrayLike,
                         peaks_template: List[Peak],
                         N: int,
                         validity: Optional[ValidityMaskConfig] = None,
                         height_spec: Optional[VariationSpec] = None,
                         baseline_fn: Optional[Callable[[np.ndarray, int, int], np.ndarray]] = None,
                         noise: Optional[NoiseConfig] = None,
                         irf_fwhm: Optional[float] = None,
                         drift_ppm_fn: Optional[Callable[[int, int], float]] = None,
                         peak_jitter_std: float = 0.0,
                         groupwise_jitter: bool = False,
                         seed0: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Simulate a time series of N spectra with optional validity mask (per-frame),
    height variation, baseline/drift schedules.
    baseline_fn(x, i, N) -> baseline array for frame i
    drift_ppm_fn(i, N) -> drift ppm for frame i
    """
    rng = np.random.default_rng(seed0 if seed0 is not None else sim.seed)
    x = np.asarray(x, dtype=float)
    n_pts = x.size
    # validity mask over time
    if validity is not None:
        mask_t = make_validity_mask_1d(validity, N)
    else:
        mask_t = np.ones(N, dtype=bool)
    # base heights
    base_heights = np.array([pk.height for pk in peaks_template], dtype=float)
    Y = np.zeros((N, n_pts), dtype=float)
    meta = {'valid_mask_t': mask_t, 'coords': {'r': x}}
    for i in range(N):
        # heights
        if height_spec is not None:
            h_i = _sample_heights(rng, base_heights, height_spec)
        else:
            h_i = base_heights
        peaks_i = [
            Peak(
                pos=pk.pos,
                height=float(h_i[j]),
                fwhm=pk.fwhm,
                shape=pk.shape,
                eta=pk.eta,
                group=getattr(pk, 'group', None)
            )
            for j, pk in enumerate(peaks_template)
        ]
        # baseline schedule
        base = None
        if baseline_fn is not None:
            base = lambda xx, i=i: baseline_fn(xx, i, N)
        # drift schedule
        dppm = drift_ppm_fn(i, N) if drift_ppm_fn is not None else 0.0
        # simulate signal or baseline+noise only
        if mask_t[i]:
            y, _ = sim.simulate(x, peaks=peaks_i, baseline=base, noise=noise, irf_fwhm=irf_fwhm,
                                drift_ppm=dppm, peak_jitter_std=peak_jitter_std, groupwise_jitter=groupwise_jitter,
                                return_components=False)
        else:
            y, _ = sim.simulate(x, peaks=[], baseline=base, noise=noise, irf_fwhm=irf_fwhm,
                                drift_ppm=dppm, return_components=False)
        Y[i, :] = y
    return Y, meta

# -------------------- Advanced statistical & anomaly utilities --------------------

@dataclass
class CorrelatedHeightSpec:
    distribution: str  # 'mvnormal' | 'mvlognormal'
    mean: np.ndarray   # (P,)
    cov: np.ndarray    # (P,P)


def sample_correlated_heights(rng: np.random.Generator, spec: CorrelatedHeightSpec) -> np.ndarray:
    if spec.distribution == 'mvnormal':
        return rng.multivariate_normal(mean=np.asarray(spec.mean), cov=np.asarray(spec.cov))
    elif spec.distribution == 'mvlognormal':
        z = rng.multivariate_normal(mean=np.asarray(spec.mean), cov=np.asarray(spec.cov))
        return np.exp(z)
    else:
        raise ValueError("distribution must be 'mvnormal' or 'mvlognormal'")


def generate_spectra_n_correlated(sim: SpectrumSimulator,
                                   x: ArrayLike,
                                   peaks_template: List[Peak],
                                   n: int,
                                   corr_spec: CorrelatedHeightSpec,
                                   peak_jitter_std: float = 0.0,
                                   groupwise_jitter: bool = False,
                                   seed0: Optional[int] = None,
                                   **simulate_kwargs) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Generate N spectra using a multivariate (log)normal for peak heights.
    Returns (Y, meta) where Y shape is (n, len(x)) and meta['heights'] stores sampled heights.
    """
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed0 if seed0 is not None else sim.seed)
    P = len(peaks_template)
    Y, H = [], []
    for _ in range(n):
        h = sample_correlated_heights(rng, corr_spec)
        if h.shape[0] != P:
            raise ValueError("CorrelatedHeightSpec length must equal number of peaks")
        H.append(h)
        peaks_i = [Peak(pos=pk.pos, height=float(h[j]), fwhm=pk.fwhm, shape=pk.shape, eta=pk.eta, group=getattr(pk,'group',None))
                   for j, pk in enumerate(peaks_template)]
        y, _ = sim.simulate(x, peaks=peaks_i, peak_jitter_std=peak_jitter_std, groupwise_jitter=groupwise_jitter,
                            return_components=False, **simulate_kwargs)
        Y.append(y)
    return np.stack(Y,0), {'heights': np.stack(H,0), 'coords': {'r': x}}

# -------------------- Heteroscedastic noise (post-processing) --------------------

def apply_heteroscedastic_gaussian(y: np.ndarray,
                                   ref: Union[float, np.ndarray],
                                   base_sigma: float,
                                   scale: float = 0.0,
                                   power: float = 1.0,
                                   seed: Optional[int] = None) -> np.ndarray:
    """Add Gaussian noise with sigma = base_sigma * (1 + scale * ref**power)."""
    rng = np.random.default_rng(seed)
    if np.isscalar(ref):
        eff = base_sigma * (1.0 + scale * (ref ** power))
        return y + rng.normal(0.0, eff, size=y.shape)
    ref = np.asarray(ref)
    eff = base_sigma * (1.0 + scale * (np.maximum(ref,0.0) ** power))
    return y + rng.normal(0.0, 1.0, size=y.shape) * eff

# -------------------- Anomalies & contamination (post-processing) --------------------

@dataclass
class AnomalyConfig:
    spurious_peak_rate: float = 0.0                    # expected count per spectrum = rate * n_points
    spurious_fwhm_range: Tuple[float,float] = (5.0,20.0)
    spurious_height_range: Tuple[float,float] = (0.2,1.0)
    saturation_level: Optional[float] = None          # clip upper bound
    axis_warp_quad: float = 0.0                       # simple quadratic warp coefficient


def inject_spurious_peaks(x: np.ndarray, y: np.ndarray, cfg: AnomalyConfig, seed: Optional[int]=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(x)
    n_spur = rng.poisson(n * max(cfg.spurious_peak_rate, 0.0))
    out = y.copy()
    for _ in range(n_spur):
        p = rng.uniform(x.min(), x.max())
        f = rng.uniform(*cfg.spurious_fwhm_range)
        h = rng.uniform(*cfg.spurious_height_range)
        out += gaussian(x, p, f, h)
    return out


def apply_axis_warp(x: np.ndarray, y: np.ndarray, quad: float) -> np.ndarray:
    if quad == 0.0:
        return y
    xc = (x - x.mean()) / (x.ptp() + 1e-12)
    xw = x + quad * (xc**2) * x.ptp()
    return np.interp(x, xw, y, left=y[0], right=y[-1])


def apply_saturation(y: np.ndarray, level: Optional[float]) -> np.ndarray:
    if level is None:
        return y
    return np.clip(y, None, level)

# -------------------- Map validity + anomalies wrapper --------------------

@dataclass
class AnomalyMapConfig:
    bad_pixel_rate: float = 0.0
    seed: Optional[int] = None


def simulate_map2d_with_validity_and_anomalies(spectrum_sim: SpectrumSimulator,
                                               x: ArrayLike,
                                               analytes: List[List[Peak]],
                                               analyte_weights: List[float],
                                               cfg: MapConfig,
                                               validity: 'ValidityMaskConfig',
                                               baseline: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                                               noise: Optional['NoiseConfig'] = None,
                                               irf_fwhm: Optional[float] = None,
                                               drift_ppm: float = 0.0,
                                               anomalies: Optional[AnomalyMapConfig] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    cube, aux = simulate_map2d_with_validity(spectrum_sim, x, analytes, analyte_weights, cfg, validity, baseline, noise, irf_fwhm, drift_ppm)
    if anomalies is not None and anomalies.bad_pixel_rate > 0.0:
        ny, nx, _ = cube.shape
        rng = np.random.default_rng(anomalies.seed)
        bad = rng.random((ny, nx)) < anomalies.bad_pixel_rate
        cube[bad, :] = np.nan
        aux['bad_mask'] = bad
    return cube, aux

# -------------------- Label noise & domain shift helpers --------------------

def corrupt_labels(labels: np.ndarray, prob: float, num_classes: int, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).copy()
    mask = rng.random(labels.shape) < prob
    noisy = labels.copy()
    rand = rng.integers(0, num_classes, size=mask.sum())
    noisy[mask] = rand
    return noisy


def make_domain_shift_configs(map_cfg: MapConfig,
                              valid_cfg: 'ValidityMaskConfig',
                              train_delta: Dict[str, float],
                              test_delta: Dict[str, float]) -> Tuple[MapConfig, 'ValidityMaskConfig', MapConfig, 'ValidityMaskConfig']:
    """Return (map_train, valid_train, map_test, valid_test) after applying parameter deltas.
    Supported keys (if present): 'variability','cluster_std','smooth_sigma' on map; 'proportion_valid' on validity.
    """
    def adjust_map(cfg: MapConfig, d: Dict[str,float]) -> MapConfig:
        c = MapConfig(**{**cfg.__dict__})
        for k,v in d.items():
            if hasattr(c, k):
                setattr(c, k, getattr(c,k) + v)
        return c
    def adjust_valid(cfg: 'ValidityMaskConfig', d: Dict[str,float]) -> 'ValidityMaskConfig':
        c = ValidityMaskConfig(**{**cfg.__dict__})
        for k,v in d.items():
            if hasattr(c, k):
                setattr(c, k, getattr(c,k) + v)
        return c
    return adjust_map(map_cfg, train_delta), adjust_valid(valid_cfg, train_delta), adjust_map(map_cfg, test_delta), adjust_valid(valid_cfg, test_delta)

# -------------------- End of advanced utilities --------------------
