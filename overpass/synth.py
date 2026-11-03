"""Synthetic training tiles for the flyover detector.

A tile is a whitened spectrogram crop (F, T) in dB above the per-bin median, exactly the
quantity dopplertrack.audio.compute_spectrogram produces. Positive tiles carry one or more
Doppler curves rendered from *real* ADS-B segments seen by a random ground observer, so the
detector learns the real shapes (turns, climbs, speed changes), not the straight-pass ideal.
Negatives carry the same background model without a curve. Backgrounds are either synthetic
(noise + stationary tone residues + broadband bursts + slow non-Doppler sweeps) or, when real
spectrogram crops are supplied, real quiet windows of the benchmark recordings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dopplertrack.doppler import curve_from_trajectory, speed_of_sound
from dopplertrack.search import Segment


@dataclass
class TileSpec:
    n_f: int = 256          # frequency bins per tile (df = 4000/8192 = 0.488 Hz -> 125 Hz)
    n_t: int = 480          # frames per tile (hop 0.25 s -> 120 s)
    df: float = 4000.0 / 8192.0
    dt: float = 0.25
    f_lo: float = 40.0      # lowest bin of the full spectrogram the tile may start at
    f_hi: float = 450.0


def curve_mask(t: np.ndarray, f: np.ndarray, t_obs: np.ndarray, ratio: np.ndarray, f0: float,
               harmonics: int = 2, half_width_hz: float = 1.2) -> np.ndarray:
    """Soft (F, T) mask of a Doppler comb: Gaussian ridge around h*f0*ratio(t), h = 1..harmonics."""
    ok = np.isfinite(ratio)
    fc = np.interp(t, t_obs[ok], ratio[ok], left=np.nan, right=np.nan) * f0
    M = np.zeros((len(f), len(t)), np.float32)
    for h in range(1, harmonics + 1):
        d = np.abs(f[:, None] - h * fc[None, :])
        g = np.exp(-0.5 * (d / half_width_hz) ** 2)
        M = np.maximum(M, np.nan_to_num(g, nan=0.0).astype(np.float32))
    return M


def random_observer(seg: Segment, rng: np.random.Generator, d_range=(300.0, 9000.0)) -> tuple[np.ndarray, int]:
    """Ground point at a random slant distance and azimuth from a random track sample."""
    j = int(rng.integers(len(seg.t) // 5, max(len(seg.t) * 4 // 5, len(seg.t) // 5 + 1)))
    p = seg.enu[j]
    d = float(rng.uniform(*d_range))
    h = p[2] - float(rng.uniform(0.0, 300.0))       # observer below the aircraft
    horiz = np.sqrt(max(d * d - h * h, 100.0 ** 2))
    az = float(rng.uniform(0, 2 * np.pi))
    obs = np.array([p[0] + horiz * np.cos(az), p[1] + horiz * np.sin(az), p[2] - h])
    return obs, j


def synthetic_background(spec: TileSpec, rng: np.random.Generator) -> np.ndarray:
    """Whitened-dB background: noise, tone residues, bursts, slow sweeps (non-Doppler lines)."""
    F, T = spec.n_f, spec.n_t
    S = rng.normal(0.0, float(rng.uniform(1.5, 3.5)), (F, T)).astype(np.float32)
    # a few stationary tone residues (mains hum harmonics survive whitening as faint flicker)
    for _ in range(int(rng.integers(0, 4))):
        k = int(rng.integers(0, F)); S[k] += rng.normal(0, 2.0, T).astype(np.float32)
    # broadband bursts (doors, voices, vehicles)
    for _ in range(int(rng.integers(0, 6))):
        a = int(rng.integers(0, T)); w = int(rng.integers(1, 12))
        S[:, a:a + w] += float(rng.uniform(3, 12))
    # slow non-Doppler sweeps (engines revving, sirens) – linear or parabolic lines
    for _ in range(int(rng.integers(0, 3))):
        a = int(rng.integers(0, T)); L = int(rng.integers(40, T))
        k0 = float(rng.uniform(0, F)); k1 = float(rng.uniform(0, F))
        tt = np.arange(a, min(a + L, T)); kk = k0 + (k1 - k0) * (tt - a) / max(L, 1)
        for ti, ki in zip(tt, kk):
            lo = int(max(ki - 1, 0)); hi = int(min(ki + 2, F))
            S[lo:hi, ti] += float(rng.uniform(3, 9))
    return S


def render_tile(segs: list[Segment], spec: TileSpec, rng: np.random.Generator, c: float | None = None,
                background: np.ndarray | None = None, p_positive: float = 0.7, max_curves: int = 2):
    """One (tile, mask) pair. Returns (S (F,T) float32, M (F,T) float32, n_curves)."""
    c = speed_of_sound() if c is None else c
    S = (synthetic_background(spec, rng) if background is None else background.astype(np.float32).copy())
    M = np.zeros_like(S)
    band = spec.n_f * spec.df
    # the tile band is chosen so that a harmonic of the first curve's fundamental falls inside it
    f0 = float(rng.uniform(60.0, 230.0)); h_in = int(rng.integers(1, 4))
    f_start = float(np.clip(h_in * f0 - rng.uniform(10.0, band - 10.0), spec.f_lo, spec.f_hi - band))
    f = f_start + np.arange(spec.n_f) * spec.df
    t = np.arange(spec.n_t) * spec.dt
    n = 0
    if rng.uniform() < p_positive and segs:
        n = int(rng.integers(1, max_curves + 1))
        for k in range(n):
            seg = segs[int(rng.integers(len(segs)))]
            if len(seg.t) < 60:
                continue
            obs, j = random_observer(seg, rng)
            t_obs, ratio, _ = curve_from_trajectory(seg.t, seg.enu, obs, f0=1.0, c=c)
            # place the heard CPA somewhere inside the tile
            t_cpa_heard = t_obs[j]
            shift = float(rng.uniform(0.15, 0.85)) * t[-1] - (t_cpa_heard - t_obs[0]) + 0.0
            t_local = t_obs - t_obs[0] + shift
            if k > 0:
                f0 = float(rng.uniform(60.0, 230.0))
            harm = int(rng.integers(1, 4))
            amp = float(rng.uniform(3.0, 12.0))
            m = curve_mask(t, f, t_local, ratio, f0, harmonics=harm, half_width_hz=float(rng.uniform(0.8, 1.8)))
            # harmonics weaker than the fundamental; amplitude flickers along the curve
            flicker = (1.0 + 0.25 * rng.normal(size=len(t))).clip(0.3, 1.7).astype(np.float32)
            S += amp * m * flicker[None, :]
            M = np.maximum(M, (m > 0.3).astype(np.float32))
    return S, M, n


def make_dataset(segs: list[Segment], n_tiles: int, seed: int = 0, spec: TileSpec | None = None,
                 backgrounds: list[np.ndarray] | None = None, c: float | None = None):
    """Arrays X (N,1,F,T) float32 and Y (N,1,F,T) float32 for training the detector."""
    spec = spec or TileSpec()
    rng = np.random.default_rng(seed)
    X = np.zeros((n_tiles, 1, spec.n_f, spec.n_t), np.float32)
    Y = np.zeros_like(X)
    for i in range(n_tiles):
        bg = None
        if backgrounds and rng.uniform() < 0.6:
            b = backgrounds[int(rng.integers(len(backgrounds)))]
            if b.shape[0] >= spec.n_f and b.shape[1] >= spec.n_t:
                r0 = int(rng.integers(0, b.shape[0] - spec.n_f + 1)); c0 = int(rng.integers(0, b.shape[1] - spec.n_t + 1))
                bg = b[r0:r0 + spec.n_f, c0:c0 + spec.n_t]
        S, M, _ = render_tile(segs, spec, rng, c=c, background=bg)
        X[i, 0] = S; Y[i, 0] = M
    return X, Y


def synth_audio(segs: list[Segment], obs_enu: np.ndarray, f0s: list[float], t0_unix: float, dur_s: float,
                sr: int = 4000, snr_amp: float = 0.05, noise: float = 0.4, c: float | None = None, seed: int = 1,
                harmonics: int = 2) -> np.ndarray:
    """Time-domain synthetic recording heard at obs_enu (for tests and for the scrubber evaluation)."""
    c = speed_of_sound() if c is None else c
    rng = np.random.default_rng(seed)
    tt = np.arange(0, dur_s, 1 / sr)
    x = noise * rng.standard_normal(len(tt)).astype(np.float32)
    for seg, f0 in zip(segs, f0s):
        to, ratio, _ = curve_from_trajectory(seg.t, seg.enu, obs_enu, f0=1.0, c=c)
        finst = np.interp(tt + t0_unix, to, ratio, left=np.nan, right=np.nan) * f0
        m = np.isfinite(finst)
        ph = 2 * np.pi * np.cumsum(np.nan_to_num(finst, nan=0.0)) / sr
        for h in range(1, harmonics + 1):
            x[m] += (snr_amp / h) * np.sin(h * ph[m]).astype(np.float32)
    return x
