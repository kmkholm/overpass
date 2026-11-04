"""The countermeasure: track-and-fill removal of propeller Doppler combs.

High-pass filtering, loud masking and telephone codecs were shown not to remove the leak
(the comb survives at 60-400 Hz under speech). The scrubber instead (1) locates each flyover
with the detector (or takes known events), (2) tracks the fundamental ridge frame by frame
with the Viterbi ridge of dopplertrack.tracks, (3) in an STFT of the original-rate audio
replaces the bins within +-half_width_hz of h*f(t), h = 1..harmonics, by the magnitude of the
flanking bins with random phase ("fill"), so neither a bright comb nor a dark trench is left
for the attacker's whitened spectrogram. Speech above 400 Hz is untouched; the affected band
is a few hertz wide and moves with the tone, so PESQ/STOI change is expected to be small.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as ss

from dopplertrack.audio import Spectrogram, compute_spectrogram
from dopplertrack.doppler import ideal_curve, speed_of_sound
from dopplertrack.tracks import EventFit, local_whiten, ridge


@dataclass
class TonalTrack:
    t: np.ndarray        # recording seconds (frame centres)
    f: np.ndarray        # fundamental frequency per frame (Hz)
    harmonics: int


def track_from_event(sp: Spectrogram, ev: EventFit, harmonics: int = 3, band_hz: float = 10.0,
                     c: float | None = None, pad_s: float = 20.0) -> TonalTrack:
    """Follow the measured ridge around the fitted ideal curve (+-band_hz) over the event window."""
    c = speed_of_sound() if c is None else c
    t0, t1 = ev.t_start - pad_s, ev.t_end + pad_s
    w = local_whiten(sp.crop(t0, t1, max(ev.f_post * 0.8, sp.f[0]), min(ev.f_pre * 1.25, sp.f[-1])))
    guide = ideal_curve(w.t, ev.f0, ev.v, ev.d, ev.t_cpa_emit, c=c)
    guide = np.where(np.isfinite(guide), guide, ev.f0)
    # keep only bins within band_hz of the guide; everything else -> very negative
    S = np.where(np.abs(w.f[:, None] - guide[None, :]) <= band_hz, w.S, -1e3).astype(np.float32)
    f_track = ridge(Spectrogram(w.t, w.f, S), max_step_bins=3, smooth=1.0)
    # where the ridge has no support (contrast ~ 0) fall back to the guide
    on = w.sample(w.t, f_track) > 1.0
    f_track = np.where(on, f_track, guide)
    f_track = ss.medfilt(f_track, 5)
    return TonalTrack(w.t, f_track, harmonics)


def scrub(x: np.ndarray, sr: int, tracks: list[TonalTrack], half_width_hz: float = 2.5,
          nperseg: int = 4096, hop: int = 512, mode: str = "fill", atten_db: float = 40.0,
          seed: int = 0, fill_window_s: float = 60.0) -> np.ndarray:
    """Return the scrubbed waveform (same length as x).

    mode 'fill'  : each comb bin is replaced by the median magnitude of the *same* bin over the
                   surrounding +-fill_window_s, taken only from frames in which the comb is not
                   near that bin, with random phase; the per-bin level is preserved, so a whitened
                   spectrogram shows neither a bright comb nor a dark trench.
    mode 'flank' : the earlier variant, magnitude of the flanking bins at the same frame.
    mode 'notch' : attenuation by atten_db (leaves a trench; ablation)."""
    rng = np.random.default_rng(seed)
    f, t, Z = ss.stft(x, sr, nperseg=nperseg, noverlap=nperseg - hop, window="hann", boundary="zeros", padded=True)
    Z = Z.copy(); A = np.abs(Z)
    df = f[1] - f[0]; dt = t[1] - t[0]
    wb = max(1, int(round(half_width_hz / df)))
    g = 10 ** (-atten_db / 20)
    W = int(round(fill_window_s / dt))
    # comb occupancy: for every frame the bins occupied by any harmonic of any track (with margin)
    occ = np.zeros(Z.shape, bool)
    spans = []       # (frame k, lo, hi) to replace
    for tr in tracks:
        fi = np.interp(t, tr.t, tr.f, left=np.nan, right=np.nan)
        for k in np.flatnonzero(np.isfinite(fi)):
            for h in range(1, tr.harmonics + 1):
                b = int(round(h * fi[k] / df))
                lo, hi = max(b - wb, 1), min(b + wb + 1, len(f) - 1)
                if hi > lo:
                    spans.append((k, lo, hi)); occ[max(lo - 2 * wb, 0):min(hi + 2 * wb, len(f)), k] = True
    # The filler must be a *consistent* STFT (one real noise signal masked and scaled), not independent
    # random phases per frame: with 7/8 overlap, independent phases cancel in the overlap-add and leave a
    # trench 8-10 dB deep. So the comb bins are zeroed, and a white-noise signal's STFT, masked to the same
    # bins and scaled to the target magnitude, is added back.
    noise = rng.standard_normal(len(x)).astype(np.float32)
    _, _, N = ss.stft(noise, sr, nperseg=nperseg, noverlap=nperseg - hop, window="hann", boundary="zeros", padded=True)
    n_ref = float(np.median(np.abs(N)))
    G = np.zeros(Z.shape, np.float32)
    for k, lo, hi in spans:
        if mode == "notch":
            Z[lo:hi, k] *= g; continue
        if mode == "flank":
            fl = A[max(lo - wb, 1):lo, k]; fr = A[hi:min(hi + wb, len(f)), k]
            mag = np.full(hi - lo, np.concatenate([fl, fr]).mean() if (len(fl) + len(fr)) else 0.0)
        else:
            k0, k1 = max(k - W, 0), min(k + W + 1, Z.shape[1])
            blk = A[lo:hi, k0:k1]; free = ~occ[lo:hi, k0:k1]
            mag = np.array([np.median(row[m]) if m.sum() >= 8 else np.median(row) for row, m in zip(blk, free)])
        Z[lo:hi, k] = 0.0
        G[lo:hi, k] = mag / n_ref
    if mode != "notch":
        _, fill = ss.istft(N * G, sr, nperseg=nperseg, noverlap=nperseg - hop, window="hann", input_onesided=True, boundary=True)
    else:
        fill = np.zeros(len(x))
    _, y = ss.istft(Z, sr, nperseg=nperseg, noverlap=nperseg - hop, window="hann", input_onesided=True, boundary=True)
    y = y[:len(x)] + fill[:len(x)][:len(y[:len(x)])]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y.astype(np.float32)


def highpass(x: np.ndarray, sr: int, fc: float = 450.0, order: int = 6) -> np.ndarray:
    """Baseline defence 1: high-pass filter (shown to fail in the video)."""
    sos = ss.butter(order, fc, btype="highpass", fs=sr, output="sos")
    return ss.sosfiltfilt(sos, x).astype(np.float32)


def add_masker(x: np.ndarray, sr: int, level: float = 0.3, seed: int = 0) -> np.ndarray:
    """Baseline defence 2: loud broadband masking noise played over the recording."""
    rng = np.random.default_rng(seed)
    return (x + level * rng.standard_normal(len(x))).astype(np.float32)


def codec_roundtrip(x: np.ndarray, sr: int, bitrate: str = "12k") -> np.ndarray:
    """Baseline defence 3: narrowband speech codec (Opus at a phone-call rate) via ffmpeg."""
    import subprocess, tempfile, os, soundfile as sf
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "in.wav"); b = os.path.join(d, "out.opus"); cc = os.path.join(d, "out.wav")
        sf.write(a, x, sr)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a, "-c:a", "libopus", "-b:a", bitrate, "-application", "voip", b], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", b, "-ar", str(sr), "-ac", "1", cc], check=True)
        y, _ = sf.read(cc, dtype="float32")
    y = y[:len(x)]
    return np.pad(y, (0, len(x) - len(y))) if len(y) < len(x) else y


def residual_contrast(x_before: np.ndarray, x_after: np.ndarray, sr: int, ev: EventFit, harmonics: int = 2,
                      c: float | None = None) -> tuple[float, float]:
    """Mean sideband contrast under the fitted curve before and after a defence (dB).
    The attacker's score at the true point; a defence works when the 'after' value is ~0."""
    from dopplertrack.tracks import curve_contrast, sweep_weights
    c = speed_of_sound() if c is None else c
    out = []
    for x in (x_before, x_after):
        sp = compute_spectrogram(x, sr, nperseg=8192 * sr // 4000, hop_s=0.25, fmin=40, fmax=450, whiten_window_s=None)
        w = local_whiten(sp.crop(ev.t_start - 30, ev.t_end + 30, max(ev.f_post * 0.7, sp.f[0]), min(ev.f_pre * 1.3 * harmonics, sp.f[-1])))
        tt = np.linspace(ev.t_start, ev.t_end, 400)
        base = ideal_curve(tt, ev.f0, ev.v, ev.d, ev.t_cpa_emit, c=c)
        wt = sweep_weights(base / ev.f0, tt)
        s = sum(float(np.sum(wt * curve_contrast(w, tt, h * base)) / np.sum(wt)) for h in range(1, harmonics + 1)) / harmonics
        out.append(s)
    return out[0], out[1]


def speech_quality(ref: np.ndarray, deg: np.ndarray, sr: int) -> dict:
    """PESQ (wideband if sr=16000, narrowband if 8000) and STOI, when the packages are installed."""
    out = {}
    try:
        from pesq import pesq
        if sr in (8000, 16000):
            out["pesq"] = float(pesq(sr, ref, deg, "wb" if sr == 16000 else "nb"))
    except Exception as e:      # pragma: no cover
        out["pesq_error"] = str(e)
    try:
        from pystoi import stoi
        out["stoi"] = float(stoi(ref, deg, sr, extended=False))
    except Exception as e:      # pragma: no cover
        out["stoi_error"] = str(e)
    return out
