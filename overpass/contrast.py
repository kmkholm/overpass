"""Contrastive locked scoring.

dopplertrack scores a candidate by the mean whitened brightness under the modelled curve.
Wherever the spectrogram is busy (bursts, voices, machinery) a wrong curve scores well, which
is why the baseline's margin over a wrong flight pair is only 4.9 %. Here each curve is scored
by its *contrast* (on-curve minus sideband brightness, as in tracks.curve_contrast) and the
contrast is z-scored against a null distribution obtained by sliding the same curve to times
well away from its placement. A candidate therefore scores high only when the curve is bright
*relative to the same recording's own clutter*.

The locking rule is unchanged: event 1 absorbs the measurement error (time shift and f0 scale
are swept), events 2..N are placed with the same time shift, so their timing is fixed by flight
geometry; each event keeps its own f0 scale because the aircraft differ.
"""
from __future__ import annotations

import numpy as np

from dopplertrack.audio import Spectrogram
from dopplertrack.doppler import speed_of_sound
from dopplertrack.score import EventWindow, modelled_curve
from dopplertrack.search import Segment
from dopplertrack.tracks import EventFit, local_whiten, sweep_weights

CONTRAST_DF = 0.04
NULL_OFFSETS = np.r_[-np.arange(24.0, 61.0, 6.0), np.arange(24.0, 61.0, 6.0)]    # s, 14 offsets


class ContrastWindow(EventWindow):
    """EventWindow whose sweep uses sideband contrast and sweep weights, plus a null z-score."""

    @classmethod
    def from_fit(cls, sp: Spectrogram, fit: EventFit, harmonics: int = 1, pad_s: float = 80.0):
        w = sp.crop(fit.t_start - pad_s, fit.t_end + pad_s, max(fit.f_post * 0.75, sp.f[0]),
                    min(fit.f_pre * 1.3 * harmonics, sp.f[-1]))
        return cls(fit, local_whiten(w), harmonics)

    def _sample_contrast(self, T, F):
        v = self.sp.sample(T, F) - 0.5 * (self.sp.sample(T, F * (1 + CONTRAST_DF)) + self.sp.sample(T, F * (1 - CONTRAST_DF)))
        return v

    def sweep_scores(self, t_audio, ratio, shifts, f0_scales, n: int = 160) -> np.ndarray:
        """Weighted mean contrast under the curve for every (shift, f0_scale): (S, K)."""
        tq = np.linspace(self.fit.t_start, self.fit.t_end, n)
        rel = tq[None, :] - np.asarray(shifts, float)[:, None]
        ok = np.isfinite(ratio)
        rat = np.interp(rel, t_audio[ok], ratio[ok], left=np.nan, right=np.nan)       # (S, n)
        valid = np.isfinite(rat).sum(1) >= 10
        w = np.where(np.isnan(rat), 0.0, sweep_weights(np.nan_to_num(rat, nan=1.0), tq, axis=1))
        wsum = np.maximum(w.sum(1), 1e-12)
        total = np.zeros((len(shifts), len(f0_scales)))
        for h in range(1, self.harmonics + 1):
            F = h * self.fit.f0 * np.asarray(f0_scales)[None, :, None] * rat[:, None, :]      # (S, K, n)
            T = np.broadcast_to(tq[None, None, :], F.shape)
            v = self._sample_contrast(T.ravel(), np.nan_to_num(F.ravel(), nan=-1.0)).reshape(F.shape)
            v = np.where(np.isfinite(F), v, 0.0)
            total += np.einsum("sn,skn->sk", w, v) / wsum[:, None]
        total[~valid] = 0.0
        return total / self.harmonics

    def z_scores(self, t_audio, ratio, shifts, f0_scales, offsets=NULL_OFFSETS):
        """(z (S,K), raw (S,K)): raw sweep scores, and the z-score of the best (shift, f0) cell
        against the same curve placed at `offsets` seconds away (null distribution of clutter
        under this curve shape). The null is evaluated only at the best cell, which keeps the
        cost at one full sweep plus len(offsets) single-curve samples; other cells get -inf."""
        shifts = np.asarray(shifts, float); f0_scales = np.asarray(f0_scales, float)
        raw = self.sweep_scores(t_audio, ratio, shifts, f0_scales)
        i, j = np.unravel_index(np.argmax(raw), raw.shape)
        null = np.array([self.sweep_scores(t_audio, ratio, [shifts[i] + o], [f0_scales[j]])[0, 0] for o in offsets])
        z = np.full(raw.shape, -np.inf)
        z[i, j] = (raw[i, j] - null.mean()) / (null.std() + 1e-3)
        return z, raw


def contrast_locked_score(wins: list[ContrastWindow], segs: list[Segment], obs_enu, t_heard1: float,
                          c: float | None = None, shifts=np.arange(-8, 8.01, 0.5),
                          f0_scales=np.arange(0.95, 1.0501, 0.005), raw: bool = False):
    """N-event locked score. wins[k] pairs with segs[k]. Returns (total_z, per_event_z, shift, per_event_raw)."""
    c = speed_of_sound() if c is None else c
    t0_map = wins[0].fit.t_cpa - t_heard1
    to, r = modelled_curve(segs[0], obs_enu, c)
    if raw:
        z1 = wins[0].sweep_scores(to + t0_map, r, shifts, f0_scales); raw1 = z1
    else:
        z1, raw1 = wins[0].z_scores(to + t0_map, r, shifts, f0_scales)
    i, j = np.unravel_index(np.argmax(z1), z1.shape)
    sh = float(shifts[i]); zs = [float(z1[i, j])]; raws = [float(raw1[i, j])]
    for w, s in zip(wins[1:], segs[1:]):
        to, r = modelled_curve(s, obs_enu, c)
        if raw:
            zk = w.sweep_scores(to + t0_map, r, [sh], f0_scales); rk = zk
        else:
            zk, rk = w.z_scores(to + t0_map, r, [sh], f0_scales)
        k = int(np.argmax(zk)); zs.append(float(zk[0, k])); raws.append(float(rk[0, k]))
    return float(sum(zs)), zs, sh, raws
