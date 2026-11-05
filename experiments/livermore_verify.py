"""Known-time verification on the 40 Livermore smartphone flyovers: the forensic use case in which an examiner
has a recording, its start time (the dataset publishes the UTC start) and a claimed position, and asks whether
the flyover heard in the recording is consistent with the public flight tracks seen from that position.

For the inferred recorder site (results/livermore_site.json) and for decoy sites on rings around it, every
propeller flight of that day whose heard closest approach at the site falls within `--tol` s of the detected
event's heard CPA is scored with the contrastive locked score; the site's score is the best consistent flight
(zero when no flight is consistent). The true site should outrank the decoys.

python experiments/livermore_verify.py --fits results/livermore_fits.csv --out results/livermore_verify.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dopplertrack.adsb import load_traces
from dopplertrack.audio import load_audio, compute_spectrogram
from dopplertrack.doppler import speed_of_sound
from dopplertrack.search import build_segments
from dopplertrack.tracks import EventFit
from overpass.contrast import ContrastWindow, contrast_locked_score

BEARINGS = np.arange(0, 360, 45)
RINGS_M = (500.0, 1000.0, 2000.0, 5000.0, 10000.0)


def site_score(win, segs, obs, t_target, tol, c, f0_scales, max_range):
    """Best locked z over the flights consistent with the known heard time at this observer; (z, n_consistent, icao)."""
    best, n, who = 0.0, 0, ""
    for s in segs:
        dist = np.linalg.norm(s.enu - obs, axis=1); j = int(np.argmin(dist))
        if dist[j] > max_range or j in (0, len(dist) - 1):
            continue
        t_heard = s.t[j] + dist[j] / c
        if abs(t_heard - t_target) > tol:
            continue
        n += 1
        z = contrast_locked_score([win], [s], obs, float(t_heard), c=c, f0_scales=f0_scales)[0]
        if z > best:
            best, who = z, s.icao
    return best, n, who


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest_livermore.csv"); ap.add_argument("--work", default="work_livermore")
    ap.add_argument("--site", default="results/livermore_site.json"); ap.add_argument("--fits", default="results/livermore_fits.csv")
    ap.add_argument("--out", default="results/livermore_verify.csv"); ap.add_argument("--tol", type=float, default=15.0)
    ap.add_argument("--max-range", type=float, default=12000.0); ap.add_argument("--harmonics", type=int, default=1)
    a = ap.parse_args()
    c = speed_of_sound(); site = json.load(open(a.site)); fits = pd.read_csv(a.fits).set_index("case")
    man = pd.read_csv(a.manifest).set_index("clip_id")
    scales = {"narrow": np.arange(0.95, 1.0501, 0.01), "wide": np.arange(0.80, 1.2501, 0.01)}
    decoys = [(float(r * np.sin(np.radians(b))), float(r * np.cos(np.radians(b))), r, b) for r in RINGS_M for b in BEARINGS]
    rows = []
    for case, m in man.iterrows():
        tr_path = Path(a.work) / f"{case}.traces.npz"
        if case not in fits.index or not np.isfinite(fits.loc[case].get("v_mps", np.nan)) or not tr_path.exists():
            rows.append({"case": case, "status": "no_fit" if case in fits.index and not np.isfinite(fits.loc[case].get("v_mps", np.nan)) else "no_traces"})
            print(case, rows[-1]["status"], flush=True); continue
        f = fits.loc[case]
        x, sr = load_audio(m.path); sp = compute_spectrogram(x, sr, whiten_window_s=None); dur = len(x) / sr
        fit = EventFit(t_cpa=float(f.t_cpa_s), t_cpa_emit=float(f.t_cpa_s) - float(f.d_m) / c, f0=float(f.f0_hz), v=float(f.v_mps), d=float(f.d_m),
                       t_start=0.0, t_end=dur, score=float(f.fit_score), f_pre=float(f.f0_hz) * c / (c - float(f.v_mps)), f_post=float(f.f0_hz) * c / (c + float(f.v_mps)))
        win = ContrastWindow.from_fit(sp, fit, a.harmonics, pad_s=10.0)
        traces = load_traces(tr_path)
        R = max(RINGS_M) + a.max_range + 2000.0
        segs = build_segments(traces, site["lat"], site["lon"], clip_enu=(-R, R, -R, R))
        t_s = pd.Timestamp(m.utc_start).timestamp(); t_target = t_s + fit.t_cpa
        rec = {"case": case, "status": "ok", "n_segments": len(segs), "fit_v": fit.v, "fit_d": fit.d, "fit_f0": fit.f0}
        for name, sc in scales.items():
            obs = np.array([0.0, 0.0, site["h_m"]])
            z_true, n_true, who = site_score(win, segs, obs, t_target, a.tol, c, sc, a.max_range)
            zd = []
            for dx, dy, r, b in decoys:
                z, n, _ = site_score(win, segs, np.array([dx, dy, site["h_m"]]), t_target, a.tol, c, sc, a.max_range)
                zd.append((z, n, r, b))
            zs = np.array([z for z, _, _, _ in zd]); rank = int((zs > z_true).sum() + 1)
            k = int(np.argmax(zs))
            rec.update({f"{name}_z_true": round(z_true, 2), f"{name}_n_consistent_true": n_true, f"{name}_icao": who,
                        f"{name}_rank_true": rank, f"{name}_best_decoy_z": round(float(zs[k]), 2), f"{name}_best_decoy_r": zd[k][2],
                        f"{name}_decoys_with_flight": int(sum(1 for _, n, _, _ in zd if n > 0)),
                        f"{name}_decoy_z_median": round(float(np.median(zs)), 2)})
        rows.append(rec)
        print({k: rec[k] for k in ("case", "n_segments", "narrow_z_true", "narrow_n_consistent_true", "narrow_rank_true", "narrow_best_decoy_z", "wide_z_true", "wide_rank_true")}, flush=True)
    df = pd.DataFrame(rows); Path(a.out).parent.mkdir(parents=True, exist_ok=True); df.to_csv(a.out, index=False)
    ok = df[df.status == "ok"]
    print(f"\n{len(ok)} of {len(df)} recordings scored")
    for name in scales:
        if len(ok):
            zt = ok[f"{name}_z_true"].to_numpy(); r1 = (ok[f"{name}_rank_true"] == 1).mean()
            print(f"{name}: true site first in {r1 * 100:.0f} % ({int((ok[f'{name}_rank_true'] == 1).sum())}/{len(ok)}); "
                  f"median z true {np.median(zt):.2f}, median best decoy {ok[f'{name}_best_decoy_z'].median():.2f}; "
                  f"true site had a consistent flight in {int((ok[f'{name}_n_consistent_true'] > 0).sum())}/{len(ok)}")


if __name__ == "__main__":
    main()
