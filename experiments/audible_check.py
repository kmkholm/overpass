"""Audible-signal check: for every real propeller pass within `--max-km` of a camera during a clip, model the curve
the camera should have heard from the real trajectory at the true position, sweep the unknown fundamental over
60-250 Hz, and record the best sweep-weighted sideband contrast (dB). Appends to results/audible.csv (clips already
present are skipped) so the table in the paper covers every session.

python experiments/audible_check.py --manifest data/manifest.csv --work work --out results/audible.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dopplertrack.adsb import load_traces
from dopplertrack.audio import Spectrogram
from dopplertrack.doppler import curve_from_trajectory, speed_of_sound
from dopplertrack.search import build_segments
from dopplertrack.tracks import curve_contrast, local_whiten, sweep_weights
from overpass.benchmark import load_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv"); ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="results/audible.csv"); ap.add_argument("--max-km", type=float, default=12.0)
    a = ap.parse_args()
    c = speed_of_sound(); man = load_manifest(a.manifest)
    out = Path(a.out); prev = pd.read_csv(out) if out.exists() else pd.DataFrame()
    done = set(prev.clip_id) if len(prev) else set()
    f0s = np.arange(60.0, 250.1, 5.0); rows = []
    for row in man.itertuples():
        if row.clip_id in done or not row.utc_start:
            continue
        sp_p = Path(a.work) / f"{row.clip_id}.spec.npz"; tr_p = Path(a.work) / f"{row.clip_id}.traces.npz"
        if not sp_p.exists() or not tr_p.exists():
            continue
        sp = Spectrogram.load(sp_p); t_s = pd.Timestamp(row.utc_start).timestamp(); dur = float(sp.t[-1])
        segs = build_segments(load_traces(tr_p), float(row.lat), float(row.lon), clip_enu=(-15000, 15000, -15000, 15000))
        obs = np.zeros(3); n = 0
        for s in segs:
            dist = np.linalg.norm(s.enu - obs, axis=1); j = int(np.argmin(dist))
            if dist[j] > a.max_km * 1000 or j in (0, len(dist) - 1):
                continue
            t_heard = s.t[j] + dist[j] / c - t_s
            if not (30 < t_heard < dur - 30):
                continue
            t_obs, ratio, _ = curve_from_trajectory(s.t, s.enu, obs, f0=1.0, c=c); t_obs = t_obs - t_s
            win = local_whiten(sp.crop(t_heard - 70, t_heard + 70, 40, 450))
            tq = np.linspace(t_heard - 45, t_heard + 45, 120); rq = np.interp(tq, t_obs, ratio)
            w = sweep_weights(rq, tq); best, bf = -np.inf, np.nan
            for f0 in f0s:
                val = float(np.sum(w * curve_contrast(win, tq, f0 * rq)) / np.sum(w))
                if val > best:
                    best, bf = val, f0
            rows.append({"clip_id": row.clip_id, "region": row.region, "t_audio": int(t_heard), "d_km": round(dist[j] / 1000, 2),
                         "v": int(round(float(s.speed[j]))), "icao": s.icao, "best_contrast_db": round(best, 2), "f0": bf}); n += 1
            print(rows[-1], flush=True)
        print(f"{row.clip_id}: {n} passes", flush=True)
    df = pd.concat([prev, pd.DataFrame(rows)], ignore_index=True) if len(prev) else pd.DataFrame(rows)
    df.to_csv(out, index=False); print(f"{len(rows)} new passes; {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
