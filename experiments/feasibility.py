"""Feasibility of every clip: how many propeller aircraft actually passed within `--range` km of
the true position during the recording (from the ADS-B traces and the clip's UTC start). A clip
with fewer than two passes cannot be located by any method; the benchmark reports results on
all clips and on the feasible subset separately.

python experiments/feasibility.py --manifest data/manifest.csv --work work --out results/feasibility.csv
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from dopplertrack.adsb import load_traces
from dopplertrack.audio import Spectrogram
from dopplertrack.search import Grid, build_segments, cpa_table
from dopplertrack.terrain import Terrain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--work", default="work"); ap.add_argument("--out", required=True)
    ap.add_argument("--range", type=float, default=12.0, help="km")
    a = ap.parse_args()
    man = pd.read_csv(a.manifest); terrain = Terrain(); rows = []
    for r in man.itertuples():
        tp = Path(a.work) / f"{r.clip_id}.traces.npz"; sp_p = Path(a.work) / f"{r.clip_id}.spec.npz"
        if not tp.exists() or not isinstance(r.utc_start, str) or not r.utc_start:
            rows.append({"clip_id": r.clip_id, "region": r.region, "split": r.split, "n_passes": np.nan}); continue
        t0 = datetime.fromisoformat(r.utc_start).timestamp()
        dur = float(Spectrogram.load(sp_p).t[-1]) if sp_p.exists() else 1800.0
        tr = load_traces(tp)
        g = Grid.points([r.lat], [r.lon], terrain.elevation(np.array([r.lat]), np.array([r.lon])))
        segs = build_segments(tr, g.lat0, g.lon0)
        cpa = cpa_table(g, segs, max_range_m=a.range * 1000.0)
        cpa["t_audio"] = cpa.t_heard - t0
        inclip = cpa[(cpa.t_audio > 0) & (cpa.t_audio < dur)].sort_values("t_audio")
        rows.append({"clip_id": r.clip_id, "region": r.region, "split": r.split, "duration_s": dur, "n_traces_day": len(tr),
                     "n_passes": int(len(inclip)), "min_d_km": float(inclip.d.min() / 1000) if len(inclip) else np.nan,
                     "passes": " ".join(f"{t:.0f}s@{d / 1000:.1f}km" for t, d in zip(inclip.t_audio, inclip.d))})
        print(rows[-1]["clip_id"], rows[-1]["n_passes"], flush=True)
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    ok = df[df.n_passes >= 2]
    print(f"{len(ok)} of {len(df)} clips have >= 2 passes within {a.range:.0f} km:", list(ok.clip_id))


if __name__ == "__main__":
    main()
