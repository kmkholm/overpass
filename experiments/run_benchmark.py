"""Fixed-grid benchmark run: every clip x every configuration, one CSV of results.

Configurations (same-family ablations; the full method is 'full'):
  full        learned detector, contrastive z-score, N=3 events, silence filter, DEM
  raw         ... raw brightness score instead of contrastive
  two         ... only the two strongest events (baseline search strategy)
  nosilence   ... silence filter off
  nodem       ... observer elevation = 0
  scan        ... classical grid-scan detector (dopplertrack.tracks.detect_events) instead of the learned one

python experiments/run_benchmark.py --manifest data/manifest.csv --work work --model models/ridgenet.pt \
    --out results/bench.csv --configs full raw two nosilence nodem scan --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dopplertrack.geo import haversine_m
from dopplertrack.terrain import Terrain
from dopplertrack.tracks import detect_events
from overpass import detect as D
from overpass import nsearch as NS
from overpass.benchmark import bbox_around, load_manifest, prepare_clip

CONFIGS = {
    "full":      dict(raw=False, n_events=3, silence=True, dem=True, scan=False),
    "raw":       dict(raw=True,  n_events=3, silence=True, dem=True, scan=False),
    "two":       dict(raw=False, n_events=2, silence=True, dem=True, scan=False),
    "nosilence": dict(raw=False, n_events=3, silence=False, dem=True, scan=False),
    "nodem":     dict(raw=False, n_events=3, silence=True, dem=False, scan=False),
    "scan":      dict(raw=False, n_events=3, silence=True, dem=True, scan=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--work", default="work")
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--configs", nargs="+", default=["full"]); ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--spacing", type=float, default=200.0); ap.add_argument("--region-km", type=float, default=150.0)
    ap.add_argument("--archive-root"); ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--clips", nargs="*", help="subset of clip_ids")
    ap.add_argument("--max-cands", type=int, default=20000, help="cap on anchor candidates scored per search (closest offset agreement first)")
    a = ap.parse_args()
    man = load_manifest(a.manifest)
    if a.clips:
        man = man[man.clip_id.isin(a.clips)]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    model = D.load_model(a.model)
    terrain = Terrain()
    rows = [] if done.empty else done.to_dict("records")
    for row in man.itertuples():
        try:
            sp, traces, sp_path, tr_path = prepare_clip(row, Path(a.work), a.archive_root, fetch=not a.no_fetch)
        except FileNotFoundError as e:
            print(f"{row.clip_id}: skipped ({e})", flush=True); continue
        det_path = Path(a.work) / f"{row.clip_id}.det.json"
        if det_path.exists():
            dets = D.load_detections(det_path)
        else:
            dets = D.detect(model, sp, top=8, log=print); D.save_detections(dets, det_path)
        scan_path = Path(a.work) / f"{row.clip_id}.scan.json"
        for cfg in a.configs:
            C = CONFIGS[cfg]
            # event-choice seeds for the full method; ablations run with the first seed only (fixed budget)
            for seed in (a.seeds if cfg == "full" else a.seeds[:1]):
                if not done.empty and ((done.clip_id == row.clip_id) & (done.config == cfg) & (done.seed == seed)).any():
                    continue
                t0 = time.time()
                if C["scan"]:
                    if scan_path.exists():
                        evs = [D.EventFit(**d) for d in json.load(open(scan_path))]
                    else:
                        evs = detect_events(sp, 60, 260, top=8); json.dump([e.to_dict() for e in evs], open(scan_path, "w"), indent=1)
                else:
                    evs = [d.fit for d in dets if d.fit is not None and D.plausible(d.fit)]
                # seed = which detections are used when more than n_events are available: seed 0 = strongest;
                # other seeds drop one of the top n+1 (robustness of the result to the event choice)
                rng = np.random.default_rng(seed)
                evs = evs[:C["n_events"] + 1] if seed else evs[:C["n_events"]]
                if seed and len(evs) > C["n_events"]:
                    drop = int(rng.integers(len(evs))); evs = [e for i, e in enumerate(evs) if i != drop]
                evs = sorted(evs, key=lambda e: e.t_cpa)
                rec = {"clip_id": row.clip_id, "region": row.region, "split": row.split, "config": cfg, "seed": seed,
                       "n_events_used": len(evs), "lat_true": row.lat, "lon_true": row.lon}
                if len(evs) < 2:
                    rec.update({"status": "too_few_events"}); rows.append(rec); continue
                bbox = bbox_around(float(row.lat), float(row.lon), a.region_km)
                try:
                    res, segs = NS.search(sp, evs, traces, bbox, elevation=terrain.elevation if C["dem"] else None,
                                          spacing_m=a.spacing, max_region_km=a.region_km + 1, raw_score=C["raw"],
                                          silence_d_m=6000.0 if C["silence"] else None, max_candidates=a.max_cands,
                                          cpa_cache=str(Path(a.work) / f"{row.clip_id}.cpa_{'dem' if C['dem'] else 'flat'}_{int(a.spacing)}m.npz"), progress=False)
                except Exception as e:
                    rec.update({"status": f"error: {e}"}); rows.append(rec); continue
                if len(res) == 0:
                    rec.update({"status": "no_candidates"}); rows.append(rec); continue
                best = NS.best_per_point(res)
                res.to_csv(Path(a.work) / f"{row.clip_id}.{cfg}.s{seed}.res.csv", index=False)
                top = NS.top_point(res)          # best tuple, then best localisation cell within it
                err = float(haversine_m(top.lat, top.lon, row.lat, row.lon))
                d_all = haversine_m(best.lat.to_numpy(), best.lon.to_numpy(), row.lat, row.lon)
                true_mask = d_all <= a.spacing * 0.75
                true_score = float(best.score[true_mask].max()) if true_mask.any() else float("nan")
                rank_true = int((best.score > true_score).sum() + 1) if np.isfinite(true_score) else -1
                # flight tuple of the top candidate vs the tuple that is best at the truth
                segcols = [c for c in res.columns if c.startswith("icao")]
                tuple_top = tuple(top[c] for c in segcols)
                near = res[haversine_m(res.lat.to_numpy(), res.lon.to_numpy(), row.lat, row.lon) <= a.spacing * 0.75]
                tuple_true = tuple(near.sort_values("score", ascending=False).iloc[0][c] for c in segcols) if len(near) else None
                runner = res[[tuple(r[c] for c in segcols) != tuple_true for _, r in res.iterrows()]] if tuple_true else res
                margin = float((true_score - runner.score.max()) / abs(runner.score.max())) if (len(runner) and np.isfinite(true_score) and runner.score.max() != 0) else float("nan")
                rec.update({"status": "ok", "top_error_m": err, "top_score": float(top.score), "true_score": true_score,
                            "rank_true": rank_true, "n_points": len(best), "n_candidates": len(res),
                            "tuple_correct": bool(tuple_top == tuple_true) if tuple_true else False,
                            "margin_true_vs_best_wrong_tuple": margin, "seconds": time.time() - t0})
                rows.append(rec)
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"{row.clip_id} {cfg} s{seed}: err {err:.0f} m rank {rank_true} margin {margin:.3f} ({time.time() - t0:.0f} s)")
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
