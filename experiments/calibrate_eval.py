"""Conformal calibration on the calib split, evaluation on the external test split.

Reads the per-clip result CSVs written by run_benchmark.py (work/<clip>.<config>.s<seed>.res.csv).

python experiments/calibrate_eval.py --manifest data/manifest.csv --work work --bench results/bench.csv \
    --config full --seed 0 --alphas 0.1 0.2 --out results/conformal.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from overpass import conformal as CF
from overpass import nsearch as NS
from overpass.benchmark import load_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--work", default="work")
    ap.add_argument("--bench", required=True); ap.add_argument("--config", default="full"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.2]); ap.add_argument("--spacing", type=float, default=200.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--feasible", help="feasibility.csv: keep only clips with >= 2 real passes (the set on which coverage is meaningful)")
    ap.add_argument("--min-score", type=float, default=None, help="identification-score gate: drop clips whose top score is below it")
    a = ap.parse_args()
    man = load_manifest(a.manifest).set_index("clip_id")
    bench = pd.read_csv(a.bench)
    bench = bench[(bench.config == a.config) & (bench.seed == a.seed) & (bench.status == "ok")]
    if a.feasible:
        feas = pd.read_csv(a.feasible); ok = set(feas[feas.n_passes >= 2].clip_id)
        bench = bench[bench.clip_id.isin(ok)]
    if a.min_score is not None:
        bench = bench[bench.top_score >= a.min_score]
    items = {}
    for r in bench.itertuples():
        p = Path(a.work) / f"{r.clip_id}.{a.config}.s{a.seed}.res.csv"
        if not p.exists():
            continue
        res = pd.read_csv(p)
        items[r.clip_id] = {"clip": r.clip_id, "res_best": NS.best_per_point(res), "lat": float(man.loc[r.clip_id].lat),
                            "lon": float(man.loc[r.clip_id].lon), "split": man.loc[r.clip_id].split, "region": man.loc[r.clip_id].region}
    calib = [it for it in items.values() if it["split"] == "calib"]
    test = [it for it in items.values() if it["split"] == "test"]
    print(f"{len(calib)} calibration clips, {len(test)} test clips")
    gaps = [CF.gap_nonconformity(it["res_best"], it["lat"], it["lon"], a.spacing) for it in calib]
    rads = [CF.radius_nonconformity(it["res_best"], it["lat"], it["lon"]) for it in calib]
    rows = []; summary = {}
    for alpha in a.alphas:
        qg = CF.conformal_quantile(gaps, alpha); qr = CF.conformal_quantile(rads, alpha)
        for kind, q in (("gap", qg), ("radius", qr)):
            ev = CF.evaluate_sets(test, q, a.spacing, kind=kind)
            ev["alpha"] = alpha; ev["q"] = q
            ev["region"] = [items[c]["region"] for c in ev["clip"]]
            rows.append(ev)
            for reg, g in ev.groupby("region"):
                summary[f"{kind}_a{alpha}_{reg}"] = {"coverage": float(g.covered.mean()), "median_area_km2": float(g.area_km2.median()), "n": int(len(g))}
            summary[f"{kind}_a{alpha}_all"] = {"coverage": float(ev.covered.mean()), "median_area_km2": float(ev.area_km2.median()), "n": int(len(ev)), "q": q}
    pd.concat(rows, ignore_index=True).to_csv(a.out, index=False)
    json.dump(summary, open(a.out.replace(".csv", "_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
