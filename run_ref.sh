#!/usr/bin/env bash
# Reference-clip pipeline after the detector is trained: ONNX export, learned detection,
# blind search over the same 100 x 150 km box used by the dopplertrack README, 50 m refinement.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="D:/Drive D/Ajloun Papers/systmatic review/dopplertrack/src;D:/Drive D/Ajloun Papers/Doppler-Geoloc 2026/code"
export PYTHONIOENCODING=utf-8
mkdir -p results/ref app/models
python experiments/export_onnx.py --model models/ridgenet.pt --out app/models/ridgenet.onnx 2>&1 | tail -2
python -m overpass.cli detect work/ref_oysterbay.spec.npz --model models/ridgenet.pt --out work/ref_oysterbay.det.json --top 12 2>&1 | tail -16
for cfg in full raw two; do
  case $cfg in
    full) extra="--top-events 3";;
    raw)  extra="--top-events 3 --raw";;
    two)  extra="--top-events 2";;
  esac
  echo "=== $cfg $(date +%H:%M)"
  python -m overpass.cli search work/ref_oysterbay.spec.npz --traces work/ref_oysterbay.traces.npz \
      --detections work/ref_oysterbay.det.json $extra --bbox 40.4 41.3 -74.4 -72.63 --spacing 200 \
      --cpa-cache work/ref_oysterbay.cpa_dem.npz --out results/ref/search_$cfg.csv --refine 2>&1 | tail -8
done
python - <<'EOF'
import pandas as pd
from dopplertrack.geo import haversine_m
for cfg in ["full", "raw", "two"]:
    try:
        r = pd.read_csv(f"results/ref/search_{cfg}.csv")
    except FileNotFoundError:
        print(cfg, "missing"); continue
    best = r.sort_values("score", ascending=False).drop_duplicates("gidx").reset_index(drop=True)
    top = best.iloc[0]; err = haversine_m(top.lat, top.lon, 40.876466, -73.524951)
    d = haversine_m(best.lat.to_numpy(), best.lon.to_numpy(), 40.876466, -73.524951)
    near = best[d <= 150]; ts = float(near.score.max()) if len(near) else float("nan")
    rank = int((best.score > ts).sum() + 1)
    print(f"{cfg}: top error {err:.0f} m, rank of truth {rank}, top score {top.score:.2f}, truth score {ts:.2f}, candidates {len(r)}, points {len(best)}")
    try:
        f = pd.read_csv(f"results/ref/search_{cfg}_refined.csv"); ft = f.iloc[0]
        print(f"   refined: {haversine_m(ft.lat, ft.lon, 40.876466, -73.524951):.0f} m")
    except FileNotFoundError:
        pass
EOF
echo "ref done $(date +%H:%M)"
