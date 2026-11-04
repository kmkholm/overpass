#!/usr/bin/env bash
# Day 2: after the daytime recording sessions, wait for the ADS-B archive of 2026-11-04, then extend the
# benchmark to the new clips (everything resumable) and redo feasibility, conformal, scrub and figures.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="D:/Drive D/Ajloun Papers/systmatic review/dopplertrack/src;D:/Drive D/Ajloun Papers/Doppler-Geoloc 2026/code"
export PYTHONIOENCODING=utf-8
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "waiting for the night pipeline and both daytime sessions"
until grep -q "ALL DONE" logs/night.log 2>/dev/null; do sleep 300; done
until grep -q "collect exit" logs/collect_eu_day.log 2>/dev/null && grep -q "collect exit" logs/collect_us_day.log 2>/dev/null; do sleep 300; done
log "clips now: $(python -c "import pandas as pd;m=pd.read_csv('data/manifest.csv');print(len(m), m.groupby(['region','split']).size().to_dict())")"

TODAY=$(date -u +%F)
for d in $(python -c "import pandas as pd;print(' '.join(sorted(set(pd.read_csv('data/manifest.csv').date))))"); do
  [ -f "work/.adsb_$d" ] && continue
  [[ "$d" > "$TODAY" || "$d" == "$TODAY" ]] && { log "skipping $d (archive not due yet)"; continue; }
  until curl -s -o /dev/null -w "%{http_code}" "https://api.github.com/repos/adsblol/globe_history_2026/releases/tags/v$(echo $d | tr - .)-planes-readsb-prod-0" | grep -q 200; do
    log "ADS-B $d not published yet"; sleep 1800
  done
  log "ADS-B $d available"; touch "work/.adsb_$d"
done
log "fetching ADS-B regions for the new clips"
python experiments/fetch_days.py --manifest data/manifest.csv --work work > logs/fetch_days2.log 2>&1 || log "fetch_days returned $?"
python - <<'EOF' > logs/prepare2.log 2>&1
from pathlib import Path
from overpass.benchmark import load_manifest, prepare_clip
for row in load_manifest("data/manifest.csv").itertuples():
    try:
        prepare_clip(row, Path("work"), fetch=False)
    except Exception as e:
        print("FAILED", row.clip_id, e)
EOF
log "feasibility"
python experiments/feasibility.py --manifest data/manifest.csv --work work --out results/feasibility.csv > logs/feasibility.log 2>&1
tail -1 logs/feasibility.log
log "benchmark (resumable: only new clip/config rows run)"
python experiments/run_benchmark.py --manifest data/manifest.csv --work work --model models/ridgenet.pt --out results/bench.csv \
    --configs full raw two nosilence --seeds 0 --region-km 100 --spacing 300 --no-fetch > logs/bench2.log 2>&1 || log "benchmark returned $?"
SUB=$(python -c "import pandas as pd;f=pd.read_csv('results/feasibility.csv');print(' '.join(f[(f.n_passes>=2)&(f.split=='calib')].clip_id.head(8)))")
python experiments/run_benchmark.py --manifest data/manifest.csv --work work --model models/ridgenet.pt --out results/bench.csv \
    --configs nodem scan --seeds 0 --region-km 100 --spacing 300 --no-fetch --clips $SUB >> logs/bench2.log 2>&1 || log "benchmark (subset) returned $?"
log "conformal"
python experiments/calibrate_eval.py --manifest data/manifest.csv --work work --bench results/bench.csv --config full --seed 0 --out results/conformal_all.csv > logs/conformal2.log 2>&1 || log "conformal (all) returned $?"
python experiments/calibrate_eval.py --manifest data/manifest.csv --work work --bench results/bench.csv --config full --seed 0 --feasible results/feasibility.csv --out results/conformal.csv >> logs/conformal2.log 2>&1 || log "conformal (feasible) returned $?"
log "scrub"
python experiments/scrub_eval.py --manifest data/manifest.csv --work work --speech data/speech_16k.wav --out results/scrub.csv > logs/scrub2.log 2>&1 || log "scrub returned $?"
log "figures"
Q=$(python -c "import json;print(json.load(open('results/conformal_summary.json'))['gap_a0.1_all']['q'])" 2>/dev/null || echo 1.0)
python experiments/figures.py --work work --bench results/bench.csv --conformal results/conformal.csv --scrub results/scrub.csv \
    --model models/ridgenet.pt --clip ref_oysterbay --q "$Q" --out figures > logs/figures2.log 2>&1 || log "figures returned $?"
log "DAY2 DONE"
