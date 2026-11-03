#!/usr/bin/env bash
# Unattended end-to-end run on the laptop. Every step is idempotent; re-running continues.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="D:/Drive D/Ajloun Papers/systmatic review/dopplertrack/src;D:/Drive D/Ajloun Papers/Doppler-Geoloc 2026/code"
export PYTHONIOENCODING=utf-8
mkdir -p logs results figures models app/models
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
STAGE=logs/stage.txt

# 1. wait for the detector
log "stage 1: waiting for detector training"
until grep -q "train exit" logs/train.log 2>/dev/null; do sleep 60; done
grep -E "AUC|train exit" logs/train.log | tail -2
[ -f models/ridgenet.pt ] || { log "no model, abort"; exit 1; }
echo "1 train done" > $STAGE

# 2. reference clip: ONNX export, learned detection, blind search
if [ ! -f results/ref/search_full.csv ]; then
  log "stage 2: reference blind search"
  bash run_ref.sh > logs/ref.log 2>&1
fi
tail -8 logs/ref.log; echo "2 ref done" > $STAGE

# 3. wait for the second recording pass, then for the ADS-B archive of the recording day(s)
until grep -q "collect exit" logs/collect_retry.log 2>/dev/null; do sleep 60; done
log "stage 3: clips: $(python -c "import pandas as pd;m=pd.read_csv('data/manifest.csv');print(len(m), m.groupby('region').size().to_dict())")"
for d in $(python -c "import pandas as pd;print(' '.join(sorted(set(pd.read_csv('data/manifest.csv').date))))"); do
  [ -f "work/.adsb_$d" ] && continue
  until curl -s -o /dev/null -w "%{http_code}" "https://api.github.com/repos/adsblol/globe_history_2026/releases/tags/v$(echo $d | tr - .)-planes-readsb-prod-0" | grep -q 200; do
    log "ADS-B $d not published yet"; sleep 1800
  done
  log "ADS-B $d available"; touch "work/.adsb_$d"
done
log "stage 3: fetching ADS-B regions (one pass per day)"
python experiments/fetch_days.py --manifest data/manifest.csv --work work > logs/fetch_days.log 2>&1 || log "fetch_days returned $?"
tail -3 logs/fetch_days.log; echo "3 adsb done" > $STAGE

# 4. spectrograms
log "stage 4: spectrograms"
python - <<'EOF' > logs/prepare.log 2>&1
from pathlib import Path
from overpass.benchmark import load_manifest, prepare_clip
for row in load_manifest("data/manifest.csv").itertuples():
    try:
        prepare_clip(row, Path("work"), fetch=False)
    except Exception as e:
        print("FAILED", row.clip_id, e)
EOF
grep -c FAILED logs/prepare.log; echo "4 spec done" > $STAGE

# 5. benchmark (full x 3 seeds, ablations seed 0), resumable
log "stage 5: benchmark"
python experiments/run_benchmark.py --manifest data/manifest.csv --work work --model models/ridgenet.pt --out results/bench.csv \
    --configs full raw two nosilence nodem scan --seeds 0 1 2 --no-fetch > logs/bench.log 2>&1 || log "benchmark returned $?"
tail -5 logs/bench.log; echo "5 bench done" > $STAGE

# 6. conformal
log "stage 6: conformal"
python experiments/calibrate_eval.py --manifest data/manifest.csv --work work --bench results/bench.csv --config full --seed 0 --out results/conformal.csv > logs/conformal.log 2>&1 || log "conformal returned $?"
tail -3 logs/conformal.log; echo "6 conformal done" > $STAGE

# 7. countermeasure (public speech sample: LibriSpeech dev-clean utterance, ~10 s)
if [ ! -f data/speech_16k.wav ]; then
  python - <<'EOF'
import tarfile, urllib.request, io, subprocess, os
# stream the tar and extract the first flac only
url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
with urllib.request.urlopen(url, timeout=120) as r:
    tf = tarfile.open(fileobj=r, mode="r|gz")
    for m in tf:
        if m.name.endswith(".flac"):
            open("data/speech_src.flac", "wb").write(tf.extractfile(m).read()); break
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "data/speech_src.flac", "-ac", "1", "-ar", "16000", "data/speech_16k.wav"], check=True)
os.remove("data/speech_src.flac"); print("speech sample ready")
EOF
fi
log "stage 7: scrub evaluation"
python experiments/scrub_eval.py --manifest data/manifest.csv --work work --speech data/speech_16k.wav --out results/scrub.csv > logs/scrub.log 2>&1 || log "scrub returned $?"
tail -3 logs/scrub.log; echo "7 scrub done" > $STAGE

# 8. figures
log "stage 8: figures"
Q=$(python -c "import json;print(json.load(open('results/conformal_summary.json'))['gap_a0.1_all']['q'])" 2>/dev/null || echo 1.0)
python experiments/figures.py --work work --bench results/bench.csv --conformal results/conformal.csv --scrub results/scrub.csv \
    --model models/ridgenet.pt --clip ref_oysterbay --q "$Q" --out figures > logs/figures.log 2>&1 || log "figures returned $?"
echo "8 figures done" > $STAGE

# 9. desktop app EXE
log "stage 9: building Ayna.exe"
( cd app && cmd //c Build_EXE.cmd < /dev/null > ../logs/build_app.log 2>&1 ) || log "app build returned $?"
ls -la app/dist 2>/dev/null; echo "9 app done" > $STAGE
log "ALL DONE"
