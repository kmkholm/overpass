#!/usr/bin/env bash
# OVERPASS: one pass over the fixed grid. Edit the three variables, then: bash run_server.sh
set -euo pipefail
MANIFEST=data/manifest.csv
DOPPLERTRACK="../../systmatic review/dopplertrack"
SEEDS="0 1 2 3 4"

mkdir -p work models results figures logs
if [ ! -d .venv ]; then
  python -m venv .venv && source .venv/bin/activate
  pip install -e "$DOPPLERTRACK" && pip install -r requirements.txt && pip install -e .
else
  source .venv/bin/activate
fi
pytest -q tests 2>&1 | tee logs/tests.log

# 0. spectrograms + ADS-B regions for every clip (cached in work/)
python - <<'EOF' 2>&1 | tee logs/prepare.log
from pathlib import Path
from overpass.benchmark import load_manifest, prepare_clip
for row in load_manifest("data/manifest.csv").itertuples():
    prepare_clip(row, Path("work"))
EOF

# 1. detector (tiles rendered from the real tracks of all clips; real backgrounds from calib clips)
python experiments/train_detector.py --traces "work/*.traces.npz" --manifest $MANIFEST --work work \
    --n-tiles 6000 --epochs 25 --seed 0 --out models/ridgenet.pt 2>&1 | tee logs/train.log

# 2. benchmark: full method + same-family ablations, five event-choice seeds
python experiments/run_benchmark.py --manifest $MANIFEST --work work --model models/ridgenet.pt \
    --out results/bench.csv --configs full raw two nosilence nodem scan --seeds $SEEDS 2>&1 | tee logs/bench.log

# 3. conformal calibration (calib split) -> external test coverage/area
python experiments/calibrate_eval.py --manifest $MANIFEST --work work --bench results/bench.csv \
    --config full --seed 0 --out results/conformal.csv 2>&1 | tee logs/conformal.log

# 4. countermeasure (speech file optional: put a 16 kHz mono public speech wav at data/speech_16k.wav)
SPEECH=""; [ -f data/speech_16k.wav ] && SPEECH="--speech data/speech_16k.wav"
python experiments/scrub_eval.py --manifest $MANIFEST --work work $SPEECH --out results/scrub.csv 2>&1 | tee logs/scrub.log

# 5. figures
Q=$(python -c "import json;print(json.load(open('results/conformal_summary.json'))['gap_a0.1_all']['q'])")
python experiments/figures.py --work work --bench results/bench.csv --conformal results/conformal.csv \
    --scrub results/scrub.csv --model models/ridgenet.pt --clip ref_oysterbay --q "$Q" --out figures 2>&1 | tee logs/figures.log
echo "done: results/ and figures/"
