"""Writes OVERPASS_colab.ipynb: the whole pipeline on Colab Pro+ (GPU), results persisted on Drive.
Every step is idempotent, so the notebook can be re-run after a session ends and it continues."""
import json
from pathlib import Path

REPO = "https://github.com/kmkholm/overpass"
DT = "https://github.com/mjshiggins/dopplertrack"

cells = []


def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})


md("# OVERPASS — full run on Colab\n\nRuntime → Change runtime type → **GPU**. Then Runtime → **Run all**. "
   "Everything (clips, ADS-B extracts, model, results, figures) is written to `MyDrive/OVERPASS`, so a second "
   "*Run all* after a disconnect continues where it stopped.")
code("""import subprocess, os, sys
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True, text=True).stdout)
from google.colab import drive
drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/OVERPASS'
os.makedirs(ROOT, exist_ok=True)
%cd /content""")
code(f"""!apt-get -qq install -y ffmpeg > /dev/null
!pip -q install yt-dlp soundfile pesq pystoi > /dev/null
![ -d dopplertrack ] || git clone -q {DT}
![ -d overpass ] || git clone -q {REPO}
!cd overpass && git pull -q
!pip -q install -e dopplertrack > /dev/null && pip -q install -e overpass > /dev/null
%cd /content/overpass
# persist data, work, models, results, figures, logs on Drive
for d in ['data', 'work', 'models', 'results', 'figures', 'logs']:
    os.makedirs(f'{{ROOT}}/{{d}}', exist_ok=True)
    if not os.path.islink(d):
        if os.path.isdir(d):
            !cp -rn {{d}}/. {{ROOT}}/{{d}}/ && rm -rf {{d}}
        os.symlink(f'{{ROOT}}/{{d}}', d)
!ls data""")
md("## 1. Benchmark clips\nReference clip (public, known answer) + 30-minute recordings from public 24/7 webcams "
   "with fixed, known positions (`data/streams.csv`). All webcams are recorded at the same time so one UTC day of ADS-B "
   "serves them. Streams that are offline or silent are skipped automatically.")
code("""import os
if not os.path.exists('data/ref_oysterbay.wav'):
    !python -m yt_dlp -q -x --audio-format wav --audio-quality 0 -o data/ref_oysterbay.%(ext)s https://youtu.be/4CxVxuMVkMc
    !ffmpeg -v error -y -i data/ref_oysterbay.wav -ac 1 -ar 16000 data/ref_16k.wav && mv data/ref_16k.wav data/ref_oysterbay.wav
import csv
rows = list(csv.DictReader(open('data/manifest.csv'))) if os.path.exists('data/manifest.csv') else []
if not any(r['clip_id'] == 'ref_oysterbay' for r in rows):
    from overpass.benchmark import reference_manifest_row
    cols = ['clip_id','path','lat','lon','utc_start','date','region','split','source','notes']
    with open('data/manifest.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows + [reference_manifest_row()]: w.writerow({k: r.get(k, '') for k in cols})
print(open('data/manifest.csv').read()[:600])""")
code("""# ~35 min wall time: all live streams recorded in parallel. Re-running skips nothing (new recordings replace old ones),
# so guard with the manifest size.
import pandas as pd, os
n = len(pd.read_csv('data/manifest.csv')) if os.path.exists('data/manifest.csv') else 0
if n < 20:
    !python experiments/collect_clips.py --streams data/streams.csv --minutes 30 --parallel 10 --out data --manifest data/manifest.csv 2>&1 | tee logs/collect.log
m = pd.read_csv('data/manifest.csv'); print(len(m), 'clips'); print(m.groupby(['region','split']).size())""")
md("## 2. ADS-B (one streaming pass per recording day) and spectrograms")
code("""!python experiments/fetch_days.py --manifest data/manifest.csv --work work 2>&1 | tail -n 60 | tee logs/fetch.log""")
code("""from pathlib import Path
from overpass.benchmark import load_manifest, prepare_clip
for row in load_manifest('data/manifest.csv').itertuples():
    prepare_clip(row, Path('work'), fetch=False)
!ls work | head -50""")
md("## 3. Flyover detector (tiles rendered from the real tracks of all clips)")
code("""import os
if not os.path.exists('models/ridgenet.pt'):
    !python experiments/train_detector.py --traces "work/*.traces.npz" --manifest data/manifest.csv --work work --n-tiles 6000 --epochs 25 --seed 0 --out models/ridgenet.pt 2>&1 | tee logs/train.log
else:
    print('model exists')""")
md("## 4. Benchmark: full method (5 event-choice seeds) + same-family ablations (seed 0). Resumable.")
code("""!python experiments/run_benchmark.py --manifest data/manifest.csv --work work --model models/ridgenet.pt --out results/bench.csv \\
    --configs full raw two nosilence nodem scan --seeds 0 1 2 3 4 --no-fetch 2>&1 | tee -a logs/bench.log | tail -n 80""")
md("## 5. Conformal sets (calibrate on US, test on EU), countermeasure, figures")
code("""!python experiments/calibrate_eval.py --manifest data/manifest.csv --work work --bench results/bench.csv --config full --seed 0 --out results/conformal.csv 2>&1 | tee logs/conformal.log
# public speech sample for PESQ/STOI: LibriSpeech dev-clean, one utterance
import os
if not os.path.exists('data/speech_16k.wav'):
    !wget -q https://www.openslr.org/resources/12/dev-clean.tar.gz -O /tmp/dc.tar.gz && tar -xzf /tmp/dc.tar.gz -C /tmp LibriSpeech/dev-clean/1272 && \\
     ffmpeg -v error -y -i /tmp/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac -ac 1 -ar 16000 data/speech_16k.wav
!python experiments/scrub_eval.py --manifest data/manifest.csv --work work --speech data/speech_16k.wav --out results/scrub.csv 2>&1 | tee logs/scrub.log | tail -n 40""")
code("""import json
Q = json.load(open('results/conformal_summary.json'))['gap_a0.1_all']['q']
!python experiments/figures.py --work work --bench results/bench.csv --conformal results/conformal.csv --scrub results/scrub.csv --model models/ridgenet.pt --clip ref_oysterbay --q {Q} --out figures
import pandas as pd
b = pd.read_csv('results/bench.csv'); print(b[b.status=='ok'].groupby(['config','region']).agg(n=('clip_id','count'), median_err_m=('top_error_m','median'), rank1=('rank_true', lambda s: (s==1).mean())))
print(json.load(open('results/conformal_summary.json')))""")
md("Done. `results/` and `figures/` are in `MyDrive/OVERPASS`.")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).with_name("OVERPASS_colab.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
