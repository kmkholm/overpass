# OVERPASS

Forensic verification of audio recording location from inaudible propeller-aircraft Doppler
signatures and public ADS-B tracks, with conformal location sets and a Doppler-tone scrubber.

Built on the open `dopplertrack` proof of concept (MIT, github.com/mjshiggins/dopplertrack),
which is vendored unchanged in `../../systmatic review/dopplertrack` and installed as a
dependency. OVERPASS adds, in `overpass/`:

| module | what it adds |
|---|---|
| `synth.py` | training tiles: Doppler combs rendered from **real** ADS-B segments onto synthetic or real backgrounds |
| `detect.py` | U-Net ridge segmenter (`RidgeNet`) → components → harmonic groups → `EventFit`; pixel ROC, event matching |
| `contrast.py` | contrastive locked score: sideband contrast z-scored against the same curve placed elsewhere |
| `nsearch.py` | N-event chain candidates with the silence constraint; `verify` (claimed point) and bounded `search` modes |
| `conformal.py` | split-conformal location sets (score-gap and radius variants), coverage/area evaluation |
| `scrub.py` | countermeasure: Viterbi tracking of the comb + STFT track-and-fill; baselines high-pass / masker / Opus codec |
| `benchmark.py` | manifest, per-clip preparation (spectrogram, ADS-B region via adsb.lol, DEM) |
| `cli.py` | `overpass detect / verify / search / scrub` |

## Easiest: Colab Pro+ (nothing to install)

Open `colab/OVERPASS_colab.ipynb` in Google Colab (File → Upload notebook, or open it from GitHub), set the
runtime to GPU and choose *Run all*. The notebook records the webcam clips (`data/streams.csv`), fetches the
reference clip and the ADS-B days, trains the detector, runs the benchmark, calibration, countermeasure and
figures, and keeps everything in `MyDrive/OVERPASS`. If the session ends, *Run all* again continues.

## Install (server)

```bash
cd "Doppler-Geoloc 2026/code"
python -m venv .venv && source .venv/bin/activate
pip install -e "../../systmatic review/dopplertrack"      # dopplertrack (needs ffmpeg on PATH)
pip install -r requirements.txt && pip install -e .
pytest -q tests                                            # numpy-only tests, ~1 min
```

## Data

* `data/manifest.csv` – one row per clip (see `data/manifest_template.csv`). Columns: `clip_id, path, lat, lon,
  utc_start, date, region, split, source, notes`. `split` is `calib` (US) or `test` (EU, JO).
* ADS-B: fetched per clip and per day from adsb.lol GitHub releases (streamed, ~3.5 GB read per day, tens of MB kept).
  To reuse a day for several clips put the extracted archive under `--archive-root/<date>/traces/...`.
* DEM: AWS terrain tiles, cached in `~/.cache/dopplertrack`.
* Reference clip: the raw one-hour recording from youtu.be/4CxVxuMVkMc saved as `data/ref_oysterbay.wav`
  (row already in the template).

## Run order (fixed grid, one pass)

```bash
bash run_server.sh            # steps 1-6 below with the default paths; edit the variables at the top
```

1. `python experiments/train_detector.py --traces "work/*.traces.npz" --manifest data/manifest.csv --n-tiles 6000 --epochs 25 --out models/ridgenet.pt`
   (needs the traces of at least the calibration clips: run step 2 once with `--configs full --clips <one clip>` first, or
   run `prepare_clip` for all clips via `python -c` as in `run_server.sh`).
2. `python experiments/run_benchmark.py --manifest data/manifest.csv --model models/ridgenet.pt --out results/bench.csv --configs full raw two nosilence nodem scan --seeds 0 1 2 3 4`
3. `python experiments/calibrate_eval.py --manifest data/manifest.csv --bench results/bench.csv --out results/conformal.csv`
4. `python experiments/scrub_eval.py --manifest data/manifest.csv --speech data/speech_16k.wav --out results/scrub.csv`
5. `python experiments/figures.py --bench results/bench.csv --conformal results/conformal.csv --scrub results/scrub.csv --model models/ridgenet.pt --clip ref_oysterbay`

Budget: the baseline search costs ~19 min per 100 × 150 km region on a laptop CPU; with the CPA cache per clip the
six configurations × five seeds reuse the same (point, flight) table, so the per-clip cost is dominated by scoring
(~1–3 ms per candidate tuple).

## Forensic use

```bash
overpass detect  work/clip.spec.npz --model models/ridgenet.pt --out work/clip.det.json
overpass verify  work/clip.spec.npz --traces work/clip.traces.npz --detections work/clip.det.json --point 40.8765 -73.5250 --out verify.csv
overpass search  work/clip.spec.npz --traces work/clip.traces.npz --detections work/clip.det.json --bbox 40.4 41.3 -74.4 -72.6 --out search.csv --refine
overpass scrub   clip.mp4 clip_scrubbed.wav --spec work/clip.spec.npz --detections work/clip.det.json
```

`search` refuses regions larger than 150 km on a side by design.
