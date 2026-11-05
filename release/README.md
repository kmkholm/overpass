# OVERPASS data release

Data behind the paper *OVERPASS: Forensic Verification of Audio Recording Location from Inaudible
Propeller-Aircraft Doppler Signatures and Public ADS-B Tracks, with Conformal Location Sets and a
Doppler-Tone Scrubber* (M. Tawfik, Ajloun National University). The code that produced every file is in
the parent directory of this repository.

## Contents

| Folder | What it holds |
|---|---|
| `manifests/` | `manifest.csv`: the 33 recordings of the study (reference + 32 webcam clips) with camera position, UTC start, region, split and the resolved stream title. `manifest_excluded.csv`: the 28 recordings removed by the title audit, with the reason. `manifest_livermore.csv`: the 40 smartphone recordings of the Livermore dataset with the inferred recorder site. `streams*.csv`: the camera searches of each session. `coverage_points.csv`: the 14 Middle East cities of the coverage check. |
| `adsb_extracts/webcam/` | Propeller and turboprop aircraft traces (ICAO Doc 8643 engine class) within 150 km of each camera on the recording day, one `.npz` per clip (30 clips; the three Middle East recordings of 5 November have no archive in this release). |
| `adsb_extracts/livermore/` | The same for the Livermore site, one file per recording day (6 days, 60 km box). |
| `detections/` | Automatic flyover detections and ideal-curve fits per clip (`.det.json`), and the classical scan detector's events (`.scan.json`). |
| `model/` | The ridge-segmentation U-Net weights (`ridgenet.pt`), its training history and its pixel ROC. |
| `results/` | Every table and figure source of the paper: `bench.csv` (blind search, all configurations), `feasibility.csv`, `audible.csv`, `scrub.csv`, `conformal*.csv/json`, `coverage_2026-11-03.csv`, `livermore_site.json`, `livermore_fits.csv`, `livermore_verify*.csv`, `ref/` (reference-recording searches), `known_point/`. |

Load a trace extract with `dopplertrack.adsb.load_traces(path)`.

## What is not included

No audio is redistributed. The webcam streams belong to their operators, and the reference recording is
the public recording of the 2024 video demonstration; `manifests/manifest.csv` identifies each stream and
the exact capture time. The Livermore recordings are available from their authors under CC BY 4.0
(P. Pathak, Zenodo, doi:10.5281/zenodo.23424278). Whitened spectrograms are recomputed from the audio by
`dopplertrack.pipeline.make_spectrogram`.

## Licences

* Code in this repository: MIT (see `LICENSE`).
* `adsb_extracts/`: derived from the adsb.lol daily archives (https://github.com/adsblol/globe_history_2026)
  and made available under the Open Database License 1.0 (ODbL), with attribution to adsb.lol.
* Everything else in `release/`: Creative Commons Attribution 4.0 (CC BY 4.0).
