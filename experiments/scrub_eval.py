"""Countermeasure evaluation: attack success before/after each defence, and speech quality.

For every test clip with detections: the attacker's contrast at the true point for each event
(residual_contrast) under: none | highpass | masker | codec | notch | fill (ours). Speech quality
(PESQ/STOI) is measured on the clip mixed with a public speech file (--speech), since the
benchmark clips carry no speech; the reference is the mixed-but-unscrubbed signal.

python experiments/scrub_eval.py --manifest data/manifest.csv --work work --speech data/speech_16k.wav --out results/scrub.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from dopplertrack.audio import load_audio
from overpass import detect as D
from overpass import scrub as SC
from overpass.benchmark import load_manifest

DEFENCES = ["none", "highpass", "masker", "codec", "notch", "flank", "fill"]


def apply(name, x, sr, tracks):
    if name == "none": return x
    if name == "highpass": return SC.highpass(x, sr)
    if name == "masker": return SC.add_masker(x, sr)
    if name == "codec": return SC.codec_roundtrip(x, sr)
    if name == "notch": return SC.scrub(x, sr, tracks, mode="notch")
    if name == "flank": return SC.scrub(x, sr, tracks, mode="flank")
    if name == "fill": return SC.scrub(x, sr, tracks, mode="fill")
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--work", default="work")
    ap.add_argument("--speech", help="16 kHz mono speech wav mixed in for PESQ/STOI")
    ap.add_argument("--defences", nargs="+", default=DEFENCES); ap.add_argument("--out", required=True)
    ap.add_argument("--max-events", type=int, default=3)
    a = ap.parse_args()
    man = load_manifest(a.manifest)
    speech = None
    if a.speech:
        speech, ssr = sf.read(a.speech, dtype="float32")
        assert ssr == 16000, "speech file must be 16 kHz"
        if speech.ndim > 1: speech = speech.mean(1)
    rows = []
    for row in man.itertuples():
        det_path = Path(a.work) / f"{row.clip_id}.det.json"
        if not det_path.exists():
            continue
        from dopplertrack.audio import Spectrogram
        sp = Spectrogram.load(Path(a.work) / f"{row.clip_id}.spec.npz")
        evs = [d.fit for d in D.load_detections(det_path) if d.fit is not None][:a.max_events]
        if not evs:
            continue
        x, sr = load_audio(row.path, sr=16000)
        x_raw = x.copy()
        if speech is not None:
            rep = int(np.ceil(len(x) / len(speech))); mix = np.tile(speech, rep)[:len(x)]
            x = (x + 0.5 * mix / (np.abs(mix).max() + 1e-6) * (np.abs(x).max() + 1e-6)).astype(np.float32)
        tracks = [SC.track_from_event(sp, e, harmonics=4) for e in evs]
        for name in a.defences:
            try:
                y = apply(name, x, sr, tracks)
            except Exception as e:
                rows.append({"clip_id": row.clip_id, "defence": name, "status": f"error: {e}"}); continue
            # PESQ is defined for utterance-length signals (it aborts on an hour of audio): score a 30 s excerpt
            # centred on the first flyover, where the scrubber acts and the mixed-in speech is present
            q = {}
            if speech is not None and name != "none":
                c0 = int(max(evs[0].t_cpa - 15, 0) * sr); c1 = min(c0 + 30 * sr, len(x))
                q = SC.speech_quality(x[c0:c1], y[c0:c1], sr)
            for k, e in enumerate(evs):
                b, af = SC.residual_contrast(x, y, sr, e)
                rows.append({"clip_id": row.clip_id, "region": row.region, "defence": name, "event": k, "t_cpa": e.t_cpa,
                             "contrast_before_db": b, "contrast_after_db": af, "status": "ok", **q})
            if name == "fill":
                sf.write(Path(a.work) / f"{row.clip_id}.scrubbed.wav", y, sr)
                # speech-free copy for the paper figure (same tracks, no mixed-in speech)
                sf.write(Path(a.work) / f"{row.clip_id}.scrubbed_nospeech.wav", SC.scrub(x_raw, sr, tracks, mode="fill"), sr)
            pd.DataFrame(rows).to_csv(a.out, index=False)
            print(f"{row.clip_id} {name}: " + ", ".join(f"{r['contrast_before_db']:.1f}->{r['contrast_after_db']:.1f}" for r in rows if r.get('clip_id') == row.clip_id and r.get('defence') == name and r.get('status') == 'ok'))
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
