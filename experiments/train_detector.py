"""Train the flyover ridge detector on tiles rendered from real ADS-B tracks.

python experiments/train_detector.py --traces work/*.traces.npz --manifest data/manifest.csv \
    --work work --n-tiles 6000 --epochs 25 --out models/ridgenet.pt --seed 0
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from dopplertrack.adsb import load_traces
from dopplertrack.search import build_segments
from overpass import detect as D
from overpass import synth
from overpass.benchmark import backgrounds_from_clips, load_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True, help="traces .npz files (globs ok)")
    ap.add_argument("--manifest"); ap.add_argument("--work", default="work")
    ap.add_argument("--n-tiles", type=int, default=6000); ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    ap.add_argument("--max-segs", type=int, default=3000)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    files = [f for pat in a.traces for f in glob.glob(pat)]
    segs = []
    for f in files:
        tr = load_traces(f)
        lat0 = float(np.nanmean([t.lat.mean() for t in tr])); lon0 = float(np.nanmean([t.lon.mean() for t in tr]))
        segs += build_segments(tr, lat0, lon0, min_len=90)
    if len(segs) > a.max_segs:
        segs = [segs[i] for i in rng.choice(len(segs), a.max_segs, replace=False)]
    print(f"{len(segs)} real segments from {len(files)} trace files")
    bgs = backgrounds_from_clips(load_manifest(a.manifest), Path(a.work)) if a.manifest else None
    print(f"{len(bgs) if bgs else 0} real background crops")
    X, Y = synth.make_dataset(segs, a.n_tiles, seed=a.seed, backgrounds=bgs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    model, hist = D.train(X, Y, epochs=a.epochs, seed=a.seed, out_path=a.out)
    # held-out synthetic ROC (fresh seed)
    Xt, Yt = synth.make_dataset(segs, 400, seed=a.seed + 1000, backgrounds=bgs)
    import torch
    dev = next(model.parameters()).device
    with torch.no_grad():
        P = [torch.sigmoid(model(torch.from_numpy(Xt[i:i + 1]).to(dev)))[0, 0].cpu().numpy() for i in range(len(Xt))]
    fpr, tpr, auc = D.pixel_roc(P, [y[0] for y in Yt])
    json.dump({"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc, "n_segs": len(segs), "n_tiles": a.n_tiles},
              open(a.out + ".roc.json", "w"))
    print(f"held-out synthetic pixel AUC = {auc:.4f}")


if __name__ == "__main__":
    main()
