"""Export the trained RidgeNet to ONNX for the torch-free desktop app and check it against torch.

python experiments/export_onnx.py --model models/ridgenet.pt --out app/models/ridgenet.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from overpass import detect as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    model = D.load_model(a.model, device="cpu")
    D.export_onnx(model, a.out, device="cpu")
    import torch
    x = np.random.default_rng(0).normal(0, 3, (1, 1, 264, 512)).astype(np.float32)
    with torch.no_grad():
        ref = torch.sigmoid(model(torch.from_numpy(x)))[0, 0].numpy()
    out = D.OnnxRidge(a.out)(x)[0, 0]
    print("onnx vs torch max abs diff:", float(np.abs(ref - out).max()), "size MB:", round(Path(a.out).stat().st_size / 1e6, 1))


if __name__ == "__main__":
    main()
