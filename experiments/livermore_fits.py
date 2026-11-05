"""Automatic detector + ideal-curve fit on the 40 Livermore smartphone flyovers (Pathak, Zenodo 10.5281/zenodo.23424278)
against the ADS-B track values published with the dataset (speed at closest approach, minimum slant distance).

python experiments/livermore_fits.py --root ../data_livermore --model models/ridgenet.pt --out results/livermore_fits.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dopplertrack.audio import load_audio, compute_spectrogram
from overpass import detect as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data_livermore"); ap.add_argument("--model", default="models/ridgenet.pt")
    ap.add_argument("--out", default="results/livermore_fits.csv"); ap.add_argument("--top", type=int, default=4)
    a = ap.parse_args()
    truth = pd.read_csv(Path(a.root) / "final_results_40_cases.csv").set_index("case")
    model = D.load_model(a.model)
    rows = []
    for wav in sorted(glob.glob(str(Path(a.root) / "raw" / "*" / "*" / "audio.wav"))):
        case = os.path.basename(os.path.dirname(wav)); t0 = time.time()
        x, sr = load_audio(wav)
        sp = compute_spectrogram(x, sr, whiten_window_s=None)
        # short clips: one pass fills most of the file, so the length and area gates are relaxed
        dets = D.detect(model, sp, top=a.top, min_len_s=12.0, min_area=30, max_fit=8)
        fits = [d.fit for d in dets if d.fit is not None]
        tr = truth.loc[case]
        v_true = float(tr.V_ADSB_at_closest_kmh) / 3.6; d_true = float(tr.R_ADSB_m)
        row = {"case": case, "category": tr.category, "duration_s": round(len(x) / sr, 1), "n_detections": len(dets), "n_fits": len(fits),
               "v_true_mps": round(v_true, 1), "d_true_m": round(d_true), "f_traced_hz": float(tr.MII_f0_Hz),
               "pathak_v_err_pct": float(tr.MII_V_error_percent), "pathak_rmin_m": float(tr.MII_Rmin_m)}
        if fits:
            best = max(fits, key=lambda f: f.score)
            ratio = best.f0 / float(tr.MII_f0_Hz) if tr.MII_f0_Hz else np.nan
            row.update({"f0_hz": round(best.f0, 2), "v_mps": round(best.v, 1), "d_m": round(best.d), "t_cpa_s": round(best.t_cpa, 2),
                        "fit_score": round(best.score, 2), "v_err_pct": round(100 * (best.v - v_true) / v_true, 1),
                        "d_err_pct": round(100 * (best.d - d_true) / d_true, 1), "harmonic_ratio": round(ratio, 3),
                        "plausible": bool(D.plausible(best, v_max=260.0))})
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row); print({k: row.get(k) for k in ("case", "n_fits", "v_true_mps", "v_mps", "d_true_m", "d_m", "fit_score", "seconds")}, flush=True)
    df = pd.DataFrame(rows); Path(a.out).parent.mkdir(parents=True, exist_ok=True); df.to_csv(a.out, index=False)
    ok = df.dropna(subset=["v_mps"]) if "v_mps" in df else df.iloc[0:0]
    print(f"\n{len(ok)} of {len(df)} passes fitted")
    if len(ok):
        print("median |v err| %:", round(ok.v_err_pct.abs().median(), 1), " median |d err| %:", round(ok.d_err_pct.abs().median(), 1))
        print("Pathak traced-ridge median |v err| %:", round(ok.pathak_v_err_pct.abs().median(), 1))
        pr = 100 * (ok.pathak_rmin_m - ok.d_true_m) / ok.d_true_m
        print("Pathak traced-ridge median |Rmin err| %:", round(pr.abs().median(), 1))


if __name__ == "__main__":
    main()
