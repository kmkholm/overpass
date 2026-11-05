"""Diagnostic plot for one Livermore case: whitened spectrogram, the dataset's traced harmonic, our fitted ideal curve
and its harmonics, plus the curve predicted from the ADS-B track seen from the inferred recorder site.

python experiments/livermore_diag.py --case N757AH_R41
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dopplertrack.audio import load_audio, compute_spectrogram
from dopplertrack.doppler import ideal_curve, speed_of_sound
from overpass import detect as D

sys.path.insert(0, os.path.dirname(__file__))
import livermore_site as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True); ap.add_argument("--root", default="../data_livermore")
    ap.add_argument("--model", default="models/ridgenet.pt"); ap.add_argument("--site", default="results/livermore_site.json")
    ap.add_argument("--out", default="figures/livermore_diag.png")
    a = ap.parse_args()
    d = glob.glob(str(Path(a.root) / "raw" / "*" / a.case))[0]
    truth = pd.read_csv(Path(a.root) / "final_results_40_cases.csv").set_index("case").loc[a.case]
    t0 = pd.Timestamp(truth.received_start_utc).timestamp()
    x, sr = load_audio(os.path.join(d, "audio.wav")); sp = compute_spectrogram(x, sr, whiten_window_s=None)
    dets = D.detect(model=D.load_model(a.model), sp=sp, top=4, min_len_s=12.0, min_area=30, max_fit=8)
    fits = [k.fit for k in dets if k.fit is not None]
    tr = pd.read_csv(os.path.join(d, "frequency_vs_time_used.csv"))
    tt = pd.to_datetime(tr.time_utc, utc=True).astype("int64").to_numpy() / 1e9 - t0
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.imshow(sp.S, origin="lower", aspect="auto", extent=[sp.t[0], sp.t[-1], sp.f[0], sp.f[-1]], cmap="gray_r", vmin=0, vmax=np.percentile(sp.S, 99.5))
    ax.plot(tt, tr.frequency_hz, ".", ms=2, color="#E63946", label="dataset traced harmonic")
    c = speed_of_sound()
    for i, f in enumerate(fits):
        for k in range(1, 5):
            ax.plot(sp.t, k * ideal_curve(sp.t, f.f0, f.v, f.d, f.t_cpa_emit, c), "-", lw=0.8, color="#1F6F8B", alpha=0.9 if i == 0 else 0.4,
                    label=f"fit {i}: f0 {f.f0:.1f} Hz, v {f.v:.0f} m/s, d {f.d:.0f} m, score {f.score:.1f}" if k == 1 else None)
    # track-predicted curve from the inferred site, for the harmonic the dataset traced
    site = json.load(open(a.site)); kml = glob.glob(os.path.join(d, "*.kml"))[0]
    t, xyz = L.read_kml(kml); te, dense_xyz = L.dense(t, xyz, 0.25)
    e = L.enu(dense_xyz[:, 0], dense_xyz[:, 1], dense_xyz[:, 2], site["lon"], site["lat"], site["h_m"]); r = np.sqrt((e ** 2).sum(1))
    rr = np.gradient(r, te); f_rest = float(truth.MII_f0_Hz)
    f_heard = f_rest * c / (c + rr); t_heard = te + r / c - t0
    m = (t_heard > sp.t[0]) & (t_heard < sp.t[-1])
    ax.plot(t_heard[m], f_heard[m], "--", lw=1.2, color="#2A9D8F", label=f"ADS-B track from inferred site, f_rest {f_rest:.0f} Hz")
    ax.set_ylim(40, 450); ax.set_xlabel("time in recording (s)"); ax.set_ylabel("Hz")
    ax.set_title(f"{a.case}: truth v {float(truth.V_ADSB_at_closest_kmh) / 3.6:.0f} m/s, R {float(truth.R_ADSB_m):.0f} m", fontsize=10)
    ax.legend(fontsize=7, loc="upper right"); fig.tight_layout(); fig.savefig(a.out, dpi=130)
    print("saved", a.out)
    for f in fits:
        print(f"fit: f0 {f.f0:.2f} v {f.v:.1f} d {f.d:.0f} t_cpa {f.t_cpa:.1f} t_cpa_emit {f.t_cpa_emit:.1f} f_pre {f.f_pre:.1f} f_post {f.f_post:.1f} score {f.score:.2f}")


if __name__ == "__main__":
    main()
