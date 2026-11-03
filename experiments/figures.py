"""Paper figures from the result files (run after the benchmark, calibration and scrub scripts).

Figure set (results checklist): detector ROC with AUC and training curves; error CDF per
configuration; conformal coverage vs set area; score-margin distributions; score map of one
clip with the conformal set; modelled curves on the real spectrogram at the true point;
spectrograms before/after scrubbing. Distinctive colours, Times font, no default blue/orange.

python experiments/figures.py --work work --bench results/bench.csv --conformal results/conformal.csv \
    --scrub results/scrub.csv --model models/ridgenet.pt --clip ref_oysterbay --out figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
PAL = {"full": "#7B1E3A", "raw": "#1F6F8B", "two": "#D9A441", "nosilence": "#5C7F3C", "nodem": "#8C6BB1", "scan": "#4D4D4D"}
LBL = {"full": "OVERPASS (full)", "raw": "raw brightness score", "two": "two events only", "nosilence": "no silence filter",
       "nodem": "no DEM", "scan": "grid-scan detector"}


def fig_roc(model_path, out):
    roc = json.load(open(model_path + ".roc.json")); hist = json.load(open(model_path + ".history.json"))
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax[0].plot(roc["fpr"], roc["tpr"], color=PAL["full"], lw=1.6, label=f"ridge detector, AUC = {roc['auc']:.3f}")
    ax[0].plot([0, 1], [0, 1], ":", color="#999999", lw=0.8); ax[0].set_xlabel("false-positive rate (pixels)"); ax[0].set_ylabel("true-positive rate"); ax[0].legend(frameon=False)
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    ax[1].plot(ep, hist["train_loss"], color=PAL["full"], lw=1.4, label="training loss"); ax[1].plot(ep, hist["val_loss"], color=PAL["raw"], lw=1.4, label="validation loss")
    ax2 = ax[1].twinx(); ax2.plot(ep, hist["val_dice"], color=PAL["two"], lw=1.2, label="validation Dice"); ax2.set_ylabel("Dice"); ax2.spines["top"].set_visible(False)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("BCE + Dice loss"); h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_detector_roc.pdf"); fig.savefig(out / "fig_detector_roc.png", dpi=300); plt.close(fig)


def fig_error_cdf(bench, out):
    b = bench[bench.status == "ok"]
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for cfg, g in b.groupby("config"):
        e = np.sort(g.top_error_m.to_numpy()); y = np.arange(1, len(e) + 1) / len(e)
        ax[0].step(e, y, where="post", color=PAL.get(cfg, "k"), lw=1.5, label=LBL.get(cfg, cfg))
    ax[0].set_xscale("log"); ax[0].set_xlabel("horizontal error of the top point (m)"); ax[0].set_ylabel("fraction of clips"); ax[0].legend(frameon=False, fontsize=7)
    for cfg, g in b.groupby("config"):
        m = g.margin_true_vs_best_wrong_tuple.dropna()
        ax[1].hist(m, bins=np.linspace(-0.5, 1.5, 21), histtype="step", color=PAL.get(cfg, "k"), lw=1.4, label=LBL.get(cfg, cfg))
    ax[1].axvline(0, color="#999999", ls=":", lw=0.8); ax[1].set_xlabel("margin of the true flight tuple over the best wrong tuple"); ax[1].set_ylabel("clips")
    fig.tight_layout(); fig.savefig(out / "fig_error_cdf_margin.pdf"); fig.savefig(out / "fig_error_cdf_margin.png", dpi=300); plt.close(fig)


def fig_conformal(conf, out):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for (kind, alpha), g in conf.groupby(["kind", "alpha"]):
        col = PAL["full"] if kind == "gap" else PAL["raw"]
        ax.scatter(g.area_km2, g.covered.astype(float) + np.random.default_rng(0).normal(0, 0.02, len(g)),
                   s=12, color=col, alpha=0.7, marker="o" if alpha == 0.1 else "s", label=f"{kind} set, 1-α = {1 - alpha:.1f} (coverage {g.covered.mean():.2f})")
    ax.set_xscale("log"); ax.set_xlabel("area of the returned set (km²)"); ax.set_yticks([0, 1]); ax.set_yticklabels(["missed", "covered"]); ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout(); fig.savefig(out / "fig_conformal.pdf"); fig.savefig(out / "fig_conformal.png", dpi=300); plt.close(fig)


def fig_scoremap(work, clip, cfg, lat, lon, q, spacing, out):
    res = pd.read_csv(Path(work) / f"{clip}.{cfg}.s0.res.csv")
    best = res.sort_values("score", ascending=False).drop_duplicates("gidx")
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    sc = ax.scatter(best.lon, best.lat, c=best.score, s=3, cmap="cividis")
    plt.colorbar(sc, ax=ax, label="contrastive locked score (z)", shrink=0.8)
    inset = best[best.score >= best.score.max() - q]
    ax.scatter(inset.lon, inset.lat, s=6, facecolor="none", edgecolor=PAL["full"], lw=0.5, label="90 % conformal set")
    ax.plot(lon, lat, "+", color="#E63946", ms=11, mew=1.6, label="true location")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_aspect(1 / np.cos(np.radians(lat))); ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(out / f"fig_scoremap_{clip}.pdf"); fig.savefig(out / f"fig_scoremap_{clip}.png", dpi=300); plt.close(fig)


def fig_scrub(work, clip, out):
    from dopplertrack.audio import load_audio, compute_spectrogram
    import soundfile as sf
    y, sr = sf.read(Path(work) / f"{clip}.scrubbed.wav", dtype="float32")
    man = pd.read_csv(Path(work).parent / "data" / "manifest.csv").set_index("clip_id")
    x, _ = load_audio(man.loc[clip].path, sr=sr)
    det = json.load(open(Path(work) / f"{clip}.det.json"))[0]["fit"]
    t0, t1 = det["t_start"] - 30, det["t_end"] + 30
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True)
    for a_, sig, ttl in zip(ax, (x, y), ("before scrubbing", "after track-and-fill scrubbing")):
        seg = sig[int(t0 * sr):int(t1 * sr)]
        sp = compute_spectrogram(seg, sr, nperseg=8192 * sr // 4000, hop_s=0.25, fmin=40, fmax=450, whiten_window_s=None)
        a_.imshow(sp.S, origin="lower", aspect="auto", extent=[t0, t1, sp.f[0], sp.f[-1]], vmin=0, vmax=10, cmap="magma")
        a_.set_title(ttl); a_.set_xlabel("time (s)")
    ax[0].set_ylabel("frequency (Hz)")
    fig.tight_layout(); fig.savefig(out / f"fig_scrub_{clip}.pdf"); fig.savefig(out / f"fig_scrub_{clip}.png", dpi=300); plt.close(fig)


def fig_scrub_bars(scrub, out):
    s = scrub[scrub.status == "ok"]
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    order = ["none", "highpass", "masker", "codec", "notch", "fill"]
    g = s.groupby("defence").contrast_after_db.agg(["mean", "std"]).reindex(order)
    ax.bar(range(len(g)), g["mean"], yerr=g["std"], color=[PAL["raw"]] * 4 + [PAL["two"], PAL["full"]], capsize=2)
    ax.set_xticks(range(len(g))); ax.set_xticklabels(["none", "high-pass", "masker", "codec", "notch", "fill (ours)"], rotation=20)
    ax.set_ylabel("attacker's curve contrast at truth (dB)"); ax.axhline(0, color="#999999", lw=0.8, ls=":")
    fig.tight_layout(); fig.savefig(out / "fig_scrub_bars.pdf"); fig.savefig(out / "fig_scrub_bars.png", dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work"); ap.add_argument("--bench"); ap.add_argument("--conformal"); ap.add_argument("--scrub")
    ap.add_argument("--model"); ap.add_argument("--clip", default="ref_oysterbay"); ap.add_argument("--out", default="figures")
    ap.add_argument("--q", type=float, default=1.0); ap.add_argument("--spacing", type=float, default=200.0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.model: fig_roc(a.model, out)
    if a.bench:
        bench = pd.read_csv(a.bench); fig_error_cdf(bench, out)
        r = bench[(bench.clip_id == a.clip) & (bench.config == "full") & (bench.seed == 0)]
        if len(r): fig_scoremap(a.work, a.clip, "full", float(r.lat_true.iloc[0]), float(r.lon_true.iloc[0]), a.q, a.spacing, out)
    if a.conformal: fig_conformal(pd.read_csv(a.conformal), out)
    if a.scrub:
        fig_scrub_bars(pd.read_csv(a.scrub), out)
        if (Path(a.work) / f"{a.clip}.scrubbed.wav").exists(): fig_scrub(a.work, a.clip, out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
