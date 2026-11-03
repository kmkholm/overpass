"""Known-point check on the reference clip (Oyster Bay, 2026-09-08, start 20:17:30 UTC).

1. Every prop flight's CPA at the true point (heard time, slant distance, speed).
2. The classical scan detector finds Doppler events in the recording; events are matched to
   modelled CPAs (same heard time within +-30 s) -> which flights are audible.
3. For the matched flights the exact modelled curve is drawn over the spectrogram, the ideal
   fit is run, and the contrastive locked z-score at the truth is reported against a wrong
   point 20 km away (raw brightness as well, for the paper's motivation figure).

python experiments/known_point.py --spec work/ref_oysterbay.spec.npz --traces work/ref_oysterbay.traces.npz \
    --lat 40.876466 --lon -73.524951 --utc 2026-09-08T20:17:30+00:00 --out results/known_point
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dopplertrack.adsb import load_traces
from dopplertrack.audio import Spectrogram
from dopplertrack.doppler import speed_of_sound
from dopplertrack.geo import lla_to_enu
from dopplertrack.search import Grid, build_segments, cpa_table
from dopplertrack.terrain import Terrain
from dopplertrack.tracks import detect_events, fit_event, local_whiten
from overpass.contrast import ContrastWindow, contrast_locked_score, modelled_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True); ap.add_argument("--traces", required=True)
    ap.add_argument("--lat", type=float, required=True); ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--utc", required=True, help="clip start, ISO-8601 UTC"); ap.add_argument("--out", default="results/known_point")
    ap.add_argument("--band", nargs=2, type=float, default=[60.0, 260.0]); ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--no-dem", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    c = speed_of_sound()
    sp = Spectrogram.load(a.spec); traces = load_traces(a.traces)
    t0_unix = datetime.fromisoformat(a.utc.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
    elev = 0.0 if a.no_dem else float(Terrain().elevation(np.array([a.lat]), np.array([a.lon]))[0])
    grid = Grid.points([a.lat], [a.lon], [elev])
    segs = build_segments(traces, grid.lat0, grid.lon0)
    cpa = cpa_table(grid, segs, max_range_m=12000.0, c=c)
    cpa["t_audio"] = cpa.t_heard - t0_unix
    cpa["icao"] = [segs[i].icao for i in cpa.seg]
    inclip = cpa[(cpa.t_audio > 0) & (cpa.t_audio < sp.t[-1])].sort_values("t_audio")
    inclip.to_csv(out / "cpa_at_truth.csv", index=False)
    print(f"{len(segs)} segments; {len(inclip)} CPAs within 12 km during the clip (elev {elev:.0f} m)")
    print(inclip[["icao", "t_audio", "d", "v"]].to_string(index=False))

    print("scanning the recording for Doppler events (classical detector) ...", flush=True)
    evs = detect_events(sp, a.band[0], a.band[1], top=a.top, workers=4)
    json.dump([e.to_dict() for e in evs], open(out / "scan_events.json", "w"), indent=1)
    rows = []
    for e in evs:
        d = (inclip.t_audio - e.t_cpa).abs()
        j = d.idxmin() if len(d) else None
        match = inclip.loc[j] if j is not None and d[j] <= 30 else None
        rows.append({"t_cpa": e.t_cpa, "f0": e.f0, "v_fit": e.v, "d_fit": e.d, "score": e.score,
                     "icao": match.icao if match is not None else "", "t_cpa_model": float(match.t_audio) if match is not None else np.nan,
                     "d_model": float(match.d) if match is not None else np.nan, "v_model": float(match.v) if match is not None else np.nan})
    tab = pd.DataFrame(rows); tab.to_csv(out / "events_vs_model.csv", index=False)
    print(tab.to_string(index=False))

    matched = tab[tab.icao != ""].sort_values("t_cpa")
    if len(matched) >= 1:
        obs = grid.enu[0]
        wrong = lla_to_enu(a.lat + 0.18, a.lon, elev, grid.lat0, grid.lon0)
        wins, ss, fits = [], [], []
        for r in matched.itertuples():
            ev = fit_event(sp, r.t_cpa - 110, r.t_cpa + 110, max(r.f0 * 0.75, sp.f[0] + 1), min(r.f0 * 1.3, sp.f[-1] - 1), c=c, harmonics=2)
            seg = segs[int(inclip[inclip.icao == r.icao].seg.iloc[0])]
            wins.append(ContrastWindow.from_fit(sp, ev, 2)); ss.append(seg); fits.append(ev)
        d = np.linalg.norm(ss[0].enu - obs, axis=1); j = int(np.argmin(d)); th = ss[0].t[j] + d[j] / c
        dw = np.linalg.norm(ss[0].enu - wrong, axis=1); jw = int(np.argmin(dw)); thw = ss[0].t[jw] + dw[jw] / c
        z_t, zs_t, sh_t, raw_t = contrast_locked_score(wins, ss, obs, th, c=c)
        z_w, zs_w, sh_w, raw_w = contrast_locked_score(wins, ss, wrong, thw, c=c)
        b_t = contrast_locked_score(wins, ss, obs, th, c=c, raw=True)[0]; b_w = contrast_locked_score(wins, ss, wrong, thw, c=c, raw=True)[0]
        summary = {"n_matched": int(len(matched)), "flights": list(matched.icao), "z_truth": z_t, "z_per_event_truth": zs_t, "shift_truth": sh_t,
                   "z_wrong_20km": z_w, "raw_contrast_truth": b_t, "raw_contrast_wrong_20km": b_w}
        json.dump(summary, open(out / "summary.json", "w"), indent=1)
        print(json.dumps(summary, indent=1))
        # figure: modelled curves at the truth over the whitened spectrogram
        fig, axes = plt.subplots(1, len(fits), figsize=(3.2 * len(fits), 2.6), squeeze=False)
        for ax, ev, seg in zip(axes[0], fits, ss):
            w = local_whiten(sp.crop(ev.t_start - 30, ev.t_end + 30, max(ev.f_post * 0.7, sp.f[0]), min(ev.f_pre * 1.3 * 2, sp.f[-1])))
            ax.imshow(w.S, origin="lower", aspect="auto", extent=[w.t[0], w.t[-1], w.f[0], w.f[-1]], vmin=0, vmax=10, cmap="magma")
            to, ratio = modelled_curve(seg, obs, c)
            ta = to - t0_unix + sh_t
            for h in (1, 2):
                ax.plot(ta, h * ev.f0 * ratio, color="#00E5FF", lw=0.8)
            ax.set_xlim(w.t[0], w.t[-1]); ax.set_ylim(w.f[0], w.f[-1]); ax.set_xlabel("recording time (s)"); ax.set_title(seg.icao, fontsize=9)
        axes[0][0].set_ylabel("frequency (Hz)")
        fig.tight_layout(); fig.savefig(out / "modelled_curves_truth.png", dpi=200); fig.savefig(out / "modelled_curves_truth.pdf")
        print("figure:", out / "modelled_curves_truth.png")
    else:
        print("no detected event matched a modelled CPA at the truth")


if __name__ == "__main__":
    main()
