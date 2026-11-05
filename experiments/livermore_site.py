"""Infer the common recorder position of the Livermore dataset from its 40 ADS-B tracks and the published minimum
slant distances R_ADSB_m (the dataset gives no recorder coordinate). Least squares over (lat, lon, h): for a candidate
point, the minimum 3-D distance to each densely interpolated track must equal the published R; passes whose residual
exceeds `--reject-m` are dropped once and the fit repeated. Mirrors the reference reconstruction used by DOPPLERGEIST.

python experiments/livermore_site.py --root ../data_livermore --out results/livermore_site.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

R_EARTH = 6371000.0


def read_kml(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    whens = re.findall(r"<when\s*>([^<]+)<", txt); coords = re.findall(r"<gx:coord\s*>([^<]+)<", txt)
    n = min(len(whens), len(coords))
    t = pd.to_datetime(whens[:n], utc=True).astype("int64").to_numpy() / 1e9
    xyz = np.array([[float(v) for v in c.split()] for c in coords[:n]])  # lon lat alt(m, MSL)
    order = np.argsort(t); t = t[order]; xyz = xyz[order]
    keep = np.concatenate([[True], np.diff(t) > 0]); return t[keep], xyz[keep]


def enu(lon, lat, alt, lon0, lat0, alt0):
    x = np.radians(lon - lon0) * R_EARTH * np.cos(np.radians(lat0)); y = np.radians(lat - lat0) * R_EARTH
    return np.stack([x, y, alt - alt0], axis=-1)


def dense(t, xyz, step_s=0.5):
    tt = np.arange(t[0], t[-1], step_s)
    return tt, np.stack([np.interp(tt, t, xyz[:, k]) for k in range(3)], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data_livermore"); ap.add_argument("--out", default="results/livermore_site.json")
    ap.add_argument("--reject-m", type=float, default=150.0); ap.add_argument("--window-s", type=float, default=300.0)
    a = ap.parse_args()
    truth = pd.read_csv(Path(a.root) / "final_results_40_cases.csv").set_index("case")
    tracks = {}
    for kml in glob.glob(str(Path(a.root) / "raw" / "*" / "*" / "*.kml")):
        case = os.path.basename(os.path.dirname(kml)); t, xyz = read_kml(kml)
        t0 = pd.Timestamp(truth.loc[case].received_start_utc).timestamp()
        m = (t >= t0 - a.window_s) & (t <= t0 + a.window_s + 120)  # the passage, not the whole flight
        if m.sum() >= 2:
            t, xyz = t[m], xyz[m]
        tracks[case] = dense(t, xyz)
    cases = sorted(tracks); R = np.array([float(truth.loc[c].R_ADSB_m) for c in cases])
    # initial guess: the centroid of the closest track points
    lon0 = np.mean([tracks[c][1][:, 0].mean() for c in cases]); lat0 = np.mean([tracks[c][1][:, 1].mean() for c in cases])
    alt0 = 150.0

    def resid(p, use):
        lon, lat, h = p; out = []
        for c, r in zip(cases, R):
            if c not in use:
                continue
            _, xyz = tracks[c]; e = enu(xyz[:, 0], xyz[:, 1], xyz[:, 2], lon, lat, h)
            out.append(np.sqrt((e ** 2).sum(1)).min() - r)
        return np.array(out)

    use = set(cases)
    for it in range(6):
        sol = least_squares(resid, [lon0, lat0, alt0], args=(use,), x_scale=[1e-3, 1e-3, 10.0], loss="soft_l1", f_scale=30.0)
        res = resid(sol.x, use); used = [c for c in cases if c in use]
        offenders = sorted([(abs(r), c) for c, r in zip(used, res) if abs(r) > a.reject_m], reverse=True)
        bad = {c for _, c in offenders[: max(1, len(offenders) // 3)]} if offenders else set()
        print(f"iteration {it}: {len(used)} passes, RMS {np.sqrt(np.mean(res ** 2)):.2f} m, rejected {sorted(bad)}")
        lon0, lat0, alt0 = sol.x
        if not bad:
            break
        use -= bad
    res = resid(sol.x, use); used = [c for c in cases if c in use]
    # a second estimate of the height from H_ADSB_above_recorder at the closest point
    out = {"lon": float(sol.x[0]), "lat": float(sol.x[1]), "h_m": float(sol.x[2]), "n_used": len(used), "rejected": sorted(set(cases) - use),
           "rms_m": float(np.sqrt(np.mean(res ** 2))), "max_abs_m": float(np.abs(res).max()),
           "residuals_m": {c: round(float(r), 1) for c, r in zip(used, res)}}
    # leave-one-out spread of the position: how much does any single pass move the answer?
    loo = []
    for c in used:
        s = least_squares(resid, sol.x, args=(use - {c},), x_scale=[1e-3, 1e-3, 10.0])
        d = enu(np.array([s.x[0]]), np.array([s.x[1]]), np.array([s.x[2]]), sol.x[0], sol.x[1], sol.x[2])[0]
        loo.append(float(np.sqrt((d[:2] ** 2).sum())))
    out["loo_horizontal_shift_m"] = {"median": float(np.median(loo)), "max": float(np.max(loo))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "residuals_m"}, indent=1))


if __name__ == "__main__":
    main()
