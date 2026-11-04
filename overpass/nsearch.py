"""Joint search over N flyovers with the silence constraint, and the two operating modes.

Generalises dopplertrack.search.pair_candidates (two events) to an ordered chain of N events:
a grid point must hear an event-1-like CPA, then event-k-like CPAs exactly dt_k later
(within tol_s) for k = 2..N, with no other propeller flight within silence_d_m during the
gaps. Each candidate (point, flight_1..flight_N) is scored with the contrastive locked score.

Modes:
  verify(...)  score one claimed point (or a small region around it) – forensic verification
  search(...)  bounded search over a region of at most max_region_km on a side
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from dopplertrack.adsb import Trace
from dopplertrack.audio import Spectrogram
from dopplertrack.doppler import speed_of_sound
from dopplertrack.geo import lla_to_enu
from dopplertrack.search import Grid, Segment, build_segments, cpa_table, load_cpa_table, pair_candidates, save_cpa_table
from dopplertrack.tracks import EventFit

from .contrast import ContrastWindow, contrast_locked_score, extra_score


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _bucket_join(a: pd.DataFrame, b: pd.DataFrame, ta: str, tb: str, dt: float, tol_s: float, suffix: str) -> pd.DataFrame:
    """Rows of a joined with rows of b on gidx where |b[tb] - a[ta] - dt| <= tol_s (bucket join)."""
    w = 2.0 * tol_s
    a = a.assign(key=np.floor(a[ta].to_numpy() / w).astype(np.int64))
    kb = np.floor((b[tb].to_numpy() - dt) / w).astype(np.int64)
    parts = [a.merge(b.assign(key=kb + off), on=["gidx", "key"], suffixes=("", suffix)) for off in (-1, 0, 1)]
    m = pd.concat(parts, ignore_index=True).drop(columns="key")
    return m[((m[tb + suffix] - m[ta] - dt).abs() <= tol_s)].copy()


def chain_candidates(cpa: pd.DataFrame, offsets: list[float], tol_s: float = 6.0,
                     v_ranges=None, d_ranges=None, silence_d_m: float | None = 6000.0,
                     silence_margin_s: float = 40.0) -> pd.DataFrame:
    """Grid points hearing event 1 then events 2..N at the given heard-CPA offsets from event 1.

    offsets[k-2] = t_cpa_k - t_cpa_1 (recording seconds). Returns columns gidx, seg1..segN,
    t_heard1..N, d1..dN, v1..vN."""
    N = len(offsets) + 1
    v_ranges = v_ranges or [(0, 1e9)] * N; d_ranges = d_ranges or [(0, 1e9)] * N
    ev = [cpa[cpa.v.between(*v_ranges[k]) & cpa.d.between(*d_ranges[k])] for k in range(N)]
    m = ev[0].rename(columns={"seg": "seg1", "t_emit": "t_emit1", "t_heard": "t_heard1", "d": "d1", "v": "v1"})
    for k in range(2, N + 1):
        b = ev[k - 1].rename(columns={"seg": f"seg{k}", "t_emit": f"t_emit{k}", "t_heard": f"t_heard{k}", "d": f"d{k}", "v": f"v{k}"})
        m = _bucket_join(m, b, "t_heard1", f"t_heard{k}", offsets[k - 2], tol_s, "")
        for j in range(1, k):
            m = m[m[f"seg{j}"] != m[f"seg{k}"]]
        if len(m) == 0:
            break
    m = m.reset_index(drop=True)
    if silence_d_m is not None and len(m):
        near = cpa[(cpa.d <= silence_d_m) & cpa.gidx.isin(m.gidx.unique())][["gidx", "seg", "t_heard"]]
        x = m[["gidx"] + [f"seg{k}" for k in range(1, N + 1)] + [f"t_heard{k}" for k in range(1, N + 1)]].reset_index().merge(near, on="gidx")
        other = np.ones(len(x), bool)
        for k in range(1, N + 1):
            other &= (x.seg != x[f"seg{k}"]).to_numpy()
        inside = np.zeros(len(x), bool)
        for k in range(1, N):
            inside |= ((x.t_heard > x[f"t_heard{k}"] + silence_margin_s) & (x.t_heard < x[f"t_heard{k + 1}"] - silence_margin_s)).to_numpy()
        bad = x[other & inside]["index"].unique()
        m = m.drop(index=bad).reset_index(drop=True)
    return m


def _segments_for(traces, grid: Grid, max_range_m: float):
    e0, e1 = grid.enu[:, 0].min() - max_range_m, grid.enu[:, 0].max() + max_range_m
    n0, n1 = grid.enu[:, 1].min() - max_range_m, grid.enu[:, 1].max() + max_range_m
    return build_segments(traces, grid.lat0, grid.lon0, clip_enu=(e0, e1, n0, n1))


def run_nsearch(sp: Spectrogram, events: list[EventFit], traces: list[Trace], grid: Grid,
                harmonics: list[int] | None = None, tol_s: float = 12.0, v_tol: float = 0.35, d_tol=(0.4, 2.5),
                silence_d_m: float | None = 6000.0, max_range_m: float = 12000.0, c: float | None = None,
                cpa_cache=None, raw_score: bool = False, max_candidates: int | None = 400000,
                progress: bool = True) -> tuple[pd.DataFrame, list[Segment]]:
    """Rank grid points for an N-event recording (events sorted by t_cpa)."""
    c = speed_of_sound() if c is None else c
    events = sorted(events, key=lambda e: e.t_cpa)
    N = len(events)
    harmonics = harmonics or [1] * N
    segs = _segments_for(traces, grid, max_range_m)
    log(f"{len(segs)} airborne segments; grid of {len(grid)} points; {N} events")
    fp = {"grid": (len(grid), float(grid.lat0), float(grid.lon0), float(grid.lat.min()), float(grid.lon.max())),
          "segs": (len(segs), int(sum(len(s.t) for s in segs))), "max_range_m": float(max_range_m), "c": float(c)}
    cpa = None
    if cpa_cache:
        try:
            cpa = load_cpa_table(cpa_cache, fp)
        except FileNotFoundError:
            cpa = None
    if cpa is None:
        cpa = cpa_table(grid, segs, max_range_m=max_range_m, c=c)
        if cpa_cache:
            save_cpa_table(cpa_cache, cpa, fp)
    log(f"{len(cpa)} (point, flight) CPA records")
    # Anchors and soft extras. The two strongest fits are the anchors: a grid point must hear an
    # anchor-1-like pass then an anchor-2-like pass at the measured offset (the baseline's two-event
    # chain). Every further event is *soft*: if a flight consistent with it passes the point at the
    # right offset it is scored and adds max(z, 0); if not, nothing is added and the point survives.
    # A badly fitted or spurious third event therefore cannot veto the true location, while every
    # genuine extra flyover sharpens the ranking.
    _vr = lambda e: (e.v * (1 - v_tol), e.v * (1 + v_tol))
    _dr = lambda e: (max(100.0, min(e.d * d_tol[0], e.d - 2000.0)), max(e.d * d_tol[1], e.d + 3000.0))
    order = sorted(range(N), key=lambda i: -events[i].score)
    a_idx = sorted(order[:2], key=lambda i: events[i].t_cpa)
    x_idx = sorted(order[2:], key=lambda i: events[i].t_cpa)
    e1, e2 = events[a_idx[0]], events[a_idx[1]]
    dt12 = e2.t_cpa - e1.t_cpa
    cands = pair_candidates(cpa, dt12, tol_s=tol_s, v1=_vr(e1), d1=_dr(e1), v2=_vr(e2), d2=_dr(e2), silence_d_m=None)
    log(f"{len(cands)} anchor (point, flight1, flight2) candidates at offset {dt12:.1f} s +- {tol_s:.0f} s")
    nx = len(x_idx)
    for k, xi in enumerate(x_idx):
        ex = events[xi]; off = ex.t_cpa - e1.t_cpa
        b = cpa[cpa.v.between(*_vr(ex)) & cpa.d.between(*_dr(ex))][["gidx", "seg", "t_heard", "d", "v"]]
        b = b.rename(columns={"seg": f"xseg{k}", "t_heard": f"xt{k}", "d": f"xd{k}", "v": f"xv{k}"})
        a = cands[["gidx", "seg1", "seg2", "t_heard1"]].reset_index()
        m = _bucket_join(a, b, "t_heard1", f"xt{k}", off, tol_s, "") if len(cands) else a.iloc[0:0]
        if len(m):
            m = m[(m[f"xseg{k}"] != m.seg1) & (m[f"xseg{k}"] != m.seg2)].copy()
            m["err"] = (m[f"xt{k}"] - m.t_heard1 - off).abs()
            m = m.sort_values("err").drop_duplicates("index").set_index("index")
        for col, fill in ((f"xseg{k}", -1), (f"xt{k}", np.nan), (f"xd{k}", np.nan), (f"xv{k}", np.nan)):
            cands[col] = (m[col].reindex(cands.index) if len(m) else pd.Series(index=cands.index, dtype=float)).fillna(fill)
        cands[f"xseg{k}"] = cands[f"xseg{k}"].astype(int)
        log(f"extra event {k + 1} (t_cpa {ex.t_cpa:.0f} s): a consistent flight at {int((cands[f'xseg{k}'] >= 0).sum())} of {len(cands)} candidates")
    if silence_d_m is not None and len(cands):
        near = cpa[(cpa.d <= silence_d_m) & cpa.gidx.isin(cands.gidx.unique())][["gidx", "seg", "t_heard"]]
        xcols = [f"xseg{k}" for k in range(nx)]
        x = cands.reset_index()[["index", "gidx", "seg1", "seg2", "t_heard1", "t_heard2"] + xcols].merge(near, on="gidx")
        other = ((x.seg != x.seg1) & (x.seg != x.seg2)).to_numpy()
        for col in xcols:
            other &= (x.seg != x[col]).to_numpy()
        inside = ((x.t_heard > x.t_heard1 + 40.0) & (x.t_heard < x.t_heard2 - 40.0)).to_numpy()
        bad = x[other & inside]["index"].unique()
        cands = cands.drop(index=bad)
        log(f"{len(cands)} candidates after the silence filter between the anchors")
    cands = cands.reset_index(drop=True)
    if max_candidates and len(cands) > max_candidates:
        log(f"capping at {max_candidates} candidates (closest anchor-offset agreement first)")
        err = (cands.t_heard2 - cands.t_heard1 - dt12).abs()
        cands = cands.loc[np.argsort(err.to_numpy())[:max_candidates]].reset_index(drop=True)
    wins_a = [ContrastWindow.from_fit(sp, e1, harmonics[a_idx[0]]), ContrastWindow.from_fit(sp, e2, harmonics[a_idx[1]])]
    wins_x = [ContrastWindow.from_fit(sp, events[xi], harmonics[xi]) for xi in x_idx]
    out = []; t0 = time.time()
    for i, row in enumerate(cands.itertuples(index=False)):
        rs = row._asdict()
        obs = grid.enu[int(rs["gidx"])]
        ss = [segs[int(rs["seg1"])], segs[int(rs["seg2"])]]
        tot, zs, sh, raws = contrast_locked_score(wins_a, ss, obs, float(rs["t_heard1"]), c=c, raw=raw_score)
        rec = {"gidx": int(rs["gidx"]), "lat": grid.lat[int(rs["gidx"])], "lon": grid.lon[int(rs["gidx"])], "score": tot, "shift": sh}
        for k in (1, 2):
            rec[f"seg{k}"] = int(rs[f"seg{k}"]); rec[f"icao{k}"] = ss[k - 1].icao; rec[f"z{k}"] = zs[k - 1]; rec[f"raw{k}"] = raws[k - 1]
            rec[f"t_heard{k}"] = float(rs[f"t_heard{k}"]); rec[f"d{k}"] = float(rs[f"d{k}"]); rec[f"v{k}"] = float(rs[f"v{k}"])
        t0_map = e1.t_cpa - float(rs["t_heard1"])
        for k in range(nx):
            j = k + 3; xs = int(rs[f"xseg{k}"])
            if xs >= 0:
                zk, rk = extra_score(wins_x[k], segs[xs], obs, t0_map, sh, c=c, raw=raw_score)
                rec["score"] += max(zk, 0.0)
                rec.update({f"seg{j}": xs, f"icao{j}": segs[xs].icao, f"z{j}": zk, f"raw{j}": rk, f"t_heard{j}": float(rs[f"xt{k}"]),
                            f"d{j}": float(rs[f"xd{k}"]), f"v{j}": float(rs[f"xv{k}"])})
            else:
                rec.update({f"seg{j}": -1, f"icao{j}": "", f"z{j}": 0.0, f"raw{j}": 0.0, f"t_heard{j}": np.nan, f"d{j}": np.nan, f"v{j}": np.nan})
        out.append(rec)
        if progress and (i + 1) % 500 == 0:
            log(f"scored {i + 1}/{len(cands)} ({(time.time() - t0) / (i + 1) * 1000:.0f} ms each)")
    res = pd.DataFrame(out)
    if len(res):
        res = res.sort_values("score", ascending=False).reset_index(drop=True)
    res.attrs["lat0"] = grid.lat0; res.attrs["lon0"] = grid.lon0
    res.attrs["events"] = [e1, e2] + [events[xi] for xi in x_idx]
    res.attrs["harmonics"] = [harmonics[a_idx[0]], harmonics[a_idx[1]]] + [harmonics[xi] for xi in x_idx]
    return res, segs


def refine_n(sp: Spectrogram, events: list[EventFit], segs: list[Segment], row, lat0: float, lon0: float,
             spacing_m: float = 50.0, half_m: float = 400.0, harmonics=None, elevation=None, c=None, raw_score=False):
    """Fine grid around one result row. `events` in the order of res.attrs["events"]: two anchors
    then the soft extras; extras whose seg is -1 in the row are skipped."""
    c = speed_of_sound() if c is None else c
    N = len(events)
    harmonics = harmonics or [1] * N
    lat, lon = float(row["lat"]), float(row["lon"])
    dlat = half_m / 111320.0; dlon = half_m / (111320.0 * np.cos(np.radians(lat)))
    g = Grid.make(lat - dlat, lat + dlat, lon - dlon, lon + dlon, spacing_m, elevation)
    ss = [segs[int(row["seg1"])], segs[int(row["seg2"])]]
    wins = [ContrastWindow.from_fit(sp, events[0], harmonics[0]), ContrastWindow.from_fit(sp, events[1], harmonics[1])]
    extras = [(ContrastWindow.from_fit(sp, events[k], harmonics[k]), segs[int(row[f"seg{k + 1}"])])
              for k in range(2, N) if f"seg{k + 1}" in row and int(row[f"seg{k + 1}"]) >= 0]
    rows = []
    for i in range(len(g)):
        obs = lla_to_enu(g.lat[i], g.lon[i], g.elev[i], lat0, lon0)
        d = np.linalg.norm(ss[0].enu - obs, axis=1); j = int(np.argmin(d)); t_heard1 = ss[0].t[j] + d[j] / c
        tot, zs, sh, _ = contrast_locked_score(wins, ss, obs, float(t_heard1), c=c, raw=raw_score)
        t0_map = events[0].t_cpa - float(t_heard1)
        zx = [extra_score(w, s, obs, t0_map, sh, c=c, raw=raw_score)[0] for w, s in extras]
        tot += sum(max(z, 0.0) for z in zx)
        rows.append((g.lat[i], g.lon[i], tot, sh, *zs, *zx))
    cols = ["lat", "lon", "score", "shift", "z1", "z2"] + [f"z{k + 3}" for k in range(len(extras))]
    return pd.DataFrame(rows, columns=cols).sort_values("score", ascending=False).reset_index(drop=True)


def verify(sp: Spectrogram, events: list[EventFit], traces: list[Trace], lat: float, lon: float,
           elevation=None, half_m: float = 1500.0, spacing_m: float = 100.0, **kw):
    """Verification mode: score a claimed point and its neighbourhood (a 3 km box by default).
    Returns (res, segs). The claimed point's own rank and score gap are what the examiner reads."""
    dlat = half_m / 111320.0; dlon = half_m / (111320.0 * np.cos(np.radians(lat)))
    grid = Grid.make(lat - dlat, lat + dlat, lon - dlon, lon + dlon, spacing_m, elevation)
    return run_nsearch(sp, events, traces, grid, **kw)


def search(sp: Spectrogram, events: list[EventFit], traces: list[Trace], bbox, elevation=None,
           spacing_m: float = 200.0, max_region_km: float = 150.0, **kw):
    """Bounded search mode. Refuses regions larger than max_region_km on a side."""
    lat_min, lat_max, lon_min, lon_max = bbox
    side_ns = (lat_max - lat_min) * 111.32
    side_ew = (lon_max - lon_min) * 111.32 * np.cos(np.radians(0.5 * (lat_min + lat_max)))
    if max(side_ns, side_ew) > max_region_km:
        raise ValueError(f"region {side_ns:.0f} x {side_ew:.0f} km exceeds the {max_region_km:.0f} km bound")
    grid = Grid.make(lat_min, lat_max, lon_min, lon_max, spacing_m, elevation)
    return run_nsearch(sp, events, traces, grid, **kw)


def best_per_point(res: pd.DataFrame) -> pd.DataFrame:
    return res.sort_values("score", ascending=False).drop_duplicates("gidx").reset_index(drop=True)
