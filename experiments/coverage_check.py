"""ADS-B coverage check: one streaming pass over an adsb.lol day, counting ALL traces and propeller traces
inside a box around each named point. Answers "does the public archive see aircraft here at all?"

python experiments/coverage_check.py --date 2026-11-03 --points data/coverage_points.csv --out results/coverage_2026-11-03.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import time

from dopplertrack.adsb import AircraftTypes, in_bbox
from dopplertrack.fetch import _ChainedHTTP, iter_tar_traces, release_asset_urls
from overpass.benchmark import bbox_around


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--points", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--region-km", type=float, default=100.0); ap.add_argument("--pod", default="planes-readsb-prod-0")
    a = ap.parse_args()
    pts = list(csv.DictReader(open(a.points, newline="", encoding="utf-8")))
    boxes = {p["name"]: bbox_around(float(p["lat"]), float(p["lon"]), a.region_km) for p in pts}
    union = (min(b[0] for b in boxes.values()), max(b[1] for b in boxes.values()),
             min(b[2] for b in boxes.values()), max(b[3] for b in boxes.values()))
    types = AircraftTypes(); allc = {k: 0 for k in boxes}; prop = {k: 0 for k in boxes}; typ = {k: {} for k in boxes}
    stream = _ChainedHTTP(release_asset_urls(a.date, a.pod)); t0 = time.time(); n = 0
    for tr in iter_tar_traces(io.BufferedReader(stream, 1 << 22), union):
        n += 1
        for k, b in boxes.items():
            if in_bbox(tr, b):
                allc[k] += 1; typ[k][tr.typ] = typ[k].get(tr.typ, 0) + 1
                if types.is_prop(tr.typ):
                    prop[k] += 1
        if n % 500 == 0:
            print(f"[{a.date}] {n} traces in union box, {stream.total / 1e9:.2f} GB read, {time.time() - t0:.0f} s", flush=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["date", "name", "lat", "lon", "region_km", "traces_all", "traces_prop", "top_types"])
        for p in pts:
            k = p["name"]; top = ",".join(f"{t}:{c}" for t, c in sorted(typ[k].items(), key=lambda x: -x[1])[:6])
            w.writerow([a.date, k, p["lat"], p["lon"], a.region_km, allc[k], prop[k], top])
            print(f"{k:12s} all {allc[k]:5d}  prop {prop[k]:4d}  {top}", flush=True)
    print(f"done: {n} traces, {stream.total / 1e9:.2f} GB, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
