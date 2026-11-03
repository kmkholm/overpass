"""One streaming pass over each adsb.lol day for ALL clips of that day (many bounding boxes).

dopplertrack.fetch_day keeps one bbox per pass; with 40 clips on the same day that would read
the 3.5 GB archive 40 times. This reads it once per date and writes work/<clip>.traces.npz for
every clip of that date (prop aircraft only).

python experiments/fetch_days.py --manifest data/manifest.csv --work work
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

from dopplertrack.adsb import AircraftTypes, in_bbox, save_traces
from dopplertrack.fetch import _ChainedHTTP, iter_tar_traces, release_asset_urls
from overpass.benchmark import bbox_around, load_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--work", default="work")
    ap.add_argument("--pod", default="planes-readsb-prod-0"); ap.add_argument("--region-km", type=float, default=150.0)
    a = ap.parse_args()
    man = load_manifest(a.manifest); work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    types = AircraftTypes()
    for date, g in man.groupby("date"):
        todo = [r for r in g.itertuples() if not (work / f"{r.clip_id}.traces.npz").exists()]
        if not todo:
            continue
        boxes = {r.clip_id: bbox_around(float(r.lat), float(r.lon), a.region_km) for r in todo}
        # union box: a trace is kept while streaming if it touches any clip box
        urls = release_asset_urls(date, a.pod)
        stream = _ChainedHTTP(urls); t0 = time.time()
        kept = {k: [] for k in boxes}
        union = (min(b[0] for b in boxes.values()), max(b[1] for b in boxes.values()),
                 min(b[2] for b in boxes.values()), max(b[3] for b in boxes.values()))
        n = 0
        for tr in iter_tar_traces(io.BufferedReader(stream, 1 << 22), union):
            n += 1
            if not types.is_prop(tr.typ):
                continue
            for k, b in boxes.items():
                if in_bbox(tr, b):
                    kept[k].append(tr)
            if n % 500 == 0:
                print(f"[{date}] {n} traces in union box, {stream.total / 1e9:.2f} GB read, {time.time() - t0:.0f} s", flush=True)
        for k, trs in kept.items():
            save_traces(trs, work / f"{k}.traces.npz")
            print(f"[{date}] {k}: {len(trs)} prop traces")


if __name__ == "__main__":
    main()
