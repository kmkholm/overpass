"""overpass command line: detect | verify | search | scrub."""
from __future__ import annotations

import argparse
import json

import numpy as np

from dopplertrack.adsb import load_traces
from dopplertrack.audio import Spectrogram, load_audio
from dopplertrack.terrain import Terrain
from dopplertrack.tracks import EventFit

from . import detect as D
from . import nsearch as NS
from . import scrub as SC


def _events_from(args, sp):
    if args.events:
        return [EventFit(**json.load(open(p))) for p in args.events]
    dets = D.load_detections(args.detections)
    evs = [d.fit for d in dets if d.fit is not None]
    if args.top_events:
        evs = evs[:args.top_events]
    return sorted(evs, key=lambda e: e.t_cpa)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="overpass")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="automatic flyover detection on a spectrogram (.npz)")
    d.add_argument("spec"); d.add_argument("--model", required=True); d.add_argument("--out", required=True)
    d.add_argument("--thr", type=float, default=0.5); d.add_argument("--top", type=int, default=8)

    v = sub.add_parser("verify", help="score a claimed point and its 3 km neighbourhood")
    v.add_argument("spec"); v.add_argument("--traces", required=True); v.add_argument("--out", required=True)
    v.add_argument("--point", nargs=2, type=float, required=True, metavar=("LAT", "LON"))
    v.add_argument("--detections"); v.add_argument("--events", nargs="*"); v.add_argument("--top-events", type=int, default=3)
    v.add_argument("--half-m", type=float, default=1500.0); v.add_argument("--spacing", type=float, default=100.0)
    v.add_argument("--no-dem", action="store_true"); v.add_argument("--raw", action="store_true", help="raw brightness score (ablation)")

    s = sub.add_parser("search", help="bounded search over a region (<= 150 km a side)")
    s.add_argument("spec"); s.add_argument("--traces", required=True); s.add_argument("--out", required=True)
    s.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"))
    s.add_argument("--detections"); s.add_argument("--events", nargs="*"); s.add_argument("--top-events", type=int, default=3)
    s.add_argument("--spacing", type=float, default=200.0); s.add_argument("--no-dem", action="store_true")
    s.add_argument("--cpa-cache"); s.add_argument("--raw", action="store_true"); s.add_argument("--no-silence", action="store_true")
    s.add_argument("--refine", action="store_true", help="50 m refinement around the top result")

    c = sub.add_parser("scrub", help="remove Doppler combs from an audio file")
    c.add_argument("audio"); c.add_argument("out"); c.add_argument("--spec", required=True, help="spectrogram .npz of the same file")
    c.add_argument("--detections"); c.add_argument("--events", nargs="*"); c.add_argument("--top-events", type=int, default=0)
    c.add_argument("--harmonics", type=int, default=3); c.add_argument("--width", type=float, default=2.5)
    c.add_argument("--mode", choices=["fill", "notch"], default="fill")

    a = ap.parse_args(argv)
    if a.cmd == "detect":
        sp = Spectrogram.load(a.spec)
        model = D.load_model(a.model)
        dets = D.detect(model, sp, thr=a.thr, top=a.top, log=print)
        D.save_detections(dets, a.out)
        print(f"{'#':>2} {'t0':>7} {'t1':>7} {'f_lo':>6} {'f_hi':>6} {'harm':>4} {'prob':>5} {'t_cpa':>8} {'f0':>6} {'v':>5} {'d':>6}")
        for i, x in enumerate(dets):
            f = x.fit
            print(f"{i + 1:>2} {x.t_start:>7.0f} {x.t_end:>7.0f} {x.f_lo:>6.1f} {x.f_hi:>6.1f} {x.harmonics:>4} {x.prob:>5.2f} "
                  + (f"{f.t_cpa:>8.1f} {f.f0:>6.1f} {f.v:>5.0f} {f.d:>6.0f}" if f else ""))
    elif a.cmd in ("verify", "search"):
        sp = Spectrogram.load(a.spec); traces = load_traces(a.traces)
        events = _events_from(a, sp)
        harmonics = [1] * len(events)
        elev = None if a.no_dem else Terrain().elevation
        if a.cmd == "verify":
            res, segs = NS.verify(sp, events, traces, a.point[0], a.point[1], elevation=elev, half_m=a.half_m,
                                  spacing_m=a.spacing, harmonics=harmonics, raw_score=a.raw)
        else:
            res, segs = NS.search(sp, events, traces, tuple(a.bbox), elevation=elev, spacing_m=a.spacing,
                                  harmonics=harmonics, raw_score=a.raw, cpa_cache=a.cpa_cache,
                                  silence_d_m=None if a.no_silence else 6000.0)
        res["lat0"] = res.attrs.get("lat0", np.nan); res["lon0"] = res.attrs.get("lon0", np.nan)
        res.to_csv(a.out, index=False)
        print(res.head(20).to_string())
        if a.cmd == "search" and a.refine and len(res):
            fine = NS.refine_n(sp, events, segs, res.iloc[0], res.attrs["lat0"], res.attrs["lon0"], elevation=elev, harmonics=harmonics, raw_score=a.raw)
            fine.to_csv(a.out.replace(".csv", "_refined.csv"), index=False)
            print(fine.head(5).to_string())
    elif a.cmd == "scrub":
        import soundfile as sf
        sp = Spectrogram.load(a.spec)
        events = _events_from(a, sp)
        x, sr = load_audio(a.audio, sr=16000)
        tracks = [SC.track_from_event(sp, e, harmonics=a.harmonics) for e in events]
        y = SC.scrub(x, sr, tracks, half_width_hz=a.width, mode=a.mode)
        sf.write(a.out, y, sr)
        for e in events:
            b, af = SC.residual_contrast(x, y, sr, e)
            print(f"event t_cpa={e.t_cpa:.0f}s contrast before {b:.2f} dB -> after {af:.2f} dB")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
