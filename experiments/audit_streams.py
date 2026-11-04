"""Audit the manifest against the resolved stream titles.

Rows whose notes show a multi-camera compilation or a music-bed stream are moved to
data/manifest_excluded.csv with the reason; first-session rows (no title recorded) inherit the
verdict of their *_d sibling, because the same search resolves to the same stream.

python experiments/audit_streams.py --manifest data/manifest.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BAD = ("top live cams", "live cams from", "compilation", "music", "relax", "rolling cam", "best of", "lofi", "radio", "camera feeds", "the world live", "earthtv", "4k |")


def verdict(notes: str) -> str:
    t = (notes or "").lower()
    for b in BAD:
        if b in t:
            return f"stream title indicates a compilation or music stream ('{b}')"
    return ""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", default="data/manifest.csv"); a = ap.parse_args()
    man = pd.read_csv(a.manifest, dtype={"notes": str}); man["notes"] = man["notes"].fillna("")
    reasons = {}
    for r in man.itertuples():
        v = verdict(r.notes)
        if v:
            reasons[r.clip_id] = v
    for r in man.itertuples():
        base = r.clip_id[:-2] if r.clip_id.endswith("_d") else r.clip_id
        sib = base + "_d"
        if r.clip_id not in reasons and sib in reasons and r.clip_id != sib:
            reasons[r.clip_id] = "same stream as " + sib + ": " + reasons[sib]
    excl = man[man.clip_id.isin(reasons)].copy(); excl["reason"] = [reasons[c] for c in excl.clip_id]
    keep = man[~man.clip_id.isin(reasons)]
    p = Path(a.manifest)
    prev = p.with_name("manifest_excluded.csv")
    if prev.exists():
        excl = pd.concat([pd.read_csv(prev), excl]).drop_duplicates("clip_id")
    excl.to_csv(prev, index=False); keep.to_csv(p, index=False)
    print(f"kept {len(keep)}, excluded {len(excl)}:"); print(excl[["clip_id", "reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
