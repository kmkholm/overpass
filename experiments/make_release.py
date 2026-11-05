"""Assemble release/ (the data deposited with the code on Zenodo): manifests with relative paths, the propeller-aircraft
ADS-B extracts per clip (ODbL, from adsb.lol), detections, the detector weights and every results table behind the
paper. No audio is included: the webcam and reference recordings belong to their stream operators.

python experiments/make_release.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "release"


def copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)


def main():
    if REL.exists():  # rebuild everything except the hand-written README
        for p in REL.iterdir():
            if p.name != "README.md":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    # manifests: audio paths reduced to file names
    for name in ("manifest.csv", "manifest_excluded.csv", "manifest_livermore.csv"):
        m = pd.read_csv(ROOT / "data" / name)
        if "path" in m.columns:
            m["path"] = m["path"].astype(str).map(lambda p: Path(p.replace("\\", "/")).name)
        (REL / "manifests").mkdir(parents=True, exist_ok=True); m.to_csv(REL / "manifests" / name, index=False)
    for name in ("streams.csv", "streams_us_day.csv", "streams_eu_day.csv", "streams_me.csv", "coverage_points.csv"):
        if (ROOT / "data" / name).exists():
            copy(ROOT / "data" / name, REL / "manifests" / name)
    man = pd.read_csv(ROOT / "data" / "manifest.csv")
    n_tr = 0
    for c in man.clip_id:
        for kind, sub in (("traces.npz", "adsb_extracts/webcam"), ("det.json", "detections"), ("scan.json", "detections")):
            p = ROOT / "work" / f"{c}.{kind}"
            if p.exists():
                copy(p, REL / sub / p.name); n_tr += kind == "traces.npz"
    # Livermore: every clip of a day shares one 60 km box around the site, so one extract per day
    lv = pd.read_csv(ROOT / "data" / "manifest_livermore.csv")
    for d, g in lv.groupby("date"):
        copy(ROOT / "work_livermore" / f"{g.clip_id.iloc[0]}.traces.npz", REL / "adsb_extracts" / "livermore" / f"livermore_{d}.traces.npz")
    for p in (ROOT / "models").glob("ridgenet*"):
        copy(p, REL / "model" / p.name)
    for p in sorted((ROOT / "results").glob("*")):
        if p.suffix in (".csv", ".json") and "backup" not in p.name:
            copy(p, REL / "results" / p.name)
    for sub in ("ref", "known_point"):
        for p in (ROOT / "results" / sub).rglob("*"):
            if p.is_file() and p.suffix in (".csv", ".json"):
                copy(p, REL / "results" / sub / p.relative_to(ROOT / "results" / sub))
    assert not any(p.suffix.lower() in (".wav", ".flac", ".webm", ".mp3", ".mp4") for p in REL.rglob("*")), "audio leaked into release/"
    tot = sum(p.stat().st_size for p in REL.rglob("*") if p.is_file())
    print(f"release/: {sum(1 for p in REL.rglob('*') if p.is_file())} files, {tot / 1e6:.0f} MB; webcam extracts {n_tr}, livermore days {lv.date.nunique()}")
    big = [(p, p.stat().st_size) for p in REL.rglob("*") if p.is_file() and p.stat().st_size > 50e6]
    print("files > 50 MB:", [(str(p.relative_to(REL)), round(s / 1e6)) for p, s in big])


if __name__ == "__main__":
    main()
