"""Benchmark manifest and per-clip preparation (spectrogram, ADS-B region, DEM).

manifest.csv columns:
  clip_id, path, lat, lon, utc_start (ISO-8601 or ''), date (YYYY-MM-DD UTC), region (US|EU|JO),
  split (calib|test), source (webcam|own|reference), notes
Only `date`, `lat`, `lon` and the audio file are needed by the pipeline; utc_start is used for
the single-event ablation and for reporting. Clips must contain no intelligible speech.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from dopplertrack import pipeline
from dopplertrack.adsb import load_traces
from dopplertrack.audio import Spectrogram

REGION_BOX_KM = 150.0


def load_manifest(path) -> pd.DataFrame:
    m = pd.read_csv(path, dtype={"clip_id": str, "utc_start": str, "date": str})
    m["utc_start"] = m["utc_start"].fillna("")
    return m


def bbox_around(lat: float, lon: float, side_km: float = REGION_BOX_KM):
    dlat = side_km / 2 / 111.32
    dlon = side_km / 2 / (111.32 * np.cos(np.radians(lat)))
    return (lat - dlat, lat + dlat, lon - dlon, lon + dlon)


def utc_to_unix(s: str) -> float | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()


def prepare_clip(row, work: Path, archive_root: Path | None = None, fetch: bool = True, log=print):
    """Spectrogram .npz and prop-aircraft traces .npz for one clip (cached)."""
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    sp_path = work / f"{row.clip_id}.spec.npz"; tr_path = work / f"{row.clip_id}.traces.npz"
    if not sp_path.exists():
        log(f"[{row.clip_id}] spectrogram")
        pipeline.make_spectrogram(row.path, sp_path)
    if not tr_path.exists():
        bbox = bbox_around(float(row.lat), float(row.lon))
        if archive_root and (Path(archive_root) / row.date).exists():
            pipeline.extract_region(Path(archive_root) / row.date, bbox, tr_path)
        elif fetch:
            pipeline.fetch_region(row.date, bbox, tr_path)
        else:
            raise FileNotFoundError(f"no ADS-B archive for {row.date} and fetch=False")
    return Spectrogram.load(sp_path), load_traces(tr_path), sp_path, tr_path


def backgrounds_from_clips(manifest: pd.DataFrame, work: Path, n_per_clip: int = 6, size=(256, 480), seed: int = 0):
    """Real quiet background crops (whitened dB) from calibration clips for detector training.
    'Quiet' = windows whose max whitened brightness is modest (no obvious tonal ridges)."""
    rng = np.random.default_rng(seed); out = []
    for row in manifest[manifest.split == "calib"].itertuples():
        p = Path(work) / f"{row.clip_id}.spec.npz"
        if not p.exists():
            continue
        sp = Spectrogram.load(p)
        F, T = sp.S.shape
        tries = 0
        while len([o for o in out if o is not None]) < n_per_clip * (1 + len(out) // max(n_per_clip, 1)) and tries < 40:
            tries += 1
            r0 = int(rng.integers(0, max(F - size[0], 1))); c0 = int(rng.integers(0, max(T - size[1], 1)))
            crop = sp.S[r0:r0 + size[0], c0:c0 + size[1]]
            if crop.shape == size and np.percentile(crop, 99.5) < 8.0:
                out.append(crop.astype(np.float32))
                if len(out) % n_per_clip == 0:
                    break
    return out


def reference_manifest_row() -> dict:
    """The public reference clip (colsto, Oyster Bay NY). Audio must be fetched by the user."""
    return {"clip_id": "ref_oysterbay", "path": "data/ref_oysterbay.wav", "lat": 40.876466, "lon": -73.524951,
            "utc_start": "2026-09-08T20:17:30+00:00", "date": "2026-09-08", "region": "US", "split": "calib",
            "source": "reference", "notes": "youtu.be/4CxVxuMVkMc; answer given in the video"}
