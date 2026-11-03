"""Record benchmark clips from public 24/7 live webcams (YouTube) with known fixed positions.

Reads data/streams.csv (stream_id, url, lat, lon, region, place, split) and records `--minutes`
of audio from every stream *at the same time* (so one UTC day of ADS-B serves all clips), with
yt-dlp resolving the stream and ffmpeg capturing mono 16 kHz WAV. Streams that are offline or
have no audio track are skipped and logged. Writes/updates data/manifest.csv with the UTC start
time of each recording and the stream's coordinates.

python experiments/collect_clips.py --streams data/streams.csv --minutes 30 --out data --parallel 8
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


def resolve(url: str) -> dict | None:
    """Direct media URL + flags via yt-dlp -j (no download).

    `url` is either a watch/channel URL or `ytsearch:<words>`; a search is expanded to the first
    ten results and the first one that is live with an audio track is used (stream IDs of 24/7
    cams change when the operator restarts them, so a search is more durable than a video ID)."""
    target = f"ytsearch10:{url[9:]}" if url.startswith("ytsearch:") else url
    try:
        out = subprocess.run([sys.executable, "-m", "yt_dlp", "-j", "--no-playlist", "--ignore-errors",
                              "-f", "bestaudio/best", target], capture_output=True, text=True, timeout=240)
        for line in out.stdout.splitlines():
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if j.get("is_live") and j.get("url") and j.get("acodec") not in (None, "none"):
                return {"media": j["url"], "acodec": j.get("acodec"), "title": j.get("title"), "id": j.get("id")}
        return None
    except Exception:
        return None


def record(stream_id: str, media_url: str, seconds: int, out_wav: Path) -> tuple[bool, str]:
    t0 = datetime.now(timezone.utc)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", media_url, "-t", str(seconds), "-vn", "-ac", "1", "-ar", "16000",
           "-acodec", "pcm_s16le", str(out_wav)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 300)
    ok = r.returncode == 0 and out_wav.exists() and out_wav.stat().st_size > 16000 * 2 * seconds * 0.8
    return ok, t0.isoformat(timespec="seconds").replace("+00:00", "+00:00")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default="data/streams.csv"); ap.add_argument("--out", default="data")
    ap.add_argument("--minutes", type=int, default=30); ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--manifest", default="data/manifest.csv")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    streams = list(csv.DictReader(open(a.streams, newline="", encoding="utf-8")))
    print(f"{len(streams)} streams; resolving ...")
    with ThreadPoolExecutor(a.parallel) as ex:
        resolved = list(ex.map(lambda s: resolve(s["url"]), streams))
    live = [(s, r) for s, r in zip(streams, resolved) if r and r["media"] and r["acodec"] not in (None, "none")]
    print(f"{len(live)} live streams with audio")
    for s, r in zip(streams, resolved):
        if not (r and r["media"] and r["acodec"] not in (None, "none")):
            print(f"  skip {s['stream_id']}: offline or no audio")
    secs = a.minutes * 60

    def job(item):
        s, r = item
        wav = out / f"{s['stream_id']}.wav"
        ok, t0 = record(s["stream_id"], r["media"], secs, wav)
        return s, ok, t0

    rows = []
    with ThreadPoolExecutor(a.parallel) as ex:
        for s, ok, t0 in ex.map(job, live):
            print(f"  {s['stream_id']}: {'ok' if ok else 'FAILED'} start {t0}")
            if ok:
                rows.append({"clip_id": s["stream_id"], "path": str(out / f"{s['stream_id']}.wav"), "lat": s["lat"], "lon": s["lon"],
                             "utc_start": t0, "date": t0[:10], "region": s["region"], "split": s["split"], "source": "webcam",
                             "notes": f"{s.get('place', '')}; {s['url']}"})
    man = Path(a.manifest)
    existing = list(csv.DictReader(open(man, newline="", encoding="utf-8"))) if man.exists() else []
    keep = [r for r in existing if r["clip_id"] not in {x["clip_id"] for x in rows}]
    cols = ["clip_id", "path", "lat", "lon", "utc_start", "date", "region", "split", "source", "notes"]
    with open(man, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in keep + rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"manifest: {len(keep) + len(rows)} rows -> {man}")


if __name__ == "__main__":
    main()
