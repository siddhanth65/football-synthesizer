"""Split a full-match mp4 into ~10-minute ``chunk_NNN.mp4`` files (the pipeline's expected input).

Uses ffmpeg stream-copy segmentation (no re-encode -> fast, lossless; cuts land on the nearest keyframe,
which is fine since chunks are analysed independently). Output names match the convention the rest of the
pipeline expects: ``chunk_000.mp4, chunk_001.mp4, ...``.

    python tools/chunk_video.py --video "France vs Senegal ....mp4" --out-dir matches/fra_sen/chunks
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def find_ffmpeg(explicit: str | None) -> str:
    """Locate ffmpeg: the explicit path, then PATH, then the user's Desktop build."""
    if explicit:
        return explicit
    onpath = shutil.which("ffmpeg")
    if onpath:
        return onpath
    for p in Path.home().glob("OneDrive/Desktop/ffmpeg-*/bin/ffmpeg.exe"):
        return str(p)
    raise FileNotFoundError("ffmpeg not found; pass --ffmpeg <path/to/ffmpeg>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, help="full-match mp4")
    ap.add_argument("--out-dir", required=True, help="directory for chunk_*.mp4")
    ap.add_argument("--seconds", type=int, default=600, help="chunk length (default 600 = 10 min)")
    ap.add_argument("--ffmpeg", default=None, help="path to ffmpeg (else PATH / Desktop build)")
    args = ap.parse_args()

    ffmpeg = find_ffmpeg(args.ffmpeg)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pattern = str(out / "chunk_%03d.mp4")
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-i", args.video,
           "-c", "copy", "-map", "0", "-segment_time", str(args.seconds),
           "-f", "segment", "-reset_timestamps", "1", pattern]
    print("ffmpeg:", ffmpeg)
    print("splitting", args.video, "->", pattern, f"({args.seconds}s chunks)")
    subprocess.run(cmd, check=True)
    chunks = sorted(out.glob("chunk_*.mp4"))
    print(f"\nwrote {len(chunks)} chunks to {out}:")
    for c in chunks:
        print(f"  {c.name}  ({c.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
