"""Regenerate the linked-ball parquets for every France match from the v5 detector.

Registry-driven end-to-end ball regeneration. For each rostered France match in
:mod:`core.registry` we walk that match's processed chunks (the per-chunk dense position
parquets under ``outputs/<match>/<half>/match/chunk_<NNN>_dense.parquet``) and, for each chunk,
run the *exact* validated probe path -- v5 fine-tuned detector at 512x288 -> per-frame
player-correspondence homography projection -> :func:`generator.ball.link_ball` at the video's
TRUE fps -- then write ``outputs/<match>/final/ball/ball_<half>_chunk<NNN>.parquet`` (schema
``frame, x, y, observed``). Each pre-existing parquet is backed up to ``<name>.v4bak.parquet``
before overwrite.

The chunk set is the dense-parquet set (the real processed-chunk universe), not the pre-existing
ball parquets: several ball directories are incomplete (e.g. norway h1 chunks 000-003 have dense
positions + video but no ball parquet), and driving off the incomplete ball dir would leave those
chunks permanently unlinked. Match identity, ball_dir and aligned still come from the registry.

Run (this is a GPU job -- one at a time):
    python -m tools.regen_ball
    python -m tools.regen_ball --match france_norway          # single match
    python -m tools.regen_ball --dry-run                      # list the work, no GPU
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.registry import Match, matches
from generator.ball import link_ball
from tools.ball_possession import detect_ball_imagexy, project_ball

VIDEO_ROOT = Path("matches")
HALVES = ("h1", "h2")
PLAYER_ROLES = ("player", "goalkeeper")
_CHUNK_RE = re.compile(r"chunk_(\d+)_dense\.parquet$")
DEFAULT_WEIGHTS = "outputs/ball_finetuned/tracknetv2_v5.pth"


@dataclass(frozen=True)
class ChunkJob:
    """One regeneration unit: a chunk's video + dense positions + output ball parquet."""

    match_id: str
    half: str
    num: str  # zero-padded chunk number, e.g. "003"
    video: Path
    dense: Path
    out: Path

    @property
    def key(self) -> str:
        """Chunk key matching ``aligned['chunk']`` (``h1_chunk_003``)."""
        return f"{self.half}_chunk_{self.num}"


_DONE_MATCH_RE = re.compile(r"^=== (\S+):")
_DONE_CHUNK_RE = re.compile(r"^\s+(h\d_chunk_\d+)\s")


def _match_root(m: Match) -> Path:
    """``outputs/<match>`` derived from the registry ball_dir (``.../final/ball``)."""
    return m.ball_dir.parents[1]


def _completed_from_log(log: Path) -> set[tuple[str, str]]:
    """Parse a prior run's stdout log into the ``(match_id, chunk_key)`` pairs already regenerated.

    Lets a killed run resume without redoing finished chunks (the v5 output + v4 backup already
    exist for those). The log format is this tool's own per-chunk output.
    """
    done: set[tuple[str, str]] = set()
    current = None
    for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        mo = _DONE_MATCH_RE.match(line)
        if mo:
            current = mo.group(1)
            continue
        ch = _DONE_CHUNK_RE.match(line)
        if ch and current:
            done.add((current, ch.group(1)))
    return done


def plan_jobs(m: Match) -> list[ChunkJob]:
    """Every regeneratable chunk for one match: a dense parquet whose chunk video exists."""
    root = _match_root(m)
    jobs: list[ChunkJob] = []
    for half in HALVES:
        dense_dir = root / half / "match"
        if not dense_dir.exists():
            continue
        for dense in sorted(dense_dir.glob("chunk_*_dense.parquet")):
            mo = _CHUNK_RE.search(dense.name)
            if not mo:
                continue
            num = mo.group(1)
            video = VIDEO_ROOT / m.id / half / f"chunk_{num}.mp4"
            if not video.exists():
                continue
            out = m.ball_dir / f"ball_{half}_chunk{num}.parquet"
            jobs.append(ChunkJob(m.id, half, num, video, dense, out))
    return jobs


def _true_fps(video: Path) -> float:
    """Read the container FPS straight from the video (per-chunk; broadcasts vary)."""
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps if fps and fps > 1.0 else 25.0


def _coverage(n_rows: int, n_dense: int) -> float:
    """Post-link usable-track coverage: linked samples / distinct dense frames (fraction)."""
    return n_rows / n_dense if n_dense else 0.0


def regen_chunk(job: ChunkJob, model, dev, *, thr: float, backup: bool) -> dict:
    """Detect -> project -> link one chunk and (re)write its ball parquet.

    Returns a summary dict with the old and new post-link coverage so progress is visible per chunk.
    """
    dense = pd.read_parquet(job.dense)
    players = dense[dense["role"].isin(PLAYER_ROLES)].dropna(
        subset=["image_x", "image_y", "pitch_x", "pitch_y"]
    )
    n_dense = int(dense["frame"].nunique())
    frames = sorted(int(f) for f in players["frame"].unique())

    old_rows = int(len(pd.read_parquet(job.out))) if job.out.exists() else 0

    if not frames:
        track = pd.DataFrame(columns=["frame", "x", "y", "observed"])
    else:
        fps = _true_fps(job.video)
        ball_img = detect_ball_imagexy(str(job.video), frames, model, dev, thr=thr)
        ball_pitch = project_ball(ball_img, players)
        track = link_ball(ball_pitch, fps=fps)

    if backup and job.out.exists():
        bak = job.out.with_suffix(".v4bak.parquet")
        if not bak.exists():  # never clobber the original v4 backup on a re-run
            bak.write_bytes(job.out.read_bytes())

    job.out.parent.mkdir(parents=True, exist_ok=True)
    track.to_parquet(job.out, index=False)

    new_rows = int(len(track))
    obs = int(track["observed"].sum()) if new_rows else 0
    return {
        "key": job.key, "n_dense": n_dense, "old_rows": old_rows, "new_rows": new_rows,
        "observed": obs, "old_cov": _coverage(old_rows, n_dense),
        "new_cov": _coverage(new_rows, n_dense),
    }


def main() -> None:
    """Regenerate ball parquets for the rostered France matches and print per-chunk coverage."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--base", default="tracknetv2")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--match", default=None, help="restrict to one match id")
    ap.add_argument("--resume-from", default=None,
                    help="prior run's log; skip chunks it already regenerated")
    ap.add_argument("--no-backup", action="store_true", help="skip the .v4bak.parquet backup")
    ap.add_argument("--dry-run", action="store_true", help="list the chunk jobs, load no model")
    args = ap.parse_args()

    france = [m for m in matches() if m.roster and m.ball_dir is not None]
    if args.match:
        france = [m for m in france if m.id == args.match]
    if not france:
        print("no rostered France matches with a ball_dir in the registry")
        return

    if args.dry_run:
        for m in france:
            jobs = plan_jobs(m)
            print(f"{m.id}: {len(jobs)} chunks")
            for j in jobs:
                exists = "exists" if j.out.exists() else "MISSING"
                print(f"  {j.key:14s} out={j.out.name} ({exists})")
        return

    done = _completed_from_log(Path(args.resume_from)) if args.resume_from else set()
    if done:
        print(f"resume: skipping {len(done)} chunks already in {args.resume_from}", flush=True)

    from tools.ball_possession import _load_model  # noqa: PLC0415

    model, dev = _load_model(args.weights, args.base)
    print(f"loaded {args.base} v5 on {dev}; weights={args.weights} thr={args.thr}", flush=True)

    grand: list[dict] = []
    for m in france:
        jobs = [j for j in plan_jobs(m) if (m.id, j.key) not in done]
        print(f"\n=== {m.id}: {len(jobs)} chunks ===", flush=True)
        per: list[dict] = []
        for j in jobs:
            r = regen_chunk(j, model, dev, thr=args.thr, backup=not args.no_backup)
            per.append(r)
            grand.append({**r, "match": m.id})
            print(
                f"  {r['key']:14s} dense={r['n_dense']:4d}  "
                f"old {r['old_rows']:4d} ({r['old_cov']:5.1%}) -> "
                f"new {r['new_rows']:4d} ({r['new_cov']:5.1%})  obs={r['observed']}",
                flush=True,
            )
        if per:
            old_mean = sum(x["old_cov"] for x in per) / len(per)
            new_mean = sum(x["new_cov"] for x in per) / len(per)
            print(f"  -- {m.id} mean coverage: {old_mean:.1%} -> {new_mean:.1%}", flush=True)

    print("\n=== overall per-match mean post-link coverage (old -> new) ===", flush=True)
    for m in france:
        rows = [x for x in grand if x["match"] == m.id]
        if not rows:
            continue
        om = sum(x["old_cov"] for x in rows) / len(rows)
        nm = sum(x["new_cov"] for x in rows) / len(rows)
        print(f"  {m.id:16s} chunks={len(rows):2d}  {om:.1%} -> {nm:.1%}", flush=True)


if __name__ == "__main__":
    main()
