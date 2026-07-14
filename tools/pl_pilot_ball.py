"""Ball ingestion for the Brighton-Man Utd pilot: v6 detector + temporal carry-over -> linked track.

Per chunk (resumable): detect the v6 fine-tuned ball on EVERY dense frame at 512x288, project to pitch
metres via per-frame player-correspondence homography WITH the camera-continuous carry-over lever ON
(:func:`generator.ball_carry.project_ball_carry`, +/-2 s window, cut-safe), then
:func:`generator.ball.link_ball` at the chunk's true fps. Writes
``outputs/brighton_manutd/final/ball/ball_<half>_chunk<NNN>.parquet`` (schema ``frame, x, y,
observed``) so the registry's ``ball_chunks()`` finds them.

The ONLY coverage number that counts is post-``link_ball`` usable-track / dense-frames. Each chunk
also reports the carry-over contribution (OFF vs ON post-link) so the lever's effect is auditable.

Resumable: a chunk whose output parquet already exists is skipped unless ``--overwrite``. Progress is
appended to ``results/pl_pilot/ball_progress.log``.

Run (GPU job -- one at a time; never with the extraction batch or the full pytest suite):
    python -m tools.pl_pilot_ball                 # all chunks (detect + project + link)
    python -m tools.pl_pilot_ball --half h2       # one half
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from generator.ball import link_ball
from generator.ball_carry import project_ball_carry

MATCH_ID = "brighton_manutd"
OUT_ROOT = Path("outputs") / MATCH_ID
VIDEO_ROOT = Path("matches") / MATCH_ID
BALL_DIR = OUT_ROOT / "final" / "ball"
LOG = Path("results/pl_pilot/ball_progress.log")
V6_WEIGHTS = "outputs/ball_finetuned/tracknetv2_v6.pth"
HALVES = ("h1", "h2")
PLAYER_ROLES = ("player", "goalkeeper")
_CHUNK_RE = re.compile(r"chunk_(\d+)_dense\.parquet$")


@dataclass(frozen=True)
class ChunkJob:
    """One ball-ingestion unit."""

    half: str
    num: str
    video: Path
    dense: Path
    out: Path


def plan_jobs(half: str | None) -> list[ChunkJob]:
    """Every chunk with a dense parquet + existing video, across the requested half(s)."""
    jobs: list[ChunkJob] = []
    for h in (HALVES if half is None else (half,)):
        for dense in sorted((OUT_ROOT / h / "match").glob("chunk_*_dense.parquet")):
            num = _CHUNK_RE.search(dense.name).group(1)
            video = VIDEO_ROOT / h / f"chunk_{num}.mp4"
            if video.exists():
                jobs.append(ChunkJob(h, num, video, dense,
                                     BALL_DIR / f"ball_{h}_chunk{num}.parquet"))
    return jobs


def _true_fps(video: Path) -> float:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps if fps and fps > 1.0 else 25.0


def _log(msg: str) -> None:
    print(msg, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def run(half: str | None, *, weights: str, thr: float, overwrite: bool) -> None:
    """Detect + carry-project + link each chunk; write ball parquets and per-chunk coverage."""
    from tools.ball_possession import _load_model, detect_ball_imagexy  # noqa: PLC0415

    BALL_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in plan_jobs(half) if overwrite or not j.out.exists()]
    if not jobs:
        _log("[ball] nothing to do (all outputs present; use --overwrite to force)")
        return
    model, dev = _load_model(weights)
    _log(f"[ball] loaded v6 on {dev}; weights={weights} thr={thr}; {len(jobs)} chunks")

    summary: list[dict] = []
    for j in jobs:
        dense = pd.read_parquet(j.dense)
        n_dense = int(dense["frame"].nunique())
        frames = sorted(int(f) for f in dense["frame"].unique())
        fps = _true_fps(j.video)
        ball_img = detect_ball_imagexy(str(j.video), frames, model, dev, thr=thr)

        off_df, _ = project_ball_carry(ball_img, dense, carry=False)
        on_df, st = project_ball_carry(ball_img, dense, carry=True, interpolate=True)
        off_track = link_ball(off_df[["frame", "x", "y"]], fps=fps)
        on_track = link_ball(on_df[["frame", "x", "y"]], fps=fps)
        on_track.to_parquet(j.out, index=False)

        off_cov = len(off_track) / n_dense if n_dense else 0.0
        on_cov = len(on_track) / n_dense if n_dense else 0.0
        obs = int(on_track["observed"].sum()) if len(on_track) else 0
        row = {"half": j.half, "num": j.num, "n_dense": n_dense, "fired": len(ball_img),
               "own": st.own, "carried": st.carried, "interp": st.interpolated,
               "off_cov": off_cov, "on_cov": on_cov, "observed": obs}
        summary.append(row)
        _log(f"[ball] {j.half}_chunk{j.num}: dense={n_dense} fired={len(ball_img)} "
             f"own={st.own} carry={st.carried} interp={st.interpolated} "
             f"OFF={off_cov:.1%} -> ON={on_cov:.1%} (+{(on_cov - off_cov) * 100:.1f}pp) "
             f"obs={obs} -> {j.out.name}")

    if summary:
        off_m = sum(r["off_cov"] for r in summary) / len(summary)
        on_m = sum(r["on_cov"] for r in summary) / len(summary)
        _log(f"[ball] MEAN post-link coverage: OFF {off_m:.1%} -> ON {on_m:.1%} "
             f"(carry-over +{(on_m - off_m) * 100:.1f}pp) over {len(summary)} chunks")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--half", choices=HALVES, default=None)
    ap.add_argument("--weights", default=V6_WEIGHTS)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run(args.half, weights=args.weights, thr=args.thr, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
