"""Generic align + ball driver for one match: kit-anchor both halves, then run TrackNet v6 + link_ball.

Promotes the ephemeral scratchpad ``run_align_ball_generic.py`` (deleted mid-run during the reverse-
fixtures chain) to a durable, registry-driven script. It takes a match id, resolves every path via
:mod:`core.registry` (``data/matches.yaml`` -- NEVER hardcodes ``outputs/`` paths), and reuses the
pilot stages unchanged (:mod:`tools.pl_pilot_align`, :mod:`tools.pl_pilot_ball`) by pointing their
module-level path constants at the requested match, exactly as the temp driver did.

Two stages, both RESUMABLE BY DISK STATE:
  * align -- skipped when the match's aligned parquet already exists; else anchors teams across BOTH
    halves together and writes ``<final>/match_aligned.parquet`` (+ per-half tables).
  * ball  -- v6 detect + carry-project + ``link_ball`` per chunk; ``pl_pilot_ball`` already skips any
    chunk whose output parquet exists, so a re-run only fills the gaps.

Run:
    python -m tools.run_align_ball <match_id>            # e.g. liverpool_manutd
    python -m tools.run_align_ball <match_id> --force-align   # rebuild the aligned parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import registry  # noqa: E402
from tools import pl_pilot_align as align  # noqa: E402
from tools import pl_pilot_ball as ball  # noqa: E402


def _point_modules_at(match_id: str) -> tuple[Path, Path]:
    """Repoint the pilot modules' path constants at ``match_id`` (all derived from the registry).

    Args:
        match_id: registered match id (key in ``data/matches.yaml``).

    Returns:
        ``(aligned_parquet, ball_progress_log)`` for this match.
    """
    m = registry.get(match_id)
    if m.ball_dir is None:
        raise SystemExit(f"{match_id}: no ball_dir in registry -- cannot run the ball stage")
    out_root = m.aligned.parent.parent  # outputs/<id>  (aligned = outputs/<id>/final/match_aligned...)
    video_root = registry.VIDEO_ROOT / match_id  # matches/<id>
    log = Path("results/pl_pilot") / f"ball_progress_{match_id}.log"

    align.MATCH_ID = ball.MATCH_ID = match_id
    align.OUT_ROOT = ball.OUT_ROOT = out_root
    align.VIDEO_ROOT = ball.VIDEO_ROOT = video_root
    ball.BALL_DIR = m.ball_dir
    ball.LOG = log
    return m.aligned, log


def main() -> None:
    """Run align (once) then ball (both halves) for the requested match, resumable by disk state."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("match_id", help="registered match id (key in data/matches.yaml)")
    ap.add_argument("--force-align", action="store_true", help="rebuild the aligned parquet even if present")
    ap.add_argument("--thr", type=float, default=0.5, help="ball detector confidence threshold")
    args = ap.parse_args()

    aligned, _ = _point_modules_at(args.match_id)

    print("align: start", flush=True)
    if aligned.exists() and not args.force_align:
        print(f"align: skip (exists) {aligned}", flush=True)
    else:
        align.main()
    print("align: done", flush=True)

    print("ball: start", flush=True)
    ball.run(None, weights=ball.V6_WEIGHTS, thr=args.thr, overwrite=False)
    print("ball: done", flush=True)


if __name__ == "__main__":
    main()
