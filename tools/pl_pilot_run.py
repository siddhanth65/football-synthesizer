"""Turnkey, resumable driver for the Brighton-Man Utd full-match pilot (Phase-B first brick).

Runs the whole pipeline in order, each stage resumable so a session-kill loses at most the chunk in
flight. Stages (GPU stages never overlap -- one process, sequential):

1. EXTRACT (GPU): detection + tracking + PnLCalib per chunk -> per-half dense parquets.
2. ALIGN (CPU): jersey-colour team anchoring across both halves -> final/match_aligned.parquet.
3. BALL (GPU): v6 detector + carry-over projection + link_ball -> final/ball/ball_*.parquet.
4. FACTS (CPU): the grounded fact store -> outputs/facts/brighton_manutd.json.
5. GATE (CPU + one network fetch): FBref aggregate-agreement comparison -> results/pl_pilot/.

Re-running is safe: extraction/ball skip finished chunks; align/facts/gate recompute (cheap). Use
``--from`` to start at a later stage once earlier ones are done.

Run:
    python -m tools.pl_pilot_run                 # all stages from extraction
    python -m tools.pl_pilot_run --from ball     # resume at the ball stage
"""

from __future__ import annotations

import argparse
from pathlib import Path

MATCH_ID = "brighton_manutd"
OUT_ROOT = Path("outputs") / MATCH_ID
VIDEO_ROOT = Path("matches") / MATCH_ID
STAGES = ("extract", "align", "ball", "facts", "gate")


def stage_extract() -> None:
    """Per-half CV extraction into the WC dense layout (skip finished chunks)."""
    from tools.batch_match import run_batch  # noqa: PLC0415

    for half in ("h1", "h2"):
        chunks_dir = VIDEO_ROOT / half
        if not any(chunks_dir.glob("chunk_*.mp4")):
            continue
        print(f"[run] EXTRACT {half}", flush=True)
        run_batch(chunks_dir, OUT_ROOT / half / "match", sample_every=5, calib_period=25,
                  calib_drift=2.0, detector="football", tracker="bytetrack", skip_existing=True,
                  limit=None)


def stage_align() -> None:
    """Anchor teams + write match_aligned + diagnostics."""
    from tools.pl_pilot_align import main as align_main  # noqa: PLC0415

    print("[run] ALIGN", flush=True)
    align_main()


def stage_ball() -> None:
    """v6 + carry-over ball ingestion (skip finished chunks)."""
    from tools.pl_pilot_ball import run as ball_run  # noqa: PLC0415

    print("[run] BALL", flush=True)
    ball_run(None, weights="outputs/ball_finetuned/tracknetv2_v6.pth", thr=0.5, overwrite=False)


def stage_facts() -> None:
    """Grounded fact store for the pilot (degrades gracefully with no PMSR)."""
    from report.facts import write_facts  # noqa: PLC0415

    print("[run] FACTS", flush=True)
    print(f"[run] wrote {write_facts(MATCH_ID)}", flush=True)


def stage_gate() -> None:
    """FBref aggregate-agreement gate."""
    from tools.pl_pilot_gate import main as gate_main  # noqa: PLC0415

    print("[run] GATE", flush=True)
    gate_main()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="start", choices=STAGES, default="extract")
    args = ap.parse_args()
    runners = {"extract": stage_extract, "align": stage_align, "ball": stage_ball,
               "facts": stage_facts, "gate": stage_gate}
    for st in STAGES[STAGES.index(args.start):]:
        runners[st]()


if __name__ == "__main__":
    main()
