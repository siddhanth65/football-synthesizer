"""True held-out ball recall of a detector on the hand-labelled PL probe frames.

Stage 1 of the PL fine-tune gate (``docs/PL_PIVOT_PLAN.md`` Phase A). The user hand-clicked 120 frames
(118 ball-visible) on the three PL probe seg_1 videos. This scores a detector against those clicks and
reports recall per match + pooled plus localisation error on hits.

Scoring reuses :func:`tools.validate_ball.evaluate_annotated` **unchanged** -- the exact function and
radius that produced v5's WC held-out recall (senegal 65 / norway 50 / iraq 80 / mun 86%). The hit
radius is ``tol_px`` measured in the fixed 512x288 model grid (validate_ball's documented ``--tol-px``
default of 8.0), so it is resolution-agnostic: labels are scaled into 512x288 via ``512/w, 288/h`` and
the peak is compared there. For the 1920x1080 PL footage 8 px @512x288 == 30 px native (isotropic, since
both frames are 16:9). Localisation error is reported in the 512x288 grid (as v5 was, "~1 px") and also
converted to native pixels.

Run (GPU job -- one at a time; ball net only):
    python -m tools.pl_ball_recall --weights outputs/ball_finetuned/tracknetv2_v5.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# (match label, annotation CSV, seg_1 video) -- the three hand-labelled PL probe segments.
PL_SEGS: tuple[tuple[str, str, str], ...] = (
    ("brighton", "data/ball_annotations/pl_probe/brighton_seg1.csv",
     "matches/pl_probe/brighton_manutd/seg_1.mp4"),
    ("fulham", "data/ball_annotations/pl_probe/fulham_seg1.csv",
     "matches/pl_probe/manutd_fulham/seg_1.mp4"),
    ("liverpool", "data/ball_annotations/pl_probe/liverpool_seg1.csv",
     "matches/pl_probe/manutd_liverpool/seg_1.mp4"),
)
NATIVE_W, MODEL_W = 1920.0, 512.0  # 512x288 grid -> native scale (isotropic for 16:9)
PX_512_TO_NATIVE = NATIVE_W / MODEL_W


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="outputs/ball_finetuned/tracknetv2_v5.pth")
    ap.add_argument("--base", default="tracknetv2")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--tol-px", type=float, default=8.0, dest="tol_px",
                    help="hit radius in the 512x288 grid (validate_ball default; reused for parity)")
    args = ap.parse_args()

    from tools.validate_ball import _load_model, evaluate_annotated  # noqa: PLC0415

    model, dev = _load_model(args.weights, args.base)
    print(f"loaded {args.base} on {dev}; weights={args.weights} thr={args.thr} "
          f"tol={args.tol_px:.0f}px@512x288 (== {args.tol_px * PX_512_TO_NATIVE:.0f}px native)\n")

    tot_vis = tot_hits = 0
    all_hit_dists: list[float] = []
    print(f"{'match':<12}{'visible':>9}{'hits':>7}{'recall':>9}"
          f"{'med_err':>10}{'mean_err':>10}   (512x288 px)")
    for label, ann, vid in PL_SEGS:
        if not (Path(ann).exists() and Path(vid).exists()):
            print(f"{label:<12}  MISSING annotation or video -> skip")
            continue
        r = evaluate_annotated(model, dev, ann, vid, thr=args.thr, tol_px=args.tol_px,
                               sample=None, include_frames=None)
        hit_dists = [d for d in r["dists"] if d <= args.tol_px]
        med = float(np.median(hit_dists)) if hit_dists else float("nan")
        mean = float(np.mean(hit_dists)) if hit_dists else float("nan")
        tot_vis += r["visible_frames"]
        tot_hits += r["hits"]
        all_hit_dists += hit_dists
        print(f"{label:<12}{r['visible_frames']:>9}{r['hits']:>7}{r['recall']:>8.1%}"
              f"{med:>10.2f}{mean:>10.2f}")

    pooled_recall = tot_hits / tot_vis if tot_vis else 0.0
    pmed = float(np.median(all_hit_dists)) if all_hit_dists else float("nan")
    pmean = float(np.mean(all_hit_dists)) if all_hit_dists else float("nan")
    print(f"{'POOLED':<12}{tot_vis:>9}{tot_hits:>7}{pooled_recall:>8.1%}"
          f"{pmed:>10.2f}{pmean:>10.2f}")
    print(f"\npooled localisation on hits: median {pmed:.2f}px / mean {pmean:.2f}px @512x288  "
          f"(== {pmed * PX_512_TO_NATIVE:.1f} / {pmean * PX_512_TO_NATIVE:.1f} px native)")
    print(f"POOLED RECALL = {pooled_recall:.1%}  (gate: >= 40% to proceed to fine-tune)")


if __name__ == "__main__":
    main()
