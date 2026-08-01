"""Per-crop jersey OCR over a real broadcast match (the GSR densification path, on our own footage).

The shipped real-match identity chain reads numbers only from **gated close-up shots**
(``tools/wire_anchors.py`` -> ``outputs/identity/<match>_named_tracks_both2_prtreid.parquet``),
which names 166-282 of ~8,000 tracklets. This tool runs the same per-track crop OCR the GSR
benchmark uses (:mod:`eval.gsr_jersey`) over every tracklet of a match, persists **every crop's**
read, and re-aggregates with the densifying rule frozen in ``results/ocr_density_rule.json``.

Crops come from the chunk videos via :func:`generator.team_anchor.estimate_player_box` -- the same
foot-point box the identity chain uses, so no re-detection is needed (the pattern of
:mod:`tools.gta_match`). Paths come from :mod:`core.registry`.

CLI::

    python -m tools.ocr_match --match manutd_liverpool --max-crops 20   # GPU, resumable per chunk
    python -m tools.ocr_match --match manutd_liverpool --report          # CPU, d before/after
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from core import registry
from eval.gsr_jersey import percrop_frame
from generator.jersey_id import OCR_PERCROP_VERSION
from generator.track_relink import _sample_frames
from tools.gta_match import MIN_BOX_H, NAMED_TRACKS, video_for
from tools.ocr_density import RULE_PATH, reads_for_sequence

logger = logging.getLogger("ocr_match")


def percrop_dir(match_id: str, variant: str = "") -> Path:
    """Per-crop OCR cache for a match, derived from the registry (never hardcoded).

    Args:
        match_id: Registry match id.
        variant: Suffix on the directory name, so a re-run at a different crop geometry lands
            beside the shipped cache instead of overwriting it (e.g. ``"_w125"``).
    """
    return Path(registry.get(match_id).aligned).parent / f"ocr_percrop{variant}"


def scale_box(box: tuple[int, int, int, int], scale: float, frame_h: int, frame_w: int
              ) -> tuple[int, int, int, int]:
    """Scale a player box about its foot point (bottom centre), clipped to the frame.

    :func:`generator.team_anchor.estimate_player_box` returns 0.814x the annotated player height on
    FOOTPASS game_18 (``results/OCR_DOMAIN_SHIFT.md`` §7), and the OCR legibility gate is very
    sensitive to that. This widens the **OCR crop only** -- nothing geometric consumes it.

    Args:
        box: ``(x1, y1, x2, y2)`` from :func:`generator.team_anchor.estimate_player_box`.
        scale: Multiplier on both height and width; ``1.0`` is a no-op.
        frame_h: Frame height, for clipping.
        frame_w: Frame width, for clipping.

    Returns:
        The scaled ``(x1, y1, x2, y2)``; a small margin below the foot point is kept so a
        mis-placed foot point does not crop the boots away.
    """
    x1, y1, x2, y2 = box
    h, w, cx = y2 - y1, x2 - x1, (x1 + x2) / 2
    return (int(max(cx - scale * w / 2, 0)), int(max(y2 - scale * h, 0)),
            int(min(cx + scale * w / 2, frame_w)), int(min(y2 + 0.05 * h, frame_h)))


def write_chunk_crops(video: Path, chunk_df: pd.DataFrame, out_dir: Path, *, max_crops: int,
                      crop_scale: float = 1.0) -> list[Path]:
    """Cut up to ``max_crops`` foot-point boxes per track from one chunk video (IO).

    Args:
        video: The chunk's source video.
        chunk_df: Person rows of this chunk (``frame, track_id, image_x, image_y``).
        out_dir: Destination directory for ``t<tid>_<frame>.jpg`` crops (created).
        max_crops: Crops sampled per track, evenly spread over the track's frames.
        crop_scale: Widening applied to the estimated box about the foot point (:func:`scale_box`).
            ``1.0`` reproduces the shipped geometry exactly.

    Returns:
        The written crop paths, ascending by frame.
    """
    import cv2  # noqa: PLC0415

    from generator.team_anchor import estimate_player_box  # noqa: PLC0415

    need: dict[int, list[tuple[int, float, float]]] = {}
    for tid, grp in chunk_df.groupby("track_id"):
        grp = grp.sort_values("frame")
        keep = set(_sample_frames(grp["frame"].astype(int).tolist(), max_crops))
        for r in grp.itertuples():
            if int(r.frame) in keep:
                need.setdefault(int(r.frame), []).append(
                    (int(tid), float(r.image_x), float(r.image_y)))

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    paths: list[Path] = []
    for frame_idx in sorted(need):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, bgr = cap.read()
        if not ok:
            continue
        for tid, ix, iy in need[frame_idx]:
            box = estimate_player_box(ix, iy, fh, fw)
            x1, y1, x2, y2 = box if crop_scale == 1.0 else scale_box(box, crop_scale, fh, fw)
            if y2 - y1 < MIN_BOX_H or x2 - x1 < 8:  # noqa: PLR2004
                continue
            crop = bgr[y1:y2, x1:x2]
            if crop.size:
                p = out_dir / f"t{tid}_{frame_idx:06d}.jpg"
                cv2.imwrite(str(p), crop)
                paths.append(p)
    cap.release()
    return paths


def run_match(match_id: str, *, max_crops: int, crop_scale: float = 1.0, variant: str = "") -> None:
    """GPU stage: one per-crop OCR parquet per chunk (resumable by disk state).

    Args:
        match_id: Registry match id.
        max_crops: Crops per track.
        crop_scale: Crop widening about the foot point; ``1.0`` is the shipped geometry.
        variant: Suffix on the output directory (see :func:`percrop_dir`).
    """
    from eval.gsr_jersey import _build_recognizer  # noqa: PLC0415

    df = pd.read_parquet(registry.get(match_id).aligned)
    people = df[df["role"].isin(["player", "goalkeeper"])
                & np.isfinite(df["image_x"]) & np.isfinite(df["image_y"])]
    dest_dir = percrop_dir(match_id, variant)
    dest_dir.mkdir(parents=True, exist_ok=True)
    chunks = sorted(people["chunk"].unique())
    for i, chunk in enumerate(chunks):
        dest = dest_dir / f"{chunk}.parquet"
        if dest.exists():
            continue
        video = video_for(match_id, chunk)
        if not video.exists():
            logger.warning("missing video %s -- skipping %s", video, chunk)
            continue
        t0 = time.time()
        work = dest_dir / "_crops" / chunk
        shutil.rmtree(work, ignore_errors=True)
        paths = write_chunk_crops(video, people[people["chunk"] == chunk], work,
                                  max_crops=max_crops, crop_scale=crop_scale)
        if not paths:
            shutil.rmtree(work, ignore_errors=True)
            continue
        recog = _build_recognizer()
        probs, detail = recog.crop_reads(paths)
        index = [int(p.stem.split("_")[0][1:]) for p in paths]
        frame = percrop_frame({}, probs, detail, index, paths).assign(
            chunk=chunk, crop_scale=float(crop_scale))
        frame.to_parquet(dest, index=False)
        shutil.rmtree(work, ignore_errors=True)
        logger.info("[%d/%d] %s: %d crops, %d with a number, %d tracks (%.0fs)",
                    i + 1, len(chunks), chunk, len(frame), int((frame["number"] > 0).sum()),
                    frame["track_id"].nunique(), time.time() - t0)


def report(match_id: str, results_dir: Path, *, rule_floor: str | None = None,
           variant: str = "") -> dict:
    """CPU: read density before (close-up anchor chain) vs after (per-crop OCR + frozen rule).

    Args:
        match_id: Registry match id.
        results_dir: Where the JSON summary is written.
        rule_floor: Which frozen DEV precision floor's rule to apply (default: the primary).
        variant: Which per-crop cache to read (see :func:`percrop_dir`).
    """
    frozen = json.loads(RULE_PATH.read_text(encoding="utf-8"))
    src = frozen["rules"][rule_floor] if rule_floor else frozen
    rule = {k: v for k, v in src.items()
            if k in {"min_crop_conf", "min_votes", "emit_all", "min_legibility"}}
    after: set[tuple] = set()
    dominant: dict[tuple, int] = {}
    n_crops = n_reads = 0
    done: list[str] = []
    for p in sorted(percrop_dir(match_id, variant).glob("*.parquet")):
        frame = pd.read_parquet(p)
        chunk = str(frame["chunk"].iloc[0])
        done.append(chunk)
        n_crops += len(frame)
        reads = reads_for_sequence(frame, rule)
        n_reads += sum(len(v) for v in reads.values())
        after |= {(chunk, int(t)) for t in reads}
        for t, votes in reads.items():
            dominant[(chunk, int(t))] = Counter(n for n, _c in votes).most_common(1)[0][0]

    # Restrict every denominator to the chunks that actually have a per-crop cache, so a partial
    # (still-running or interrupted) pass reports an honest d rather than a diluted one.
    df = pd.read_parquet(registry.get(match_id).aligned)
    people = df[df["role"].isin(["player", "goalkeeper"]) & df["chunk"].isin(done)]
    tracks = people[["chunk", "track_id"]].drop_duplicates()
    gk = people[people["role"] == "goalkeeper"][["chunk", "track_id"]].drop_duplicates()
    gk_keys = set(map(tuple, gk.to_numpy()))

    named = pd.read_parquet(Path(str(NAMED_TRACKS).format(match=match_id)))
    named = named[named["chunk"].isin(done)]
    before = set(map(tuple, named[["chunk", "track_id"]].drop_duplicates().to_numpy()))
    n_tracks = len(tracks)

    # The only pseudo-truth this match carries: the close-up anchor chain's jersey numbers (98.6%
    # verified). On tracklets BOTH mechanisms read, do they say the same number? This is an
    # agreement rate, not a precision -- the anchor chain is itself fallible -- but it is the first
    # real-broadcast check on the per-crop reads at all.
    agree = graded = 0
    for c, t, j in zip(named["chunk"], named["track_id"], named["jersey_number"]):
        got = dominant.get((str(c), int(t)))
        if got is None or pd.isna(j):
            continue
        graded += 1
        agree += int(got == int(j))

    out = {
        "match": match_id, "rule": rule, "ocr_version": OCR_PERCROP_VERSION, "variant": variant,
        "chunks": done, "n_chunks_total": int(df["chunk"].nunique()),
        "n_tracks": n_tracks, "n_gk_tracks": len(gk_keys), "n_crops": n_crops, "n_reads": n_reads,
        "d_before_closeup_anchors": len(before) / max(n_tracks, 1),
        "d_after_percrop": len(after) / max(n_tracks, 1),
        "n_tracks_before": len(before), "n_tracks_after": len(after),
        "gk_tracks_read_before": len(before & gk_keys), "gk_tracks_read_after": len(after & gk_keys),
        "overlap_before_after": len(before & after),
        "n_anchor_graded": graded, "n_anchor_agree": agree,
        "agreement_with_anchor": (agree / graded if graded else float("nan")),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"ocr_match_{match_id}{variant}.json").write_text(json.dumps(out, indent=2),
                                                                     encoding="utf-8")
    logger.info("%s: d %.4f (%d/%d) -> %.4f (%d/%d); GK tracks read %d -> %d; %d reads over %d crops",
                match_id, out["d_before_closeup_anchors"], len(before), n_tracks,
                out["d_after_percrop"], len(after), n_tracks, out["gk_tracks_read_before"],
                out["gk_tracks_read_after"], n_reads, n_crops)
    return out


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", required=True)
    ap.add_argument("--max-crops", type=int, default=20)
    ap.add_argument("--crop-scale", type=float, default=1.0,
                    help="widen the OCR crop about the foot point (1.25 = the measured fix)")
    ap.add_argument("--variant", default="",
                    help="suffix on the ocr_percrop cache dir, e.g. _w125")
    ap.add_argument("--report", action="store_true", help="CPU: d before/after from the cache")
    ap.add_argument("--rule-floor", default=None, help="frozen DEV precision floor to apply")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()
    if args.report:
        report(args.match, args.results_dir, rule_floor=args.rule_floor, variant=args.variant)
        return
    run_match(args.match, max_crops=args.max_crops, crop_scale=args.crop_scale,
              variant=args.variant)


if __name__ == "__main__":
    main()
