"""Paired crop-geometry trial for the per-crop OCR chain (the validator of OCR_DOMAIN_SHIFT.md).

``tools/ocr_match.py`` cuts its OCR crops with :func:`generator.team_anchor.estimate_player_box`, a
two-constant depth model fitted to nothing. Measured against FOOTPASS's annotated player ROIs it
returns **0.814x** the true player height, and the OCR chain's legibility gate is very sensitive to
that: on game_18 the same crops widened by 1.25 produce **2.56x** the confident jersey reads at
*higher* precision.

This tool re-cuts a sample of the crops a match's per-crop OCR pass already made, once per box
geometry, runs the identical reader over each arm, and reports the paired difference. It never
touches the match cache and never re-runs the pipeline.

Arms:

* ``est``    -- the shipped ``estimate_player_box``.
* ``scaled`` -- the same box scaled about the foot point (``--scale``, default 1.25).
* ``gtbox``  -- the FOOTPASS annotated ROI; only for ``footpass_*`` matches whose VAL HDF5 is on
  disk (see ``--gt-h5``), and the only arm that yields a precision number.

CLI::

    python -m tools.ocr_box_trial --match manutd_brighton --frames 900
    python -m tools.ocr_box_trial --match footpass_game_18 --gt-h5 <path>/val_tactical_data.h5
    python -m tools.ocr_box_trial --match footpass_game_18 --grade --variant _w125 \
        --gt-h5 <path>/val_tactical_data.h5
    python -m tools.ocr_box_trial --selftest
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from core import registry
from generator.team_anchor import estimate_player_box
from tools.gta_match import MIN_BOX_H, video_for
from tools.ocr_match import percrop_dir, scale_box

logger = logging.getLogger("ocr_box_trial")

#: Confident-read floor, the shipped rule's ``min_crop_conf`` at the 0.85 DEV floor.
P_CONF = 0.99


def gt_rois(h5_path: Path, game: str) -> dict[tuple[int, int], tuple[float, float, float, float]]:
    """FOOTPASS annotated player ROIs as ``{(global_frame, shirt): (x1, y1, x2, y2)}``."""
    import h5py  # noqa: PLC0415

    out: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    with h5py.File(h5_path, "r") as f:
        for half in ("H1", "H2"):
            a = np.asarray(f[f"{game}_{half}"][:])
            a = a[~np.isnan(a[:, 9])]
            for r in a:
                out[(int(r[0]), int(r[3]))] = (r[9], r[10], r[9] + r[11], r[10] + r[12])
    return out


def sample_crops(match_id: str, frames: int, seed: int) -> pd.DataFrame:
    """Rows of the match's per-crop OCR cache on ``frames`` randomly chosen frames.

    Args:
        match_id: Registry match id whose ``ocr_percrop`` cache exists.
        frames: Number of distinct ``(chunk, frame)`` pairs to sample.
        seed: RNG seed for the frame sample.

    Returns:
        ``chunk, frame, track_id, image_x, image_y, team`` for every cached crop on those frames.
    """
    per = pd.concat([pd.read_parquet(p, columns=["track_id", "frame", "chunk"])
                     for p in sorted(percrop_dir(match_id).glob("*.parquet"))])
    per = per[per["track_id"] >= 0]
    al = pd.read_parquet(registry.get(match_id).aligned)
    al = al[al["role"].isin(["player", "goalkeeper"])][
        ["chunk", "frame", "track_id", "image_x", "image_y", "team"]]
    for c in ("frame", "track_id"):
        al[c] = al[c].astype("int64")
        per[c] = per[c].astype("int64")
    j = per.merge(al, on=["chunk", "frame", "track_id"], how="inner")
    keys = j[["chunk", "frame"]].drop_duplicates()
    return j.merge(keys.sample(n=min(frames, len(keys)), random_state=seed),
                   on=["chunk", "frame"])


def cut_arms(match_id: str, sample: pd.DataFrame, work: Path, *, scale: float,
             gt: dict | None, offsets: dict[str, int] | None) -> pd.DataFrame:
    """Write one crop per arm per sampled row; return the per-crop metadata (IO)."""
    import cv2  # noqa: PLC0415

    arms = ["est", "scaled"] + (["gtbox"] if gt is not None else [])
    for arm in arms:
        (work / arm).mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for chunk, grp in sample.groupby("chunk"):
        cap = cv2.VideoCapture(str(video_for(match_id, chunk)))
        for frame_idx, sub in grp.groupby("frame"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, bgr = cap.read()
            if not ok:
                continue
            fh, fw = bgr.shape[:2]
            for r in sub.itertuples():
                est = estimate_player_box(r.image_x, r.image_y, fh, fw)
                boxes = {"est": est, "scaled": scale_box(est, scale, fh, fw)}
                if gt is not None:
                    key = ((offsets or {}).get(chunk, 0) + int(frame_idx), int(r.gt_shirt))
                    roi = gt.get(key)
                    if roi is None:
                        continue
                    boxes["gtbox"] = (int(max(roi[0], 0)), int(max(roi[1], 0)),
                                      int(min(roi[2], fw)), int(min(roi[3], fh)))
                name = f"{chunk}_{int(frame_idx):06d}_t{int(r.track_id)}.jpg"
                for arm, (x1, y1, x2, y2) in boxes.items():
                    if y2 - y1 < MIN_BOX_H or x2 - x1 < 8:  # noqa: PLR2004
                        continue
                    crop = bgr[y1:y2, x1:x2]
                    if crop.size:
                        cv2.imwrite(str(work / arm / name), crop)
                rows.append({"file": name, "chunk": chunk, "frame": int(frame_idx),
                             "track_id": int(r.track_id), "team": int(r.team),
                             "est_h": int(est[3] - est[1]),
                             "gt_shirt": int(getattr(r, "gt_shirt", -1))})
        cap.release()
    return pd.DataFrame(rows).drop_duplicates("file")


def read_arm(work: Path, arm: str, files: list[str]) -> pd.DataFrame:
    """Run the shipped OCR chain over one arm's crops (GPU)."""
    from eval.gsr_jersey import _build_recognizer  # noqa: PLC0415

    paths = [work / arm / f for f in files if (work / arm / f).exists()]
    recog = _build_recognizer()
    probs, _detail = recog.crop_reads(paths)
    best = probs[:, 1:].argmax(axis=1) + 1
    conf = probs[np.arange(len(probs)), best]
    number = np.where(probs.argmax(axis=1) == 0, -1, best)
    return pd.DataFrame({"arm": arm, "file": [p.name for p in paths], "number": number,
                         "p_number": np.where(number < 0, 0.0, conf)})


def summarise(reads: pd.DataFrame, meta: pd.DataFrame, scale: float) -> dict:
    """Per-arm read rates, the paired gain over ``est`` and (with GT) read precision."""
    reads = reads.merge(meta[["file", "gt_shirt"]], on="file", how="left")
    out: dict = {"scale": scale, "arms": {}, "paired_vs_est": {}}
    base = reads[reads["arm"] == "est"].set_index("file")
    for arm, d in reads.groupby("arm"):
        conf = d[(d["number"] >= 0) & (d["p_number"] >= P_CONF)]
        graded = conf[conf["gt_shirt"] >= 0]
        out["arms"][arm] = {
            "n": int(len(d)), "read_rate": float((d["number"] >= 0).mean()),
            "confident_rate": float((d["p_number"] >= P_CONF).mean()),
            "n_confident": int(len(conf)), "n_graded": int(len(graded)),
            "precision": (float((graded["number"] == graded["gt_shirt"]).mean())
                          if len(graded) else None)}
        if arm == "est":
            continue
        a = d.set_index("file")
        idx = a.index.intersection(base.index)
        a, b = a.loc[idx], base.loc[idx]
        ac, bc = a["p_number"] >= P_CONF, b["p_number"] >= P_CONF
        both = ac & bc
        add = a[ac & ~bc]
        addg = add[add["gt_shirt"] >= 0]
        out["paired_vs_est"][arm] = {
            "n_paired": int(len(idx)), "n_added": int((ac & ~bc).sum()),
            "n_lost": int((bc & ~ac).sum()),
            "p_added_given_est_missed": float((ac & ~bc).sum() / max((~bc).sum(), 1)),
            "agreement_where_both_confident": (
                float((a["number"][both] == b["number"][both]).mean()) if both.sum() else None),
            "n_both_confident": int(both.sum()),
            "added_read_precision": (float((addg["number"] == addg["gt_shirt"]).mean())
                                     if len(addg) else None)}
    return out


def run(match_id: str, *, scale: float, frames: int, seed: int, gt_h5: Path | None,
        results_dir: Path, keep: bool) -> dict:
    """Cut, read and score every arm for one match."""
    sample = sample_crops(match_id, frames, seed)
    gt = offsets = None
    if gt_h5 is not None:
        from tools.footpass_prep import chunk_offsets  # noqa: PLC0415

        game = match_id.removeprefix("footpass_")
        gt, offsets = gt_rois(gt_h5, game), chunk_offsets(game)
        sample = _attach_gt_shirt(sample, gt, offsets)
    work = percrop_dir(match_id) / "_boxtrial"
    shutil.rmtree(work, ignore_errors=True)
    meta = cut_arms(match_id, sample, work, scale=scale, gt=gt, offsets=offsets)
    logger.info("%s: %d crops per arm", match_id, len(meta))
    files = meta["file"].tolist()
    reads = pd.concat([read_arm(work, arm, files)
                       for arm in (["est", "scaled"] + (["gtbox"] if gt else []))])
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    out = {"match": match_id, "n_crops": int(len(meta)),
           "est_h_median": float(meta["est_h"].median()), **summarise(reads, meta, scale)}
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"ocr_box_trial_{match_id}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    for arm, v in out["arms"].items():
        logger.info("  %-7s confident %.4f (n=%d) precision %s", arm, v["confident_rate"],
                    v["n_confident"], v["precision"])
    return out


def _attach_gt_shirt(sample: pd.DataFrame, gt: dict, offsets: dict[str, int]) -> pd.DataFrame:
    """Label each sampled row with the shirt of the nearest annotated ROI (foot-point match).

    The match is accepted only when the nearest ROI is clearly nearest (its normalised distance is
    at most 0.35 and the runner-up is at least 1.5x further), which is the criterion whose team
    labels agree with our kit anchor on 96-98% of crops within a half.
    """
    by_frame: dict[int, list[tuple[int, tuple]]] = {}
    for (gframe, shirt), roi in gt.items():
        by_frame.setdefault(gframe, []).append((shirt, roi))
    shirts = np.full(len(sample), -1, np.int64)
    for i, (chunk, frame_idx, fx, fy) in enumerate(zip(
            sample["chunk"], sample["frame"], sample["image_x"], sample["image_y"])):
        cand = by_frame.get(offsets.get(chunk, 0) + int(frame_idx))
        if not cand:
            continue
        boxes = np.array([r for _s, r in cand], float)
        half_h = np.maximum((boxes[:, 3] - boxes[:, 1]) * 0.5, 1.0)
        d = np.hypot((fx - (boxes[:, 0] + boxes[:, 2]) / 2) / half_h,
                     (fy - boxes[:, 3]) / half_h)
        k = int(d.argmin())
        runner = np.partition(d, 1)[1] if len(d) > 1 else np.inf
        if d[k] <= 0.35 and runner >= 1.5 * d[k]:  # noqa: PLR2004
            shirts[i] = cand[k][0]
    return sample.assign(gt_shirt=shirts)[lambda f: f["gt_shirt"] >= 0]


#: Matcher thresholds, identical to :func:`_attach_gt_shirt` (whose team labels agree with our kit
#: anchor on 96-98% of crops within a half -- the validation of ``OCR_DOMAIN_SHIFT.md`` §0).
MAX_NORM_DIST = 0.35
MIN_RUNNER_RATIO = 1.5


def gt_track_shirts(match_id: str, gt_h5: Path, cache: Path) -> dict[tuple[str, int], int]:
    """Dominant annotated shirt per ``(chunk, track_id)``, over EVERY aligned player row.

    Arm-independent on purpose: the label must not move when the crop geometry moves, so it is
    built from the aligned table (not from whichever crops a given OCR pass happened to cut) and
    cached.

    Args:
        match_id: Registry match id (``footpass_<game>``).
        gt_h5: FOOTPASS ``val_tactical_data.h5``.
        cache: JSON cache path; read if present, written otherwise.

    Returns:
        ``{(chunk, track_id): shirt}`` for tracks whose matched annotations have a dominant shirt.
    """
    if cache.exists():
        return {(c, int(t)): int(s) for c, t, s in json.loads(cache.read_text(encoding="utf-8"))}
    from tools.footpass_prep import chunk_offsets  # noqa: PLC0415

    game = match_id.removeprefix("footpass_")
    gt, offsets = gt_rois(gt_h5, game), chunk_offsets(game)
    by_frame: dict[int, list[tuple[int, tuple]]] = {}
    for (gframe, shirt), roi in gt.items():
        by_frame.setdefault(gframe, []).append((shirt, roi))

    al = pd.read_parquet(registry.get(match_id).aligned)
    al = al[al["role"].isin(["player", "goalkeeper"])
            & np.isfinite(al["image_x"]) & np.isfinite(al["image_y"])]
    tally: dict[tuple[str, int], Counter] = {}
    for (chunk, frame_idx), grp in al.groupby(["chunk", "frame"]):
        cand = by_frame.get(offsets.get(str(chunk), 0) + int(frame_idx))
        if not cand:
            continue
        boxes = np.array([r for _s, r in cand], float)
        half_h = np.maximum((boxes[:, 3] - boxes[:, 1]) * 0.5, 1.0)
        fx = grp["image_x"].to_numpy(float)[:, None]
        fy = grp["image_y"].to_numpy(float)[:, None]
        d = np.hypot((fx - (boxes[:, 0] + boxes[:, 2])[None, :] / 2) / half_h[None, :],
                     (fy - boxes[:, 3][None, :]) / half_h[None, :])
        k = d.argmin(axis=1)
        best = d[np.arange(len(d)), k]
        runner = (np.partition(d, 1, axis=1)[:, 1] if d.shape[1] > 1
                  else np.full(len(d), np.inf))
        ok = (best <= MAX_NORM_DIST) & (runner >= MIN_RUNNER_RATIO * best)
        for tid, j in zip(grp["track_id"].to_numpy()[ok], k[ok]):
            tally.setdefault((str(chunk), int(tid)), Counter())[int(cand[int(j)][0])] += 1
    out = {key: c.most_common(1)[0][0] for key, c in tally.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([[c, t, s] for (c, t), s in sorted(out.items())]),
                     encoding="utf-8")
    logger.info("GT track shirts: %d tracks labelled from %d annotated frames", len(out),
                len(by_frame))
    return out


def grade(match_id: str, gt_h5: Path, *, variant: str, floor: str, min_votes: int | None,
          results_dir: Path) -> dict:
    """Ground-truth-graded read density ``d`` and read precision for one per-crop cache.

    ``d`` counts distinct player/GK ``(chunk, track_id)`` pairs of the aligned table carrying at
    least one read (``OCR_REALMATCH.md`` §2's denominator); read precision compares each read
    track's dominant number against :func:`gt_track_shirts`.

    Args:
        match_id: Registry match id.
        gt_h5: FOOTPASS ``val_tactical_data.h5``.
        variant: Which per-crop cache to grade (``tools.ocr_match.percrop_dir``).
        floor: Frozen DEV precision floor whose rule to apply.
        min_votes: Exploratory override of that rule's vote bar (``None`` = frozen).
        results_dir: Where the JSON summary lands.

    Returns:
        The summary dict, also written to ``ocr_box_grade_<match><variant>_v<votes>.json``.
    """
    from tools.identity_match import percrop_reads  # noqa: PLC0415
    from tools.ocr_density import RULE_PATH  # noqa: PLC0415

    rule = json.loads(RULE_PATH.read_text(encoding="utf-8"))["rules"][floor].copy()
    if min_votes is not None:
        rule["min_votes"] = int(min_votes)
    reads = percrop_reads(match_id, floor, variant=variant, min_votes=min_votes)
    gt = gt_track_shirts(match_id, gt_h5, results_dir / f"gt_track_shirts_{match_id}.json")

    al = pd.read_parquet(registry.get(match_id).aligned)
    tracks = al[al["role"].isin(["player", "goalkeeper"])][
        ["chunk", "track_id"]].drop_duplicates()
    n_tracks = len(tracks)

    read_keys, correct, graded = [], 0, 0
    for chunk, by_track in reads.items():
        for tid, votes in by_track.items():
            key = (str(chunk), int(tid))
            read_keys.append(key)
            dom = Counter(n for n, _c in votes).most_common(1)[0][0]
            if key in gt:
                graded += 1
                correct += int(dom == gt[key])

    crops = legible = conf = crop_graded = crop_correct = 0
    for path in sorted(percrop_dir(match_id, variant).glob("*.parquet")):
        frame = pd.read_parquet(path)
        crops += len(frame)
        legible += int((frame["legibility"] >= rule["min_legibility"]).sum())
        conf += int((frame["p_number"] >= rule["min_crop_conf"]).sum())
        chunk = str(frame["chunk"].iloc[0])
        hit = frame[(frame["p_number"] >= rule["min_crop_conf"]) & (frame["number"] > 0)]
        for tid, num in zip(hit["track_id"], hit["number"]):
            truth = gt.get((chunk, int(tid)))
            if truth is not None:
                crop_graded += 1
                crop_correct += int(int(num) == truth)

    out = {
        "match": match_id, "variant": variant, "floor": floor, "rule": rule,
        "n_tracks": int(n_tracks), "n_tracks_read": len(read_keys),
        "d": len(read_keys) / max(n_tracks, 1),
        "n_graded": graded, "n_correct": correct,
        "read_precision": correct / graded if graded else None,
        "n_gt_labelled_tracks": len(gt),
        "n_crops": crops, "crop_legible_rate": legible / max(crops, 1),
        "crop_confident_rate": conf / max(crops, 1),
        "crop_read_precision": crop_correct / crop_graded if crop_graded else None,
        "n_crop_graded": crop_graded,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{variant}_v{rule['min_votes']}"
    (results_dir / f"ocr_box_grade_{match_id}{tag}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    logger.info("%s%s votes=%d: d %.4f (%d/%d), read precision %s (%d graded), crops %d "
                "legible %.4f confident %.4f", match_id, variant or " (shipped)",
                rule["min_votes"], out["d"], out["n_tracks_read"], n_tracks,
                f"{out['read_precision']:.4f}" if out["read_precision"] else "n/a", graded,
                crops, out["crop_legible_rate"], out["crop_confident_rate"])
    return out


def _selftest() -> None:
    """Pin the box scaling: the foot point stays put, the box grows, clipping holds."""
    box = (100, 200, 145, 300)  # 45 x 100, foot at (122.5, 300)
    x1, y1, x2, y2 = scale_box(box, 1.25, 1080, 1920)
    assert x1 <= box[0] and x2 >= box[2], (x1, x2)
    assert y1 < box[1], (y1, box[1])
    assert abs((y2 - y1) - 1.30 * (box[3] - box[1])) <= 1.0, (y1, y2)
    assert abs((x1 + x2) / 2 - (box[0] + box[2]) / 2) <= 1.0, (x1, x2)
    assert scale_box((0, 0, 40, 90), 1.25, 90, 1920)[:2] == (0, 0)
    assert scale_box(box, 1.0, 1080, 1920)[1] == box[1]
    print("ocr_box_trial selftest OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match")
    ap.add_argument("--scale", type=float, default=1.25)
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--gt-h5", type=Path, default=None, help="FOOTPASS val_tactical_data.h5")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--keep-crops", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--grade", action="store_true",
                    help="CPU: GT-graded d + read precision for a per-crop cache")
    ap.add_argument("--variant", default="", help="per-crop cache variant to grade, e.g. _w125")
    ap.add_argument("--floor", default="0.85", help="frozen DEV precision floor")
    ap.add_argument("--min-votes", type=int, default=None,
                    help="EXPLORATORY override of the frozen rule's vote bar")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.match:
        ap.error("--match is required")
    if args.grade:
        if args.gt_h5 is None:
            ap.error("--grade needs --gt-h5")
        grade(args.match, args.gt_h5, variant=args.variant, floor=args.floor,
              min_votes=args.min_votes, results_dir=args.results_dir)
        return
    run(args.match, scale=args.scale, frames=args.frames, seed=args.seed, gt_h5=args.gt_h5,
        results_dir=args.results_dir, keep=args.keep_crops)


if __name__ == "__main__":
    main()
