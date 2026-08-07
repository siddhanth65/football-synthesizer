"""Grade the v7-V2 side-aware overlay detector against the ground-truth cluster -> side map.

``results/GSR_TEAMSIDE.md`` closed the positional family: ``meanx`` sits at its own oracle ceiling
(112/115 on ground-truth positions) and the residual failures are clips whose ground-truth geometry
*inverts*, so no depth statistic can reach them. The v7 plan's answer is a **learned** side bit: a
detector fine-tuned on SoccerNet-v3's ``team left`` / ``team right`` labels, run as an OVERLAY on
top of the shipped v6 detections rather than replacing them.

This module is the component gate, and nothing else. It never edits a shipped code path:

1. Overlay boxes (dumped server-side, one ``.npz`` per sequence) are matched to our cached
   positions rows -- the rows that already carry the kit cluster -- by foot point, at the repo's own
   tolerance ``max(0.5 * box width, 12 px)``.
2. Every matched row casts one side vote for its kit cluster.
3. The cluster -> side assignment that maximises vote agreement is the overlay's verdict, with a
   vote-share margin the fusion session can threshold on.
4. The verdict is compared with :func:`eval.gsr_score.resolve_team_map` -- the same ground-truth map
   ``tools/gsr_teamside.py`` grades against.

CLI::

    python -m tools.gsr_sideoverlay --grade --overlay-dir outputs/gsr/side_overlay
    python -m tools.gsr_sideoverlay --demo    # runnable self-check, no data needed
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from eval.gsr_score import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    load_gt_people_by_frame,
    resolve_team_map,
)
from tools.gsr_teamside import RESULTS_DIR, dev_names

logger = logging.getLogger("gsr_sideoverlay")

#: Overlay class id -> the side it votes for (classes absent cast no vote).
SIDE_OF = {1: "left", 2: "right", 3: "left", 4: "right"}
#: Overlay class ids that are goalkeepers -- reported separately, they anchor the GSR side attribute.
GK_CLS = (1, 2)
#: Foot-point match tolerance, px: the convention already used for GT-vs-positions matching.
MATCH_MIN_PX = 12.0
#: Clips whose GROUND-TRUTH geometry inverts (GSR_TEAMSIDE section 3) -- the gate's targets.
INVERSION_CLIPS = ("SNGS-038", "SNGS-092", "SNGS-111")


def match_frame(pos_xy: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    """Greedy nearest foot-point matching of positions rows to overlay boxes.

    Args:
        pos_xy: ``(N, 2)`` foot points of our cached detections (``image_x``, ``image_y``).
        box_xyxy: ``(M, 4)`` overlay boxes in the same pixel frame.

    Returns:
        ``(N,)`` index of the matched overlay box per positions row, ``-1`` where unmatched.
    """
    out = np.full(len(pos_xy), -1, int)
    if not len(pos_xy) or not len(box_xyxy):
        return out
    foot = np.stack([(box_xyxy[:, 0] + box_xyxy[:, 2]) / 2.0, box_xyxy[:, 3]], 1)
    tol = np.maximum(0.5 * (box_xyxy[:, 2] - box_xyxy[:, 0]), MATCH_MIN_PX)
    d = np.linalg.norm(pos_xy[:, None, :] - foot[None, :, :], axis=2)
    d = np.where(d <= tol[None, :], d, np.inf)
    used = np.zeros(len(box_xyxy), bool)
    for i in np.argsort(d.min(axis=1)):
        if not np.isfinite(d[i]).any():
            continue
        j = int(np.argmin(np.where(used, np.inf, d[i])))
        if used[j] or not np.isfinite(d[i, j]):
            continue
        out[i], used[j] = j, True
    return out


def vote_sequence(df: pd.DataFrame, ov: dict[str, np.ndarray]) -> dict:
    """Aggregate overlay side votes per kit cluster over one sequence.

    Args:
        df: Cached positions table (``frame, role, team, image_x, image_y, track_id``).
        ov: Overlay dump arrays ``frame, cls, conf, xyxy``.

    Returns:
        Vote counts per (cluster, side), the winning assignment, its vote share and the matched-row
        counts -- everything the conservative fusion needs.
    """
    people = df[df["role"].isin(("player", "goalkeeper")) & df["team"].isin([0, 1])
                & np.isfinite(df["image_x"]) & np.isfinite(df["image_y"])]
    by_frame: dict[int, np.ndarray] = {}
    order = np.argsort(ov["frame"], kind="stable")
    f_sorted = ov["frame"][order]
    bounds = np.searchsorted(f_sorted, np.unique(f_sorted), side="left").tolist() + [len(f_sorted)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        by_frame[int(ov["frame"][idx[0]])] = idx

    votes = np.zeros((2, 2))          # [cluster][0=left, 1=right]
    wvotes = np.zeros((2, 2))         # confidence-weighted
    gkvotes = np.zeros((2, 2))        # goalkeeper classes only
    n_rows = n_matched = 0
    iou_pairs: list[tuple[int, int, int]] = []   # (frame, track_id, overlay box index)
    for fr, g in people.groupby("frame", sort=False):
        n_rows += len(g)
        idx = by_frame.get(int(fr))
        if idx is None:
            continue
        m = match_frame(g[["image_x", "image_y"]].to_numpy(float), ov["xyxy"][idx])
        for row, j in zip(g.itertuples(), m):
            if j < 0:
                continue
            k = int(idx[j])                      # index into the sequence-wide overlay arrays
            n_matched += 1
            iou_pairs.append((int(fr), int(row.track_id), k))
            side = SIDE_OF.get(int(ov["cls"][k]))
            if side is None:
                continue
            c, s = int(row.team), 0 if side == "left" else 1
            votes[c, s] += 1
            wvotes[c, s] += float(ov["conf"][k])
            if int(ov["cls"][k]) in GK_CLS:
                gkvotes[c, s] += 1

    direct = votes[0, 0] + votes[1, 1]           # cluster 0 = left
    inverse = votes[0, 1] + votes[1, 0]          # cluster 0 = right
    total = direct + inverse
    return {
        "pred0": "left" if direct >= inverse else "right",
        "vote_share": round(float(max(direct, inverse) / total), 4) if total else float("nan"),
        "n_side_votes": int(total),
        "n_rows": int(n_rows),
        "n_matched": int(n_matched),
        "match_rate": round(n_matched / max(n_rows, 1), 4),
        "votes": {"c0_left": int(votes[0, 0]), "c0_right": int(votes[0, 1]),
                  "c1_left": int(votes[1, 0]), "c1_right": int(votes[1, 1])},
        "conf_vote_share": (round(float(max(wvotes[0, 0] + wvotes[1, 1],
                                            wvotes[0, 1] + wvotes[1, 0]) / wvotes.sum()), 4)
                            if wvotes.sum() else float("nan")),
        "gk_pred0": ("left" if gkvotes[0, 0] + gkvotes[1, 1] >= gkvotes[0, 1] + gkvotes[1, 0]
                     else "right") if gkvotes.sum() else None,
        "n_gk_votes": int(gkvotes.sum()),
        "_iou_pairs": iou_pairs,
    }


def meanx_score(df: pd.DataFrame) -> float:
    """The incumbent rule's signed score: negative means cluster 0 defends the left goal."""
    pl = df[(df["role"] == "player") & df["team"].isin([0, 1]) & np.isfinite(df["pitch_x"])]
    if pl.empty or pl["team"].nunique() < 2:
        return float("nan")
    m = pl.groupby("team")["pitch_x"].mean()
    return float(m[0] - m[1])


def iou_crosscheck(pairs: list[tuple[int, int, int]], ov: dict[str, np.ndarray],
                   cache: Path) -> dict:
    """Validate the foot-point match against the cached v6 boxes, where that cache exists."""
    if not cache.exists() or not pairs:
        return {}
    z = np.load(cache)
    key = {(int(f), int(t)): i for i, (f, t) in enumerate(zip(z["frames"], z["track_ids"]))}
    ious = []
    for fr, tid, j in pairs:
        i = key.get((fr, tid))
        if i is None:
            continue
        a, b = z["boxes"][i], ov["xyxy"][j]
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(x2 - x1, 0) * max(y2 - y1, 0)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        ious.append(inter / max(ua, 1e-9))
    if not ious:
        return {}
    v = np.asarray(ious)
    return {"n": int(len(v)), "mean_iou": round(float(v.mean()), 4),
            "frac_iou_ge_0.5": round(float((v >= 0.5).mean()), 4),
            "frac_iou_ge_0.7": round(float((v >= 0.7).mean()), 4)}


def grade(overlay_dir: Path, data_dir: Path, out_dir: Path) -> dict:
    """Run the component gate over the dev side-pool and return the full report."""
    pool = dev_names(data_dir)
    rows, cross = [], []
    for seq, pos_dir in sorted(pool.items()):
        npz = overlay_dir / f"{seq}.npz"
        if not npz.exists():
            logger.warning("no overlay dump for %s", seq)
            continue
        z = np.load(npz)
        ov = {k: z[k] for k in ("frame", "cls", "conf", "xyxy")}
        df = pd.read_parquet(pos_dir / f"{seq}.parquet")
        v = vote_sequence(df, ov)
        pairs = v.pop("_iou_pairs")
        cc = iou_crosscheck(pairs, ov, out_dir / "detbox_cache" / f"{seq}.npz")
        if cc:
            cross.append(cc)
        gt0 = resolve_team_map(df, load_gt_people_by_frame(data_dir / seq))[0]
        mx = meanx_score(df)
        rows.append({"seq": seq, "split": "train" if "probe" in str(pos_dir) else "valid",
                     "gt0": gt0, **v, "correct": v["pred0"] == gt0,
                     "gk_correct": None if v["gk_pred0"] is None else v["gk_pred0"] == gt0,
                     "meanx": round(mx, 3) if np.isfinite(mx) else None,
                     "meanx_correct": bool((mx < 0) == (gt0 == "left")) if np.isfinite(mx)
                     else None})
        logger.info("%s gt=%s pred=%s share=%.3f votes=%d meanx=%s", seq, gt0, v["pred0"],
                    v["vote_share"], v["n_side_votes"], rows[-1]["meanx"])

    d = pd.DataFrame(rows)
    n = len(d)
    gk = d[d["gk_correct"].notna()]
    return {
        "n_sequences": n,
        "accuracy": round(float(d["correct"].mean()), 4) if n else float("nan"),
        "correct": int(d["correct"].sum()),
        "wrong": d.loc[~d["correct"], "seq"].tolist(),
        "meanx_accuracy": round(float(d["meanx_correct"].mean()), 4) if n else float("nan"),
        "meanx_wrong": d.loc[~d["meanx_correct"].astype(bool), "seq"].tolist(),
        "agree_with_meanx": int((d["correct"] == d["meanx_correct"]).sum()),
        "gk_only_accuracy": round(float(gk["gk_correct"].mean()), 4) if len(gk) else float("nan"),
        "gk_only_n": int(len(gk)),
        "inversion_clips": {c: (d.loc[d["seq"] == c].to_dict("records") or [None])[0]
                            for c in INVERSION_CLIPS},
        "by_split": {s: {"n": int(len(g)), "correct": int(g["correct"].sum()),
                         "accuracy": round(float(g["correct"].mean()), 4)}
                     for s, g in d.groupby("split")},
        "match_rate_mean": round(float(d["match_rate"].mean()), 4) if n else float("nan"),
        "vote_share_mean": round(float(d["vote_share"].mean()), 4) if n else float("nan"),
        "iou_crosscheck_vs_v6_detbox": (
            {"n_sequences": len(cross),
             "n_pairs": int(sum(c["n"] for c in cross)),
             "mean_iou": round(float(np.average([c["mean_iou"] for c in cross],
                                                weights=[c["n"] for c in cross])), 4),
             "frac_iou_ge_0.5": round(float(np.average([c["frac_iou_ge_0.5"] for c in cross],
                                                       weights=[c["n"] for c in cross])), 4)}
            if cross else {}),
        "per_seq": rows,
    }


def _demo() -> None:
    """Self-check: matching, vote aggregation and the verdict (asserts; needs no data)."""
    # two clusters, two frames; cluster 1 sits at x=1500 and is the LEFT team by overlay class
    pos = pd.DataFrame({
        "frame": [0, 0, 1, 1], "role": "player", "team": [0, 1, 0, 1], "track_id": [1, 2, 1, 2],
        "image_x": [500.0, 1500.0, 505.0, 1505.0], "image_y": [800.0, 800.0, 800.0, 800.0],
        "pitch_x": [60.0, 40.0, 60.0, 40.0],
    })
    ov = {
        "frame": np.array([0, 0, 1, 1], np.int32),
        "cls": np.array([4, 3, 4, 3], np.int8),          # right, left, right, left
        "conf": np.array([0.9, 0.9, 0.9, 0.9], np.float32),
        "xyxy": np.array([[480, 700, 520, 800], [1480, 700, 1520, 800],
                          [485, 700, 525, 800], [1485, 700, 1525, 800]], np.float32),
    }
    v = vote_sequence(pos, ov)
    assert v["n_matched"] == 4, v
    assert v["votes"] == {"c0_left": 0, "c0_right": 2, "c1_left": 2, "c1_right": 0}, v["votes"]
    assert v["pred0"] == "right" and v["vote_share"] == 1.0, v
    # meanx would say the opposite here (cluster 0 has the LARGER pitch_x -> positive -> "right"
    # is also what meanx concludes); flip one cluster's votes and the verdict must flip with it
    ov2 = dict(ov, cls=np.array([3, 4, 3, 4], np.int8))
    assert vote_sequence(pos, ov2)["pred0"] == "left"
    # a box whose foot point is far from every cached row must not match
    ov3 = dict(ov, xyxy=ov["xyxy"] + np.array([0, 0, 0, 0], np.float32))
    ov3["xyxy"] = ov3["xyxy"] + np.array([900.0, 0.0, 900.0, 0.0], np.float32)
    assert vote_sequence(pos, ov3)["n_matched"] == 0, vote_sequence(pos, ov3)
    # split votes -> share 0.5, verdict defaults to the direct map
    ov4 = dict(ov, cls=np.array([4, 4, 3, 3], np.int8))
    v4 = vote_sequence(pos, ov4)
    assert v4["vote_share"] == 0.5 and v4["pred0"] == "left", v4
    assert meanx_score(pos) > 0
    print("gsr_sideoverlay demo OK: foot-point matching respects the tolerance, votes aggregate "
          "per kit cluster, and the verdict follows the majority assignment")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--overlay-dir", type=Path, default=DEFAULT_OUT_DIR / "side_overlay")
    ap.add_argument("--tag", default="dev", help="suffix of the results json")
    ap.add_argument("--grade", action="store_true", help="run the component gate on the dev pool")
    ap.add_argument("--demo", action="store_true", help="runnable self-check")
    args = ap.parse_args()

    if args.demo:
        _demo()
        return
    if not args.grade:
        ap.error("choose --grade or --demo")
    rep = grade(args.overlay_dir, args.data_dir, args.out_dir)
    path = RESULTS_DIR / f"sideoverlay_{args.tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", path)
    print(f"side accuracy {rep['correct']}/{rep['n_sequences']} = {rep['accuracy']}  "
          f"(meanx {rep['meanx_accuracy']})")
    for c, r in rep["inversion_clips"].items():
        print(f"  inversion {c}: {'n/a' if r is None else r['correct']} "
              f"(gt {None if r is None else r['gt0']}, pred {None if r is None else r['pred0']}, "
              f"share {None if r is None else r['vote_share']})")


if __name__ == "__main__":
    main()
