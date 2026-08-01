"""Where GS-AssA dies on SoccerNet-GSR: a per-sequence decomposition from cached artifacts.

CPU only. Reads the cached positions parquets (``outputs/gsr/positions``), the GSR labels and the
already-scored arms, and answers one question: of the association we lose, how much is
**fragmentation** (one GT player carried by many predicted tracks), how much is **coverage** (GT
detections nobody predicted) and how much is **contamination** (predicted detections that belong to
nobody, or to somebody else).

The bridge is the HOTA association score itself. For a matched pair (predicted track ``i``,
ground-truth track ``g``) HOTA scores::

    A(i, g) = n_ig / (N_g + M_i - n_ig)

with ``n_ig`` their overlap, ``N_g`` the GT track's length and ``M_i`` the predicted track's length;
``AssA`` is the ``n_ig``-weighted mean of ``A`` over matched pairs. Everything on the right is
computable from a greedy nearest-GT assignment, so the same table yields three counterfactuals:

* ``assa_hat``     -- the proxy for what the evaluator measures (validated against it, see ``--diag``)
* ``assa_perfect_link`` -- every fragment of one GT player merged into one track (connector ceiling)
* ``assa_perfect_det``  -- fragmentation kept, but no missed GT and no spurious predicted rows

CLI::

    python -m tools.gsr_assoc_diag --diag            # the per-sequence table + correlations
    python -m tools.gsr_assoc_diag --coverage        # image-space vs pitch-space GT coverage
    python -m tools.gsr_assoc_diag --merge           # join both + the correlation panel
    python -m tools.gsr_assoc_diag --demo            # self-check
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from eval.gsr_score import CENTRE_SHIFT_X, CENTRE_SHIFT_Y, DEFAULT_DATA_DIR, GSR_DIST_TOL_M
from generator.track_relink import load_gt_ids_by_frame

VALID_OUT = Path("outputs/gsr")
RESULTS = Path("results/gsr_benchmark")


# === The association arithmetic ==================================================================
def assa_from_pairs(pairs: dict[tuple[int, int], int], n_gt: dict[int, int],
                    n_pred: dict[int, int]) -> float:
    """HOTA AssA over a greedy matching (pure).

    Args:
        pairs: ``{(pred_track_id, gt_track_id): overlap}`` -- matched detections per pair.
        n_gt: ``{gt_track_id: length}`` -- total GT detections of that identity.
        n_pred: ``{pred_track_id: length}`` -- total predicted detections of that track.

    Returns:
        The overlap-weighted mean of ``n/(N_g + M_i - n)``; ``0.0`` when nothing matched.
    """
    tp = sum(pairs.values())
    if not tp:
        return 0.0
    acc = 0.0
    for (pid, gid), n in pairs.items():
        denom = n_gt[gid] + n_pred[pid] - n
        acc += n * (n / denom if denom else 0.0)
    return acc / tp


def _relabel(pairs: dict[tuple[int, int], int], key) -> dict[tuple[int, int], int]:
    """Merge predicted ids by ``key(pid, gid)`` (pure) -- used to build the counterfactuals."""
    out: dict[tuple[int, int], int] = defaultdict(int)
    for (pid, gid), n in pairs.items():
        out[(key(pid, gid), gid)] += n
    return dict(out)


def counterfactuals(pairs: dict[tuple[int, int], int], n_gt: dict[int, int],
                    n_pred: dict[int, int]) -> dict[str, float]:
    """The three AssA numbers that separate fragmentation, coverage and contamination (pure)."""
    tp_of_pred: dict[int, int] = defaultdict(int)
    for (pid, _g), n in pairs.items():
        tp_of_pred[pid] += n
    # perfect link: all fragments of one GT identity become a single predicted track.
    linked = _relabel(pairs, lambda _p, g: -1000 - g)
    n_pred_linked: dict[int, int] = defaultdict(int)
    for (pid, gid), n in pairs.items():
        n_pred_linked[-1000 - gid] += n_pred[pid] * (n / max(tp_of_pred[pid], 1))
    # perfect detection: no FN, no FP -- N_g and M_i shrink to what actually matched.
    n_gt_tight: dict[int, int] = defaultdict(int)
    for (_p, gid), n in pairs.items():
        n_gt_tight[gid] += n
    return {
        "assa_hat": assa_from_pairs(pairs, n_gt, n_pred),
        "assa_perfect_link": assa_from_pairs(linked, n_gt, dict(n_pred_linked)),
        "assa_perfect_det": assa_from_pairs(pairs, dict(n_gt_tight), dict(tp_of_pred)),
        "assa_both": 1.0 if pairs else 0.0,
    }


# === Per-sequence measurement ====================================================================
def match_sequence(df: pd.DataFrame, gt_by_frame: dict) -> tuple[dict, dict, dict, dict]:
    """Greedy nearest-GT matching of one sequence -> ``(pairs, n_gt, n_pred, extras)``.

    One GT identity may claim at most one predicted row per frame and vice versa (the evaluator's
    one-to-one constraint), resolved greedily by distance -- the cheap stand-in for the Hungarian
    assignment ``trackeval`` runs.
    """
    people = df[df["role"].isin(["player", "goalkeeper"])]
    pairs: dict[tuple[int, int], int] = defaultdict(int)
    n_pred = Counter(people["track_id"].astype(int))
    n_gt = Counter(g[2] for gts in gt_by_frame.values() for g in gts)
    n_gt_rows = sum(n_gt.values())
    matched_gt = 0
    for frame, grp in people.groupby("frame"):
        gts = gt_by_frame.get(int(frame), [])
        if not gts:
            continue
        rows = [(int(r.track_id), float(r.pitch_x) - CENTRE_SHIFT_X,
                 float(r.pitch_y) - CENTRE_SHIFT_Y) for r in grp.itertuples()
                if np.isfinite(r.pitch_x) and np.isfinite(r.pitch_y)]
        if not rows:
            continue
        gxy = np.array([[g[0], g[1]] for g in gts])
        pxy = np.array([[x, y] for _t, x, y in rows])
        d = np.hypot(gxy[:, None, 0] - pxy[None, :, 0], gxy[:, None, 1] - pxy[None, :, 1])
        used_g, used_p = set(), set()
        for gi, pi in zip(*np.unravel_index(np.argsort(d, axis=None), d.shape), strict=True):
            if d[gi, pi] > GSR_DIST_TOL_M:
                break
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            pairs[(rows[pi][0], int(gts[gi][2]))] += 1
            matched_gt += 1
    extras = {
        "n_gt_rows": n_gt_rows, "n_pred_rows": int(len(people)), "n_gt_ids": len(n_gt),
        "n_pred_tracks": int(people["track_id"].nunique()),
        "det_recall": matched_gt / max(n_gt_rows, 1),
        "pred_precision": matched_gt / max(len(people), 1),
        "crowding": n_gt_rows / max(len(gt_by_frame), 1),
    }
    return dict(pairs), dict(n_gt), dict(n_pred), extras


def frag_shares(pairs: dict[tuple[int, int], int], n_gt: dict[int, int]) -> dict[str, float]:
    """Fragment-share statistics per GT identity (pure): count, dominant share, Herfindahl."""
    per_gt: dict[int, list[int]] = defaultdict(list)
    for (_p, gid), n in pairs.items():
        per_gt[gid].append(n)
    if not per_gt:
        return {"frag_per_gt": 0.0, "dominant_share": 0.0, "herfindahl": 0.0, "matched_share": 0.0}
    counts, doms, hhi, cov = [], [], [], []
    for gid, ns in per_gt.items():
        tot = float(n_gt[gid])
        counts.append(len(ns))
        doms.append(max(ns) / tot)
        hhi.append(sum((n / tot) ** 2 for n in ns))
        cov.append(sum(ns) / tot)
    return {"frag_per_gt": float(np.mean(counts)), "dominant_share": float(np.mean(doms)),
            "herfindahl": float(np.mean(hhi)), "matched_share": float(np.mean(cov))}


def gt_image_boxes(seq_dir: Path) -> dict[int, list[tuple[float, float, float, dict | None]]]:
    """GT player/GK boxes per frame as ``(foot_x, foot_y, width, bbox_pitch)`` in image pixels."""
    gt = json.loads((seq_dir / "Labels-GameState.json").read_text(encoding="utf-8"))
    idx = {im["image_id"]: int(Path(im["file_name"]).stem) - 1 for im in gt["images"]}
    out: dict[int, list] = defaultdict(list)
    for ann in gt["annotations"]:
        if (ann.get("attributes") or {}).get("role") not in {"player", "goalkeeper"}:
            continue
        bb, f = ann.get("bbox_image"), idx.get(ann["image_id"])
        if bb is None or f is None or ann.get("track_id") is None:
            continue
        out[f].append((bb["x"] + bb["w"] / 2.0, bb["y"] + bb["h"], bb["w"], ann.get("bbox_pitch")))
    return dict(out)


def coverage_split(seq_dir: Path, df: pd.DataFrame) -> dict[str, float]:
    """Split one sequence's GT coverage into image-space recall vs pitch-space recall (pure-ish).

    The difference is the calibration loss: players we detected and tracked, then failed to project.
    The image-space tolerance is half a GT box width (floor 12 px) around its foot point -- a proxy
    for an IoU match, so ``img_recall`` is an upper bound.
    """
    people = df[df["role"].isin(["player", "goalkeeper"])]
    by_frame = dict(people.groupby("frame").__iter__())
    gti = gt_image_boxes(seq_dir)
    n_gt = img_hit = pitch_hit = both_lost = no_pitch = calib_ok = 0
    for f, gts in gti.items():
        n_gt += len(gts)
        grp = by_frame.get(f)
        if grp is None or grp.empty:
            continue
        px, py = grp["image_x"].to_numpy(), grp["image_y"].to_numpy()
        ppx, ppy = grp["pitch_x"].to_numpy(), grp["pitch_y"].to_numpy()
        fin = np.isfinite(ppx) & np.isfinite(ppy)
        calib_ok += int(fin.any())
        for gx, gy, gw, bp in gts:
            hit_i = np.hypot(px - gx, py - gy).min() <= max(gw * 0.5, 12.0)
            img_hit += int(hit_i)
            if bp is None:
                no_pitch += 1
                continue
            hit_p = False
            if fin.any():
                d = np.hypot(ppx[fin] - CENTRE_SHIFT_X - bp["x_bottom_middle"],
                             ppy[fin] - CENTRE_SHIFT_Y - bp["y_bottom_middle"])
                hit_p = bool(d.min() <= GSR_DIST_TOL_M)
            pitch_hit += int(hit_p)
            both_lost += int(hit_i and not hit_p)
    n = max(n_gt, 1)
    return {"n_gt": n_gt, "img_recall": img_hit / n, "pitch_recall": pitch_hit / n,
            "img_ok_pitch_lost": both_lost / n, "net_img_minus_pitch": (img_hit - pitch_hit) / n,
            "gt_no_pitch": no_pitch / n, "calib_frame_rate": calib_ok / max(len(gti), 1),
            "nan_row_share": float((~np.isfinite(people["pitch_x"])).mean())}


def coverage(data_dir: Path, out_dir: Path) -> None:
    """Per-sequence image-vs-pitch coverage split -> printed table + JSON."""
    rows = []
    for p in sorted((out_dir / "positions").glob("*.parquet")):
        rows.append({"seq": p.stem,
                     **coverage_split(data_dir / p.stem, pd.read_parquet(p))})
        print(f"  {p.stem}: img {rows[-1]['img_recall']:.3f} pitch {rows[-1]['pitch_recall']:.3f} "
              f"lost {rows[-1]['img_ok_pitch_lost']:.3f} calib {rows[-1]['calib_frame_rate']:.3f}")
    df = pd.DataFrame(rows)
    w = df["n_gt"]
    print("\nPOOLED (GT-row weighted):")
    for c in ("img_recall", "pitch_recall", "img_ok_pitch_lost", "net_img_minus_pitch",
              "gt_no_pitch"):
        print(f"  {c:20s} {np.average(df[c], weights=w):.4f}")
    print(f"  calib_frame_rate     {df['calib_frame_rate'].mean():.4f} "
          f"({int((df['calib_frame_rate'] < 0.5).sum())} sequences below 0.50)")
    print(f"  nan_row_share        {df['nan_row_share'].mean():.4f}")
    out = Path("results/gsr_benchmark/gsr_coverage_split.json")
    out.write_text(df.to_json(orient="records", indent=1), encoding="utf-8")
    print(f"wrote {out}")


def camera_motion(seq_dir: Path) -> float:
    """Median per-frame image-space displacement of GT boxes, in pixels -- a camera-pan proxy."""
    gt = json.loads((seq_dir / "Labels-GameState.json").read_text(encoding="utf-8"))
    idx = {im["image_id"]: int(Path(im["file_name"]).stem) for im in gt["images"]}
    pos: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    for ann in gt["annotations"]:
        bb, tid = ann.get("bbox_image"), ann.get("track_id")
        if not bb or tid is None or (ann.get("attributes") or {}).get("role") not in {
                "player", "goalkeeper"}:
            continue
        f = idx.get(ann["image_id"])
        if f is not None:
            pos[int(tid)][f] = (bb["x"] + bb["w"] / 2.0, bb["y"] + bb["h"] / 2.0)
    steps = [abs(p[f][0] - p[f - 1][0]) for p in pos.values() for f in p if f - 1 in p]
    return float(np.median(steps)) if steps else 0.0


# === Driver ======================================================================================
def run(data_dir: Path, out_dir: Path, seqs: list[str] | None = None) -> list[dict]:
    """Measure every valid sequence with a cached positions parquet -> per-sequence rows."""
    pos_dir = out_dir / "positions"
    names = seqs or sorted(p.stem for p in pos_dir.glob("*.parquet"))
    rows = []
    for name in names:
        seq_dir = data_dir / name
        df = pd.read_parquet(pos_dir / f"{name}.parquet")
        gt_by_frame = load_gt_ids_by_frame(seq_dir)
        pairs, n_gt, n_pred, extras = match_sequence(df, gt_by_frame)
        rows.append({"seq": name, **extras, **frag_shares(pairs, n_gt),
                     **counterfactuals(pairs, n_gt, n_pred),
                     "cam_px_per_frame": camera_motion(seq_dir)})
        print(f"  {name}: recall {extras['det_recall']:.3f} frags {rows[-1]['frag_per_gt']:.2f} "
              f"assa_hat {rows[-1]['assa_hat']:.3f} link {rows[-1]['assa_perfect_link']:.3f}")
    return rows


def _measured(path: Path, arm: str | None, cfg: str = "loc_assoc") -> dict[str, float]:
    """Per-sequence measured AssA from a cached scores file."""
    d = json.loads(path.read_text(encoding="utf-8"))
    node = d["arms"][arm]["per_seq"][cfg] if arm else d["configs"][cfg]["per_seq"]
    return {k: v["GS-AssA"] for k, v in node.items()}


def diag(data_dir: Path, out_dir: Path) -> None:
    """Print the diagnosis table and the correlations, and write the raw rows."""
    rows = run(data_dir, out_dir)
    raw = _measured(RESULTS / "gsr_scores.json", None)
    gta = _measured(RESULTS / "gsr_scores_gta.json", "tau0.040_nosplit")
    for r in rows:
        r["measured_assa_raw"] = raw.get(r["seq"], float("nan"))
        r["measured_assa_gta"] = gta.get(r["seq"], float("nan"))
    df = pd.DataFrame(rows).sort_values("measured_assa_raw")
    pd.set_option("display.width", 200)
    cols = ["seq", "measured_assa_raw", "assa_hat", "assa_perfect_link", "assa_perfect_det",
            "frag_per_gt", "herfindahl", "dominant_share", "det_recall", "pred_precision",
            "crowding", "cam_px_per_frame"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nmeans:")
    print(df[cols[1:]].mean().to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nSpearman vs measured raw AssA:")
    for c in cols[2:]:
        print(f"  {c:22s} {df['measured_assa_raw'].corr(df[c], method='spearman'):+.3f}")
    out = Path("results/gsr_benchmark/gsr_assoc_diag.json")
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


def merge() -> None:
    """Join the ``--diag`` and ``--coverage`` tables and print the correlation panel (seconds)."""
    a = pd.DataFrame(json.loads((RESULTS / "gsr_assoc_diag.json").read_text(encoding="utf-8")))
    b = pd.DataFrame(json.loads((RESULTS / "gsr_coverage_split.json").read_text(encoding="utf-8")))
    d = a.merge(b, on="seq")
    d["assa_gain_gta"] = d["measured_assa_gta"] - d["measured_assa_raw"]
    cols = ["assa_hat", "assa_perfect_link", "assa_perfect_det", "frag_per_gt", "herfindahl",
            "dominant_share", "det_recall", "pred_precision", "crowding", "cam_px_per_frame",
            "img_recall", "pitch_recall", "img_ok_pitch_lost", "calib_frame_rate", "nan_row_share"]
    print("Spearman vs measured GS-AssA (raw arm / GTA arm / the GTA gain):")
    for c in cols:
        print(f"  {c:22s} {d['measured_assa_raw'].corr(d[c], method='spearman'):+.3f}  "
              f"{d['measured_assa_gta'].corr(d[c], method='spearman'):+.3f}  "
              f"{d['assa_gain_gta'].corr(d[c], method='spearman'):+.3f}")
    d["q"] = pd.qcut(d["calib_frame_rate"], 4, labels=["Q1 worst", "Q2", "Q3", "Q4 best"])
    keep = ["calib_frame_rate", "measured_assa_raw", "measured_assa_gta", "assa_perfect_link",
            "assa_perfect_det", "frag_per_gt", "img_recall", "pitch_recall"]
    print("\nBy calibration-health quartile:")
    print(d.groupby("q", observed=True)[keep].mean().round(3).to_string())
    out = RESULTS / "gsr_assoc_diag_merged.json"
    out.write_text(d.drop(columns=["q"]).to_json(orient="records", indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


def _demo() -> None:
    """Self-check the association arithmetic on hand-computable cases."""
    # one GT of length 10, split into two predicted tracks of 5, both pure and complete.
    pairs = {(1, 100): 5, (2, 100): 5}
    n_gt, n_pred = {100: 10}, {1: 5, 2: 5}
    assert abs(assa_from_pairs(pairs, n_gt, n_pred) - 0.5) < 1e-9
    cf = counterfactuals(pairs, n_gt, n_pred)
    assert abs(cf["assa_hat"] - 0.5) < 1e-9, cf
    assert abs(cf["assa_perfect_link"] - 1.0) < 1e-9, cf   # merging the two halves is perfect
    assert abs(cf["assa_perfect_det"] - 0.5) < 1e-9, cf    # detection was already perfect
    # one GT of length 10, one predicted track covering 5 of it and 5 rows of nothing.
    pairs, n_pred = {(1, 100): 5}, {1: 10}
    assert abs(assa_from_pairs(pairs, n_gt, n_pred) - 5 / 15) < 1e-9
    cf = counterfactuals(pairs, n_gt, n_pred)
    assert abs(cf["assa_perfect_link"] - 5 / 15) < 1e-9, cf  # nothing to link
    assert abs(cf["assa_perfect_det"] - 1.0) < 1e-9, cf      # the FN/FP were the whole loss
    assert frag_shares({(1, 100): 5, (2, 100): 5}, {100: 10})["herfindahl"] == 0.5
    print("gsr_assoc_diag self-check OK")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=VALID_OUT)
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--coverage", action="store_true",
                    help="image-space vs pitch-space GT coverage (isolates the calibration loss)")
    ap.add_argument("--merge", action="store_true",
                    help="join the two tables and print the correlation panel")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return
    if args.coverage:
        coverage(args.data_dir, args.out_dir)
    if args.merge:
        merge()
    if args.diag:
        diag(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
