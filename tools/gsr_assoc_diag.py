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
    python -m tools.gsr_assoc_diag --v7              # v7 V4 step 1: the ceiling on the CURRENT stack
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


def assa_merge_only(pairs: dict[tuple[int, int], int], n_gt: dict[int, int],
                    n_pred: dict[int, int]) -> float:
    """AssA of an oracle that only MERGES: no predicted track is ever split (pure).

    :func:`counterfactuals`' ``assa_perfect_link`` lets the oracle split a contaminated predicted
    track as well as merge fragments, so it is the ceiling of a linker *plus* a splitter. A merge-only
    connector -- what the GTA connector is -- can never undo contamination, so each predicted track
    goes wholesale to the GT identity holding most of its matched rows.

    Returns:
        The overlap-weighted AssA after that relabelling.
    """
    best: dict[int, tuple[int, int]] = {}
    for (pid, gid), n in pairs.items():
        if n > best.get(pid, (0, -1))[0]:
            best[pid] = (n, gid)
    merged: dict[tuple[int, int], int] = defaultdict(int)
    n_merged: dict[int, int] = defaultdict(int)
    seen: set[int] = set()
    for (pid, gid), n in pairs.items():
        dom = best[pid][1]
        merged[(-2000 - dom, gid)] += n
        if pid not in seen:
            n_merged[-2000 - dom] += n_pred[pid]
            seen.add(pid)
    return assa_from_pairs(dict(merged), n_gt, dict(n_merged))


def rank_profile(pairs: dict[tuple[int, int], int], n_gt: dict[int, int]) -> list[list[float]]:
    """Per GT identity, its predicted fragments' shares sorted descending (pure).

    One list per GT identity that at least one predicted row matched; the shares sum to the
    identity's matched coverage, so ``1 - sum`` is the share nothing predicted.
    """
    per_gt: dict[int, list[int]] = defaultdict(list)
    for (_p, gid), n in pairs.items():
        per_gt[gid].append(n)
    return [sorted((n / float(n_gt[gid]) for n in ns), reverse=True)
            for gid, ns in per_gt.items()]


def pooled_rank_table(profile: list[list[float]], k: int = 8) -> dict[str, float]:
    """Mean share held by the k-th largest fragment, pooled over GT identities (pure)."""
    if not profile:
        return {}
    out = {f"rank{i + 1}": float(np.mean([p[i] if len(p) > i else 0.0 for p in profile]))
           for i in range(k)}
    out[f"rank>{k}"] = float(np.mean([sum(p[k:]) for p in profile]))
    out["unmatched"] = float(np.mean([max(0.0, 1.0 - sum(p)) for p in profile]))
    out["n_gt_identities"] = float(len(profile))
    return out


def submission_positions(pred_path: Path, seq_dir: Path) -> pd.DataFrame:
    """Read a scored GSR submission back into the positions schema this module measures.

    The shipped submission is the partition the official scorer actually sees (tracker + connector
    + identity solver), so measuring it answers "what does the CURRENT chain realise" on exactly the
    same instrument as the tracker-only partitions.

    Args:
        pred_path: ``.../predictions/data/<seq>.json``.
        seq_dir: The sequence's GT directory (for the ``image_id`` -> frame index map).

    Returns:
        A frame/track_id/role/pitch_xy/image_xy table; ``pitch_xy`` is un-centred so that
        :func:`match_sequence`'s own centre shift recovers the submitted coordinate.
    """
    from eval.gsr_score import load_image_id_map  # noqa: PLC0415

    frame_of = {v: k for k, v in load_image_id_map(seq_dir).items()}
    rows = []
    for ann in json.loads(pred_path.read_text(encoding="utf-8"))["predictions"]:
        bp, f = ann.get("bbox_pitch"), frame_of.get(ann["image_id"])
        if f is None:
            continue
        bi = ann.get("bbox_image") or {}
        rows.append({
            "frame": int(f), "track_id": int(ann["track_id"]),
            "role": (ann.get("attributes") or {}).get("role", "player"),
            "pitch_x": (bp["x_bottom_middle"] + CENTRE_SHIFT_X) if bp else float("nan"),
            "pitch_y": (bp["y_bottom_middle"] + CENTRE_SHIFT_Y) if bp else float("nan"),
            "image_x": bi.get("x_center", float("nan")),
            "image_y": bi.get("y", float("nan")) + bi.get("h", 0.0),
        })
    return pd.DataFrame(rows)


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


def coverage(data_dir: Path, out_dir: Path, positions_subdir: str = "positions") -> None:
    """Per-sequence image-vs-pitch coverage split -> printed table + JSON."""
    rows = []
    for p in sorted((out_dir / positions_subdir).glob("*.parquet")):
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
def run(data_dir: Path, out_dir: Path, seqs: list[str] | None = None,
        positions_subdir: str = "positions") -> list[dict]:
    """Measure every valid sequence with a cached positions parquet -> per-sequence rows."""
    pos_dir = out_dir / positions_subdir
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


# === v7 session V4 step 1: the ceiling on the CURRENT detector + tracker =========================
#: The partitions compared on DEV-20. Order is the 2x2 (detector x tracker) the v6 stack walked.
V7_PARTITIONS = (
    ("A base-det + ByteTrack", "positions_gate"),
    ("B base-det + EIoU", "positions_gate_eiou"),
    ("C v6det + ByteTrack", "positions_gate_v6det"),
    ("D v6det + EIoU  <- v6 tracker", "positions_gate_v6det_eiou"),
)
#: The shipped v6 arm's submission directory (tracker + GTA connector + identity solver).
V7_SHIPPED = Path("outputs/gsr/deleak_v6detdev_clip_e0.3r1w0.5a0.3/predictions/data")


def partition_report(data_dir: Path, out_dir: Path, seqs: list[str], *,
                     positions_subdir: str | None = None,
                     pred_dir: Path | None = None) -> tuple[list[dict], list[list[float]]]:
    """Measure one track partition on ``seqs`` -> (per-sequence rows, pooled fragment profile).

    Exactly one of ``positions_subdir`` (a cached positions parquet directory) or ``pred_dir`` (a
    materialised submission's ``predictions/data``) must be given.
    """
    rows: list[dict] = []
    profile: list[list[float]] = []
    for name in seqs:
        seq_dir = data_dir / name
        df = (submission_positions(pred_dir / f"{name}.json", seq_dir) if pred_dir is not None
              else pd.read_parquet(out_dir / positions_subdir / f"{name}.parquet"))
        pairs, n_gt, n_pred, extras = match_sequence(df, load_gt_ids_by_frame(seq_dir))
        rows.append({"seq": name, **extras, **frag_shares(pairs, n_gt),
                     **counterfactuals(pairs, n_gt, n_pred),
                     "assa_merge_only": assa_merge_only(pairs, n_gt, n_pred)})
        profile.extend(rank_profile(pairs, n_gt))
    return rows, profile


def _measured_dev(pred_root: Path, data_dir: Path, seqs: list[str]) -> dict[str, dict[str, dict]]:
    """Score the shipped arm under the ``loc_assoc`` and full configs -> per-config per-sequence."""
    from eval.gsr_score import EVAL_CONFIGS, gs_hota  # noqa: PLC0415

    out = {}
    for cfg in ("loc_assoc", "gs_hota_full"):
        res = gs_hota(pred_root, data_dir, seq_info={s: 0 for s in seqs},
                      **EVAL_CONFIGS.get(cfg, {}))
        out[cfg] = {"combined": res["combined"], "per_seq": res["per_seq"]}
        print(f"  measured {cfg:14s} AssA {res['combined']['GS-AssA']:.4f} "
              f"HOTA {res['combined']['GS-HOTA']:.4f}")
    return out


def _fit_scale(rows: list[dict], measured: dict[str, dict], cfg: str) -> dict[str, float]:
    """Validate the proxy against a measured per-sequence metric: Spearman, Pearson, scale."""
    d = pd.DataFrame(rows)
    d["measured"] = [measured[s]["GS-AssA"] / 100.0 for s in d["seq"]]
    return {"config": cfg,
            "spearman": float(d["measured"].corr(d["assa_hat"], method="spearman")),
            "pearson": float(d["measured"].corr(d["assa_hat"], method="pearson")),
            "proxy_mean": float(d["assa_hat"].mean()),
            "measured_mean": float(d["measured"].mean()),
            "scale": float(d["measured"].mean() / max(d["assa_hat"].mean(), 1e-9))}


def v7_step1(data_dir: Path, out_dir: Path, seqs: list[str], pred_root: Path) -> dict:
    """The pre-registered V4 step-1 read: connector ceiling, decomposition and the path verdict."""
    parts: dict[str, dict] = {}
    for label, sub in V7_PARTITIONS:
        if not (out_dir / sub).is_dir():
            print(f"  SKIP {label}: {sub} missing")
            continue
        rows, prof = partition_report(data_dir, out_dir, seqs, positions_subdir=sub)
        parts[label] = {"source": sub, "rows": rows, "rank_table": pooled_rank_table(prof)}
        print(f"  {label}: done ({sub})")
    rows, prof = partition_report(data_dir, out_dir, seqs, pred_dir=pred_root / "predictions/data")
    parts["E shipped (D + connector + solver)"] = {
        "source": str(pred_root), "rows": rows, "rank_table": pooled_rank_table(prof)}
    print("  E shipped: done")

    cov = [{"seq": s, **coverage_split(data_dir / s,
                                       pd.read_parquet(out_dir / "positions_gate_v6det"
                                                       / f"{s}.parquet"))} for s in seqs]
    measured = _measured_dev(pred_root, data_dir, seqs)
    scales = {c: _fit_scale(parts["E shipped (D + connector + solver)"]["rows"],
                            measured[c]["per_seq"], c) for c in measured}

    keys = ("assa_hat", "assa_merge_only", "assa_perfect_link", "assa_perfect_det", "frag_per_gt",
            "herfindahl", "dominant_share", "matched_share", "det_recall", "pred_precision",
            "n_pred_tracks")
    print(f"\n{'partition':<34}" + "".join(f"{k[:9]:>11}" for k in keys))
    for label, p in parts.items():
        d = pd.DataFrame(p["rows"])
        p["mean"] = {k: float(d[k].mean()) for k in keys}
        print(f"{label:<34}" + "".join(f"{p['mean'][k]:>11.4f}" for k in keys))
    print("\nfragment profile (mean share of a GT identity per fragment rank):")
    for label, p in parts.items():
        t = p["rank_table"]
        print(f"  {label:<34}" + " ".join(f"{t[k]:.3f}" for k in
                                          ("rank1", "rank2", "rank3", "rank4", "unmatched")))
    print("\ncoverage (v6det gated positions, DEV-20 GT-row weighted):")
    cdf = pd.DataFrame(cov)
    w = cdf["n_gt"]
    pooled_cov = {c: float(np.average(cdf[c], weights=w)) for c in
                  ("img_recall", "pitch_recall", "img_ok_pitch_lost", "net_img_minus_pitch",
                   "gt_no_pitch")}
    pooled_cov["calib_frame_rate"] = float(cdf["calib_frame_rate"].mean())
    for k, v in pooled_cov.items():
        print(f"  {k:22s} {v:.4f}")
    print("\nproxy validation on the shipped partition:")
    for c, s in scales.items():
        print(f"  {c:14s} spearman {s['spearman']:+.3f} pearson {s['pearson']:+.3f} "
              f"scale {s['scale']:.3f} (proxy {s['proxy_mean']:.4f} -> "
              f"measured {s['measured_mean']:.4f})")

    cur = parts["E shipped (D + connector + solver)"]["mean"]
    ceil_src = parts.get("D v6det + EIoU  <- v6 tracker", parts["E shipped (D + connector + solver)"])
    verdict = {
        "current_proxy_assa": cur["assa_hat"],
        "ceiling_proxy_assa": ceil_src["mean"]["assa_perfect_link"],
        "ceiling_merge_only_proxy": ceil_src["mean"]["assa_merge_only"],
        "headroom_proxy": ceil_src["mean"]["assa_perfect_link"] - cur["assa_hat"],
        "headroom_merge_only_proxy": ceil_src["mean"]["assa_merge_only"] - cur["assa_hat"],
    }
    for c, s in scales.items():
        verdict[f"headroom_merge_only_scaled_{c}"] = (verdict["headroom_merge_only_proxy"]
                                                      * s["scale"] * 100.0)
        verdict[f"headroom_scaled_{c}"] = verdict["headroom_proxy"] * s["scale"] * 100.0
        verdict[f"current_measured_{c}"] = measured[c]["combined"]["GS-AssA"]
        verdict[f"ceiling_scaled_{c}"] = (measured[c]["combined"]["GS-AssA"]
                                          + verdict["headroom_proxy"] * s["scale"] * 100.0)
    h = verdict["headroom_scaled_loc_assoc"]
    verdict["path"] = ("EIoU retune (CPU)" if h < 5.0 else "CAMELTrack")  # noqa: PLR2004
    print(f"\nCEILING: proxy {verdict['ceiling_proxy_assa']:.4f} (merge-only "
          f"{verdict['ceiling_merge_only_proxy']:.4f}) vs current proxy "
          f"{verdict['current_proxy_assa']:.4f} -> headroom {verdict['headroom_proxy']:.4f} "
          f"({h:+.2f} AssA on the loc_assoc scale; merge-only "
          f"{verdict['headroom_merge_only_scaled_loc_assoc']:+.2f})")
    print(f"PATH (pre-registered thresholds, <+5 / +5..+12 / >+12): {verdict['path']}")
    payload = {"seqs": seqs, "partitions": parts, "coverage": cov, "coverage_pooled": pooled_cov,
               "measured": measured, "proxy_scales": scales, "verdict": verdict}
    dest = RESULTS / "gsr_assoc_diag_v7.json"
    dest.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"wrote {dest}")
    return payload


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
    # merge-only oracle: two clean fragments merge (1.0); a contaminated track cannot be split.
    assert abs(assa_merge_only({(1, 100): 5, (2, 100): 5}, {100: 10}, {1: 5, 2: 5}) - 1.0) < 1e-9
    # track 1 holds 6 rows of GT 100 and 4 of GT 200; merged wholesale into 100 -> 6/(10+10-6).
    mo = assa_merge_only({(1, 100): 6, (1, 200): 4}, {100: 10, 200: 10}, {1: 10})
    assert abs(mo - (6 * (6 / 14) + 4 * (4 / 16)) / 10) < 1e-9, mo
    # fragment profile: two equal halves of one identity, plus one identity nobody covered fully.
    prof = rank_profile({(1, 100): 5, (2, 100): 3}, {100: 10})
    assert prof == [[0.5, 0.3]], prof
    t = pooled_rank_table(prof)
    assert abs(t["rank1"] - 0.5) < 1e-9 and abs(t["rank2"] - 0.3) < 1e-9, t
    assert abs(t["unmatched"] - 0.2) < 1e-9 and t["rank3"] == 0.0, t
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
    ap.add_argument("--v7", action="store_true",
                    help="v7 V4 step 1: the ceiling on the CURRENT (v6det + EIoU) DEV-20 stack")
    ap.add_argument("--positions", default="positions", help="cached positions source (--coverage)")
    ap.add_argument("--pred-root", type=Path, default=V7_SHIPPED.parent.parent,
                    help="the shipped arm directory holding predictions/data (--v7)")
    ap.add_argument("--seqs", default=None, help="comma-separated sequence names")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return
    if args.coverage:
        coverage(args.data_dir, args.out_dir, args.positions)
    if args.merge:
        merge()
    if args.diag:
        diag(args.data_dir, args.out_dir)
    if args.v7:
        if args.seqs:
            seqs = args.seqs.split(",")
        else:
            from eval.gsr_identity import split_sequences  # noqa: PLC0415

            seqs = split_sequences(args.data_dir, args.out_dir)[0]
        print(f"v7 V4 step 1: {len(seqs)} DEV sequences")
        v7_step1(args.data_dir, args.out_dir, seqs, args.pred_root)


if __name__ == "__main__":
    main()
