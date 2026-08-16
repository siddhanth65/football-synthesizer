"""v10-W7: association kill-experiments on the arm-D substrate (CPU, cached artifacts only).

Three measure-only stages, all registered in ``results/gsr_v10_w7_registered.json`` before any
number below existed:

``--oracle``    Task A. Rebuild per-row GT ownership of arm D's OWN submission with the v9 factory
                machinery, grant every same-identity tracklet merge, and score the result end to end
                (flags ON and OFF) against the untouched arm D. The answer is the merge-only
                headroom LEFT on the current substrate -- the 57.32-era number is v9-substrate stale.
``--splitter``  Task B. A learned per-row change-point model (HistGradientBoosting) over the v9
                factory train split: does it cut contaminated tracklets where the identity actually
                changes, at a precision the shipped DBSCAN splitter never reached?
``--forensic``  Task C. Why SNGS-021 is the only DEV clip arm D hurts.

Ground truth is read only to label, to audit and to score; no arm consumes it.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from tools.gsr_v10_w5 import DEV20

logger = logging.getLogger("gsr_v10_w7")

#: Arm-D lineage (v10-W5).
ARM_D_PRED = Path("outputs/gsr/deleak_v10w5_D/predictions/data")
ARM_D_POSITIONS = "positions_v10w5D_v6det"
#: Audited rows a predicted track needs before its dominant GT identity is trusted (v9 factory).
MIN_AUDITED = 5
#: v9 factory dataset (train split) -- Task B's substrate.
V9_ASSOC = Path("outputs/gsr/v9_assoc/v1")


# === Task A: the merge-only association oracle ====================================================
def submission_rows(pred_path: Path, seq_dir: Path) -> pd.DataFrame:
    """One submission JSON as a positions-shaped table (pure-ish, reads the sequence's image map).

    The factory's labeller works on positions tables (uncentred metres), while a submission carries
    centred ``bbox_pitch`` metres and GSR ``image_id`` strings, so both are undone here rather than
    re-deriving ownership from a parallel artifact that the submission may no longer agree with.

    Args:
        pred_path: ``<seq>.json`` of a predictions/data directory.
        seq_dir: GSR sequence folder (supplies the frame index <-> ``image_id`` map).

    Returns:
        Frame with ``frame, track_id, role, team, pitch_x, pitch_y, idx`` (``idx`` = position of the
        row in the submission list, so a merge can be written back in place).
    """
    from eval.gsr_score import CENTRE_SHIFT_X, CENTRE_SHIFT_Y, load_image_id_map  # noqa: PLC0415

    frame_of = {v: k for k, v in load_image_id_map(seq_dir).items()}
    preds = json.loads(pred_path.read_text(encoding="utf-8"))["predictions"]
    recs = []
    for i, p in enumerate(preds):
        bp = p.get("bbox_pitch") or {}
        attrs = p.get("attributes") or {}
        recs.append({
            "idx": i, "frame": frame_of.get(p["image_id"], -1), "track_id": int(p["track_id"]),
            "role": attrs.get("role", "other"), "team": attrs.get("team", ""),
            "pitch_x": float(bp.get("x_bottom_middle", np.nan)) + CENTRE_SHIFT_X,
            "pitch_y": float(bp.get("y_bottom_middle", np.nan)) + CENTRE_SHIFT_Y,
        })
    return pd.DataFrame(recs)


def ownership(rows: pd.DataFrame, min_audited: int = MIN_AUDITED) -> pd.DataFrame:
    """Per predicted track: dominant GT identity, audited rows and purity (pure).

    Args:
        rows: Output of :func:`tools.gsr_v9_factory.label_rows` (carries ``gt_id``).
        min_audited: Audited rows required before the dominant identity is trusted.

    Returns:
        Frame indexed by row with ``track_id, gt_id, n_rows, n_audited, purity``; ``gt_id`` is -1
        when the track is not trusted.
    """
    recs = []
    for tid, g in rows.groupby("track_id"):
        aud = g[g["gt_id"] >= 0]["gt_id"]
        votes = Counter(int(v) for v in aud)
        gid, dom = votes.most_common(1)[0] if votes else (-1, 0)
        recs.append({"track_id": int(tid), "gt_id": int(gid) if len(aud) >= min_audited else -1,
                     "n_rows": int(len(g)), "n_audited": int(len(aud)),
                     "purity": float(dom / len(aud)) if len(aud) else float("nan")})
    return pd.DataFrame(recs)


def merge_map(own: pd.DataFrame, frames: dict[int, set[int]] | None = None,
              base_id: int = 100000) -> tuple[dict[int, int], dict]:
    """``({old track id: new track id}, stats)`` granting every same-identity merge (pure).

    Only tracks with a trusted dominant GT identity move, and they only ever move TOGETHER: no track
    is split, no row is added, moved or deleted, no attribute is set from ground truth.

    Temporally OVERLAPPING same-identity tracks are not merged (registration amendment 1): the
    official evaluator refuses two rows of one id in one timestep, and a connector cannot link
    overlapping tracklets anyway (the v9 factory's candidate rule requires ``A`` to end before ``B``
    starts). Within an identity the longest track anchors the chain and every further track joins
    only if it shares no frame with the chain so far; the rest keep their own ids.

    Args:
        own: Output of :func:`ownership`.
        frames: ``{track_id: set of frames}``; when None no overlap check is applied.
        base_id: Offset of the synthesised merged ids (kept clear of the writer's own ids).
    """
    new: dict[int, int] = {}
    groups = blocked = merged_tracks = 0
    for gid, g in own[own["gt_id"] >= 0].groupby("gt_id"):
        order = g.sort_values(["n_rows", "track_id"], ascending=[False, True])
        chain: set[int] = set()
        members = []
        for t in (int(x) for x in order["track_id"]):
            f = (frames or {}).get(t, set())
            if members and frames is not None and (chain & f):
                blocked += 1
                continue
            members.append(t)
            chain |= f
        if len(members) < 2:
            continue
        groups += 1
        merged_tracks += len(members)
        for t in members:
            new[t] = base_id + int(gid)
    return new, {"tracks": int(len(own)), "trusted": int((own["gt_id"] >= 0).sum()),
                 "groups_merged": groups, "tracks_merged": merged_tracks,
                 "merges_granted": merged_tracks - groups, "merges_blocked_overlap": blocked}


def apply_merges(src: Path, dest: Path, data_dir: Path, seqs: list[str], *,
                 mode: str = "merge") -> dict:
    """Write the merge-only oracle submission; returns per-sequence ownership/merge stats.

    Args:
        src: ``predictions/data`` of the arm being priced.
        dest: Destination ``predictions/data``.
        data_dir: GSR dataset root.
        seqs: Sequences.
        mode: ``merge`` -- the registered oracle (frame-disjoint merges only); ``merge_dedup`` --
            amendment 2, every same-identity track merged with the duplicate rows of a timestep
            resolved by keeping the highest-confidence one; ``dedup`` -- amendment 2's second leg,
            the same row removal with NO id change, which isolates duplicate-track removal from
            linking.
    """
    from tools.gsr_v9_factory import label_rows  # noqa: PLC0415

    dest.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}
    for seq in seqs:
        payload = json.loads((src / f"{seq}.json").read_text(encoding="utf-8"))
        preds = payload["predictions"]
        rows = submission_rows(src / f"{seq}.json", data_dir / seq)
        labelled = label_rows(rows, data_dir / seq)
        own = ownership(labelled)
        frames = {int(t): set(g["frame"].astype(int)) for t, g in rows.groupby("track_id")}
        new, st = merge_map(own, frames if mode == "merge" else None)
        owner = dict(zip(own["track_id"].astype(int), own["gt_id"].astype(int)))
        if mode == "perfect":  # row-level: every audited row joins ITS OWN GT identity
            row_gt = dict(zip(labelled["idx"], labelled["gt_id"]))
            new = {}
            keyed = rows.assign(
                nid=[100000 + int(row_gt.get(i, -1)) if row_gt.get(i, -1) >= 0 else int(t)
                     for i, t in zip(rows["idx"], rows["track_id"])])
            keyed = keyed.assign(gid=keyed["nid"],
                                 conf=[float(preds[i].get("confidence") or 0.0)
                                       for i in rows["idx"]])
        else:
            keyed = rows.assign(
                nid=[new.get(int(t), int(t)) for t in rows["track_id"]],
                gid=[owner.get(int(t), -1) if owner.get(int(t), -1) >= 0 else -int(t) - 1
                     for t in rows["track_id"]],
                conf=[float(preds[i].get("confidence") or 0.0) for i in rows["idx"]])
        dup = keyed.groupby(["gid", "frame"]).size()
        st["mode"] = mode
        st["stacked_rows"] = int((dup[dup > 1] - 1).sum())
        st["rows"] = int(len(rows))
        st["ids_out"] = int(keyed["nid"].nunique())
        st["purity_rowweighted"] = float(
            (own["purity"] * own["n_audited"]).sum() / max(own["n_audited"].sum(), 1))
        drop: set[int] = set()
        if mode != "merge" and st["stacked_rows"]:
            best = keyed.sort_values("conf", ascending=False).groupby(["gid", "frame"])["idx"]
            drop = set(keyed["idx"]) - set(best.first())
        st["dropped_rows"] = len(drop)
        if mode != "dedup":
            for i, nid in zip(keyed["idx"], keyed["nid"]):
                preds[int(i)]["track_id"] = int(nid)
        out = [p for i, p in enumerate(preds) if i not in drop]
        (dest / f"{seq}.json").write_text(json.dumps({"predictions": out}), encoding="utf-8")
        stats[seq] = st
        logger.info("%s: %d tracks (%d trusted) -> %d merges over %d identities, stacked %d rows, "
                    "dropped %d", seq, st["tracks"], st["trusted"], st["merges_granted"],
                    st["groups_merged"], st["stacked_rows"], st["dropped_rows"])
    return stats


def run_oracle(data_dir: Path, out_dir: Path, seqs: list[str], *, mode: str = "merge") -> dict:
    """Task A: build an oracle submission (see :func:`apply_merges`) and score it against arm D."""
    from tools.gsr_v9_w7 import score_pair  # noqa: PLC0415

    work = out_dir / "v10_w7"
    name = f"oracle_{mode}_D"
    dest = work / name / "predictions" / "data"
    stats = apply_merges(ARM_D_PRED, dest, data_dir, seqs, mode=mode)
    scored = score_pair(ARM_D_PRED, dest, data_dir, work / "pair" / name, seqs)
    tot = {k: int(sum(v[k] for v in stats.values()))
           for k in ("tracks", "trusted", "groups_merged", "tracks_merged", "merges_granted",
                     "merges_blocked_overlap", "stacked_rows", "dropped_rows", "rows",
                     "ids_out")}
    head = {f: scored[f]["arm"]["combined"]["GS-HOTA"] - scored[f]["control"]["combined"]["GS-HOTA"]
            for f in ("on", "off")}
    return {"per_seq": stats, "totals": tot, "score": scored, "headroom_gs_hota": head,
            "verdict": "OPEN" if head["on"] >= 1.5 else "CLOSED"}


# === Task B: the learned tracklet splitter ========================================================
#: Rows on either side of a boundary that count as positive.
LABEL_HALFWIDTH = 2
#: Rows averaged for the running appearance centroid on each side of a candidate cut.
CENTROID_K = 10
#: Audited rows an ownership run needs on BOTH sides of a boundary for it to be a SUBSTANTIAL
#: handover (1 s at 25 fps) -- the exploratory arm's label, not the registered one.
MIN_RUN = 25
#: Non-maximum suppression window and match tolerance, in frames (25 fps).
NMS_FRAMES = 25
MATCH_TOL_FRAMES = 12
FEATURES = ("app_prev", "app_next", "app_split", "app_gain", "jump_m", "speed_mps", "speed_prev",
            "speed_next", "turn_cos", "gap_frames", "img_jump_px", "conf", "conf_prev",
            "calib_error_m", "role_change", "team_change", "pos_frac", "len_rows")


def _embeddings(seq: str, root: Path = V9_ASSOC) -> dict[tuple[int, int], np.ndarray]:
    """``{(track_id, frame): L2-normalised embedding}`` from a per-detection CLIP cache."""
    with np.load(root / "detembed" / f"{seq}.npz") as z:
        tid, fr, emb = z["track_ids"], z["frames"], z["embeddings"]
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    return {(int(t), int(f)): e for t, f, e in zip(tid, fr, emb)}


def tracklet_features(g: pd.DataFrame, embs: dict[tuple[int, int], np.ndarray]) -> pd.DataFrame:
    """Per-row change-point features for one tracklet, frame-sorted (pure).

    A row's features describe the BOUNDARY just before it: appearance disagreement between the
    preceding and following windows, the metric jump, the turn, and the local attribute changes.
    """
    g = g.sort_values("frame", ignore_index=True)
    n = len(g)
    fr = g["frame"].to_numpy(dtype=float)
    x, y = g["pitch_x"].to_numpy(dtype=float), g["pitch_y"].to_numpy(dtype=float)
    ix, iy = g["image_x"].to_numpy(dtype=float), g["image_y"].to_numpy(dtype=float)
    tid = int(g["track_id"].iloc[0])
    e = np.array([embs.get((tid, int(f)), np.full(0, 0.0)) for f in fr], dtype=object)
    dim = next((len(v) for v in e if len(v)), 0)
    emb = np.full((n, dim), np.nan) if dim else np.zeros((n, 1))
    for i, v in enumerate(e):
        if dim and len(v):
            emb[i] = v

    def centroid(lo: int, hi: int) -> np.ndarray:
        block = emb[max(lo, 0):hi]
        block = block[np.isfinite(block).all(axis=1)] if len(block) else block
        if not len(block):
            return np.full(emb.shape[1], np.nan)
        c = block.mean(axis=0)
        return c / max(float(np.linalg.norm(c)), 1e-9)

    gap = np.diff(fr, prepend=fr[0])
    dx, dy = np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])
    jump = np.hypot(dx, dy)
    dt = np.maximum(gap, 1.0) / 25.0
    speed = jump / dt
    img_jump = np.hypot(np.diff(ix, prepend=ix[0]), np.diff(iy, prepend=iy[0]))
    role = g["role"].to_numpy()
    team = g["team"].to_numpy()
    rows = []
    for i in range(n):
        prev_c = centroid(i - CENTROID_K, i)
        next_c = centroid(i, i + CENTROID_K)
        cur = emb[i]
        d_prev = float(1.0 - prev_c @ cur) if np.isfinite(prev_c).all() and np.isfinite(cur).all() \
            else np.nan
        d_next = float(1.0 - next_c @ cur) if np.isfinite(next_c).all() and np.isfinite(cur).all() \
            else np.nan
        d_split = float(1.0 - prev_c @ next_c) if (np.isfinite(prev_c).all()
                                                   and np.isfinite(next_c).all()) else np.nan
        sp_prev = float(np.nanmean(speed[max(i - 5, 1):i])) if i > 1 else np.nan
        sp_next = float(np.nanmean(speed[i + 1:i + 6])) if i + 1 < n else np.nan
        v_in = np.array([dx[i - 1], dy[i - 1]]) if i >= 1 else np.array([np.nan, np.nan])
        v_out = np.array([dx[i], dy[i]])
        nn = np.linalg.norm(v_in) * np.linalg.norm(v_out)
        rows.append({
            "app_prev": d_prev, "app_next": d_next, "app_split": d_split,
            "app_gain": (d_split - min(v for v in (d_prev, d_next) if np.isfinite(v))
                         if np.isfinite(d_split) and (np.isfinite(d_prev) or np.isfinite(d_next))
                         else np.nan),
            "jump_m": float(jump[i]), "speed_mps": float(speed[i]),
            "speed_prev": sp_prev, "speed_next": sp_next,
            "turn_cos": float(v_in @ v_out / nn) if nn > 1e-9 else np.nan,
            "gap_frames": float(gap[i]), "img_jump_px": float(img_jump[i]),
            "conf": float(g["conf"].iloc[i]),
            "conf_prev": float(g["conf"].iloc[i - 1]) if i else np.nan,
            "calib_error_m": float(g["calib_error_m"].iloc[i]),
            "role_change": float(i > 0 and role[i] != role[i - 1]),
            "team_change": float(i > 0 and team[i] != team[i - 1]),
            "pos_frac": i / max(n - 1, 1), "len_rows": float(n),
        })
    out = pd.DataFrame(rows)
    out["frame"] = fr.astype(int)
    out["track_id"] = tid
    out["gt_id"] = g["gt_id"].to_numpy()
    return out


def true_cuts(g: pd.DataFrame, min_run: int = 0) -> list[int]:
    """Frames at which one tracklet's audited GT identity changes (pure).

    Unaudited rows (``gt_id`` -1) are transparent: the boundary is placed at the first audited row
    of the new identity, which is where a cut has to land to separate the two owners.

    Args:
        g: One tracklet's rows (needs ``frame`` and ``gt_id``).
        min_run: When > 0, only boundaries whose two adjacent ownership runs are BOTH at least this
            many audited rows count -- i.e. real handovers rather than one-row matcher flicker.
    """
    aud = g.sort_values("frame")
    aud = aud[aud["gt_id"] >= 0]
    runs: list[list] = []
    for f, gid in zip(aud["frame"], aud["gt_id"]):
        if runs and runs[-1][0] == int(gid):
            runs[-1][2] += 1
        else:
            runs.append([int(gid), int(f), 1])
    return [r[1] for i, r in enumerate(runs)
            if i and (min_run <= 0 or (r[2] >= min_run and runs[i - 1][2] >= min_run))]


def build_splitter_table(seqs: list[str], root: Path = V9_ASSOC) -> pd.DataFrame:
    """Feature/label table over a set of factory sequences (one row per detection row)."""
    out = []
    for seq in seqs:
        tr = pd.read_parquet(root / "tracklets" / f"{seq}.parquet")
        embs = _embeddings(seq, root)
        for tid, g in tr.groupby("track_id"):
            if len(g) < 2 * LABEL_HALFWIDTH + 2:
                continue
            feat = tracklet_features(g, embs)
            cuts = true_cuts(g)
            fr = feat["frame"].to_numpy()
            for lcol, ccol, cs in (("label", "cut_frames", cuts),
                                   ("label_sub", "cut_frames_sub", true_cuts(g, MIN_RUN))):
                lab = np.zeros(len(feat), dtype=int)
                for c in cs:
                    i = int(np.argmin(np.abs(fr - c)))
                    lab[max(i - LABEL_HALFWIDTH, 0):i + LABEL_HALFWIDTH + 1] = 1
                feat[lcol] = lab
                feat[ccol] = [tuple(cs)] * len(feat)
            feat["seq"] = seq
            feat["uid"] = f"{seq}:{int(tid)}"
            feat["purity"] = float(g["purity"].iloc[0]) if np.isfinite(g["purity"].iloc[0]) else 1.0
            feat["n_cuts"] = len(cuts)
            out.append(feat)
        logger.info("%s: %d tracklets featurised", seq, tr["track_id"].nunique())
    return pd.concat(out, ignore_index=True)


def extract_cuts(feat: pd.DataFrame, prob: np.ndarray, thr: float) -> dict[str, list[int]]:
    """``{uid: [cut frames]}`` after thresholding and 1 s non-maximum suppression (pure).

    ``uid`` is the sequence-unique tracklet key: track ids repeat across sequences, so anything
    keyed on ``track_id`` alone silently pools different clips' tracklets.
    """
    out: dict[str, list[int]] = {}
    df = feat[["uid", "frame"]].copy()
    df["p"] = prob
    for tid, g in df[df["p"] >= thr].groupby("uid"):
        kept: list[int] = []
        for _, r in g.sort_values("p", ascending=False).iterrows():
            f = int(r["frame"])
            if all(abs(f - k) > NMS_FRAMES for k in kept):
                kept.append(f)
        out[str(tid)] = sorted(kept)
    return out


def score_cuts(feat: pd.DataFrame, cuts: dict[str, list[int]], *, impure: float = 0.80,
               pure: float = 0.99, truth_col: str = "cut_frames") -> dict:
    """Cut-point precision/recall on impure tracklets and false-cut rate on pure ones (pure)."""
    meta = feat.groupby("uid").agg(purity=("purity", "first"),
                                   cut_frames=(truth_col, "first"))
    tp = fp = n_true = 0
    pure_n = pure_hit = 0
    all_cuts = all_hit = 0
    for tid, r in meta.iterrows():
        pred = cuts.get(str(tid), [])
        truth = list(r["cut_frames"])
        all_cuts += len(pred)
        all_hit += sum(any(abs(t - f) <= MATCH_TOL_FRAMES for t in truth) for f in pred)
        if r["purity"] < impure:
            n_true += len(truth)
            free = list(truth)
            for f in pred:
                hit = [t for t in free if abs(t - f) <= MATCH_TOL_FRAMES]
                if hit:
                    free.remove(min(hit, key=lambda t: abs(t - f)))
                    tp += 1
                else:
                    fp += 1
        if r["purity"] >= pure:
            pure_n += 1
            pure_hit += int(bool(pred))
    return {"tp": tp, "fp": fp, "true_cuts": n_true,
            "precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
            "recall": float(tp / n_true) if n_true else float("nan"),
            "pure_tracklets": pure_n, "pure_with_cut": pure_hit,
            "false_cut_rate": float(pure_hit / pure_n) if pure_n else float("nan"),
            "cuts_all": all_cuts, "cuts_all_on_a_true_change": all_hit,
            "precision_all_tracklets": float(all_hit / all_cuts) if all_cuts else float("nan")}


def run_splitter(root: Path = V9_ASSOC) -> dict:
    """Task B: train the change-point model on the 45 train clips, read the 12 held-out once."""
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415
    from sklearn.metrics import average_precision_score  # noqa: PLC0415

    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    val = list(man["split"]["val_slice"])
    train = list(man["split"]["train_slice"])
    logger.info("train %d clips, held-out %d clips", len(train), len(val))
    tr = build_splitter_table(train, root)
    va = build_splitter_table(val, root)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=0)
    model.fit(tr[list(FEATURES)], tr["label"])
    p_tr = model.predict_proba(tr[list(FEATURES)])[:, 1]
    p_va = model.predict_proba(va[list(FEATURES)])[:, 1]
    grid = [round(t, 3) for t in np.arange(0.05, 0.96, 0.05)]
    train_curve = {t: score_cuts(tr, extract_cuts(tr, p_tr, t)) for t in grid}
    # registered operating point: smallest threshold whose TRAIN false-cut rate on pure is <= 1%
    ok = [t for t in grid if train_curve[t]["false_cut_rate"] <= 0.01]
    thr = min(ok) if ok else max(grid)
    val_curve = {t: score_cuts(va, extract_cuts(va, p_va, t)) for t in grid}
    # secondary, reported alongside: a purity >= 0.99 tracklet CAN still hold a true change point,
    # so the registered false-cut rate over-charges. purity == 1.0 tracklets hold none by definition.
    strict = {t: score_cuts(va, extract_cuts(va, p_va, t), pure=1.0)["false_cut_rate"]
              for t in grid}
    imp = va[va["purity"] < 0.80]["uid"].nunique()
    return {
        "split": {"train": train, "held_out": val},
        "rows": {"train": int(len(tr)), "held_out": int(len(va)),
                 "train_pos": int(tr["label"].sum()), "held_out_pos": int(va["label"].sum())},
        "populations": {"held_out_impure_tracklets": int(imp),
                        "held_out_pure_tracklets": int(va[va["purity"] >= 0.99]
                                                       ["uid"].nunique()),
                        "held_out_true_cuts": int(val_curve[grid[0]]["true_cuts"])},
        "operating_point": {"threshold": thr, "rule": "smallest train threshold with pure "
                            "false-cut rate <= 1%"},
        "train_curve": train_curve, "held_out_curve": val_curve,
        "held_out_false_cut_rate_strict_pure": strict,
        "held_out_row_average_precision": float(average_precision_score(va["label"], p_va)),
        "held_out_row_positive_rate": float(va["label"].mean()),
        "held_out_at_operating_point": val_curve[thr],
        "feature_importance_proxy": _permutation_gain(model, va, p_va),
        "label_anatomy": _label_anatomy(va),
        "exploratory_substantial": _substantial_arm(tr, va, grid),
    }


def _label_anatomy(va: pd.DataFrame) -> dict:
    """How many held-out change points are real handovers rather than one-row matcher flicker."""
    meta = va.groupby("uid").agg(purity=("purity", "first"),
                                      allc=("cut_frames", "first"), sub=("cut_frames_sub", "first"))
    imp = meta[meta["purity"] < 0.80]
    return {"tracklets": int(len(meta)), "change_points_all": int(meta["allc"].map(len).sum()),
            "change_points_substantial": int(meta["sub"].map(len).sum()),
            "impure_change_points_all": int(imp["allc"].map(len).sum()),
            "impure_change_points_substantial": int(imp["sub"].map(len).sum()),
            "min_run_frames": MIN_RUN}


def _substantial_arm(tr: pd.DataFrame, va: pd.DataFrame, grid: list[float]) -> dict:
    """Post-registration exploratory arm: train and score against SUBSTANTIAL handovers only.

    Flicker boundaries (a one-row ownership blip) are not cuttable by any splitter and dominate the
    registered label set; this arm gives the reviewer's proposal its strongest fair version.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415
    from sklearn.metrics import average_precision_score  # noqa: PLC0415

    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=0)
    model.fit(tr[list(FEATURES)], tr["label_sub"])
    p = model.predict_proba(va[list(FEATURES)])[:, 1]
    curve = {t: score_cuts(va, extract_cuts(va, p, t), truth_col="cut_frames_sub") for t in grid}
    return {"held_out_curve": curve, "train_positive_rate": float(tr["label_sub"].mean()),
            "held_out_row_average_precision": float(average_precision_score(va["label_sub"], p)),
            "best_precision_at_recall_30": max(
                (c["precision"] for c in curve.values() if c["recall"] >= 0.30), default=None)}


def flicker_anatomy(seqs: list[str], root: Path = V9_ASSOC,
                    data_dir: Path = Path("data/soccernet/gamestate-2024")) -> dict:
    """Is tracklet 'contamination' a real identity handover, or the LABELLER flickering? (measure).

    For every audited row that disagrees with its tracklet's dominant identity, this reports how
    long the disagreement lasts and how much closer the claimed GT actually is than the dominant
    one. A one-row disagreement at a 0.2 m margin in a crowd is an instrument artefact, not
    something a splitter could ever cut.
    """
    from eval.gsr_score import CENTRE_SHIFT_X, CENTRE_SHIFT_Y  # noqa: PLC0415
    from generator.track_relink import load_gt_ids_by_frame  # noqa: PLC0415

    runs: list[int] = []
    margins: list[float] = []
    n_rows = n_impure = 0
    for seq in seqs:
        tr = pd.read_parquet(root / "tracklets" / f"{seq}.parquet")
        gt = load_gt_ids_by_frame(data_dir / seq)
        for _tid, g in tr.groupby("track_id"):
            g = g.sort_values("frame")
            aud = g[g["gt_id"] >= 0]
            if not len(aud):
                continue
            dom = int(Counter(int(v) for v in aud["gt_id"]).most_common(1)[0][0])
            n_rows += len(aud)
            cur = 0
            for f, gid, px, py in zip(aud["frame"], aud["gt_id"], aud["pitch_x"], aud["pitch_y"]):
                if int(gid) == dom:
                    if cur:
                        runs.append(cur)
                    cur = 0
                    continue
                cur += 1
                n_impure += 1
                here = {int(t): (x, y) for x, y, t in gt.get(int(f), [])}
                cx, cy = float(px) - CENTRE_SHIFT_X, float(py) - CENTRE_SHIFT_Y
                if int(gid) in here and dom in here:
                    d1 = float(np.hypot(here[int(gid)][0] - cx, here[int(gid)][1] - cy))
                    d2 = float(np.hypot(here[dom][0] - cx, here[dom][1] - cy))
                    margins.append(d2 - d1)
            if cur:
                runs.append(cur)
    r = np.array(runs) if runs else np.zeros(0)
    m = np.array(margins) if margins else np.zeros(0)
    return {"sequences": len(seqs), "audited_rows": n_rows, "impure_rows": n_impure,
            "impurity_rate": float(n_impure / max(n_rows, 1)),
            "foreign_runs": int(len(r)),
            "run_len_median": float(np.median(r)) if len(r) else float("nan"),
            "run_len_p90": float(np.percentile(r, 90)) if len(r) else float("nan"),
            "share_runs_le_5_rows": float((r <= 5).mean()) if len(r) else float("nan"),
            "share_runs_le_25_rows": float((r <= 25).mean()) if len(r) else float("nan"),
            "margin_m_median": float(np.median(m)) if len(m) else float("nan"),
            "share_margin_lt_1m": float((m < 1.0).mean()) if len(m) else float("nan"),
            "share_dominant_absent": float(1.0 - len(m) / max(n_impure, 1))}


def _permutation_gain(model, va: pd.DataFrame, base: np.ndarray) -> dict[str, float]:
    """Mean absolute probability shift when one feature is shuffled (cheap importance proxy)."""
    rng = np.random.default_rng(0)
    x = va[list(FEATURES)].copy()
    out = {}
    for f in FEATURES:
        keep = x[f].to_numpy().copy()
        x[f] = rng.permutation(keep)
        out[f] = float(np.mean(np.abs(model.predict_proba(x)[:, 1] - base)))
        x[f] = keep
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# === Task C: the SNGS-021 forensic ================================================================
def forensic(seq: str, data_dir: Path, out_dir: Path, cache_dir: Path) -> dict:
    """Per-frame ctrl-vs-D geometry autopsy of one clip (measure-only, cached artifacts).

    Reproduces the W5 arm-D stages on this clip (gate pick -> temporal re-selection -> batch
    refinement), measures each stage's GT-anchored displacement per frame, and reports where the
    batch step moved the mapping the wrong way, together with the clip anomalies that could explain
    it (zoom, correspondence count, candidate-pool spread, shot cuts).
    """
    from generator.calibrate import apply_homography, estimate_homography  # noqa: PLC0415
    from generator.camera_track import CHECK_GRID, grid_shift_m  # noqa: PLC0415
    from tools.gsr_v10_w5 import (  # noqa: PLC0415
        batch_refine, foot_points_by_frame, kernel_loss, load_seq, reselect,
    )
    from tools.gsr_w2_calibswap import gt_people  # noqa: PLC0415

    df, by_frame, homs, err, n = load_seq(seq, out_dir, cache_dir)
    feet = foot_points_by_frame(df)
    picks, _st_s = reselect(homs, by_frame, feet, n, err)
    dpost, st_b = batch_refine(picks, by_frame, feet, n)
    gt = gt_people(data_dir / seq)
    rows = []
    prev_fit = None
    for fr in sorted(homs):
        g = gt.get(fr)
        if g is None or len(g) < 6:
            continue
        h_gt = estimate_homography(g[:, :2], g[:, 2:4])
        if h_gt is None or not np.isfinite(h_gt).all():
            continue
        ref = apply_homography(h_gt, g[:, :2])
        cut = grid_shift_m(prev_fit, h_gt) if prev_fit is not None else float("nan")
        prev_fit = h_gt
        cands = by_frame.get(fr) or []
        sel = next((i for i, c in enumerate(cands) if c.homography is homs[fr]), -1)
        rec = {"frame": fr, "n_gt": len(g), "pool": len(cands), "sel_index": sel,
               "sel_err_m": float(cands[sel].error_m) if sel >= 0 else float("nan"),
               "sel_n_points": int(cands[sel].n_points) if sel >= 0 else -1,
               "gt_cut_shift_m": cut,
               "zoom_m_per_px": float(np.median(np.linalg.norm(
                   np.diff(apply_homography(h_gt, CHECK_GRID), axis=0), axis=1))),
               "shift_reselect_m": grid_shift_m(homs[fr], picks[fr]),
               "shift_batch_m": grid_shift_m(picks[fr], dpost[fr])}
        for name, h in (("ctrl", homs[fr]), ("resel", picks[fr]), ("D", dpost[fr])):
            d = np.linalg.norm(apply_homography(h, g[:, :2]) - ref, axis=1)
            rec[f"{name}_med_m"] = float(np.median(d))
            rec[f"{name}_kloss"] = kernel_loss(d)
        rows.append(rec)
    t = pd.DataFrame(rows)
    dest = out_dir / "v10_w7"
    dest.mkdir(parents=True, exist_ok=True)
    t.to_csv(dest / f"forensic_{seq}.csv", index=False, encoding="utf-8")
    worse = t[t["D_med_m"] > t["ctrl_med_m"] + 0.05]
    better = t[t["D_med_m"] < t["ctrl_med_m"] - 0.05]
    return {
        "seq": seq, "frames_scored": int(len(t)),
        "median_m": {k: float(t[f"{k}_med_m"].median()) for k in ("ctrl", "resel", "D")},
        "mean_m": {k: float(t[f"{k}_med_m"].mean()) for k in ("ctrl", "resel", "D")},
        "kloss": {k: float(t[f"{k}_kloss"].mean()) for k in ("ctrl", "resel", "D")},
        "frames_worse": int(len(worse)), "frames_better": int(len(better)),
        "worse_share": float(len(worse) / max(len(t), 1)),
        "damage_m": float((worse["D_med_m"] - worse["ctrl_med_m"]).sum()),
        "gain_m": float((better["ctrl_med_m"] - better["D_med_m"]).sum()),
        "stage_split": {"reselect_worse": int((t["resel_med_m"] > t["ctrl_med_m"] + 0.05).sum()),
                        "batch_worse": int((t["D_med_m"] > t["resel_med_m"] + 0.05).sum())},
        "shift": {"reselect_median_m": float(t["shift_reselect_m"].median()),
                  "batch_median_m": float(t["shift_batch_m"].median()),
                  "batch_p90_m": float(t["shift_batch_m"].quantile(0.90))},
        "pool": {"mean_size": float(t["pool"].mean()),
                 "sel_index_median": float(t["sel_index"].median()),
                 "sel_err_m_median": float(t["sel_err_m"].median()),
                 "sel_n_points_median": float(t["sel_n_points"].median())},
        "clip": {"zoom_m_per_px_median": float(t["zoom_m_per_px"].median()),
                 "gt_cut_shift_p99_m": float(t["gt_cut_shift_m"].quantile(0.99)),
                 "gt_cut_shift_max_m": float(t["gt_cut_shift_m"].max()),
                 "frames_with_gt": int(len(t)), "solved_frames": len(homs), "frames": n},
        "batch_stats": {k: v for k, v in st_b.items() if not isinstance(v, dict)},
        "csv": str(dest / f"forensic_{seq}.csv"),
    }


def guard_probe(seqs: list[str], data_dir: Path, out_dir: Path, cache_dir: Path,
                taus: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)) -> dict:
    """Price a GT-FREE guard on arm D: revert the batch step where it leaves its own evidence.

    The frame statistic is ``min over admissible cached hypotheses of grid_shift_m(D, candidate)``
    -- how far the batch output sits from the nearest solve that frame actually supports. Where that
    exceeds ``tau`` the frame falls back to the re-selection pick (still a genuine PnLCalib solve).
    Accuracy only; the GS-HOTA column is the W5 kernel-loss prediction, not an end-to-end run.
    """
    from generator.calibrate import apply_homography, estimate_homography  # noqa: PLC0415
    from generator.camera_track import grid_shift_m  # noqa: PLC0415
    from tools.gsr_v10_w5 import (  # noqa: PLC0415
        admissible, batch_refine, foot_points_by_frame, kernel_loss, load_seq, predicted_gain,
        reselect,
    )
    from tools.gsr_w2_calibswap import gt_people  # noqa: PLC0415

    pooled: dict[str, list] = {k: [] for k in ("ctrl", "resel", "D", *[f"g{t}" for t in taus])}
    per_seq: dict[str, dict] = {}
    reverted = dict.fromkeys(taus, 0)
    n_frames = 0
    for seq in seqs:
        df, by_frame, homs, err, n = load_seq(seq, out_dir, cache_dir)
        feet = foot_points_by_frame(df)
        picks, _st = reselect(homs, by_frame, feet, n, err)
        dpost, _stb = batch_refine(picks, by_frame, feet, n)
        gt = gt_people(data_dir / seq)
        resid = {}
        for fr in homs:
            adm = admissible(by_frame.get(fr) or [], feet.get(fr))
            resid[fr] = min((grid_shift_m(dpost[fr], by_frame[fr][i].homography) for i in adm),
                            default=0.0)
        vals: dict[str, list] = {k: [] for k in pooled}
        for fr in sorted(homs):
            g = gt.get(fr)
            if g is None or len(g) < 6:
                continue
            h_gt = estimate_homography(g[:, :2], g[:, 2:4])
            if h_gt is None or not np.isfinite(h_gt).all():
                continue
            ref = apply_homography(h_gt, g[:, :2])
            n_frames += 1
            arms = {"ctrl": homs[fr], "resel": picks[fr], "D": dpost[fr]}
            for t in taus:
                arms[f"g{t}"] = picks[fr] if resid[fr] > t else dpost[fr]
                reverted[t] += int(resid[fr] > t)
            for k, h in arms.items():
                vals[k].append(np.linalg.norm(apply_homography(h, g[:, :2]) - ref, axis=1))
        for k in pooled:
            pooled[k].append(np.concatenate(vals[k]) if vals[k] else np.zeros(0))
        per_seq[seq] = {k: kernel_loss(np.concatenate(vals[k])) if vals[k] else float("nan")
                        for k in pooled}
        logger.info("%s: kloss ctrl %.4f D %.4f guard2 %.4f", seq, per_seq[seq]["ctrl"],
                    per_seq[seq]["D"], per_seq[seq]["g2.0"])
    out = {"per_seq": per_seq, "frames": n_frames,
           "reverted_frames": {str(t): reverted[t] for t in taus}}
    kl = {k: kernel_loss(np.concatenate([x for x in pooled[k] if len(x)])) for k in pooled}
    out["kernel_loss"] = kl
    out["predicted_gs_hota_vs_ctrl"] = {k: round(predicted_gain(kl["ctrl"], kl[k]), 3) for k in kl}
    out["predicted_gs_hota_vs_D"] = {k: round(predicted_gain(kl["ctrl"], kl[k])
                                              - predicted_gain(kl["ctrl"], kl["D"]), 3)
                                     for k in kl}
    return out


# === self-check ===================================================================================
def _demo() -> None:
    """Assert the pure seams: ownership, the merge map, cut extraction and cut scoring."""
    rows = pd.DataFrame({"track_id": [1] * 6 + [2] * 6 + [3] * 3,
                         "frame": list(range(6)) + list(range(10, 16)) + [0, 1, 2],
                         "gt_id": [7] * 6 + [7, 7, 7, 7, 7, 9] + [-1, -1, -1]})
    own = ownership(rows)
    assert list(own["gt_id"]) == [7, 7, -1], own
    assert abs(own.loc[1, "purity"] - 5 / 6) < 1e-9, own
    new, st = merge_map(own)
    assert new == {1: 100007, 2: 100007} and st["merges_granted"] == 1, (new, st)
    frames = {1: set(range(6)), 2: set(range(10, 16))}
    assert merge_map(own, frames)[0] == new, "disjoint tracks must still merge"
    overlap = merge_map(own, {1: set(range(6)), 2: set(range(3, 9))})
    assert overlap[0] == {} and overlap[1]["merges_blocked_overlap"] == 1, overlap

    g = pd.DataFrame({"track_id": 1, "frame": range(10), "gt_id": [4] * 5 + [8] * 5})
    assert true_cuts(g) == [5], true_cuts(g)
    assert true_cuts(g, min_run=5) == [5] and true_cuts(g, min_run=6) == []
    flick = pd.DataFrame({"track_id": 1, "frame": range(11),
                          "gt_id": [4] * 5 + [8] + [4] * 5})
    assert true_cuts(flick) == [5, 6] and true_cuts(flick, min_run=5) == []
    # two clips reusing track id 1: the uid must keep them apart (a track_id-keyed version of this
    # scored them as one tracklet and lost 90% of the change points).
    feat = pd.DataFrame({"uid": ["A:1"] * 4 + ["B:1"] * 2, "frame": [0, 3, 40, 60, 0, 3],
                         "purity": [0.5] * 4 + [1.0] * 2,
                         "cut_frames": [(3,)] * 4 + [()] * 2})
    cuts = extract_cuts(feat, np.array([0.1, 0.9, 0.8, 0.2, 0.1, 0.9]), 0.5)
    assert cuts == {"A:1": [3, 40], "B:1": [3]}, cuts
    s = score_cuts(feat, cuts)
    assert (s["tp"], s["fp"], s["true_cuts"]) == (1, 1, 1), s
    assert abs(s["precision"] - 0.5) < 1e-9 and s["recall"] == 1.0, s
    assert s["pure_tracklets"] == 1 and s["false_cut_rate"] == 1.0, s
    assert s["cuts_all"] == 3 and s["cuts_all_on_a_true_change"] == 1, s
    print("gsr_v10_w7 demo OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gsr"))
    ap.add_argument("--results", type=Path, default=Path("results/gsr_benchmark/gsr_v10_w7.json"))
    ap.add_argument("--seqs", default=None)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--oracle-dedup", action="store_true",
                    help="amendment 2: merge + dedup upper variant (measure-only)")
    ap.add_argument("--dedup-only", action="store_true",
                    help="amendment 2: duplicate-track removal alone, ids untouched")
    ap.add_argument("--oracle-perfect", action="store_true",
                    help="amendment 3: row-level split+merge association oracle (measure-only)")
    ap.add_argument("--splitter", action="store_true")
    ap.add_argument("--forensic", default=None,
                    help="comma-separated clips for the per-frame geometry autopsy")
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/gsr/v10_w2_full/ctrl"))
    ap.add_argument("--guard", action="store_true",
                    help="price the GT-free batch-residual guard on arm D (accuracy only)")
    ap.add_argument("--flicker", action="store_true",
                    help="anatomy of tracklet contamination on the held-out factory slice")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return
    seqs = args.seqs.split(",") if args.seqs else DEV20
    payload: dict = {}
    t0 = time.time()
    for flag, key in ((args.oracle, "merge"), (args.oracle_dedup, "merge_dedup"),
                      (args.dedup_only, "dedup"), (args.oracle_perfect, "perfect")):
        if not flag:
            continue
        payload[f"oracle_{key}"] = run_oracle(args.data_dir, args.out_dir, seqs, mode=key)
        o = payload[f"oracle_{key}"]
        print(f"{key}: merges granted {o['totals']['merges_granted']} over "
              f"{o['totals']['groups_merged']} identities; blocked "
              f"{o['totals']['merges_blocked_overlap']}; dropped rows "
              f"{o['totals']['dropped_rows']}")
        for f in ("on", "off"):
            s = o["score"][f]
            print(f"flags {f}: arm D {s['control']['combined']['GS-HOTA']:.4f} -> oracle "
                  f"{s['arm']['combined']['GS-HOTA']:.4f} ({o['headroom_gs_hota'][f]:+.4f}) "
                  f"DetA {s['arm']['combined']['GS-DetA']:.4f} AssA "
                  f"{s['arm']['combined']['GS-AssA']:.4f}")
        print("VERDICT:", o["verdict"])
    if args.forensic:
        payload["forensic"] = {s: forensic(s, args.data_dir, args.out_dir, args.cache_dir)
                               for s in args.forensic.split(",")}
        print(json.dumps(payload["forensic"], indent=1, default=str))
    if args.guard:
        payload["guard"] = guard_probe(seqs, args.data_dir, args.out_dir, args.cache_dir)
        g = payload["guard"]
        print("kernel loss:", {k: round(v, 5) for k, v in g["kernel_loss"].items()})
        print("predicted GS-HOTA vs ctrl:", g["predicted_gs_hota_vs_ctrl"])
        print("predicted GS-HOTA vs D:", g["predicted_gs_hota_vs_D"])
        print("reverted frames:", g["reverted_frames"], "of", g["frames"])
    if args.flicker:
        man = json.loads((V9_ASSOC / "manifest.json").read_text(encoding="utf-8"))
        payload["flicker"] = flicker_anatomy(list(man["split"]["val_slice"]))
        print(json.dumps(payload["flicker"], indent=1))
    if args.splitter:
        payload["splitter"] = run_splitter()
        s = payload["splitter"]
        op = s["held_out_at_operating_point"]
        print(f"threshold {s['operating_point']['threshold']}: precision {op['precision']:.4f} "
              f"recall {op['recall']:.4f} false-cut {op['false_cut_rate']:.4f}")
    if payload:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        prev = (json.loads(args.results.read_text(encoding="utf-8"))
                if args.results.exists() else {})
        prev.update(payload)
        args.results.write_text(json.dumps(prev, indent=1, default=str), encoding="utf-8")
        print(f"wrote {args.results} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
