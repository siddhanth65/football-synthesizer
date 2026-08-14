"""Campaign v9 W1: the training-data factory for a metric-coordinate tracklet associator.

The v9 vehicle is a learned tracklet associator that consumes **metric pitch coordinates** rather
than image boxes (the published DanceTrack-lineage associators -- TWiX, MOTIP, SUSHI -- are all
image-space). This module builds the supervision it trains on.

The CAMELTrack lesson (on record, campaign v7): an associator trained on ground-truth boxes does not
transfer to our own tracker's output. So the factory runs the **frozen v6 chain** (S4b detector ->
ByteTrack -> PnLCalib -> calibration gate/fill -> EIoU re-association) over the GSR **train** split
and labels *that* distribution. The GTA connector is deliberately NOT run: the fragmented EIoU
partition is exactly what the associator has to learn to merge.

Two artifacts per sequence, under ``outputs/gsr/v9_assoc/<version>/``:

* ``tracklets/<seq>.parquet`` -- every non-ball positions row of the EIoU partition, with the
  per-row GT identity attached (nearest GT within :data:`eval.gsr_score.GSR_DIST_TOL_M`, the same
  auditing rule :func:`tools.gsr_eiou.partition_purity` uses). Full trajectories, so W2 can consume
  them pairwise (TWiX-style) or as a graph (SUSHI-style).
* ``pairs/<seq>.parquet`` -- one row per candidate tracklet pair, with metric-kinematic features,
  jersey evidence and the same-identity label.

Candidate rule (registered before the run, `results/GSR_V9_W1.md` S3): an ordered pair (A ends
before B starts) is a candidate iff it clears the **physical** constraints the connector already
applies at inference -- no temporal overlap, and the gap is closable at
:data:`generator.track_relink.MAX_PLAYER_SPEED_MPS` plus :data:`POS_SLACK_M` slack. Team, role and
jersey agreement are recorded as *features*, never as filters: those are the parts a learned model
is supposed to beat the connector on.

CLI::

    python -m tools.gsr_v9_factory --stage eiou      # CPU: EIoU re-association of the train split
    python -m tools.gsr_v9_factory --stage dataset   # CPU: labels + pairs + manifest
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("gsr_v9_factory")

#: Version stamp of the dataset layout and the feature set (bump on any schema/feature change).
FACTORY_VERSION = "v9-factory-1.0"
#: Artifact suffixes of the frozen v6 chain this factory reads (tools.gsr_v6det VARIANT + EIoU).
POSITIONS_SUBDIR = "positions_gate_v6det_eiou"
PERCROP_SUBDIR = "koshkina_percrop_v6_v6det_eiou"
EMBED_SUBDIR = "detembed_cache_clip_v6det_eiou"
#: Every 5th train sequence (sorted) is W2's model-selection slice. DEV-20 stays the W3 gate.
VAL_STRIDE = 5


@dataclass(frozen=True)
class FactoryParams:
    """Registered knobs of the candidate generator and the feature set.

    Attributes:
        fps: Sequence frame rate.
        max_speed_mps: Speed ceiling used by the physical candidate filter.
        pos_slack_m: Positional slack absorbing calibration jitter at short gaps.
        max_gap_frames: Hard cap on the temporal gap of a candidate pair (750 = whole sequence).
        endpoint_frames: Rows averaged at each tracklet end for the velocity estimate.
        min_rows: Tracklets with fewer positions rows are dropped before pairing.
        min_audited: GT-audited rows a tracklet needs before its label is trusted.
        jersey_floor: Per-crop reader confidence a jersey read must clear to count as evidence.
    """

    fps: float = 25.0
    max_speed_mps: float = 9.0
    pos_slack_m: float = 2.0
    max_gap_frames: int = 750
    endpoint_frames: int = 5
    min_rows: int = 3
    min_audited: int = 5
    jersey_floor: float = 0.80


# === Per-sequence tracklet table =================================================================
def _referee_gt_by_frame(seq_dir: Path) -> dict[int, list[tuple[float, float, int]]]:
    """GT ``(x_centred, y_centred, track_id)`` per frame for **referees** only (pure).

    :func:`generator.track_relink.load_gt_ids_by_frame` covers players and goalkeepers; referees are
    scored identities under GS-HOTA too, and their tracklets fragment like everyone else, so they
    get the same treatment through a referee-only GT map rather than by widening the shared loader
    (which every on-record purity number depends on).
    """
    from collections import defaultdict  # noqa: PLC0415

    gt = json.loads((seq_dir / "Labels-GameState.json").read_text(encoding="utf-8"))
    id_to_idx = {im["image_id"]: int(Path(im["file_name"]).stem) - 1 for im in gt["images"]}
    out: dict[int, list[tuple[float, float, int]]] = defaultdict(list)
    for ann in gt["annotations"]:
        attrs = ann.get("attributes") or {}
        bp, idx = ann.get("bbox_pitch"), id_to_idx.get(ann["image_id"])
        if attrs.get("role") != "referee" or not bp or idx is None or ann.get("track_id") is None:
            continue
        out[idx].append((bp["x_bottom_middle"], bp["y_bottom_middle"], int(ann["track_id"])))
    return out


def label_rows(df: pd.DataFrame, seq_dir: Path) -> pd.DataFrame:
    """Attach the per-row GT identity to one sequence's EIoU positions table (pure-ish, reads GT).

    Players and goalkeepers are matched with the on-record auditing rule (:func:`eval.gsr_gta.
    gt_rows` against :func:`generator.track_relink.load_gt_ids_by_frame`); referees are matched by
    the same code against a referee-only GT map (their rows are temporarily relabelled because
    ``gt_rows`` filters on role). GT track ids are unique across roles, so the two maps cannot
    collide.

    Args:
        df: Positions table of the EIoU partition (one row per detection).
        seq_dir: GSR sequence folder holding ``Labels-GameState.json``.

    Returns:
        The people rows with an added ``gt_id`` column (``-1`` where no GT is within tolerance).
    """
    from eval.gsr_gta import gt_rows  # noqa: PLC0415
    from generator.track_relink import load_gt_ids_by_frame  # noqa: PLC0415

    people = df[df["role"].isin(["player", "goalkeeper", "referee"])].copy()
    gt = gt_rows(people, load_gt_ids_by_frame(seq_dir))
    refs = people[people["role"] == "referee"]
    if len(refs):
        gt |= gt_rows(refs.assign(role="player"), _referee_gt_by_frame(seq_dir))
    people["gt_id"] = [gt.get((int(t), int(f)), -1)
                       for t, f in zip(people["track_id"], people["frame"])]
    return people.sort_values(["track_id", "frame"], ignore_index=True)


def jersey_evidence(percrop: pd.DataFrame, params: FactoryParams) -> dict[int, dict]:
    """Per-tracklet jersey evidence from the per-crop reader table (pure).

    Confidence mass is summed per number over crops clearing ``params.jersey_floor``; the dominant
    number and its share are what a learned associator gets as a sparse long-range edge feature.

    Args:
        percrop: Per-crop OCR table (``track_id, frame, number, p_number, legibility``).
        params: Registered knobs (uses ``jersey_floor``).

    Returns:
        ``{track_id: {"number", "mass", "share", "n_reads"}}``; absent ids have no evidence.
    """
    ok = percrop[(percrop["p_number"] >= params.jersey_floor) & (percrop["number"] >= 0)]
    out: dict[int, dict] = {}
    for tid, grp in ok.groupby("track_id"):
        mass: Counter = Counter()
        for num, p in zip(grp["number"], grp["p_number"]):
            mass[int(num)] += float(p)
        num, m = mass.most_common(1)[0]
        total = sum(mass.values())
        out[int(tid)] = {"number": int(num), "mass": float(m),
                         "share": float(m / total) if total else 0.0, "n_reads": int(len(grp))}
    return out


def summarise_tracklets(rows: pd.DataFrame, jersey: dict[int, dict],
                        params: FactoryParams) -> pd.DataFrame:
    """One row per tracklet: endpoints, endpoint velocities, majority attributes, GT label (pure).

    Args:
        rows: Output of :func:`label_rows` for one sequence.
        jersey: Output of :func:`jersey_evidence`.
        params: Registered knobs.

    Returns:
        Frame indexed by ``track_id`` with kinematics, attributes, jersey evidence and
        ``gt_id`` / ``purity`` / ``n_audited``.
    """
    recs = []
    for tid, g in rows.groupby("track_id"):
        g = g.sort_values("frame")
        fin = g[np.isfinite(g["pitch_x"]) & np.isfinite(g["pitch_y"])]
        if len(g) < params.min_rows or len(fin) < 2:
            continue
        k = params.endpoint_frames
        head, tail = fin.iloc[:k], fin.iloc[-k:]
        dt_h = max(int(head["frame"].iloc[-1]) - int(head["frame"].iloc[0]), 1) / params.fps
        dt_t = max(int(tail["frame"].iloc[-1]) - int(tail["frame"].iloc[0]), 1) / params.fps
        aud = g[g["gt_id"] >= 0]["gt_id"]
        votes = Counter(int(v) for v in aud)
        gid, dom = votes.most_common(1)[0] if votes else (-1, 0)
        j = jersey.get(int(tid), {})
        recs.append({
            "track_id": int(tid),
            "start_frame": int(g["frame"].iloc[0]), "end_frame": int(g["frame"].iloc[-1]),
            "n_rows": int(len(g)), "n_finite": int(len(fin)),
            "start_x": float(fin["pitch_x"].iloc[0]), "start_y": float(fin["pitch_y"].iloc[0]),
            "end_x": float(fin["pitch_x"].iloc[-1]), "end_y": float(fin["pitch_y"].iloc[-1]),
            "vx_start": float(head["pitch_x"].iloc[-1] - head["pitch_x"].iloc[0]) / dt_h,
            "vy_start": float(head["pitch_y"].iloc[-1] - head["pitch_y"].iloc[0]) / dt_h,
            "vx_end": float(tail["pitch_x"].iloc[-1] - tail["pitch_x"].iloc[0]) / dt_t,
            "vy_end": float(tail["pitch_y"].iloc[-1] - tail["pitch_y"].iloc[0]) / dt_t,
            "role": str(g["role"].mode().iloc[0]), "team": int(g["team"].mode().iloc[0]),
            "conf_mean": float(g["conf"].mean()),
            "calib_error_m": float(g["calib_error_m"].mean()),
            "jersey_number": int(j.get("number", -1)), "jersey_mass": float(j.get("mass", 0.0)),
            "jersey_share": float(j.get("share", 0.0)), "jersey_reads": int(j.get("n_reads", 0)),
            "gt_id": int(gid) if len(aud) >= params.min_audited else -1,
            "n_audited": int(len(aud)),
            "purity": float(dom / len(aud)) if len(aud) else float("nan"),
        })
    return pd.DataFrame(recs)


# === Candidate pairs =============================================================================
def build_pairs(tr: pd.DataFrame, params: FactoryParams,
                embs: dict[int, np.ndarray] | None = None) -> pd.DataFrame:
    """All candidate ordered tracklet pairs of one sequence, featurised and labelled (pure).

    A pair ``(A, B)`` is a candidate iff ``B`` starts after ``A`` ends, the gap is at most
    ``max_gap_frames``, and the endpoint distance is closable at ``max_speed_mps`` within the gap
    (plus ``pos_slack_m``). Nothing else is filtered -- team, role and jersey agreement ride along
    as features.

    Args:
        tr: Tracklet summary from :func:`summarise_tracklets`.
        params: Registered knobs.
        embs: Optional ``{track_id: L2-normalised mean CLIP embedding}``
            (:func:`generator.gta_link.mean_embeddings`). Its cosine distance is the **entire**
            signal the incumbent GTA connector merges on, so carrying it per pair lets W2 score the
            baseline and the learned model on exactly the same rows.

    Returns:
        Frame of pair rows; ``label`` is 1 (same GT identity), 0 (different) or -1 (unlabelled --
        at least one side has no trusted GT identity, e.g. a false-positive tracklet).
    """
    if tr.empty:
        return pd.DataFrame()
    t = tr.sort_values("end_frame", ignore_index=True)
    a_end = t["end_frame"].to_numpy()
    b_start = t["start_frame"].to_numpy()
    rows = []
    for i in range(len(t)):
        gap = b_start - a_end[i]
        cand = np.flatnonzero((gap > 0) & (gap <= params.max_gap_frames))
        if cand.size == 0:
            continue
        A = t.iloc[i]
        gap_s = gap[cand] / params.fps
        dx = t["start_x"].to_numpy()[cand] - A.end_x
        dy = t["start_y"].to_numpy()[cand] - A.end_y
        dist = np.hypot(dx, dy)
        keep = cand[dist <= params.max_speed_mps * gap_s + params.pos_slack_m]
        for j in keep:
            B = t.iloc[j]
            gs = (B.start_frame - A.end_frame) / params.fps
            # Constant-velocity extrapolation residual, both directions -- the metric-space feature
            # image-coordinate associators cannot form.
            fx, fy = A.end_x + A.vx_end * gs, A.end_y + A.vy_end * gs
            bx, by = B.start_x - B.vx_start * gs, B.start_y - B.vy_start * gs
            ea = (embs or {}).get(int(A.track_id))
            eb = (embs or {}).get(int(B.track_id))
            cos_d = float(1.0 - np.dot(ea, eb)) if ea is not None and eb is not None else np.nan
            same_num = (A.jersey_number >= 0 and B.jersey_number >= 0)
            lab = (1 if A.gt_id == B.gt_id else 0) if (A.gt_id >= 0 and B.gt_id >= 0) else -1
            rows.append({
                "tid_a": int(A.track_id), "tid_b": int(B.track_id),
                "gap_frames": int(B.start_frame - A.end_frame), "gap_s": float(gs),
                "len_a": int(A.n_rows), "len_b": int(B.n_rows),
                "end_x_a": float(A.end_x), "end_y_a": float(A.end_y),
                "start_x_b": float(B.start_x), "start_y_b": float(B.start_y),
                "dist_m": float(np.hypot(B.start_x - A.end_x, B.start_y - A.end_y)),
                "req_speed_mps": float(np.hypot(B.start_x - A.end_x, B.start_y - A.end_y)
                                       / max(gs, 1e-6)),
                "vx_a": float(A.vx_end), "vy_a": float(A.vy_end),
                "vx_b": float(B.vx_start), "vy_b": float(B.vy_start),
                "fwd_resid_m": float(np.hypot(B.start_x - fx, B.start_y - fy)),
                "bwd_resid_m": float(np.hypot(A.end_x - bx, A.end_y - by)),
                "team_a": int(A.team), "team_b": int(B.team),
                "team_same": int(A.team == B.team),
                "role_a": str(A.role), "role_b": str(B.role),
                "role_same": int(A.role == B.role),
                "num_a": int(A.jersey_number), "num_b": int(B.jersey_number),
                "num_mass_a": float(A.jersey_mass), "num_mass_b": float(B.jersey_mass),
                "num_agree": int(A.jersey_number == B.jersey_number) if same_num else -1,
                "conf_a": float(A.conf_mean), "conf_b": float(B.conf_mean),
                "clip_cos_dist": cos_d,
                "purity_a": float(A.purity), "purity_b": float(B.purity),
                "gt_a": int(A.gt_id), "gt_b": int(B.gt_id), "label": int(lab),
            })
    return pd.DataFrame(rows)


def candidate_recall(tr: pd.DataFrame, pairs: pd.DataFrame, params: FactoryParams) -> dict:
    """What fraction of the true same-identity pairs the physical filter kept (pure).

    The candidate generator is the associator's recall ceiling, so it is measured rather than
    assumed: every ordered non-overlapping pair of tracklets sharing a trusted GT identity is a
    true pair, and the filter either kept it or lost it.

    Args:
        tr: Tracklet summary.
        pairs: Output of :func:`build_pairs`.
        params: Registered knobs.

    Returns:
        ``{"true_pairs", "kept", "recall"}``.
    """
    lab = tr[tr["gt_id"] >= 0]
    true_pairs = 0
    for _gid, g in lab.groupby("gt_id"):
        g = g.sort_values("end_frame")
        for i in range(len(g)):
            gap = g["start_frame"].to_numpy() - int(g["end_frame"].iloc[i])
            true_pairs += int(((gap > 0) & (gap <= params.max_gap_frames)).sum())
    kept = int((pairs["label"] == 1).sum()) if len(pairs) else 0
    return {"true_pairs": true_pairs, "kept": kept,
            "recall": float(kept / true_pairs) if true_pairs else float("nan")}


# === Stages ======================================================================================
def stage_eiou(data_dir: Path, out_dir: Path, seqs: list[str]) -> dict:
    """CPU: the frozen EIoU re-association over the train split -- no connector, no scoring.

    Args:
        data_dir: GSR dataset root.
        out_dir: Pipeline output root.
        seqs: Sequence names.

    Returns:
        Per-sequence ``{n_tracks_before, n_tracks_after}`` from
        :func:`tools.gsr_eiou.build_arm_artifacts`.
    """
    import tools.gsr_eiou as eiou  # noqa: PLC0415

    eiou.BOX_SUBDIR = "detbox_cache_v6det"
    return eiou.build_arm_artifacts(
        data_dir, out_dir, seqs, eiou.EiouParams(e=0.3, rounds=1, w_app=0.5, app_max=0.30),
        positions_subdir="positions_gate_v6det", cache_subdir="detembed_cache_clip_v6det",
        percrop_subdir="koshkina_percrop_v6_v6det")


def build_sequence(data_dir: Path, out_dir: Path, dest: Path, name: str,
                   params: FactoryParams) -> dict:
    """Label, summarise and pair one sequence, writing both parquets.

    Args:
        data_dir: GSR dataset root.
        out_dir: Pipeline output root.
        dest: Versioned dataset directory.
        name: Sequence name.
        params: Registered knobs.

    Returns:
        Per-sequence stats row.
    """
    from generator.gta_link import mean_embeddings, tracklet_purity  # noqa: PLC0415

    df = pd.read_parquet(out_dir / POSITIONS_SUBDIR / f"{name}.parquet")
    percrop = pd.read_parquet(out_dir / PERCROP_SUBDIR / f"{name}.parquet")
    rows = label_rows(df, data_dir / name)
    tr = summarise_tracklets(rows, jersey_evidence(percrop, params), params)
    embs: dict[int, np.ndarray] = {}
    npz = out_dir / EMBED_SUBDIR / f"{name}.npz"
    if npz.exists():
        z = np.load(npz)
        det: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for tid in np.unique(z["track_ids"]):
            m = z["track_ids"] == tid
            det[int(tid)] = (z["frames"][m], z["embeddings"][m])
        embs = mean_embeddings(det)[0]
    else:
        logger.warning("%s: no embedding cache, clip_cos_dist will be NaN", name)
    pairs = build_pairs(tr, params, embs)
    (dest / "tracklets").mkdir(parents=True, exist_ok=True)
    (dest / "pairs").mkdir(parents=True, exist_ok=True)
    rows.merge(tr[["track_id", "gt_id", "purity"]].rename(columns={"gt_id": "tracklet_gt_id"}),
               on="track_id", how="left").to_parquet(dest / "tracklets" / f"{name}.parquet",
                                                     index=False)
    pairs.to_parquet(dest / "pairs" / f"{name}.parquet", index=False)
    lab = tr[tr["gt_id"] >= 0]
    frag = lab.groupby("gt_id").size()
    rec = candidate_recall(tr, pairs, params)
    # The on-record instrument (tools.gsr_eiou.partition_purity): every distinct track id that
    # *touches* a GT identity counts as a fragment, and purity is row-weighted over all of them.
    aud = rows[rows["gt_id"] >= 0]
    touch = aud.groupby("gt_id")["track_id"].nunique()
    pur = tracklet_purity({(int(t), int(f)): int(g) for t, f, g
                           in zip(aud["track_id"], aud["frame"], aud["gt_id"])})
    return {
        "seq": name, "n_rows": int(len(rows)), "n_tracklets": int(len(tr)),
        "n_labelled": int(len(lab)), "n_gt_ids": int(frag.size),
        "frag_per_gt": float(frag.mean()) if frag.size else 0.0,
        "frag_per_gt_touch": float(touch.mean()) if touch.size else 0.0,
        "purity_rowweighted": float(pur["purity"]), "n_contaminated": int(pur["n_contaminated"]),
        "purity_mean": float(tr["purity"].mean()), "purity_p10": float(tr["purity"].quantile(0.10)),
        "n_pairs": int(len(pairs)),
        "n_pos": int((pairs["label"] == 1).sum()) if len(pairs) else 0,
        "n_neg": int((pairs["label"] == 0).sum()) if len(pairs) else 0,
        "n_unlab": int((pairs["label"] == -1).sum()) if len(pairs) else 0,
        "cand_recall": rec["recall"], "true_pairs": rec["true_pairs"],
        "jersey_tracklets": int((tr["jersey_number"] >= 0).sum()),
    }


def leakage_check(data_dir: Path, names: list[str]) -> dict:
    """Verify every dataset sequence belongs to the GSR **train** split (pure-ish, reads the split).

    The associator may only see train sequences: GSR valid holds DEV-20 (W3's end-to-end gate) and
    TEST-38, and test-49 carries the board submission.

    Args:
        data_dir: GSR dataset root.
        names: Sequences written into the dataset.

    Returns:
        ``{"n", "leaked", "other_splits"}`` -- ``leaked`` is the offending list (empty = clean).
    """
    from tools.gsr_deleak import split_names  # noqa: PLC0415

    train = set(split_names(data_dir, "train"))
    other = {s: sorted(set(names) & set(split_names(data_dir, s))) for s in ("valid", "test")}
    return {"n": len(names), "leaked": sorted(set(names) - train), "other_splits": other}


def stage_dataset(data_dir: Path, out_dir: Path, dest: Path, seqs: list[str],
                  params: FactoryParams) -> dict:
    """Build the labelled pair dataset for every sequence and write the manifest.

    Args:
        data_dir: GSR dataset root.
        out_dir: Pipeline output root.
        dest: Versioned dataset directory.
        seqs: Sequence names (train split).
        params: Registered knobs.

    Returns:
        The manifest dict (also written to ``dest/manifest.json``).
    """
    stats = []
    for i, name in enumerate(seqs):
        if not (out_dir / POSITIONS_SUBDIR / f"{name}.parquet").exists():
            logger.warning("%s: no EIoU positions, skipped", name)
            continue
        s = build_sequence(data_dir, out_dir, dest, name, params)
        stats.append(s)
        logger.info("[%d/%d] %s: %d tracklets, %d pairs (%d pos / %d neg / %d unlab), "
                    "frag/GT %.2f, cand-recall %.3f", i + 1, len(seqs), name, s["n_tracklets"],
                    s["n_pairs"], s["n_pos"], s["n_neg"], s["n_unlab"], s["frag_per_gt"],
                    s["cand_recall"])
    st = pd.DataFrame(stats)
    names = sorted(st["seq"])
    val = names[::VAL_STRIDE]
    leak = leakage_check(data_dir, names)
    if leak["leaked"]:
        raise SystemExit(f"LEAKAGE: non-train sequences in the dataset: {leak['leaked']}")
    manifest = {
        "version": FACTORY_VERSION, "params": asdict(params),
        "source": {"positions": POSITIONS_SUBDIR, "percrop": PERCROP_SUBDIR,
                   "embeddings": EMBED_SUBDIR,
                   "chain": "S4b detector -> ByteTrack -> PnLCalib -> calib gate/fill -> EIoU "
                            "(e0.3 r1 w_app0.5 app_max0.30); NO GTA connector, NO solver"},
        "split": {"name": "GSR train", "n_sequences": len(names),
                  "val_slice": val, "train_slice": [n for n in names if n not in set(val)],
                  "val_rule": f"every {VAL_STRIDE}th sequence of the sorted train split",
                  "leakage_check": leak,
                  "leakage": "GSR valid (DEV-20 + TEST-38), test-49 and challenge are untouched"},
        "totals": {
            "sequences": int(len(st)), "rows": int(st["n_rows"].sum()),
            "tracklets": int(st["n_tracklets"].sum()),
            "labelled_tracklets": int(st["n_labelled"].sum()),
            "gt_identities": int(st["n_gt_ids"].sum()),
            "frag_per_gt_mean": float(st["frag_per_gt"].mean()),
            "frag_per_gt_touch_mean": float(st["frag_per_gt_touch"].mean()),
            "purity_rowweighted_mean": float(st["purity_rowweighted"].mean()),
            "pairs": int(st["n_pairs"].sum()), "pos": int(st["n_pos"].sum()),
            "neg": int(st["n_neg"].sum()), "unlabelled": int(st["n_unlab"].sum()),
            "pos_rate_labelled": float(st["n_pos"].sum() / max(st["n_pos"].sum()
                                                               + st["n_neg"].sum(), 1)),
            "true_pairs": int(st["true_pairs"].sum()),
            "candidate_recall": float(st["n_pos"].sum() / max(st["true_pairs"].sum(), 1)),
            "jersey_tracklets": int(st["jersey_tracklets"].sum()),
        },
        "per_seq": stats,
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    st.to_csv(dest / "per_sequence.csv", index=False)
    t = manifest["totals"]
    print(f"sequences {t['sequences']}  tracklets {t['tracklets']}  frag/GT "
          f"{t['frag_per_gt_mean']:.2f}")
    print(f"pairs {t['pairs']}  pos {t['pos']}  neg {t['neg']}  unlab {t['unlabelled']}  "
          f"pos-rate {t['pos_rate_labelled']:.4f}  cand-recall {t['candidate_recall']:.4f}")
    print(f"LEAKAGE CHECK: {leak['n']} sequences, non-train {leak['leaked']}, "
          f"valid-split overlap {leak['other_splits']['valid']}, "
          f"test-split overlap {leak['other_splits']['test']}")
    print(f"val slice ({len(val)}): {' '.join(val)}")
    return manifest


# === Self-check ==================================================================================
def _demo() -> None:
    """Deterministic self-check of the pairing rule and the labels (asserts; runnable)."""
    p = FactoryParams()
    tr = pd.DataFrame([
        # A: frames 0-9 walking +x, ends at (10, 0). B continues it after a 10-frame gap.
        {"track_id": 1, "start_frame": 0, "end_frame": 9, "n_rows": 10, "n_finite": 10,
         "start_x": 6.0, "start_y": 0.0, "end_x": 10.0, "end_y": 0.0, "vx_start": 10.0,
         "vy_start": 0.0, "vx_end": 10.0, "vy_end": 0.0, "role": "player", "team": 0,
         "conf_mean": 0.9, "calib_error_m": 0.1, "jersey_number": 7, "jersey_mass": 3.0,
         "jersey_share": 1.0, "jersey_reads": 3, "gt_id": 42, "n_audited": 10, "purity": 1.0},
        {"track_id": 2, "start_frame": 20, "end_frame": 29, "n_rows": 10, "n_finite": 10,
         "start_x": 14.0, "start_y": 0.0, "end_x": 18.0, "end_y": 0.0, "vx_start": 10.0,
         "vy_start": 0.0, "vx_end": 10.0, "vy_end": 0.0, "role": "player", "team": 0,
         "conf_mean": 0.9, "calib_error_m": 0.1, "jersey_number": 7, "jersey_mass": 3.0,
         "jersey_share": 1.0, "jersey_reads": 3, "gt_id": 42, "n_audited": 10, "purity": 1.0},
        # C: a different player, reachable from both A and B, so it produces negative pairs.
        {"track_id": 3, "start_frame": 40, "end_frame": 49, "n_rows": 10, "n_finite": 10,
         "start_x": 20.0, "start_y": 5.0, "end_x": 24.0, "end_y": 5.0, "vx_start": 10.0,
         "vy_start": 0.0, "vx_end": 10.0, "vy_end": 0.0, "role": "player", "team": 1,
         "conf_mean": 0.9, "calib_error_m": 0.1, "jersey_number": 9, "jersey_mass": 3.0,
         "jersey_share": 1.0, "jersey_reads": 3, "gt_id": 43, "n_audited": 10, "purity": 1.0},
    ])
    pairs = build_pairs(tr, p)
    assert len(pairs) == 3, f"expected 1-2, 1-3 and 2-3, got {len(pairs)}"
    assert set(map(tuple, pairs[["tid_a", "tid_b"]].to_numpy())) == {(1, 2), (1, 3), (2, 3)}, pairs
    pos = pairs[(pairs.tid_a == 1) & (pairs.tid_b == 2)].iloc[0]
    assert pos.label == 1 and pos.num_agree == 1, pos
    # Constant velocity at 10 m/s over a 0.44 s gap predicts (14.4, 0): a 0.4 m residual.
    assert abs(pos.fwd_resid_m - 0.4) < 1e-6, pos.fwd_resid_m
    assert pairs[(pairs.tid_a == 2) & (pairs.tid_b == 3)].iloc[0].label == 0

    rec = candidate_recall(tr, pairs, p)
    assert rec == {"true_pairs": 1, "kept": 1, "recall": 1.0}, rec

    # An unreachable pair (60 m in 0.44 s) must be filtered by the physical rule.
    far = build_pairs(pd.concat([tr.iloc[[0]], tr.iloc[[2]].assign(
        track_id=4, start_frame=20, end_frame=29, start_x=70.0, end_x=74.0)]), p)
    assert far.empty, far

    ev = jersey_evidence(pd.DataFrame({"track_id": [1, 1, 1], "frame": [0, 1, 2],
                                       "number": [7, 7, 9], "p_number": [0.9, 0.95, 0.5],
                                       "legibility": [1.0, 1.0, 1.0]}), p)
    assert ev[1] == {"number": 7, "mass": 1.85, "share": 1.0, "n_reads": 2}, ev
    print("gsr_v9_factory self-check OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gsr"))
    ap.add_argument("--dest", type=Path, default=Path("outputs/gsr/v9_assoc/v1"))
    ap.add_argument("--stage", choices=("eiou", "dataset", "demo"), default="dataset")
    ap.add_argument("--seqs", default=None, help="comma-separated names (default: the train split)")
    args = ap.parse_args()
    if args.stage == "demo":
        _demo()
        return

    from tools.gsr_deleak import split_names  # noqa: PLC0415

    seqs = args.seqs.split(",") if args.seqs else split_names(args.data_dir, "train")
    logger.info("v9 factory %s: stage %s over %d sequences", FACTORY_VERSION, args.stage, len(seqs))
    if args.stage == "eiou":
        stats = stage_eiou(args.data_dir, args.out_dir, seqs)
        before = sum(v["n_tracks_before"] for v in stats.values())
        after = sum(v["n_tracks_after"] for v in stats.values())
        print(f"EIoU re-association: {before} ByteTrack tracks -> {after} tracklets "
              f"over {len(stats)} sequences")
    else:
        stage_dataset(args.data_dir, args.out_dir, args.dest, seqs, FactoryParams())


if __name__ == "__main__":
    main()
