"""Score the SoccerNet-GSR valid split with GTA-Link tracklet repair ON vs OFF.

The comparison is isolated exactly the way :mod:`eval.gsr_prtreid_relink` isolates its own arm: the
repaired submissions are produced by rewriting ONLY ``track_id`` (and the derived ``id``, plus the
optional propagated jersey) on the already-jersey-attached ``eval_koshkina`` submissions. Geometry,
role, team and confidence are copied verbatim, so the single difference against the on-record arm is
the identity partition.

**OFF arm (on record, 22.85 GS-HOTA)**: ``eval.gsr_prtreid_relink`` at cosine 0.960 plus unanimous
jersey propagation -- single-linkage merging over per-fragment *median* PRTreID embeddings, no
splitter. **ON arm**: :mod:`generator.gta_link` -- DBSCAN splitter over per-detection embeddings,
then mean-recomputing agglomerative connection, then the same unanimous jersey propagation.

The splitter needs per-detection embeddings, which the shipped ``relink_cache_prtreid`` does not
hold. ``--build-cache`` runs that GPU pass once (resumable by disk state); every other mode is CPU.

CLI::

    python -m eval.gsr_gta --build-cache                 # GPU: per-detection PRTreID embeddings
    python -m eval.gsr_gta --sweep 0.02,0.03,0.04,0.06   # CPU: tau sweep on the 3 pilot sequences
    python -m eval.gsr_gta --tau 0.04                    # CPU: full 58-sequence arm + GS-HOTA
    python -m eval.gsr_gta --tau 0.04 --no-split         # connector-only ablation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from eval.gsr_score import (
    CENTRE_SHIFT_X,
    CENTRE_SHIFT_Y,
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_RESULTS_DIR,
    EVAL_CONFIGS,
    GSR_DIST_TOL_M,
    PILOT_SEQS,
    gs_hota,
    load_image_id_map,
)
from eval.gsr_prtreid_relink import audit_propagation, gt_jersey_by_track, propagation_fill
from generator.gta_link import (
    GTA_VERSION,
    GtaParams,
    apply_splits,
    connect,
    load_or_build_det_embeddings,
    mean_embeddings,
    split_detection_embeddings,
    split_tracklets,
    tracklet_purity,
)
from generator.track_relink import load_gt_ids_by_frame, merge_precision, summarize_fragments

logger = logging.getLogger("gsr_gta")

#: Appearance embedder for the per-detection cache. ``GSR_EMBEDDER=clip`` swaps in the cluster
#: session-3 encoder (:mod:`tools.clip_embedder`); the default keeps every on-record path
#: byte-identical. Driving the cache directory off the same name stops two embedding spaces from
#: ever landing in one cache -- every consumer resolves the subdir through this constant.
EMBEDDER = os.environ.get("GSR_EMBEDDER", "prtreid")

#: Where the per-detection embeddings live (distinct from the shipped per-fragment cache).
CACHE_SUBDIR = f"detembed_cache_{EMBEDDER}"


# === GT row audit ================================================================================
def gt_rows(df: pd.DataFrame, gt_by_frame: dict) -> dict[tuple[int, int], int]:
    """Nearest-GT id per positions row -> ``{(track_id, frame): gt_track_id}`` (pure).

    Rows whose nearest GT player is beyond :data:`eval.gsr_score.GSR_DIST_TOL_M` are unauditable and
    omitted, matching :func:`generator.track_relink.fragment_gt_ids`.
    """
    out: dict[tuple[int, int], int] = {}
    people = df[df["role"].isin(["player", "goalkeeper"])]
    for frame, grp in people.groupby("frame"):
        gts = gt_by_frame.get(int(frame), [])
        if not gts:
            continue
        gxy = np.array([[g[0], g[1]] for g in gts])
        for r in grp.itertuples():
            if not (np.isfinite(r.pitch_x) and np.isfinite(r.pitch_y)):
                continue
            cx, cy = float(r.pitch_x) - CENTRE_SHIFT_X, float(r.pitch_y) - CENTRE_SHIFT_Y
            d = np.hypot(gxy[:, 0] - cx, gxy[:, 1] - cy)
            k = int(np.argmin(d))
            if d[k] <= GSR_DIST_TOL_M:
                out[(int(r.track_id), int(r.frame))] = int(gts[k][2])
    return out


def fragments_per_gt(gt_by_row: dict[tuple[int, int], int]) -> float:
    """Mean number of distinct tracklets carrying each GT player (fragmentation ratio) (pure)."""
    per_gt: dict[int, set[int]] = defaultdict(set)
    for (tid, _f), gid in gt_by_row.items():
        per_gt[gid].add(tid)
    return float(np.mean([len(v) for v in per_gt.values()])) if per_gt else 0.0


# === One sequence ================================================================================
def repair_sequence(
    df: pd.DataFrame, det: dict, params: GtaParams, *, do_split: bool = True,
    numbers: dict[int, int] | None = None,
) -> tuple[pd.DataFrame, dict, dict, dict]:
    """Split then connect one sequence -> ``(split_df, row_lookup, remap, stats)``.

    Args:
        df: The sequence's positions table.
        det: ``{track_id: (frames, embeddings)}`` per-detection embeddings.
        params: GTA hyper-parameters.
        do_split: When False the splitter is skipped (connector-only ablation).
        numbers: Optional ``track_id -> confident jersey number`` gate for the connector
            (:func:`generator.gta_link.connect`). Sub-tracklets inherit their parent's number.

    Returns:
        ``(split_df, {(old_tid, frame): sub_tid}, {sub_tid: final_tid}, stats)``.
    """
    splits: dict[int, list[tuple[int, int, int]]] = {}
    sstats = {"n_tracks_embedded": len(det), "n_tracks_split": 0, "n_new_subtracks": 0}
    if do_split:
        splits, sstats = split_tracklets(df, det, params)
    sdf, lookup = apply_splits(df, splits)
    sdet = split_detection_embeddings(det, splits)
    embs, counts = mean_embeddings(sdet)
    fragments = summarize_fragments(sdf)
    parents = {int(sub): int(old) for old, b in splits.items() for _lo, _hi, sub in b}
    nums = None if numbers is None else {
        **{int(k): int(v) for k, v in numbers.items()},
        **{sub: int(numbers[old]) for sub, old in parents.items() if old in numbers}}
    remap = connect(fragments, embs, counts, params, numbers=nums)
    n_before, n_after = len(fragments), len(set(remap.values()))
    return sdf, lookup, remap, {
        **sstats, "n_fragments_presplit": int(df[df["role"] != "ball"]["track_id"].nunique()),
        "n_fragments_before": n_before, "n_fragments_after": n_after,
        "n_merges": n_before - n_after, "n_embedded": len(embs),
        "parents": {int(sub): int(old) for old, b in splits.items() for _lo, _hi, sub in b},
    }


def remap_submission_gta(
    src: Path, lookup: dict[tuple[int, int], int], remap: dict[int, int],
    frame_of_image: dict[str, int], dest: Path, jersey_fill: dict[int, int] | None = None,
) -> tuple[int, int]:
    """Rewrite ``track_id``/``id`` per row (splits are per-row) and optionally fill jerseys.

    Args:
        src: Jersey-attached baseline submission (``eval_koshkina/.../<seq>.json``).
        lookup: ``{(original_track_id, frame): sub_track_id}`` from the splitter.
        remap: ``{sub_track_id: final_track_id}`` from the connector.
        frame_of_image: ``{image_id: frame_index}`` for this sequence.
        dest: Destination submission path (created).
        jersey_fill: ``{sub_track_id: number}`` from :func:`propagation_fill`; applied only to
            ``role == "player"`` rows whose jersey is still null (never overwrites a read).

    Returns:
        ``(n_track_ids_changed, n_jersey_rows_filled)``.
    """
    payload = json.loads(src.read_text(encoding="utf-8"))
    fill = jersey_fill or {}
    n_changed = n_filled = 0
    for pred in payload["predictions"]:
        old = int(pred["track_id"])
        frame = frame_of_image.get(pred["image_id"])
        sub = lookup.get((old, frame), old) if frame is not None else old
        num = fill.get(sub)
        if (num is not None and pred["attributes"].get("role") == "player"
                and pred["attributes"].get("jersey") is None):
            pred["attributes"]["jersey"] = str(num)
            n_filled += 1
        new = int(remap.get(sub, sub))
        if new == old:
            continue
        pred["track_id"] = new
        pred["id"] = f"{pred['image_id']}{new:04d}"
        n_changed += 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload), encoding="utf-8")
    return n_changed, n_filled


def _inherit_votes(votes: dict[int, tuple[int, float]], parents: dict[int, int]) -> dict:
    """Sub-tracklets inherit their parent tracklet's Koshkina jersey vote (pure)."""
    out = dict(votes)
    for sub, old in parents.items():
        out[int(sub)] = votes.get(int(old), (-1, 0.0))
    return out


def process_sequence(
    seq_dir: Path, df: pd.DataFrame, det: dict, params: GtaParams, *, do_split: bool,
    base_json: Path, dest: Path, vote_json: Path | None,
    numbers: dict[int, int] | None = None,
) -> dict:
    """Repair, propagate jerseys, write the submission and GT-audit one sequence -> stats."""
    gt_by_frame = load_gt_ids_by_frame(seq_dir)
    pre_rows = gt_rows(df, gt_by_frame)
    _sdf, lookup, remap, st = repair_sequence(df, det, params, do_split=do_split, numbers=numbers)
    post_rows = {(int(lookup.get((tid, f), tid)), f): g for (tid, f), g in pre_rows.items()}
    st["purity_before"] = tracklet_purity(pre_rows)
    st["purity_after_split"] = tracklet_purity(post_rows)
    st["frag_per_gt_before"] = fragments_per_gt(pre_rows)
    st["frag_per_gt_after"] = fragments_per_gt(
        {(int(remap.get(t, t)), f): g for (t, f), g in post_rows.items()})

    sub_gt: dict[int, Counter] = defaultdict(Counter)
    for (tid, _f), gid in post_rows.items():
        sub_gt[tid][gid] += 1
    gt_ids = {t: c.most_common(1)[0][0] for t, c in sub_gt.items() if c}
    correct, total = merge_precision(remap, gt_ids)
    st["merge_precision_correct"], st["merge_precision_total"] = correct, total

    fill: dict[int, int] = {}
    if vote_json is not None and vote_json.exists():
        votes = _inherit_votes(
            {int(k): (int(v[0]), float(v[1]))
             for k, v in json.loads(vote_json.read_text(encoding="utf-8"))["votes"].items()},
            st["parents"])
        fill, pst = propagation_fill(remap, votes)
        st.update(pst)
        st["propagation_audit"] = audit_propagation(fill, gt_ids, gt_jersey_by_track(seq_dir))
        st["n_tracks"] = len(votes)
        st["n_tracks_read"] = sum(1 for n, _ in votes.values() if n >= 1)
    st["n_preds_remapped"], st["n_rows_jersey_filled"] = remap_submission_gta(
        base_json, lookup, remap, {v: k for k, v in load_image_id_map(seq_dir).items()}, dest, fill)
    st.pop("parents", None)
    return st


# === Drivers =====================================================================================
def _sequences(data_dir: Path, out_dir: Path, limit: int | None,
               only: list[str] | None = None) -> list[Path]:
    """Sequences that have a positions parquet and a jersey-attached baseline submission."""
    pos_dir = out_dir / "positions"
    base = out_dir / "eval_koshkina" / "predictions" / "data"
    seqs = sorted(p for p in data_dir.iterdir()
                  if p.is_dir() and (pos_dir / f"{p.name}.parquet").exists()
                  and (base / f"{p.name}.json").exists())
    if only:
        seqs = [p for p in seqs if p.name in set(only)]
    return seqs[:limit] if limit else seqs


def build_cache(data_dir: Path, out_dir: Path, params: GtaParams, limit: int | None,
                only: list[str] | None = None) -> None:
    """GPU stage: build the per-detection embedding cache (resumable by disk state)."""
    import time  # noqa: PLC0415

    seqs = _sequences(data_dir, out_dir, limit, only)
    cache_dir = out_dir / CACHE_SUBDIR
    cold = [p for p in seqs if not (cache_dir / f"{p.name}.npz").exists()]
    logger.info("per-detection cache: %d sequences, %d cold, stride=%d",
                len(seqs), len(cold), params.frame_stride)
    if not cold:
        return
    import torch  # noqa: PLC0415

    from generator.extract import _build_detector  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if EMBEDDER == "clip":
        from tools.clip_embedder import ClipEmbedder  # noqa: PLC0415

        embedder = ClipEmbedder(device=device)
    else:
        from tools.prtreid_probe import PrtreidEmbedder  # noqa: PLC0415

        embedder = PrtreidEmbedder(device=device)
    detector = _build_detector(device, "football")
    for i, seq_dir in enumerate(cold):
        t0 = time.time()
        df = pd.read_parquet(out_dir / "positions" / f"{seq_dir.name}.parquet")
        det = load_or_build_det_embeddings(
            seq_dir, df, cache_dir / f"{seq_dir.name}.npz", params=params,
            embedder=embedder, detector=detector)
        n = sum(len(f) for f, _ in det.values())
        logger.info("[%d/%d] %s: %d tracks, %d detections embedded (%.0fs)",
                    i + 1, len(cold), seq_dir.name, len(det), n, time.time() - t0)


def _arm_tag(params: GtaParams, do_split: bool) -> str:
    """Arm identity: every hyper-parameter that changes the output must appear, or arms collide."""
    if not do_split:
        return f"tau{params.tau:.3f}_nosplit"
    return (f"tau{params.tau:.3f}_split_eps{params.eps:.3f}"
            f"_ms{params.min_samples}_mr{params.min_run}")


def run_arm(
    data_dir: Path, out_dir: Path, params: GtaParams, *, do_split: bool, limit: int | None,
    only: list[str] | None = None, arm_tag: str | None = None,
) -> tuple[Path, dict[str, dict]]:
    """Repair + write submissions for one (tau, split) arm -> ``(arm_root, per-sequence stats)``."""
    seqs = _sequences(data_dir, out_dir, limit, only)
    cache_dir = out_dir / CACHE_SUBDIR
    base = out_dir / "eval_koshkina" / "predictions" / "data"
    votes = out_dir / "koshkina_jersey"
    tag = arm_tag or _arm_tag(params, do_split)
    arm_root = out_dir / f"eval_gta_{tag}"
    stats: dict[str, dict] = {}
    for i, seq_dir in enumerate(seqs):
        name = seq_dir.name
        cache = cache_dir / f"{name}.npz"
        if not cache.exists():
            logger.warning("[%d/%d] %s: no per-detection cache, skipping", i + 1, len(seqs), name)
            continue
        df = pd.read_parquet(out_dir / "positions" / f"{name}.parquet")
        det = load_or_build_det_embeddings(seq_dir, df, cache, params=params)
        stats[name] = process_sequence(
            seq_dir, df, det, params, do_split=do_split, base_json=base / f"{name}.json",
            dest=arm_root / "predictions" / "data" / f"{name}.json",
            vote_json=votes / f"{name}.json")
        s = stats[name]
        logger.info("[%d/%d] %s: split %d->%d subtracks, frags %d -> %d (%d merges), "
                    "purity %.3f -> %.3f, merge-prec %d/%d",
                    i + 1, len(seqs), name, s["n_tracks_embedded"],
                    s["n_tracks_embedded"] + s["n_new_subtracks"], s["n_fragments_before"],
                    s["n_fragments_after"], s["n_merges"], s["purity_before"]["purity"],
                    s["purity_after_split"]["purity"], s["merge_precision_correct"],
                    s["merge_precision_total"])
    (arm_root / "gta_stats.json").parent.mkdir(parents=True, exist_ok=True)
    (arm_root / "gta_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return arm_root, stats


def _totals(stats: dict[str, dict]) -> dict:
    """Aggregate per-sequence repair stats into the reportable summary block."""
    def s(key: str) -> int:
        return int(sum(v.get(key, 0) for v in stats.values()))

    def pur(key: str) -> float:
        dom = sum(v[key]["n_dominant"] for v in stats.values())
        row = sum(v[key]["n_rows"] for v in stats.values())
        return dom / max(row, 1)

    audits = [v.get("propagation_audit", {}) for v in stats.values()]
    audit = {k: int(sum(a.get(k, 0) for a in audits))
             for k in ("correct", "wrong", "gt_unnumbered", "unauditable")}
    c, t = s("merge_precision_correct"), s("merge_precision_total")
    n_tracks, n_read, n_fill = s("n_tracks"), s("n_tracks_read"), s("n_fragments_filled")
    return {
        "n_sequences": len(stats),
        "n_tracks_split": s("n_tracks_split"), "n_new_subtracks": s("n_new_subtracks"),
        "mean_fragments_presplit": float(np.mean([v["n_fragments_presplit"] for v in
                                                  stats.values()])) if stats else 0.0,
        "mean_fragments_before": float(np.mean([v["n_fragments_before"] for v in
                                                stats.values()])) if stats else 0.0,
        "mean_fragments_after": float(np.mean([v["n_fragments_after"] for v in
                                               stats.values()])) if stats else 0.0,
        "total_merges": s("n_merges"),
        "merge_precision": {"correct": c, "total": t, "precision": c / t if t else float("nan")},
        "purity_before": pur("purity_before"), "purity_after_split": pur("purity_after_split"),
        "n_contaminated_before": sum(v["purity_before"]["n_contaminated"] for v in stats.values()),
        "n_contaminated_after": sum(v["purity_after_split"]["n_contaminated"]
                                    for v in stats.values()),
        "frag_per_gt_before": float(np.mean([v["frag_per_gt_before"] for v in
                                             stats.values()])) if stats else 0.0,
        "frag_per_gt_after": float(np.mean([v["frag_per_gt_after"] for v in
                                            stats.values()])) if stats else 0.0,
        "propagation": {
            "n_groups_agree": s("n_groups_agree"), "n_groups_disagree": s("n_groups_disagree"),
            "n_fragments_filled": n_fill, "n_rows_jersey_filled": s("n_rows_jersey_filled"),
            "audit": audit,
            "precision_vs_numbered_gt": (audit["correct"] / (audit["correct"] + audit["wrong"])
                                         if audit["correct"] + audit["wrong"] else float("nan")),
            "abstain_rate_before": round(1.0 - n_read / max(n_tracks, 1), 4),
            "abstain_rate_after": round(1.0 - (n_read + n_fill) / max(n_tracks, 1), 4),
        },
    }


def score(arm_root: Path, data_dir: Path, stats: dict[str, dict]) -> dict:
    """Score every attribute config over the sequences this arm actually wrote."""
    scored = {n: 0 for n in stats}
    return {name: gs_hota(arm_root, data_dir, seq_info=scored, **cfg)
            for name, cfg in EVAL_CONFIGS.items()}


def sweep(data_dir: Path, out_dir: Path, variants: list[tuple[float, float, bool]],
          base: GtaParams, seqs: list[str]) -> list[dict]:
    """Sweep ``(eps, tau, split)`` variants on ``seqs`` -> printed table + rows.

    Swept on the frozen pilot :data:`eval.gsr_score.PILOT_SEQS` only; the winner is then applied to
    the full split, which is the tuning discipline the earlier re-link arms used.
    """
    rows: list[dict] = []
    print(f"{'eps':>6} {'tau':>6} {'split':>6} {'nsplit':>7} {'purity':>15} {'frags':>13} "
          f"{'mergePrec':>11} {'HOTAfull':>9} {'HOTAloc':>8} {'AssAloc':>8}")
    for eps, tau, do_split in variants:
        params = GtaParams(**{**vars(base), "eps": eps, "tau": tau})
        arm, st = run_arm(data_dir, out_dir, params, do_split=do_split, limit=None, only=seqs,
                          arm_tag=f"sweep_e{eps:.3f}_t{tau:.3f}_{int(do_split)}")
        res = score(arm, data_dir, st)
        tot = _totals(st)
        rows.append({"eps": eps, "tau": tau, "split": do_split, **tot,
                     "combined": {k: v["combined"] for k, v in res.items()}})
        mp = tot["merge_precision"]
        prec = "{}/{}".format(mp["correct"], mp["total"])  # noqa: UP032
        pur = "{:.4f}->{:.4f}".format(tot["purity_before"], tot["purity_after_split"])  # noqa: UP032
        loc = rows[-1]["combined"]["loc_assoc"]
        print(f"{eps:>6.3f} {tau:>6.3f} {str(do_split):>6} {tot['n_tracks_split']:>7} {pur:>15} "
              f"{tot['mean_fragments_before']:>5.0f} ->{tot['mean_fragments_after']:>5.0f} "
              f"{prec:>11} {rows[-1]['combined']['gs_hota_full']['GS-HOTA']:>9.2f} "
              f"{loc['GS-HOTA']:>8.2f} {loc['GS-AssA']:>8.2f}")
    return rows


def _log_summary(tag: str, tot: dict, res: dict) -> None:
    """ASCII summary of one arm (cp1252-safe console)."""
    logger.info("== %s over %d seqs", tag, tot["n_sequences"])
    logger.info("   splitter: %d tracklets split -> %d new sub-tracklets; purity %.4f -> %.4f; "
                "contaminated %d -> %d", tot["n_tracks_split"], tot["n_new_subtracks"],
                tot["purity_before"], tot["purity_after_split"],
                tot["n_contaminated_before"], tot["n_contaminated_after"])
    logger.info("   connector: frags/seq %.1f -> %.1f (%d merges); merge precision %d/%d = %.1f%%",
                tot["mean_fragments_before"], tot["mean_fragments_after"], tot["total_merges"],
                tot["merge_precision"]["correct"], tot["merge_precision"]["total"],
                100.0 * tot["merge_precision"]["precision"])
    logger.info("   fragments per GT player %.2f -> %.2f",
                tot["frag_per_gt_before"], tot["frag_per_gt_after"])
    for cfg, m in res.items():
        c = m["combined"]
        logger.info("   %-12s HOTA %.2f  DetA %.2f  AssA %.2f  LocA %.2f  IDF1 %.2f",
                    cfg, c["GS-HOTA"], c["GS-DetA"], c["GS-AssA"], c["GS-LocA"], c["IDF1"])


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--tau", type=float, default=0.04, help="connector cosine-distance ceiling")
    ap.add_argument("--eps", type=float, default=0.30, help="splitter DBSCAN radius (cosine dist)")
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--min-run", type=int, default=5)
    ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument("--no-split", action="store_true", help="connector only (splitter ablation)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seqs", default=None, help="comma-separated sequence names")
    ap.add_argument("--build-cache", action="store_true", help="GPU: per-detection embeddings")
    ap.add_argument("--sweep", default=None, help="comma-separated taus, swept on the pilot")
    ap.add_argument("--sweep-eps", default=None, help="comma-separated splitter eps values")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        from generator.gta_link import _demo  # noqa: PLC0415

        _demo()
        return
    params = GtaParams(tau=args.tau, eps=args.eps, min_samples=args.min_samples,
                       min_run=args.min_run, frame_stride=args.frame_stride)
    only = args.seqs.split(",") if args.seqs else None
    if args.build_cache:
        build_cache(args.data_dir, args.out_dir, params, args.limit, only)
        return
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.sweep or args.sweep_eps:
        taus = [float(t) for t in args.sweep.split(",")] if args.sweep else [args.tau]
        epss = [float(e) for e in args.sweep_eps.split(",")] if args.sweep_eps else [args.eps]
        variants = [(e, t, s) for s in (True, False) for e in epss for t in taus
                    if s or e == epss[0]]  # the splitter is off -> eps is inert, run it once
        rows = sweep(args.data_dir, args.out_dir, variants, params, only or list(PILOT_SEQS))
        name = "gsr_gta_sweep.json"
        (args.results_dir / name).write_text(
            json.dumps({"version": GTA_VERSION, "seqs": only or list(PILOT_SEQS),
                        "params": vars(params), "rows": rows}, indent=2), encoding="utf-8")
        print(f"wrote {args.results_dir / name}")
        return
    tag = _arm_tag(params, not args.no_split)
    arm, stats = run_arm(args.data_dir, args.out_dir, params, do_split=not args.no_split,
                         limit=args.limit, only=only)
    res = score(arm, args.data_dir, stats)
    tot = _totals(stats)
    _log_summary(tag, tot, res)
    path = args.results_dir / "gsr_scores_gta.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "version": GTA_VERSION, "arms": {}}
    payload["arms"][tag] = {"params": vars(params), "split": not args.no_split, **tot,
                            "combined": {k: v["combined"] for k, v in res.items()},
                            "per_seq": {k: v["per_seq"] for k, v in res.items()},
                            "per_seq_stats": stats}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
