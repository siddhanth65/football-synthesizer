"""Re-score the GSR valid split with the jersey layer PLUS PRTreID appearance re-linking.

The shipped 19.83 arm (:mod:`eval.gsr_jersey`) attaches Koshkina jersey reads to the baseline
submissions and leaves ByteTrack's fragmented ``track_id`` alone. Re-linking was previously excluded
from the scored path because ImageNet-OSNet merge precision was only ~35% at the frozen 0.80
threshold. PRTreID (SoccerNet-trained, part-based) measures 84.7% held-out merge precision at 0.965
and 81.7% at 0.960 (``results/gsr_benchmark/prtreid_heldout_soccernet.json``), so this module tests
whether a *validated-precision* merge actually moves GS-HOTA.

Isolation of the variable: the relinked submissions are produced by rewriting ONLY ``track_id`` (and
the derived ``id``) on the already-jersey-attached ``eval_koshkina`` submissions. Pitch position,
role, team, confidence and the jersey attribute are copied verbatim, so the single difference against
the 19.83 arm is the identity partition. (This is equivalent to re-writing submissions from a
relinked positions table -- :func:`generator.track_relink.relink_sequence` only remaps ``track_id``,
and :func:`eval.gsr_score.resolve_team_map` does not read it -- but it is exact by construction.)

Thresholds are PRE-COMMITTED from the held-out precision measurement (0.965 primary, 0.960 secondary)
and are never tuned against GS-HOTA. Embeddings come from the cached PRTreID pass
(``outputs/gsr/relink_cache_prtreid``), so this run is CPU-only and resumable by disk state.

**Optional stage** ``--propagate-jersey`` (does not change the shipped relink behaviour): a merge
group is one player by an 84%-precision merge, so a number read on one member can be carried to
members that abstained. Under ``gs_hota_full`` GS-DetA is pinned at 9.89 because 91.3% of tracks
abstain and an abstaining prediction cannot match a numbered GT player, so this attacks jersey
RECALL -- the binding term -- with no new reads. Rule (pre-committed, parameter-free, see
:func:`propagation_fill`): fill only previously-abstaining members, only when every numbered member
of the group agrees; on ANY disagreement the whole group is left untouched.

CLI::

    python -m eval.gsr_prtreid_relink --threshold 0.965
    python -m eval.gsr_prtreid_relink --threshold 0.960 --limit 3
    python -m eval.gsr_prtreid_relink --threshold 0.960 --propagate-jersey
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from eval.gsr_score import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, DEFAULT_RESULTS_DIR, EVAL_CONFIGS, gs_hota
from generator.track_relink import (
    RelinkParams,
    _load_or_build_embeddings,
    fragment_gt_ids,
    greedy_merge,
    load_gt_ids_by_frame,
    merge_precision,
    pair_similarities,
    summarize_fragments,
)

logger = logging.getLogger("gsr_prtreid_relink")

#: Thresholds fixed on the pilot BEFORE the held-out precision measurement; never tuned to GS-HOTA.
PRECOMMITTED_THRESHOLDS = (0.965, 0.960)
#: Where the PRTreID fragment embeddings were cached by ``tools/prtreid_probe.py``.
CACHE_SUBDIR = "relink_cache_prtreid"


def build_remap(
    seq_dir: Path, df: pd.DataFrame, cache_path: Path, *, threshold: float, params: RelinkParams,
) -> tuple[dict[int, int], dict, dict[int, int]]:
    """Compute one sequence's ``old_track_id -> new_track_id`` map from cached PRTreID embeddings.

    Args:
        seq_dir: GSR sequence directory (for the GT used in the merge-precision audit).
        df: The sequence's positions table.
        cache_path: Warm ``.npz`` PRTreID embedding cache (built by ``tools/prtreid_probe.py``).
        threshold: Pre-committed cosine merge threshold.
        params: Re-link hyper-parameters (motion budget, crops); identical across arms.

    Returns:
        ``(remap, stats, gt_ids)`` -- ``stats`` carries fragment counts and GT merge precision,
        ``gt_ids`` the dominant GT track id per fragment (reused by the propagation audit).

    Raises:
        RuntimeError: If the embedding cache is cold (this module never runs the GPU stage).
    """
    fragments = summarize_fragments(df)
    embeddings, _counts = _load_or_build_embeddings(
        seq_dir, df, params=params, embedder=None, detector=None, cache_path=cache_path)
    sims = pair_similarities(fragments, embeddings, fps=params.fps,
                             max_speed_mps=params.max_speed_mps, pos_slack_m=params.pos_slack_m)
    remap = greedy_merge(fragments, sims, threshold, fps=params.fps,
                         max_speed_mps=params.max_speed_mps, pos_slack_m=params.pos_slack_m)
    gt_ids = fragment_gt_ids(df, load_gt_ids_by_frame(seq_dir))
    correct, total = merge_precision(remap, gt_ids)
    n_before = len(fragments)
    n_after = len(set(remap.values())) if remap else n_before
    return remap, {
        "n_fragments_before": n_before, "n_fragments_after": n_after,
        "n_merges": n_before - n_after, "n_embedded": len(embeddings), "n_pairs": len(sims),
        "merge_precision_correct": correct, "merge_precision_total": total,
    }, gt_ids


def propagation_fill(
    remap: dict[int, int], votes: dict[int, tuple[int, float]],
) -> tuple[dict[int, int], dict]:
    """Carry a merge group's jersey number to its abstaining members (pure).

    Pre-committed rule, chosen before any GS-HOTA number was computed and stated here so it cannot
    be re-tuned:

    1. Scope: propagation happens ONLY inside a validated merge group (fragments the PRTreID relink
       put on one ``new_track_id``). Singleton groups are untouched by construction.
    2. Direction: only previously-ABSTAINING members are filled. A member that already read a number
       keeps it verbatim -- propagation is strictly additive, it never overwrites or deletes a read.
    3. Disagreement: if the group's numbered members do not all report the same number, the WHOLE
       group is left untouched (no fill). Chosen over a confidence-margin winner because it has zero
       free parameters -- there is nothing to tune against the composite -- and because a group with
       internally contradictory reads is exactly the case where the merge or the read is wrong, so
       spreading a number there is the backfire mode this experiment must not manufacture. Where the
       group does agree, the confidence-weighted vote is trivially that same number.

    Args:
        remap: ``{old_track_id: new_track_id}`` for the sequence (identity for unmerged fragments).
        votes: ``{track_id: (number, conf)}`` from the Koshkina read; ``number <= 0`` == abstain.

    Returns:
        ``(fill, stats)`` where ``fill`` is ``{abstaining_track_id: number}`` and ``stats`` counts
        groups, agreeing/disagreeing groups and filled fragments.
    """
    groups: dict[int, list[int]] = {}
    for tid, rep in remap.items():
        groups.setdefault(int(rep), []).append(int(tid))
    fill: dict[int, int] = {}
    n_multi = n_with_read = n_agree = n_disagree = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        n_multi += 1
        read = {t: votes.get(t, (-1, 0.0)) for t in members}
        numbered = {t: v for t, v in read.items() if v[0] >= 1}
        if not numbered:
            continue
        n_with_read += 1
        nums = {v[0] for v in numbered.values()}
        if len(nums) > 1:
            n_disagree += 1
            continue
        n_agree += 1
        num = nums.pop()
        for t in members:
            if read[t][0] < 1:
                fill[t] = num
    return fill, {
        "n_groups_multi": n_multi, "n_groups_with_read": n_with_read,
        "n_groups_agree": n_agree, "n_groups_disagree": n_disagree,
        "n_fragments_filled": len(fill),
    }


def gt_jersey_by_track(seq_dir: Path) -> dict[int, str | None]:
    """Map GT ``track_id`` -> its labelled jersey string (``None`` when GT carries no number)."""
    gt = json.loads((seq_dir / "Labels-GameState.json").read_text(encoding="utf-8"))
    out: dict[int, str | None] = {}
    for ann in gt["annotations"]:
        attrs = ann.get("attributes") or {}
        if attrs.get("role") not in {"player", "goalkeeper"} or ann.get("track_id") is None:
            continue
        j = attrs.get("jersey")
        out[int(ann["track_id"])] = None if j in (None, "") else str(j)
    return out


def audit_propagation(
    fill: dict[int, int], gt_ids: dict[int, int], gt_jersey: dict[int, str | None],
) -> dict:
    """GT-audit every propagated number -> correct / wrong / spread onto an unnumbered GT player.

    A fill is *correct* when the fragment's dominant GT track carries exactly that number, *wrong*
    when it carries a different one, and ``gt_unnumbered`` when GT gives that player no number at
    all -- the case the jersey work already showed is damaging, because it turns a ``null == null``
    match into a miss. Fragments with no dominant GT id are unauditable and excluded from precision.
    """
    correct = wrong = unnumbered = unauditable = 0
    for tid, num in fill.items():
        gid = gt_ids.get(int(tid))
        if gid is None or gid not in gt_jersey:
            unauditable += 1
            continue
        gt_num = gt_jersey[gid]
        if gt_num is None:
            unnumbered += 1
        elif gt_num == str(num):
            correct += 1
        else:
            wrong += 1
    return {"correct": correct, "wrong": wrong, "gt_unnumbered": unnumbered,
            "unauditable": unauditable}


def remap_submission(
    src: Path, remap: dict[int, int], dest: Path, jersey_fill: dict[int, int] | None = None,
) -> tuple[int, int]:
    """Rewrite ``track_id``/``id`` (and optionally fill propagated jerseys) -> rows changed.

    Args:
        src: Jersey-attached baseline submission (``eval_koshkina/.../<seq>.json``).
        remap: ``{old_track_id: new_track_id}`` for this sequence.
        dest: Destination submission path (created).
        jersey_fill: Optional ``{old_track_id: number}`` from :func:`propagation_fill`; applied only
            to ``role == "player"`` rows whose jersey is still null (never overwrites a read).

    Returns:
        ``(n_track_ids_changed, n_jersey_rows_filled)``.
    """
    payload = json.loads(src.read_text(encoding="utf-8"))
    fill = jersey_fill or {}
    n_changed = n_filled = 0
    for pred in payload["predictions"]:
        old = int(pred["track_id"])
        num = fill.get(old)
        if (num is not None and pred["attributes"].get("role") == "player"
                and pred["attributes"].get("jersey") is None):
            pred["attributes"]["jersey"] = str(num)
            n_filled += 1
        new = int(remap.get(old, old))
        if new == old:
            continue
        pred["track_id"] = new
        pred["id"] = f"{pred['image_id']}{new:04d}"
        n_changed += 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload), encoding="utf-8")
    return n_changed, n_filled


def _load_votes(vote_json: Path) -> dict[int, tuple[int, float]]:
    """Load one sequence's Koshkina per-track votes checkpoint (``{tid: (number, conf)}``)."""
    blob = json.loads(vote_json.read_text(encoding="utf-8"))
    return {int(k): (int(v[0]), float(v[1])) for k, v in blob["votes"].items()}


def run(
    data_dir: Path, out_dir: Path, results_dir: Path, *, threshold: float, limit: int | None,
    score_only: bool = False, propagate: bool = False,
) -> dict:
    """Relink the jersey-attached submissions at ``threshold``, re-score, and write the payload.

    Args:
        data_dir: GSR ground-truth folder.
        out_dir: Pipeline output root (``positions/``, ``eval_koshkina/``, PRTreID cache).
        results_dir: Where ``gsr_scores_prtreid_relink.json`` (or ``..._propagate.json``) is written.
        threshold: Pre-committed merge threshold (0.965 or 0.960).
        limit: Process only the first N sequences (pilot).
        score_only: Reuse existing relinked submissions; skip the (CPU) merge stage.
        propagate: Also carry each merge group's jersey number to its abstaining members
            (:func:`propagation_fill`). Writes a separate arm dir and a separate results file, so
            the shipped relink artifacts are never touched.

    Returns:
        The arm's result dict (also merged into the on-disk multi-threshold payload).
    """
    pos_dir = out_dir / "positions"
    cache_dir = out_dir / CACHE_SUBDIR
    base_data = out_dir / "eval_koshkina" / "predictions" / "data"
    vote_dir = out_dir / "koshkina_jersey"
    suffix = "_propagate" if propagate else ""
    arm_root = out_dir / f"eval_prtreid_relink_{threshold:.3f}{suffix}"
    arm_data = arm_root / "predictions" / "data"
    seqs = sorted(p for p in data_dir.iterdir()
                  if p.is_dir() and (pos_dir / f"{p.name}.parquet").exists()
                  and (base_data / f"{p.name}.json").exists())
    if limit:
        seqs = seqs[:limit]
    if not seqs:
        raise SystemExit(f"no sequences with positions + jersey submissions under {out_dir}")
    logger.info("PRTreID relink rescore: %d sequences, threshold=%.3f", len(seqs), threshold)

    params = RelinkParams(threshold=threshold)
    stats: dict[str, dict] = {}
    stats_path = arm_root / "relink_stats.json"
    if score_only and stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for i, seq_dir in enumerate(seqs):
        name = seq_dir.name
        if score_only and (arm_data / f"{name}.json").exists() and name in stats:
            continue
        df = pd.read_parquet(pos_dir / f"{name}.parquet")
        remap, st, gt_ids = build_remap(seq_dir, df, cache_dir / f"{name}.npz",
                                        threshold=threshold, params=params)
        fill: dict[int, int] = {}
        if propagate:
            votes = _load_votes(vote_dir / f"{name}.json")
            fill, pst = propagation_fill(remap, votes)
            st.update(pst)
            st["propagation_audit"] = audit_propagation(fill, gt_ids, gt_jersey_by_track(seq_dir))
            st["n_tracks"] = len(votes)
            st["n_tracks_read"] = sum(1 for n, _ in votes.values() if n >= 1)
        st["n_preds_remapped"], st["n_rows_jersey_filled"] = remap_submission(
            base_data / f"{name}.json", remap, arm_data / f"{name}.json", fill)
        stats[name] = st
        prec = st["merge_precision_correct"] / max(st["merge_precision_total"], 1)
        logger.info("[%d/%d] %s: frags %d -> %d (%d merges), merge-prec %d/%d = %.1f%%",
                    i + 1, len(seqs), name, st["n_fragments_before"], st["n_fragments_after"],
                    st["n_merges"], st["merge_precision_correct"], st["merge_precision_total"],
                    100.0 * prec)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=1), encoding="utf-8")

    scored = {p.name: 0 for p in seqs if (arm_data / f"{p.name}.json").exists()}
    results: dict[str, dict] = {}
    for cfg_name, cfg in EVAL_CONFIGS.items():
        logger.info("scoring relinked config '%s' over %d sequences", cfg_name, len(scored))
        results[cfg_name] = gs_hota(arm_root, data_dir, seq_info=scored, **cfg)

    kosh = json.loads((results_dir / "gsr_scores_koshkina.json").read_text(encoding="utf-8"))
    relink_ref = None
    relink_path = results_dir / "gsr_scores_prtreid_relink.json"
    if propagate and relink_path.exists():
        ref_payload = json.loads(relink_path.read_text(encoding="utf-8"))
        ref_arm = ref_payload["arms"].get(f"{threshold:.3f}")
        if ref_arm:
            relink_ref = {"combined": ref_arm["combined"],
                          "per_seq": ref_arm["per_seq"]["gs_hota_full"]}
    arm = _assemble_arm(results, kosh, stats, scored, threshold, relink_ref=relink_ref)
    out_name = "gsr_scores_prtreid_propagate.json" if propagate else \
        "gsr_scores_prtreid_relink.json"
    _merge_into_payload(results_dir / out_name, arm, kosh, threshold)
    _log_summary(arm)
    return arm


def _propagation_totals(stats: dict, relink_ref: dict | None) -> dict:
    """Aggregate the propagation stage over sequences: fills, audit precision, abstention rate."""
    def tot(key: str) -> int:
        return int(sum(s.get(key, 0) for s in stats.values()))

    audits = [s.get("propagation_audit", {}) for s in stats.values()]
    audit = {k: int(sum(a.get(k, 0) for a in audits))
             for k in ("correct", "wrong", "gt_unnumbered", "unauditable")}
    auditable = audit["correct"] + audit["wrong"] + audit["gt_unnumbered"]
    n_tracks, n_read = tot("n_tracks"), tot("n_tracks_read")
    n_filled = tot("n_fragments_filled")
    return {
        "rule": "unanimous within merge group; abstain whole group on disagreement; fill only "
                "previously-abstaining members",
        "n_groups_multi": tot("n_groups_multi"), "n_groups_with_read": tot("n_groups_with_read"),
        "n_groups_agree": tot("n_groups_agree"), "n_groups_disagree": tot("n_groups_disagree"),
        "n_fragments_filled": n_filled, "n_rows_jersey_filled": tot("n_rows_jersey_filled"),
        "audit": audit,
        "precision_vs_numbered_gt": (audit["correct"] / (audit["correct"] + audit["wrong"])
                                     if audit["correct"] + audit["wrong"] else float("nan")),
        "precision_vs_all_auditable": (audit["correct"] / auditable if auditable
                                       else float("nan")),
        "tracks_total": n_tracks, "tracks_read_number": n_read,
        "tracks_numbered_after": n_read + n_filled,
        "abstain_rate_before": round(1.0 - n_read / max(n_tracks, 1), 4),
        "abstain_rate_after": round(1.0 - (n_read + n_filled) / max(n_tracks, 1), 4),
        "relink_only_combined": relink_ref["combined"] if relink_ref else None,
    }


def _assemble_arm(
    results: dict, kosh: dict, stats: dict, scored: dict, threshold: float,
    relink_ref: dict | None = None,
) -> dict:
    """Assemble one threshold arm: combined metrics, per-seq deltas vs 19.83, and hurt counts."""
    base_seq = kosh["attach_per_seq"]["gs_hota_full"]
    new_seq = results["gs_hota_full"]["per_seq"]
    per_seq: dict[str, dict] = {}
    for name in scored:
        new = new_seq.get(name, {}).get("GS-HOTA")
        base = base_seq.get(name, {}).get("GS-HOTA")
        st = stats.get(name, {})
        ref = (relink_ref or {}).get("per_seq", {}).get(name, {}).get("GS-HOTA")
        per_seq[name] = {
            "gs_hota_relink": new, "gs_hota_jersey_only": base,
            "delta": None if new is None or base is None else round(new - base, 3),
            "gs_hota_relink_only": ref,
            "delta_vs_relink_only": None if new is None or ref is None else round(new - ref, 3),
            "n_merges": st.get("n_merges", 0),
            "merge_precision_correct": st.get("merge_precision_correct", 0),
            "merge_precision_total": st.get("merge_precision_total", 0),
            "n_fragments_filled": st.get("n_fragments_filled", 0),
            "propagation_audit": st.get("propagation_audit"),
        }
    hurt = {n: d["delta"] for n, d in per_seq.items() if d["delta"] is not None and d["delta"] < 0}
    helped = {n: d["delta"] for n, d in per_seq.items()
              if d["delta"] is not None and d["delta"] > 0}
    hurt_r = {n: d["delta_vs_relink_only"] for n, d in per_seq.items()
              if d["delta_vs_relink_only"] is not None and d["delta_vs_relink_only"] < 0}
    helped_r = {n: d["delta_vs_relink_only"] for n, d in per_seq.items()
                if d["delta_vs_relink_only"] is not None and d["delta_vs_relink_only"] > 0}
    correct = int(sum(s.get("merge_precision_correct", 0) for s in stats.values()))
    total = int(sum(s.get("merge_precision_total", 0) for s in stats.values()))
    return {
        "threshold": threshold, "n_sequences": len(scored), "params": vars(RelinkParams(
            threshold=threshold)),
        "mean_fragments_before": float(np.mean(
            [s["n_fragments_before"] for s in stats.values()])) if stats else 0.0,
        "mean_fragments_after": float(np.mean(
            [s["n_fragments_after"] for s in stats.values()])) if stats else 0.0,
        "total_merges": int(sum(s["n_merges"] for s in stats.values())) if stats else 0,
        "merge_precision": {"correct": correct, "total": total,
                            "precision": (correct / total if total else float("nan"))},
        "combined": {k: v["combined"] for k, v in results.items()},
        "per_seq": {k: v["per_seq"] for k, v in results.items()},
        "per_seq_delta": per_seq,
        "n_seqs_hurt": len(hurt), "seqs_hurt": hurt,
        "n_seqs_helped": len(helped),
        "mean_delta": round(float(np.mean([d["delta"] for d in per_seq.values()
                                           if d["delta"] is not None])), 4) if per_seq else 0.0,
        "n_seqs_hurt_vs_relink_only": len(hurt_r), "seqs_hurt_vs_relink_only": hurt_r,
        "n_seqs_helped_vs_relink_only": len(helped_r),
        "propagation": _propagation_totals(stats, relink_ref),
    }


def _merge_into_payload(path: Path, arm: dict, kosh: dict, threshold: float) -> None:
    """Write/update the multi-threshold payload without disturbing the shipped koshkina artifacts."""
    payload = {"reference_jersey_only": kosh["arms"]["attach"],
               "reference_pre_jersey": kosh["arms"]["abstain"],
               "embedder": "prtreid-soccernet-baseline", "arms": {}}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["reference_jersey_only"] = kosh["arms"]["attach"]
        payload["reference_pre_jersey"] = kosh["arms"]["abstain"]
    payload["arms"][f"{threshold:.3f}"] = arm
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _log_summary(arm: dict) -> None:
    """ASCII summary of one threshold arm (cp1252-safe console)."""
    c = arm["combined"]
    mp = arm["merge_precision"]
    logger.info("== threshold %.3f over %d seqs; frags/seq %.0f -> %.0f (%d merges)",
                arm["threshold"], arm["n_sequences"], arm["mean_fragments_before"],
                arm["mean_fragments_after"], arm["total_merges"])
    logger.info("   merge precision %d/%d = %.1f%% (GT-audited, all sequences)",
                mp["correct"], mp["total"], 100.0 * mp["precision"])
    for cfg in EVAL_CONFIGS:
        m = c[cfg]
        logger.info("   %-12s HOTA %.2f  DetA %.2f  AssA %.2f  LocA %.2f  IDF1 %.2f",
                    cfg, m["GS-HOTA"], m["GS-DetA"], m["GS-AssA"], m["GS-LocA"], m["IDF1"])
    logger.info("   sequences hurt %d, helped %d, mean per-seq delta %+.3f",
                arm["n_seqs_hurt"], arm["n_seqs_helped"], arm["mean_delta"])
    p = arm.get("propagation", {})
    if p.get("n_fragments_filled"):
        a = p["audit"]
        logger.info("   propagation: %d groups w/ a read (%d agree, %d disagree) -> %d fragments, "
                    "%d rows filled", p["n_groups_with_read"], p["n_groups_agree"],
                    p["n_groups_disagree"], p["n_fragments_filled"], p["n_rows_jersey_filled"])
        logger.info("   propagation precision: %d correct / %d wrong / %d onto GT-unnumbered / "
                    "%d unauditable = %.1f%% vs numbered GT",
                    a["correct"], a["wrong"], a["gt_unnumbered"], a["unauditable"],
                    100.0 * p["precision_vs_numbered_gt"])
        logger.info("   track abstention %.1f%% -> %.1f%%; hurt vs relink-only %d, helped %d",
                    100.0 * p["abstain_rate_before"], 100.0 * p["abstain_rate_after"],
                    arm["n_seqs_hurt_vs_relink_only"], arm["n_seqs_helped_vs_relink_only"])


def _demo() -> None:
    """Self-check of the submission remap seam (asserts; runnable)."""
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "s.json"
        src.write_text(json.dumps({"predictions": [
            {"id": "img10007", "image_id": "img1", "track_id": 7,
             "attributes": {"role": "player", "jersey": "9", "team": "left"}},
            {"id": "img10009", "image_id": "img1", "track_id": 9,
             "attributes": {"role": "player", "jersey": None, "team": "left"}},
        ]}), encoding="utf-8")
        dest = Path(td) / "d.json"
        assert remap_submission(src, {9: 7}, dest) == (1, 0)
        out = json.loads(dest.read_text(encoding="utf-8"))["predictions"]
        assert [p["track_id"] for p in out] == [7, 7]
        assert out[1]["id"] == "img10007"  # id rederived from the new track id
        assert out[0]["attributes"]["jersey"] == "9"  # every other field untouched
        assert out[1]["attributes"]["jersey"] is None

        # Propagation rule: agreeing group fills the abstainer; disagreeing group fills nobody.
        remap = {7: 7, 9: 7, 11: 11, 12: 11, 13: 11, 20: 20}
        votes = {7: (9, 0.8), 9: (-1, 0.0), 11: (5, 0.9), 12: (6, 0.4), 13: (-1, 0.0),
                 20: (-1, 0.0)}
        fill, st = propagation_fill(remap, votes)
        assert fill == {9: 9}, fill                       # only the abstainer of the agreeing group
        assert st["n_groups_disagree"] == 1 and st["n_groups_agree"] == 1
        assert 13 not in fill and 20 not in fill          # disagreement + singleton stay abstained
        gt_ids = {9: 101}                                 # fragment 9 -> GT track 101
        assert audit_propagation(fill, gt_ids, {101: "9"})["correct"] == 1
        assert audit_propagation(fill, gt_ids, {101: "4"})["wrong"] == 1
        assert audit_propagation(fill, gt_ids, {101: None})["gt_unnumbered"] == 1
        dest2 = Path(td) / "d2.json"
        assert remap_submission(src, remap, dest2, fill) == (1, 1)
        out2 = json.loads(dest2.read_text(encoding="utf-8"))["predictions"]
        assert out2[1]["attributes"]["jersey"] == "9"     # abstainer inherited the group's number
        assert out2[0]["attributes"]["jersey"] == "9"     # existing read untouched
    print("gsr_prtreid_relink demo OK: remap + unanimous-group jersey propagation")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--threshold", type=float, default=PRECOMMITTED_THRESHOLDS[0])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--score-only", action="store_true",
                    help="reuse relinked submissions on disk; skip the merge stage")
    ap.add_argument("--propagate-jersey", action="store_true",
                    help="carry each merge group's number to its abstaining members (new arm dir "
                         "+ gsr_scores_prtreid_propagate.json; shipped artifacts untouched)")
    ap.add_argument("--demo", action="store_true", help="run the self-check and exit")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return
    run(args.data_dir, args.out_dir, args.results_dir, threshold=args.threshold,
        limit=args.limit, score_only=args.score_only, propagate=args.propagate_jersey)


if __name__ == "__main__":
    main()
