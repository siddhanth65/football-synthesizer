"""Repair the calibration dropout on SoccerNet-GSR and measure what it buys (CPU only).

`results/GSR_ASSOCIATION.md` measures the defect: on the valid split 25% of frames carry **no**
pitch position at all, and 14.4% of all ground-truth player rows are detected in image space but
have no pitch coordinate. The calibrator's own keypoint-reprojection gate passes those frames -- it
is the projection that is wrong, so :func:`generator.postprocess.clamp_to_pitch` (and then
`reject_implausible_frames`) discards them.

:func:`generator.postprocess.fill_calibration_gaps` re-derives each surviving frame's homography
from the positions table itself and re-projects the dropped rows. Nothing here re-runs the GPU: the
detections, track ids, jersey votes and PRTreID embeddings are all keyed by ``(track_id, frame)``
and are unchanged by the repair.

Partitions are the ones already declared in :func:`eval.gsr_identity.split_sequences` -- DEV-20
(every third valid sequence) for choosing ``max_gap``, TEST-38 and the full 58 for the frozen run.

CLI::

    python -m tools.gsr_calibfill --recover-audit          # GT audit of the recovery (no scoring)
    python -m tools.gsr_calibfill --dev                    # max_gap sweep on DEV-20, base + GTA arms
    python -m tools.gsr_calibfill --run --max-gap 0        # frozen setting over the full valid split
    python -m tools.gsr_calibfill --solve-valid            # GT-free solver, no-fill vs fill (T38)
    python -m tools.gsr_calibfill --freeze --max-gap 10    # pre-declare the combined recipe
    python -m tools.gsr_calibfill --solve-test             # the ONE test run + submission package
    python -m tools.gsr_calibfill --demo
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
    DEFAULT_RESULTS_DIR,
    EVAL_CONFIGS,
    GSR_DIST_TOL_M,
    CENTRE_SHIFT_X,
    CENTRE_SHIFT_Y,
    gs_hota,
    load_image_id_map,
    _write_submission,
)
from generator.postprocess import fill_calibration_gaps
from generator.track_relink import load_gt_ids_by_frame

logger = logging.getLogger("gsr_calibfill")

#: Where the repaired parquets live (the originals are never overwritten).
FILLED_SUBDIR = "positions_filled"


def dev_test_split(out_dir: Path) -> tuple[list[str], list[str]]:
    """The declared DEV-20 / TEST-38 partition of the valid split (every third sequence is DEV)."""
    names = sorted(p.stem for p in (out_dir / "positions").glob("*.parquet"))
    dev = names[::3]
    return dev, [n for n in names if n not in set(dev)]


# === Repair ======================================================================================
def fill_split(out_dir: Path, seqs: list[str], *, max_gap: int | None) -> dict[str, dict]:
    """Write repaired parquets for ``seqs`` -> per-sequence row counts before/after."""
    dest = out_dir / FILLED_SUBDIR
    dest.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name in seqs:
        df = pd.read_parquet(out_dir / "positions" / f"{name}.parquet")
        out = fill_calibration_gaps(df, max_gap=max_gap)
        out.to_parquet(dest / f"{name}.parquet", index=False)
        before = int(np.isfinite(df["pitch_x"]).sum())
        after = int(np.isfinite(out["pitch_x"]).sum())
        stats[name] = {"rows": len(df), "pitch_rows_before": before, "pitch_rows_after": after,
                       "recovered": after - before}
    return stats


def recover_audit(data_dir: Path, out_dir: Path, seqs: list[str],
                  max_gap: int | None) -> dict[str, float]:
    """GT-audit the repair: what share of the recovered rows lands within the evaluator's tolerance.

    Ground truth is used **only to score**; the repair itself reads nothing but the positions table.
    """
    tp_new = fp_new = 0
    gt_before = gt_after = gt_tot = 0
    for name in seqs:
        df = pd.read_parquet(out_dir / "positions" / f"{name}.parquet")
        out = fill_calibration_gaps(df, max_gap=max_gap)
        gt = load_gt_ids_by_frame(data_dir / name)
        people = out[out["role"].isin(["player", "goalkeeper"])]
        was_nan = ~np.isfinite(df.loc[people.index, "pitch_x"].to_numpy())
        for frame, grp in people.groupby("frame"):
            gts = gt.get(int(frame), [])
            gt_tot += len(gts)
            if not gts:
                continue
            gxy = np.array([[g[0], g[1]] for g in gts])
            new = was_nan[people.index.get_indexer(grp.index)]
            fin = np.isfinite(grp["pitch_x"].to_numpy())
            pxy = np.column_stack([grp["pitch_x"].to_numpy() - CENTRE_SHIFT_X,
                                   grp["pitch_y"].to_numpy() - CENTRE_SHIFT_Y])
            for lbl, mask in (("old", fin & ~new), ("new", fin & new)):
                if not mask.any():
                    continue
                d = np.hypot(gxy[:, None, 0] - pxy[None, mask, 0],
                             gxy[:, None, 1] - pxy[None, mask, 1])
                hit = (d.min(axis=0) <= GSR_DIST_TOL_M)
                if lbl == "new":
                    tp_new += int(hit.sum())
                    fp_new += int((~hit).sum())
            if fin.any():
                d = np.hypot(gxy[:, None, 0] - pxy[None, fin, 0],
                             gxy[:, None, 1] - pxy[None, fin, 1])
                gt_after += int((d.min(axis=1) <= GSR_DIST_TOL_M).sum())
            old = fin & ~new
            if old.any():
                d = np.hypot(gxy[:, None, 0] - pxy[None, old, 0],
                             gxy[:, None, 1] - pxy[None, old, 1])
                gt_before += int((d.min(axis=1) <= GSR_DIST_TOL_M).sum())
    return {
        "n_seqs": len(seqs), "recovered_rows": tp_new + fp_new,
        "recovered_precision": tp_new / max(tp_new + fp_new, 1),
        "gt_recall_before": gt_before / max(gt_tot, 1),
        "gt_recall_after": gt_after / max(gt_tot, 1),
    }


# === Scoring =====================================================================================
def build_arm(data_dir: Path, out_dir: Path, pos_dir: Path, arm: Path, seqs: list[str]) -> None:
    """Write base-arm submissions for ``seqs`` from ``pos_dir`` into ``arm``."""
    for name in seqs:
        df = pd.read_parquet(pos_dir / f"{name}.parquet")
        _write_submission(df, data_dir / name, arm / "predictions" / "data" / f"{name}.json")


def score_arm(arm: Path, data_dir: Path, seqs: list[str]) -> dict[str, dict]:
    """Score one arm over ``seqs`` under every attribute configuration."""
    info = {n: len(load_image_id_map(data_dir / n)) for n in seqs}
    return {k: gs_hota(arm, data_dir, seq_info=info, **cfg) for k, cfg in EVAL_CONFIGS.items()}


def jersey_arm(out_dir: Path, src: Path, dest: Path, seqs: list[str]) -> None:
    """Attach the cached Koshkina jersey votes, so the filled arm matches the on-record lineage.

    The votes are keyed by ``track_id``, which the repair never changes -- no OCR is re-run.
    """
    from eval.gsr_jersey import patch_submission  # noqa: PLC0415

    votes_dir = out_dir / "koshkina_jersey"
    for name in seqs:
        vj = votes_dir / f"{name}.json"
        votes = ({int(k): (int(v[0]), float(v[1]))
                  for k, v in json.loads(vj.read_text(encoding="utf-8"))["votes"].items()}
                 if vj.exists() else {})
        patch_submission(src / "predictions" / "data" / f"{name}.json", votes,
                         dest / "predictions" / "data" / f"{name}.json")


def gta_arm(data_dir: Path, out_dir: Path, pos_dir: Path, arm: Path, seqs: list[str],
            *, tau: float) -> None:
    """Run the frozen GTA connector (splitter OFF) over ``seqs`` on top of the jersey-attached arm."""
    from eval.gsr_gta import CACHE_SUBDIR, process_sequence  # noqa: PLC0415
    from generator.gta_link import GtaParams, load_or_build_det_embeddings  # noqa: PLC0415

    params = GtaParams(tau=tau, eps=0.30, min_samples=5, min_run=5, frame_stride=2)
    base = out_dir / "eval_calibfill_koshkina" / "predictions" / "data"
    votes = out_dir / "koshkina_jersey"
    for name in seqs:
        cache = out_dir / CACHE_SUBDIR / f"{name}.npz"
        if not cache.exists():
            logger.warning("%s: no per-detection embedding cache, skipped", name)
            continue
        df = pd.read_parquet(pos_dir / f"{name}.parquet")
        det = load_or_build_det_embeddings(data_dir / name, df, cache, params=params)
        process_sequence(data_dir / name, df, det, params, do_split=False,
                         base_json=base / f"{name}.json",
                         dest=arm / "predictions" / "data" / f"{name}.json",
                         vote_json=votes / f"{name}.json")


def _paired(base_per_seq: dict, arm: dict[str, dict], seqs: list[str]) -> dict:
    """Paired per-sequence deltas + Wilcoxon, per attribute configuration."""
    from scipy.stats import wilcoxon  # noqa: PLC0415

    out = {}
    for cfg in ("loc_assoc", "role_only", "no_jersey", "gs_hota_full"):
        a = np.array([base_per_seq[cfg][s]["GS-HOTA"] for s in seqs])
        b = np.array([arm[cfg]["per_seq"][s]["GS-HOTA"] for s in seqs])
        d = b - a
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        out[cfg] = {"mean": float(d.mean()), "median": float(np.median(d)),
                    "helped": int((d > 0).sum()), "hurt": int((d < 0).sum()),
                    "worst": float(d.min()), "best": float(d.max()), "wilcoxon_p": p}
    return out


def _row(tag: str, res: dict[str, dict]) -> dict:
    """Flatten one arm's four configs into a printable row."""
    out = {"arm": tag}
    for cfg in ("loc_assoc", "role_only", "no_jersey", "gs_hota_full"):
        c = res[cfg]["combined"]
        out[cfg] = (c["GS-HOTA"], c["GS-DetA"], c["GS-AssA"])
    return out


def _print(rows: list[dict]) -> None:
    """ASCII table: HOTA/DetA/AssA per configuration per arm."""
    print(f"{'arm':<28} " + " ".join(f"{c:>26}" for c in
                                     ("loc_assoc", "role_only", "no_jersey", "gs_hota_full")))
    print(f"{'':<28} " + " ".join(f"{'HOTA':>8}{'DetA':>9}{'AssA':>9}" for _ in range(4)))
    for r in rows:
        cells = []
        for cfg in ("loc_assoc", "role_only", "no_jersey", "gs_hota_full"):
            h, d, a = r[cfg]
            cells.append(f"{h:>8.2f}{d:>9.2f}{a:>9.2f}")
        print(f"{r['arm']:<28} " + " ".join(cells))


def run(data_dir: Path, out_dir: Path, results_dir: Path, seqs: list[str], tag: str,
        gaps: list[int | None], *, tau: float, with_gta: bool) -> dict:
    """Score the unrepaired baseline and one arm per ``max_gap`` over ``seqs``."""
    base_res = score_arm(out_dir / "eval", data_dir, seqs)
    rows = [_row("baseline base", base_res)]
    payload = {"tag": tag, "seqs": seqs, "tau": tau, "arms": {
        "baseline_base": {"combined": {k: v["combined"] for k, v in base_res.items()},
                          "per_seq": {k: v["per_seq"] for k, v in base_res.items()}}}}
    if with_gta:
        gres = score_arm(out_dir / "eval_gta_tau0.040_nosplit", data_dir, seqs)
        rows.append(_row("baseline +GTA tau0.04", gres))
        payload["arms"]["baseline_gta"] = {
            "combined": {k: v["combined"] for k, v in gres.items()},
            "per_seq": {k: v["per_seq"] for k, v in gres.items()}}
    for gap in gaps:
        gtag = "inf" if gap is None else str(gap)
        st = fill_split(out_dir, seqs, max_gap=gap)
        rec = sum(v["recovered"] for v in st.values())
        tot = sum(v["rows"] for v in st.values())
        logger.info("max_gap=%s: recovered %d of %d rows (%.1f%%)", gtag, rec, tot, 100 * rec / tot)
        base = out_dir / "eval_calibfill_base"
        build_arm(data_dir, out_dir, out_dir / FILLED_SUBDIR, base, seqs)
        res = score_arm(base, data_dir, seqs)
        rows.append(_row(f"fill gap={gtag} base", res))
        payload["arms"][f"gap{gtag}_base"] = {
            "recovered_rows": rec, "total_rows": tot,
            "combined": {k: v["combined"] for k, v in res.items()},
            "per_seq": {k: v["per_seq"] for k, v in res.items()}}
        if with_gta:
            jersey_arm(out_dir, base, out_dir / "eval_calibfill_koshkina", seqs)
            arm = out_dir / "eval_calibfill_gta"
            gta_arm(data_dir, out_dir, out_dir / FILLED_SUBDIR, arm, seqs, tau=tau)
            gres = score_arm(arm, data_dir, seqs)
            rows.append(_row(f"fill gap={gtag} +GTA", gres))
            payload["arms"][f"gap{gtag}_gta"] = {
                "combined": {k: v["combined"] for k, v in gres.items()},
                "per_seq": {k: v["per_seq"] for k, v in gres.items()}}
            payload["paired_gta"] = _paired(
                payload["arms"]["baseline_gta"]["per_seq"], gres, seqs)
    _print(rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    dest = results_dir / f"gsr_calibfill_{tag}.json"
    dest.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {dest}")
    return payload


# === The GT-free identity solver on repaired positions ===========================================
#: Bundle cache for the repaired positions (never shares a directory with the on-record one).
FILLED_BUNDLES = "identity_bundles_percrop_filled"
#: The GTA connector arm built on the repaired positions -- the base every solver row rewrites.
FILLED_GTA_ARM = "eval_calibfill_gta"
#: Where the combined (GT-free + fill) recipe is pre-declared.
COMBINED_FROZEN = Path("results/gsr_calibfill_frozen.json")
#: The DEV-20 frozen repair setting (results/GSR_ASSOCIATION.md section 4.1).
FROZEN_MAX_GAP = 10


def solve_arm(data_dir: Path, out_dir: Path, names: list[str], *, tag: str, filled: bool) -> dict:
    """Run the frozen GT-free solver recipe (free team map + self roster) on one positions source.

    Args:
        data_dir: GSR ground-truth folder (scoring only).
        out_dir: Pipeline output root.
        names: Sequences to solve.
        tag: Arm tag (``outputs/.../deleak_<tag>``, and the results key).
        filled: Consume :data:`FILLED_SUBDIR` and the repaired GTA arm instead of the on-record ones.

    Returns:
        The arm payload from :func:`tools.gsr_deleak.run_arm`.
    """
    from generator.identity_solve import SolverConfig  # noqa: PLC0415

    from eval.gsr_identity import load_bundles  # noqa: PLC0415
    from tools.gsr_deleak import BASE_ARM, SOLVER_CONFIG, run_arm  # noqa: PLC0415

    pos = FILLED_SUBDIR if filled else "positions"
    bundles = load_bundles(data_dir, out_dir, names, votes_subdir="koshkina_percrop_votes",
                           cache_subdir=FILLED_BUNDLES if filled else "identity_bundles_percrop",
                           positions_subdir=pos)
    return run_arm(bundles, SolverConfig.load(SOLVER_CONFIG), data_dir, out_dir, names,
                   team_src="free", roster="self", tag=tag, score_hota=True,
                   positions_subdir=pos, base_arm=FILLED_GTA_ARM if filled else BASE_ARM)


def verify_gtfree(arm_dir: Path, base_arm: Path, data_dir: Path, pos_dir: Path,
                  names: list[str]) -> dict:
    """Audit the shipped artifact for the one place a label could still reach it: ``team``.

    The cached base arm's ``attributes.team`` was written under the GT-agreement map, so the solver
    arm must carry exactly ``free_map[cluster]``: the base value, flipped on precisely the sequences
    where the free map disagrees with the GT one. Any other row is a violation.
    """
    from eval.gsr_score import (  # noqa: PLC0415
        load_gt_people_by_frame,
        resolve_team_map,
        resolve_team_map_free,
    )

    swap = {"left": "right", "right": "left"}
    flipped, violations, n_rows = [], 0, 0
    for name in names:
        df = pd.read_parquet(pos_dir / f"{name}.parquet")
        gt_map = resolve_team_map(df, load_gt_people_by_frame(data_dir / name))
        free_map, _margin = resolve_team_map_free(df)
        flip = gt_map[0] != free_map[0]
        if flip:
            flipped.append(name)
        base = json.loads((base_arm / "predictions" / "data" / f"{name}.json")
                          .read_text(encoding="utf-8"))["predictions"]
        got = json.loads((arm_dir / "predictions" / "data" / f"{name}.json")
                         .read_text(encoding="utf-8"))["predictions"]
        if len(base) != len(got):
            raise SystemExit(f"{name}: {len(base)} base rows vs {len(got)} shipped rows")
        for b, g in zip(base, got):
            n_rows += 1
            want = b["attributes"].get("team")
            if flip and want in swap:
                want = swap[want]
            violations += int(g["attributes"].get("team") != want)
    return {"n_sequences": len(names), "n_rows": n_rows, "violations": violations,
            "flipped_sequences": flipped, "n_flipped": len(flipped)}


def freeze_combined(max_gap: int = FROZEN_MAX_GAP, path: Path = COMBINED_FROZEN) -> dict:
    """Pre-declare the combined recipe (frozen GT-free chain + the calibration-gap repair)."""
    from datetime import datetime, timezone  # noqa: PLC0415

    from generator.postprocess import MIN_ONPITCH_PLAYERS  # noqa: PLC0415
    from tools.gsr_deleak import FROZEN_PATH  # noqa: PLC0415

    base = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    record = {
        "declared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parent": {"file": str(FROZEN_PATH), "declared_at": base["declared_at"]},
        "team_map": base["team_map"],
        "roster": base["roster"],
        "solver_config": base["solver_config"],
        "solver_config_sha256": base["solver_config_sha256"],
        "calibration_fill": {
            "function": "generator.postprocess.fill_calibration_gaps",
            "max_gap": max_gap, "min_donor_rows": MIN_ONPITCH_PLAYERS,
            "applied_to": "the cached positions parquets, before the GTA connector and the solver",
            "chosen_on": "valid DEV-20 (results/GSR_ASSOCIATION.md section 4.1)",
        },
        "note": "fill chosen on DEV-20, GT-free chain chosen on train + DEV-20; the valid TEST-38 "
                "arm and the official test split are verification only",
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("frozen combined recipe -> %s", path)
    return record


def run_testsplit(data_dir: Path, out_dir: Path, results_dir: Path, *,
                  max_gap: int = FROZEN_MAX_GAP, tau: float = 0.04) -> dict:
    """The ONE test run: repair the cached test positions, reconnect, solve, score, package.

    The unrepaired GT-free arm (on record at GS-HOTA 31.88) is re-derived through this same harness
    first, so the delta is like-for-like rather than a comparison against a copied number.
    """
    from eval.gsr_identity import paired_stats  # noqa: PLC0415
    from tools.gsr_deleak import package_free, split_names  # noqa: PLC0415

    if not COMBINED_FROZEN.exists():
        raise SystemExit(f"{COMBINED_FROZEN} missing -- freeze before touching the test split")
    frozen = json.loads(COMBINED_FROZEN.read_text(encoding="utf-8"))
    names = split_names(data_dir, "test")
    control = solve_arm(data_dir, out_dir, names, tag="t49_nofill_free_self", filled=False)
    st = fill_split(out_dir, names, max_gap=max_gap)
    rec, tot = sum(v["recovered"] for v in st.values()), sum(v["rows"] for v in st.values())
    logger.info("test fill: recovered %d of %d rows (%.1f%%)", rec, tot, 100 * rec / tot)
    pos = out_dir / FILLED_SUBDIR
    base = out_dir / "eval_calibfill_base"
    build_arm(data_dir, out_dir, pos, base, names)
    jersey_arm(out_dir, base, out_dir / "eval_calibfill_koshkina", names)
    gta_arm(data_dir, out_dir, pos, out_dir / FILLED_GTA_ARM, names, tau=tau)
    arm = solve_arm(data_dir, out_dir, names, tag="t49_calibfill_free_self", filled=True)
    arm_dir = out_dir / "deleak_t49_calibfill_free_self"
    arm["legitimacy"] = verify_gtfree(arm_dir, out_dir / FILLED_GTA_ARM, data_dir, pos, names)
    arm["fill"] = {"max_gap": max_gap, "recovered_rows": rec, "total_rows": tot,
                   "per_seq": st}
    arm["manifest"] = package_free(arm_dir, names, frozen, stem="gtfree_calibfill")
    arm["control_nofill"] = control
    arm["paired_vs_control"] = paired_stats(control["gs_hota_per_seq"], arm["gs_hota_per_seq"],
                                            names)
    results_dir.mkdir(parents=True, exist_ok=True)
    dest = results_dir / "gsr_calibfill_testsplit.json"
    dest.write_text(json.dumps(arm, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", dest)
    logger.info("test-49 GT-free: control %.2f -> calibfill %.2f (paired mean %+.2f, helped %d, "
                "hurt %d, p=%.3g)", control["gs_hota"]["GS-HOTA"], arm["gs_hota"]["GS-HOTA"],
                arm["paired_vs_control"]["mean"], arm["paired_vs_control"]["helped"],
                arm["paired_vs_control"]["hurt"], arm["paired_vs_control"]["wilcoxon_p"])
    return arm


def _demo() -> None:
    """Self-check: the repair adds pitch rows and never moves an existing one."""
    from generator.postprocess import clamp_to_pitch  # noqa: PLC0415

    h = np.array([[0.06, 0.004, -8.0], [0.0008, 0.045, -3.0], [8e-6, 2.5e-4, 1.0]])
    rows = []
    for f in range(10):
        for i in range(12):
            ix, iy = 300 + 100 * i, 400 + 20 * (i % 5)
            p = np.array([ix, iy, 1.0]) @ h.T
            px, py = clamp_to_pitch(p[0] / p[2], p[1] / p[2])
            blank = f in (4, 5)
            rows.append({"frame": f, "track_id": i, "role": "player", "team": i % 2,
                         "pitch_x": np.nan if blank else px, "pitch_y": np.nan if blank else py,
                         "image_x": float(ix), "image_y": float(iy)})
    df = pd.DataFrame(rows)
    out = fill_calibration_gaps(df)
    assert np.isfinite(out["pitch_x"]).sum() > np.isfinite(df["pitch_x"]).sum()
    keep = np.isfinite(df["pitch_x"])
    assert np.allclose(out.loc[keep, "pitch_x"], df.loc[keep, "pitch_x"])
    _demo_verify()
    print("gsr_calibfill self-check OK")


def _demo_verify() -> None:
    """Self-check for :func:`verify_gtfree`: a shipped side that is not ``free_map[cluster]`` fails."""
    import tempfile  # noqa: PLC0415

    root = Path(tempfile.mkdtemp(prefix="gsr_calibfill_verify_"))
    # Cluster 0 deep left -> free map {0: left}; the GT people put cluster 0 on the RIGHT, so the
    # GT-agreement map disagrees and every base row must come out flipped.
    df = pd.DataFrame({"frame": [0] * 4, "track_id": [0, 1, 2, 3], "role": ["player"] * 4,
                       "team": [0, 0, 1, 1], "pitch_x": [20.0, 25.0, 80.0, 85.0],
                       "pitch_y": [30.0, 34.0, 30.0, 34.0]})
    (root / "pos").mkdir(parents=True)
    df.to_parquet(root / "pos" / "S.parquet", index=False)
    for arm, sides in (("base", ["right", "right", "left", "left"]),
                       ("ship", ["left", "left", "right", "right"])):
        d = root / arm / "predictions" / "data"
        d.mkdir(parents=True)
        (d / "S.json").write_text(json.dumps({"predictions": [
            {"track_id": i, "attributes": {"role": "player", "team": s}}
            for i, s in enumerate(sides)]}), encoding="utf-8")
    seq = root / "gt" / "S"
    seq.mkdir(parents=True)
    (seq / "Labels-GameState.json").write_text(json.dumps({"images": [], "annotations": []}),
                                               encoding="utf-8")

    def _fake_gt(_seq_dir: Path) -> dict:
        # Same centred points as the df rows, but cluster 0's players are GT 'right' -> the
        # GT-agreement map is {0: right} while the geometry (deeper mean x) says {0: left}.
        return {0: [(-32.5, -4.0, "player", "right"), (-27.5, 0.0, "player", "right"),
                    (27.5, -4.0, "player", "left"), (32.5, 0.0, "player", "left")]}

    import eval.gsr_score as gs  # noqa: PLC0415

    real, gs.load_gt_people_by_frame = gs.load_gt_people_by_frame, _fake_gt
    try:
        ok = verify_gtfree(root / "ship", root / "base", root / "gt", root / "pos", ["S"])
        bad = verify_gtfree(root / "base", root / "base", root / "gt", root / "pos", ["S"])
    finally:
        gs.load_gt_people_by_frame = real
    assert ok == {"n_sequences": 1, "n_rows": 4, "violations": 0,
                  "flipped_sequences": ["S"], "n_flipped": 1}, ok
    assert bad["violations"] == 4, bad


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--tau", type=float, default=0.04)
    ap.add_argument("--max-gap", type=int, default=None, help="0 or negative means unlimited")
    ap.add_argument("--gaps", default=None, help="comma-separated max_gap sweep (DEV only)")
    ap.add_argument("--dev", action="store_true", help="max_gap sweep on the declared DEV-20")
    ap.add_argument("--run", action="store_true", help="frozen max_gap over the full valid split")
    ap.add_argument("--recover-audit", action="store_true")
    ap.add_argument("--no-gta", action="store_true")
    ap.add_argument("--solve-valid", action="store_true",
                    help="GT-free solver on valid TEST-38, no-fill control + fill arm")
    ap.add_argument("--freeze", action="store_true", help="pre-declare the combined recipe")
    ap.add_argument("--solve-test", action="store_true",
                    help="the ONE test run: fill -> connector -> solve -> score -> package")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return
    if args.freeze:
        freeze_combined(args.max_gap or FROZEN_MAX_GAP)
        return
    if args.solve_test:
        run_testsplit(args.data_dir, args.out_dir, args.results_dir,
                      max_gap=args.max_gap or FROZEN_MAX_GAP, tau=args.tau)
        return
    if args.solve_valid:
        from eval.gsr_identity import paired_stats, split_sequences  # noqa: PLC0415

        _dev, t38 = split_sequences(args.data_dir, args.out_dir)
        arms = {}
        for filled in (False, True):
            tag = f"t38_{'calibfill' if filled else 'nofill'}_free_self"
            arms[tag] = solve_arm(args.data_dir, args.out_dir, t38, tag=tag, filled=filled)
        a, b = arms["t38_nofill_free_self"], arms["t38_calibfill_free_self"]
        paired = paired_stats(a["gs_hota_per_seq"], b["gs_hota_per_seq"], t38)
        print(f"no-fill {a['gs_hota']['GS-HOTA']:.2f} -> fill {b['gs_hota']['GS-HOTA']:.2f} "
              f"(paired mean {paired['mean']:+.2f}, helped {paired['helped']}, "
              f"hurt {paired['hurt']}, p={paired['wilcoxon_p']:.3g})")
        args.results_dir.mkdir(parents=True, exist_ok=True)
        dest = args.results_dir / "gsr_calibfill_solve_valid38.json"
        dest.write_text(json.dumps({"test38": t38, "arms": arms, "paired": paired}, indent=2,
                                   default=str), encoding="utf-8")
        print(f"wrote {dest}")
        return
    dev, test = dev_test_split(args.out_dir)
    gap = None if args.max_gap is None or args.max_gap <= 0 else args.max_gap
    if args.recover_audit:
        for g in ([int(x) for x in args.gaps.split(",")] if args.gaps else [gap]):
            g2 = None if g is None or g <= 0 else g
            print(f"max_gap={g2}", recover_audit(args.data_dir, args.out_dir, dev, g2))
        return
    if args.dev:
        gaps = [None if int(x) <= 0 else int(x)
                for x in (args.gaps or "0").split(",")]
        run(args.data_dir, args.out_dir, args.results_dir, dev, "dev20", gaps,
            tau=args.tau, with_gta=not args.no_gta)
        return
    if args.run:
        run(args.data_dir, args.out_dir, args.results_dir, sorted(dev + test), "valid58", [gap],
            tau=args.tau, with_gta=not args.no_gta)
        return
    ap.error("choose --recover-audit / --dev / --run / --solve-valid / --freeze / --solve-test "
             "/ --demo")


if __name__ == "__main__":
    main()
