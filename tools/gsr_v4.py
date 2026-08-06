"""GSR campaign v4: measure four candidate levers on top of the frozen v3 (calibgate) recipe.

v3 (`results/GSR_CALIBGATE.md`) ships at GS-HOTA **35.40** on the official test split, built from
re-gated positions (``positions_gate``) + the GTA connector at ``tau = 0.040`` + the frozen GT-free
solver. This module measures, on the declared **DEV-20** partition and nowhere else, whether four
unfired levers move that arm:

1. **crop_scale 1.25 OCR** -- the widened crop that bought 1.37x reads on our EPL broadcast. On GSR
   the crop is the *detector's own box*, not the ``estimate_player_box`` reconstruction the 1.25
   constant was fitted to repair, so the transfer is a genuine open question (measured: it is not).
2. **Aggregation floor** -- the shipped 0.85-floor rule (``min_crop_conf 0.99``, ``min_votes 5``)
   against the denser 0.80-floor rule (``0.90`` / ``3``). The 80% bar of ``tools/prtreid_probe.py``
   governs per-player FACTS; GS-HOTA is a different objective and may want a different point.
3. **Connector tau** -- re-swept on the CALIBGATE-repaired positions (18% more pitch rows = a
   stronger spatial-feasibility constraint), with and without the new **jersey-compatibility merge
   gate** (:func:`generator.gta_link.connect`'s ``numbers``: two tracklets whose confident reads
   disagree may never merge, whatever appearance says).
4. **Truncation-aware digit prior** -- *not built*; see ``--trunc-probe``, which measures the
   mechanism's frequency on GSR before anyone spends code on it.

Every arm is the SAME chain with one variable changed, and the v3 arm is re-derived inside each run
rather than quoted. CPU only (the crop_scale pass is ``eval.gsr_jersey --percrop --crop-scale``).

CLI::

    python -m tools.gsr_v4 --trunc-probe             # component 4's premise, on DEV reads
    python -m tools.gsr_v4 --cropscale --seqs <DEV>  # component 1: d + precision at 1.0 vs 1.25
    python -m tools.gsr_v4 --dev                     # components 2 + 3 + combinations, DEV-20
    python -m tools.gsr_v4 --freeze --bundle <tag>   # pre-declare the winning bundle
    python -m tools.gsr_v4 --valid                   # ONE verification run on valid TEST-38
    python -m tools.gsr_v4 --demo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eval.gsr_score import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, DEFAULT_RESULTS_DIR
from tools.gsr_calibgate import GATED_SUBDIR, apply_split

logger = logging.getLogger("gsr_v4")

#: Where the winning bundle is pre-declared, before any valid/test run.
FROZEN_PATH = Path("results/gsr_v4_frozen.json")
#: The frozen aggregation rules of `results/OCR_DENSIFICATION.md`, one per DEV precision floor.
RULE_PATH = Path("results/ocr_density_rule.json")
#: v3's operating point: the 0.85-floor rule, tau 0.040, no jersey gate, crop_scale 1.0.
V3 = {"floor": "0.85", "tau": 0.04, "gate": False, "variant": "", "refit": False,
      "app_gain": None, "sim_none": None}


def rule_for(floor: str, variant: str = "") -> dict:
    """The frozen aggregation rule for a DEV read-precision floor (pure lookup).

    An evidence variant with its OWN frozen sweep (``results/ocr_density_rule<variant>.json``, written
    by ``tools.ocr_density --sweep --variant``) uses that rule; anything else falls back to the
    on-record freeze, so every existing arm is unchanged.
    """
    from tools.ocr_density import rule_path  # noqa: PLC0415

    p = rule_path(variant)
    return json.loads((p if p.exists() else RULE_PATH).read_text(encoding="utf-8"))["rules"][floor]


def config_key(floor: str, tau: float, gate: bool, variant: str, refit: bool) -> str:
    """Directory-safe identity of one arm: every knob that changes its output appears.

    The appearance embedder is part of the arm's identity too -- without it a CLIP arm at a
    tau an on-record PRTreID arm already used would silently reuse that arm's bundle cache.
    The default embedder contributes no suffix, so every on-record directory name is
    unchanged.
    """
    from eval.gsr_gta import EMBEDDER  # noqa: PLC0415

    return (f"f{floor.replace('.', '')}{variant}_t{tau:.3f}"
            f"{'_jg' if gate else ''}{'_refit' if refit else ''}"
            f"{'' if EMBEDDER == 'prtreid' else '_' + EMBEDDER}")


def votes_dir(out_dir: Path, floor: str, variant: str) -> Path:
    """Per-track densified vote cache for one (floor, crop-geometry) pair, built on demand.

    The directory name encodes the floor and the evidence variant but NOT the aggregation rule, so
    two arms that differ only in the rule (S3's arm A vs arm B: ``results/GSR_S3_READER.md`` §7)
    used to overwrite each other's votes silently. The rule is therefore persisted next to the votes
    and any mismatch wipes the cache instead of reusing it.
    """
    from tools.ocr_density import emit_votes  # noqa: PLC0415

    dest = out_dir / f"koshkina_percrop_votes_f{floor.replace('.', '')}{variant}"
    src = out_dir / f"koshkina_percrop{variant}"
    rule = rule_for(floor, variant)
    stamp = dest / "_rule.json"
    if dest.exists() and (not stamp.exists()
                          or json.loads(stamp.read_text(encoding="utf-8")) != rule):
        logger.warning("votes cache %s was built under a different rule -- rebuilding", dest.name)
        shutil.rmtree(dest, ignore_errors=True)
    missing = [p.stem for p in sorted(src.glob("*.parquet"))
               if not (dest / f"{p.stem}.json").exists()]
    if missing:
        emit_votes(out_dir, missing, rule, votes_subdir=dest.name, variant=variant)
    dest.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps(rule), encoding="utf-8")
    return dest


# === One arm =====================================================================================
def solver_config(app_gain: float | None = None, sim_none: float | None = None):
    """The frozen solver config with its appearance scale optionally overridden.

    ``app_gain`` and ``sim_none`` are in **cosine-similarity units** and were fitted to PRTreID's
    distribution, so a different embedding space needs them re-scaled (``results/GSR_V5.md``).
    Both ``None`` returns the on-disk config untouched, which is what every on-record arm used.

    Args:
        app_gain: Inverse temperature on the appearance likelihood, or ``None`` to keep the frozen
            value.
        sim_none: Similarity credited to the ``unknown`` class, or ``None`` to keep the frozen one.

    Returns:
        A :class:`generator.identity_solve.SolverConfig`.
    """
    from generator.identity_solve import SolverConfig  # noqa: PLC0415

    from tools.gsr_deleak import SOLVER_CONFIG  # noqa: PLC0415

    cfg = SolverConfig.load(SOLVER_CONFIG)
    over = {k: v for k, v in (("app_gain", app_gain), ("sim_none", sim_none)) if v is not None}
    return replace(cfg, **over) if over else cfg


def solve_v4_arm(data_dir: Path, out_dir: Path, seqs: list[str], *, tag: str, floor: str = "0.85",
                 tau: float = 0.04, gate: bool = False, variant: str = "", refit: bool = False,
                 dev_bundles_for_prior: dict | None = None, app_gain: float | None = None,
                 sim_none: float | None = None, positions_subdir: str | None = None) -> dict:
    """Build and score one v4 arm on the re-gated positions -> the ``run_arm`` payload + diagnostics.

    Args:
        data_dir: GSR ground-truth folder (scoring only).
        out_dir: Pipeline output root (must already hold ``positions_gate``).
        seqs: Sequences to score.
        tag: Arm tag (names ``deleak_<tag>`` and the results key).
        floor: Which frozen aggregation rule supplies the jersey evidence.
        tau: Connector merge ceiling.
        gate: Switch on the jersey-compatibility merge gate.
        variant: Per-crop evidence variant (``""`` = crop_scale 1.0, ``"_w125"`` = 1.25).
        refit: Refit the digit-confusion prior on ``dev_bundles_for_prior`` (which MUST be DEV).
        dev_bundles_for_prior: DEV bundles of this same configuration; required when ``refit``.
        app_gain: Override the solver's appearance inverse temperature (``None`` = frozen value).
        sim_none: Override the solver's ``unknown``-class similarity (``None`` = frozen value).
        positions_subdir: Positions source for the whole arm (submission, connector, bundles and
            the free team map). ``None`` keeps :data:`tools.gsr_calibgate.GATED_SUBDIR`, which is
            what every on-record arm used; :mod:`tools.gsr_eiou` points it at its re-associated
            variant.

    Returns:
        The arm payload with ``gs_hota``, ``gs_hota_per_seq`` and a ``v4`` diagnostics block.
    """
    from generator.gta_link import GtaParams  # noqa: PLC0415

    from eval.gsr_identity import fit_confusion, load_bundles  # noqa: PLC0415
    from tools.gsr_calibfill import build_arm, gta_arm, jersey_arm  # noqa: PLC0415
    from tools.gsr_calibgate import BASE_ARM_DIR, KOSHKINA_ARM_DIR  # noqa: PLC0415
    from tools.gsr_deleak import run_arm  # noqa: PLC0415

    key = config_key(floor, tau, gate, variant, refit)
    pos_sub = positions_subdir or GATED_SUBDIR
    pos = out_dir / pos_sub
    vdir = votes_dir(out_dir, floor, variant)
    gta_dir = out_dir / f"eval_v4_gta_{key}"
    params = GtaParams(tau=tau, eps=0.30, min_samples=5, min_run=5, frame_stride=2)

    build_arm(data_dir, out_dir, pos, out_dir / BASE_ARM_DIR, seqs)
    jersey_arm(out_dir, out_dir / BASE_ARM_DIR, out_dir / KOSHKINA_ARM_DIR, seqs)
    st = gta_arm(data_dir, out_dir, pos, gta_dir, seqs, tau=tau,
                 jersey_dir=out_dir / KOSHKINA_ARM_DIR,
                 gate_votes_dir=vdir if gate else None)
    bundles = load_bundles(data_dir, out_dir, seqs, votes_subdir=vdir.name,
                           cache_subdir=f"identity_bundles_v4_{key}",
                           positions_subdir=pos_sub, params=params, jersey_gate=gate)
    cfg = solver_config(app_gain, sim_none)
    prior = None
    if refit:
        src = dev_bundles_for_prior if dev_bundles_for_prior is not None else bundles
        _pc, prior = fit_confusion(src, data_dir)
        cfg = replace(cfg, confusion=prior)
    payload = run_arm(bundles, cfg, data_dir, out_dir, seqs, team_src="free", roster="self",
                      tag=tag, score_hota=True, positions_subdir=pos_sub,
                      base_arm=gta_dir.name)
    c, t = (int(sum(v.get("merge_precision_correct", 0) for v in st.values())),
            int(sum(v.get("merge_precision_total", 0) for v in st.values())))
    payload["v4"] = {
        "key": key, "floor": floor, "tau": tau, "jersey_gate": gate, "variant": variant,
        "confusion_refit": refit, "app_gain": cfg.app_gain, "sim_none": cfg.sim_none,
        "merge_precision": {"correct": c, "total": t, "precision": c / t if t else float("nan")},
        "n_merges": int(sum(v.get("n_merges", 0) for v in st.values())),
        "frags_after": int(sum(v.get("n_fragments_after", 0) for v in st.values())),
        "frag_per_gt_after": float(np.mean([v["frag_per_gt_after"] for v in st.values()]))
        if st else float("nan"),
        "confusion_n": None if prior is None else {k: prior[k] for k in ("n", "n_wrong",
                                                                        "n_aligned")},
    }
    v = payload["v4"]
    logger.info("%-26s GS-HOTA %.4f DetA %.4f AssA %.4f LocA %.4f IDF1 %.4f | merges %d prec %.3f",
                tag, payload["gs_hota"]["GS-HOTA"], payload["gs_hota"]["GS-DetA"],
                payload["gs_hota"]["GS-AssA"], payload["gs_hota"]["GS-LocA"],
                payload["gs_hota"]["IDF1"], v["n_merges"], v["merge_precision"]["precision"])
    return payload


# === Component 1: the widened crop ===============================================================
def cropscale_report(data_dir: Path, out_dir: Path, seqs: list[str]) -> dict:
    """Paired crop-level and tracklet-level comparison of crop_scale 1.0 vs 1.25 on ``seqs``.

    The two passes sample the identical ``(track_id, frame)`` crops, so the comparison is paired at
    the crop level rather than sampled.
    """
    import pandas as pd  # noqa: PLC0415

    from tools.ocr_density import load_gt_cache, load_percrop, measure  # noqa: PLC0415

    gt = load_gt_cache(data_dir, out_dir, seqs)
    a, b = load_percrop(out_dir, seqs), load_percrop(out_dir, seqs, "_w125")
    seqs = [s for s in seqs if s in a and s in b]
    crop = {"n": 0, "shared": 0, "conf_10": 0, "conf_125": 0, "leg_10": 0, "leg_125": 0,
            "gain": 0, "lost": 0, "both": 0, "agree": 0}
    for s in seqs:
        m = a[s].merge(b[s], on=["track_id", "frame"], suffixes=("_a", "_b"))
        ca = (m["number_a"] > 0) & (m["p_number_a"] >= 0.99)  # noqa: PLR2004
        cb = (m["number_b"] > 0) & (m["p_number_b"] >= 0.99)  # noqa: PLR2004
        crop["n"] += len(a[s])
        crop["shared"] += len(m)
        crop["leg_10"] += int((a[s]["legibility"] >= 0.5).sum())  # noqa: PLR2004
        crop["leg_125"] += int((b[s]["legibility"] >= 0.5).sum())  # noqa: PLR2004
        crop["conf_10"] += int(ca.sum())
        crop["conf_125"] += int(cb.sum())
        crop["gain"] += int((cb & ~ca).sum())
        crop["lost"] += int((ca & ~cb).sum())
        crop["both"] += int((ca & cb).sum())
        crop["agree"] += int((m.loc[ca & cb, "number_a"] == m.loc[ca & cb, "number_b"]).sum())
    arms = {}
    for floor in ("0.80", "0.85"):
        rule = rule_for(floor)
        arms[floor] = {"crop10": measure({s: a[s] for s in seqs}, gt, rule),
                       "crop125": measure({s: b[s] for s in seqs}, gt, rule)}
    del pd
    return {"seqs": seqs, "crop_level": crop, "rule_arms": arms}


# === Component 4: the truncation premise, measured before it is built ============================
def truncation_probe(data_dir: Path, out_dir: Path, seqs: list[str]) -> dict:
    """How often a wrong GSR read is a *truncation* of the true number (the '81 -> 8' mode).

    ``results/OCR_DOMAIN_SHIFT.md`` §3.1 measured 44% of wrong confident FOOTPASS reads as exactly
    the first digit of the true number. A truncation-aware digit prior is only worth building if
    that mode exists here; this is the measurement, on DEV reads only.
    """
    from tools.ocr_density import load_gt_cache, load_percrop, reads_for_sequence  # noqa: PLC0415

    gt = load_gt_cache(data_dir, out_dir, seqs)
    per = load_percrop(out_dir, seqs)
    out: dict[str, dict] = {}
    for floor in ("0.80", "0.85"):
        pairs: list[tuple[int, int]] = []
        for name, df in per.items():
            g = gt.get(name, {})
            for tid, votes in reads_for_sequence(df, rule_for(floor)).items():
                true = g.get(tid)
                if true in (None, "missing"):
                    continue
                pairs.extend((int(true), int(n)) for n, _c in votes)
        wrong = [(t, r) for t, r in pairs if t != r]
        trunc = [(t, r) for t, r in wrong
                 if len(str(t)) > len(str(r)) and str(t).startswith(str(r))]
        lenmis = [(t, r) for t, r in wrong if len(str(t)) != len(str(r))]
        out[floor] = {
            "n_reads": len(pairs), "n_wrong": len(wrong),
            "read_precision": 1.0 - len(wrong) / max(len(pairs), 1),
            "n_length_mismatch": len(lenmis), "n_truncation": len(trunc),
            "truncation_share_of_wrong": len(trunc) / max(len(wrong), 1),
            "top_wrong": [[list(k), v] for k, v in Counter(wrong).most_common(8)],
        }
        logger.info("floor %s: %d reads, %d wrong (prec %.4f), %d length mismatch, "
                    "%d truncation (%.1f%% of wrong)", floor, out[floor]["n_reads"],
                    out[floor]["n_wrong"], out[floor]["read_precision"],
                    out[floor]["n_length_mismatch"], out[floor]["n_truncation"],
                    100 * out[floor]["truncation_share_of_wrong"])
    lens = Counter(len(str(j)) for n in seqs for j in gt.get(n, {}).values() if j)
    out["gt_jersey_digit_lengths"] = dict(lens)
    return out


# === DEV campaign ================================================================================
def dev_campaign(data_dir: Path, out_dir: Path, dev: list[str], arms: list[dict]) -> dict:
    """Score every declared arm on DEV-20, paired against the re-derived v3 control."""
    from eval.gsr_identity import load_bundles, paired_stats  # noqa: PLC0415
    from generator.gta_link import GtaParams  # noqa: PLC0415

    apply_split(out_dir, dev, fill_gap=10)
    out: dict[str, dict] = {}
    for spec in arms:
        cfg = {**V3, **spec}
        tag = f"v4dev_{config_key(cfg['floor'], cfg['tau'], cfg['gate'], cfg['variant'], cfg['refit'])}"
        if cfg["app_gain"] is not None or cfg["sim_none"] is not None:
            tag += f"_a{cfg['app_gain'] or 0:g}s{cfg['sim_none'] or 0:g}"
        prior_src = None
        if cfg["refit"]:
            key = config_key(cfg["floor"], cfg["tau"], cfg["gate"], cfg["variant"], cfg["refit"])
            prior_src = load_bundles(
                data_dir, out_dir, dev, votes_subdir=votes_dir(out_dir, cfg["floor"],
                                                               cfg["variant"]).name,
                cache_subdir=f"identity_bundles_v4_{key}", positions_subdir=GATED_SUBDIR,
                params=GtaParams(tau=cfg["tau"], eps=0.30, min_samples=5, min_run=5,
                                 frame_stride=2), jersey_gate=cfg["gate"])
        out[spec.get("name", tag)] = solve_v4_arm(
            data_dir, out_dir, dev, tag=tag, floor=cfg["floor"], tau=cfg["tau"], gate=cfg["gate"],
            variant=cfg["variant"], refit=cfg["refit"], dev_bundles_for_prior=prior_src,
            app_gain=cfg["app_gain"], sim_none=cfg["sim_none"])
    base = out.get("v3")
    if base is not None:
        for name, a in out.items():
            if name == "v3":
                continue
            a["paired_vs_v3"] = paired_stats(base["gs_hota_per_seq"], a["gs_hota_per_seq"], dev)
    return out


def freeze_bundle(spec: dict, dev_result: dict, path: Path = FROZEN_PATH) -> dict:
    """Pre-declare the winning v4 bundle, hash-linked to the frozen v3 recipe."""
    from tools.gsr_calibgate import FROZEN_PATH as V3_PATH  # noqa: PLC0415

    parent = json.loads(V3_PATH.read_text(encoding="utf-8"))
    cfg = {**V3, **spec}
    record = {
        "declared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parent": {"file": str(V3_PATH), "declared_at": parent["declared_at"],
                   "sha256": hashlib.sha256(V3_PATH.read_bytes()).hexdigest()},
        "team_map": parent["team_map"], "roster": parent["roster"],
        "solver_config": parent["solver_config"],
        "solver_config_sha256": parent["solver_config_sha256"],
        "calibration_fill": parent["calibration_fill"],
        "calibration_gate": parent["calibration_gate"],
        "v4_bundle": {
            "aggregation_floor": cfg["floor"], "aggregation_rule": rule_for(cfg["floor"]),
            "connector_tau": cfg["tau"], "jersey_compatible_merge": cfg["gate"],
            "crop_scale": 1.25 if cfg["variant"] else 1.0,
            "confusion_prior_refit_on_dev": cfg["refit"],
            "chosen_on": "valid DEV-20 (results/GSR_V4.md section 3)",
        },
        "dev20_gs_hota": dev_result["gs_hota"]["GS-HOTA"],
        "note": "bundle chosen on DEV-20; valid TEST-38 and the official test split are "
                "verification only",
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("frozen v4 bundle -> %s", path)
    return record


def verify_valid(data_dir: Path, out_dir: Path, t38: list[str], dev: list[str]) -> dict:
    """The ONE valid TEST-38 run at the frozen bundle, with the v3 arm re-derived as the control."""
    from eval.gsr_identity import load_bundles, paired_stats  # noqa: PLC0415
    from generator.gta_link import GtaParams  # noqa: PLC0415

    if not FROZEN_PATH.exists():
        raise SystemExit(f"{FROZEN_PATH} missing -- freeze the bundle before touching TEST-38")
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))["v4_bundle"]
    spec = {"floor": frozen["aggregation_floor"], "tau": frozen["connector_tau"],
            "gate": frozen["jersey_compatible_merge"],
            "variant": "_w125" if frozen["crop_scale"] != 1.0 else "",
            "refit": frozen["confusion_prior_refit_on_dev"]}
    apply_split(out_dir, sorted(set(t38) | set(dev)), fill_gap=10)
    arms = {"v3": solve_v4_arm(data_dir, out_dir, t38, tag="v4t38_v3", **V3)}
    prior_src = None
    if spec["refit"]:
        key = config_key(spec["floor"], spec["tau"], spec["gate"], spec["variant"], spec["refit"])
        prior_src = load_bundles(
            data_dir, out_dir, dev,
            votes_subdir=votes_dir(out_dir, spec["floor"], spec["variant"]).name,
            cache_subdir=f"identity_bundles_v4_{key}", positions_subdir=GATED_SUBDIR,
            params=GtaParams(tau=spec["tau"], eps=0.30, min_samples=5, min_run=5, frame_stride=2),
            jersey_gate=spec["gate"])
    arms["v4"] = solve_v4_arm(data_dir, out_dir, t38, tag="v4t38_v4",
                              dev_bundles_for_prior=prior_src, **spec)
    paired = paired_stats(arms["v3"]["gs_hota_per_seq"], arms["v4"]["gs_hota_per_seq"], t38)
    logger.info("TEST-38: v3 %.4f -> v4 %.4f (paired mean %+.3f helped %d hurt %d worst %+.3f "
                "p=%.3g)", arms["v3"]["gs_hota"]["GS-HOTA"], arms["v4"]["gs_hota"]["GS-HOTA"],
                paired["mean"], paired["helped"], paired["hurt"], paired["worst"],
                paired["wilcoxon_p"])
    return {"seqs": t38, "spec": spec, "arms": arms, "paired_v4_vs_v3": paired}


def run_testsplit(data_dir: Path, out_dir: Path, results_dir: Path) -> dict:
    """The ONE official-test run: v3 control re-derived, v4 at the frozen bundle, audit, package.

    CPU only -- the frozen bundle moved no GPU-side knob, so every cached test-49 artifact (candidate
    homographies, per-crop OCR, PRTreID embeddings) is consumed unchanged.
    """
    from eval.gsr_identity import paired_stats  # noqa: PLC0415
    from tools.gsr_calibfill import verify_gtfree  # noqa: PLC0415
    from tools.gsr_calibgate import zip_selfscore  # noqa: PLC0415
    from tools.gsr_deleak import package_free, split_names  # noqa: PLC0415

    if not FROZEN_PATH.exists():
        raise SystemExit(f"{FROZEN_PATH} missing -- freeze before touching the test split")
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    b = frozen["v4_bundle"]
    spec = {"floor": b["aggregation_floor"], "tau": b["connector_tau"],
            "gate": b["jersey_compatible_merge"],
            "variant": "_w125" if b["crop_scale"] != 1.0 else "",
            "refit": b["confusion_prior_refit_on_dev"]}
    names = split_names(data_dir, "test")
    apply_split(out_dir, names, fill_gap=10)
    arms = {"v3": solve_v4_arm(data_dir, out_dir, names, tag="t49_v4campaign_v3", **V3),
            "v4": solve_v4_arm(data_dir, out_dir, names, tag="t49_v4campaign_v4", **spec)}
    arm_dir = out_dir / "deleak_t49_v4campaign_v4"
    gta_dir = out_dir / ("eval_v4_gta_" + config_key(spec["floor"], spec["tau"], spec["gate"],
                                                     spec["variant"], spec["refit"]))
    arms["v4"]["legitimacy"] = verify_gtfree(arm_dir, gta_dir, data_dir,
                                             out_dir / GATED_SUBDIR, names)
    arms["v4"]["manifest"] = package_free(arm_dir, names, frozen, stem="gtfree_v4")
    zip_path = Path(arms["v4"]["manifest"]["zip"])
    arms["v4"]["zip_selfscore"] = zip_selfscore(
        zip_path, data_dir, names, zip_path.parent / "zip_selfscore_gtfree_v4.json")["combined"]
    res = {"seqs": names, "spec": spec, "arms": arms,
           "paired_v4_vs_v3": paired_stats(arms["v3"]["gs_hota_per_seq"],
                                           arms["v4"]["gs_hota_per_seq"], names)}
    results_dir.mkdir(parents=True, exist_ok=True)
    dest = results_dir / "gsr_v4_testsplit.json"
    dest.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    p = res["paired_v4_vs_v3"]
    logger.info("test-49: v3 %.4f -> v4 %.4f (paired mean %+.3f helped %d hurt %d worst %+.3f "
                "p=%.3g)", arms["v3"]["gs_hota"]["GS-HOTA"], arms["v4"]["gs_hota"]["GS-HOTA"],
                p["mean"], p["helped"], p["hurt"], p["worst"], p["wilcoxon_p"])
    logger.info("legitimacy: %s", arms["v4"]["legitimacy"])
    logger.info("zip self-score: %s", arms["v4"]["zip_selfscore"])
    logger.info("wrote %s", dest)
    return res


def _demo() -> None:
    """Self-check: the jersey gate blocks a conflicting merge and the config key is injective."""
    from generator.gta_link import GtaParams, connect  # noqa: PLC0415
    from generator.track_relink import Fragment  # noqa: PLC0415

    from eval.gsr_identity import dominant_numbers  # noqa: PLC0415

    e = {1: np.array([1.0, 0.0]), 2: np.array([1.0, 0.0])}
    f1 = Fragment(1, 0, 10, (10.0, 10.0), (12.0, 10.0), "player", 0, 11)
    f2 = Fragment(2, 15, 25, (12.5, 10.0), (14.0, 10.0), "player", 0, 11)
    p = GtaParams()
    assert connect([f1, f2], e, {1: 5, 2: 5}, p)[2] == 1
    assert connect([f1, f2], e, {1: 5, 2: 5}, p, numbers={1: 7, 2: 9})[2] == 2
    assert dominant_numbers({3: [(7, 0.4), (9, 0.9)], 4: [], 5: [(-1, 0.9)]}) == {3: 9}
    keys = {config_key(f, t, g, v, r) for f in ("0.80", "0.85") for t in (0.04, 0.05)
            for g in (False, True) for v in ("", "_w125") for r in (False, True)}
    assert len(keys) == 32, keys
    # The solver-scale hook is default-preserving: no override -> the frozen config, byte-for-byte.
    from tools.gsr_deleak import SOLVER_CONFIG  # noqa: PLC0415

    frozen = json.loads(SOLVER_CONFIG.read_text(encoding="utf-8"))
    assert solver_config() == solver_config(None, None)
    assert solver_config().app_gain == frozen["app_gain"], solver_config()
    assert solver_config().sim_none == frozen["sim_none"], solver_config()
    tuned = solver_config(3.0, 0.67)
    assert (tuned.app_gain, tuned.sim_none) == (3.0, 0.67), tuned
    assert tuned.p_correct == frozen["p_correct"], tuned  # nothing else moves
    print("gsr_v4 self-check OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--trunc-probe", action="store_true")
    ap.add_argument("--cropscale", action="store_true")
    ap.add_argument("--dev", action="store_true", help="components 2+3 and combinations on DEV-20")
    ap.add_argument("--arms", default=None, help="JSON list of arm specs (default: the declared set)")
    ap.add_argument("--split", choices=("dev", "t38"), default="dev",
                    help="partition the --dev arms run on (t38 is a VERIFICATION spend)")
    ap.add_argument("--freeze", default=None, help="JSON spec of the winning bundle")
    ap.add_argument("--valid", action="store_true", help="the ONE valid TEST-38 verification run")
    ap.add_argument("--test", action="store_true",
                    help="the ONE official test-49 run: solve -> score -> audit -> package")
    ap.add_argument("--tag", default="dev20")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return

    from eval.gsr_identity import split_sequences  # noqa: PLC0415

    dev, t38 = split_sequences(args.data_dir, args.out_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.trunc_probe:
        res = truncation_probe(args.data_dir, args.out_dir, dev)
        (args.results_dir / "gsr_v4_truncation_probe.json").write_text(
            json.dumps(res, indent=2), encoding="utf-8")
        return
    if args.cropscale:
        res = cropscale_report(args.data_dir, args.out_dir, dev)
        c = res["crop_level"]
        print(f"crops {c['n']} paired {c['shared']} | legible {c['leg_10']} -> {c['leg_125']} | "
              f"confident {c['conf_10']} -> {c['conf_125']} "
              f"({c['conf_125'] / max(c['conf_10'], 1):.3f}x) | gained {c['gain']} lost {c['lost']} "
              f"| both {c['both']} agree {c['agree']}")
        for floor, arm in res["rule_arms"].items():
            for k, m in arm.items():
                print(f"  floor {floor} {k:<8} d {m['d']:.4f} ({m['n_tracks_read']}/"
                      f"{m['n_tracks']}) read precision {m['read_precision']:.4f} "
                      f"(n={m['n_reads_auditable']})")
        (args.results_dir / "gsr_v4_cropscale_dev20.json").write_text(
            json.dumps(res, indent=2, default=str), encoding="utf-8")
        return
    if args.dev:
        specs = json.loads(args.arms) if args.arms else DECLARED_ARMS
        seqs = t38 if args.split == "t38" else dev
        res = dev_campaign(args.data_dir, args.out_dir, seqs, specs)
        dest = args.results_dir / f"gsr_v4_{args.tag}.json"
        dest.write_text(json.dumps({args.split: seqs, "arms": res}, indent=1, default=str),
                        encoding="utf-8")
        print(f"wrote {dest}")
        return
    if args.freeze:
        spec = json.loads(args.freeze)
        blob = json.loads((args.results_dir / f"gsr_v4_{args.tag}.json").read_text(
            encoding="utf-8"))
        print(json.dumps(freeze_bundle(spec, blob["arms"][spec["name"]]), indent=2))
        return
    if args.test:
        run_testsplit(args.data_dir, args.out_dir, args.results_dir)
        return
    if args.valid:
        res = verify_valid(args.data_dir, args.out_dir, t38, dev)
        dest = args.results_dir / "gsr_v4_valid_t38.json"
        dest.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        print(f"wrote {dest}")
        return
    ap.error("choose --trunc-probe / --cropscale / --dev / --freeze / --valid / --demo")


#: The arms declared for the DEV-20 campaign (component-alone first, then combinations).
DECLARED_ARMS: list[dict] = [
    {"name": "v3"},
    {"name": "C1_w125", "variant": "_w125"},
    {"name": "C2_f080", "floor": "0.80"},
    {"name": "C2_f080_refit", "floor": "0.80", "refit": True},
    {"name": "C3_tau030", "tau": 0.030},
    {"name": "C3_tau045", "tau": 0.045},
    {"name": "C3_tau050", "tau": 0.050},
    {"name": "C3_tau060", "tau": 0.060},
    {"name": "C3_jg_tau040", "gate": True},
    {"name": "C3_jg_tau050", "tau": 0.050, "gate": True},
    {"name": "C3_jg_tau060", "tau": 0.060, "gate": True},
]

if __name__ == "__main__":
    main()
