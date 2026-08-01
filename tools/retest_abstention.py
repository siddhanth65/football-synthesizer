"""DEV-only tuning of the Stage-2 abstention regime, for the pre-registered confirmatory retest.

``results/IDENTITY_SOLVER_STAGE2.md`` section 8 closes with a instruction the project has not yet
acted on: *"Any re-run of this stage should pre-declare coverage at a precision floor (e.g. maximise
named-row coverage subject to jersey precision >= 0.80) and redo the DEV/TEST discipline from
scratch."* ``results/OCR_REALMATCH.md`` then measured what happens when it is not acted on: the
``r_abstain = 0`` name-everything regime loses to both controls at p < 1e-98.

This module does the DEV half of that re-run. It sweeps **only** ``r_abstain`` and ``sim_none`` --
the two knobs that shape the abstention regime -- over the DEV-20 per-crop bundles, holding every
other value of ``results/identity_solver_config_percrop.json`` frozen, and reports for each point:

* the arm's **own** operating point (coverage and jersey precision of the tracklets it actually
  names) -- the only point that transfers to a real match, where no posterior threshold can be
  calibrated because there is no ground truth;
* the post-hoc posterior dial's coverage at precision floors 0.85 and 0.60
  (:func:`eval.gsr_identity.coverage_curve`), reported for continuity with
  ``results/OCR_DENSIFICATION.md`` section 6.

The greedy Stage-1 rule (unanimous read inside a merge group, roster-gated) is scored on the
identical bundles so greedy-vs-solver is decided on DEV numbers alone.

CLI::

    python -m tools.retest_abstention            # CPU, ~minutes; writes the DEV frontier JSON
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from eval.gsr_identity import (
    coverage_curve,
    load_bundles,
    solve_bundle_scored,
    split_sequences,
)
from eval.gsr_score import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, DEFAULT_RESULTS_DIR
from generator.identity_solve import SolverConfig

logger = logging.getLogger("retest_abstention")

#: Per-crop evidence caches (frozen by ``results/OCR_DENSIFICATION.md``).
VOTES_SUBDIR = "koshkina_percrop_votes"
CACHE_SUBDIR = "identity_bundles_percrop"
#: Frozen per-crop Stage-2 config; only ``r_abstain`` and ``sim_none`` are reopened.
CONFIG_PATH = Path("results/identity_solver_config_percrop.json")
#: The sweep. Both lists are the values already present in the Stage-2 DEV grid.
R_ABSTAIN = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
SIM_NONE = (0.85, 0.92)
#: Pre-declared selection rule: maximise the arm's own named-row coverage subject to its own
#: jersey precision clearing this floor.
PRECISION_FLOOR = 0.85


def greedy_percrop(bundle: dict) -> tuple[list[int | None], list[float]]:
    """Stage-1 naming on per-crop evidence, at merged-tracklet granularity (pure).

    The shipped real-match rule (:func:`tools.gta_carrier.propagate_names` via
    :func:`eval.gsr_prtreid_relink.propagation_fill`) spreads a merge group's read to its unnamed
    members and leaves a group whose reads disagree untouched. A bundle tracklet *is* a merge group
    carrying its members' pooled reads, so the rule reduces to: assign the read number when the
    group's reads are unanimous and that number exists on the tracklet's team roster, else abstain.

    Returns:
        ``([jersey number or None per tracklet], [1.0 when named else 0.0])``.
    """
    slots = {(i.team, i.number) for i in bundle["identities"]}
    assign: list[int | None] = []
    conf: list[float] = []
    for trk in bundle["tracklets"]:
        nums = {int(n) for n, _c in trk.reads}
        num = next(iter(nums)) if len(nums) == 1 else None
        if num is None or (trk.team, num) not in slots:
            assign.append(None)
            conf.append(0.0)
            continue
        assign.append(num)
        conf.append(1.0)
    return assign, conf


def _point(bundles: dict, assigns: dict, confs: dict) -> dict:
    """Coverage/precision at the arm's own operating point plus the post-hoc dial (pure)."""
    cur = coverage_curve(bundles, assigns, confs)
    full = cur["full_coverage"] or {"coverage": 0.0, "precision": 0.0}
    return {
        "coverage": float(full["coverage"]), "precision": float(full["precision"]),
        "n_named_tracklets": int(cur["n_named_tracklets"]), "n_rows": int(cur["n_rows"]),
        "dial_at_0.85": cur["at_0.85"], "dial_at_0.6": cur["at_0.6"],
    }


def sweep(bundles: dict, cfg: SolverConfig) -> dict:
    """Sweep ``r_abstain`` x ``sim_none`` on DEV plus the greedy arm -> the frontier payload."""
    rows: list[dict] = []
    for sim_none in SIM_NONE:
        for r in R_ABSTAIN:
            c = replace(cfg, sim_none=sim_none, r_abstain=r)
            scored = {n: solve_bundle_scored(b, c) for n, b in bundles.items()}
            pt = _point(bundles, {n: v[0] for n, v in scored.items()},
                        {n: v[1] for n, v in scored.items()})
            rows.append({"arm": "solver", "sim_none": sim_none, "r_abstain": r, **pt})
            logger.info("solver sim_none=%.2f r_abstain=%.2f: coverage %.4f precision %.4f "
                        "(%d named)", sim_none, r, pt["coverage"], pt["precision"],
                        pt["n_named_tracklets"])
    g = {n: greedy_percrop(b) for n, b in bundles.items()}
    pt = _point(bundles, {n: v[0] for n, v in g.items()}, {n: v[1] for n, v in g.items()})
    rows.append({"arm": "greedy_percrop", "sim_none": None, "r_abstain": None, **pt})
    logger.info("greedy_percrop: coverage %.4f precision %.4f (%d named)", pt["coverage"],
                pt["precision"], pt["n_named_tracklets"])

    ok = [r for r in rows if r["precision"] >= PRECISION_FLOOR]
    best = max(ok, key=lambda r: r["coverage"], default=None)
    return {"precision_floor": PRECISION_FLOOR, "rows": rows, "selected": best,
            "n_qualifying": len(ok)}


def verdict(loto_path: Path, registration: Path, comparison: str) -> dict:
    """Read the single registered comparison out of a ``--loto`` payload -> the one-sided verdict.

    The grader (:func:`tools.gta_carrier.loto_query_hits`) and the paired-key rule
    (:func:`tools.gta_carrier.mcnemar`) are used unmodified; only the sidedness declared in the
    registration -- one-sided, ``arm_only_right > control_only_right`` -- is applied here, because
    the shipped helper reports the two-sided p.

    Args:
        loto_path: ``results/retest_loto.json`` written by ``tools.gta_carrier --loto``.
        registration: The pre-registration JSON (echoed into the verdict for auditability).
        comparison: Key of the registered comparison inside ``payload["paired"]``.
    """
    from scipy.stats import binomtest  # noqa: PLC0415

    payload = json.loads(loto_path.read_text(encoding="utf-8"))
    reg = json.loads(registration.read_text(encoding="utf-8"))
    m = payload["paired"][comparison]
    n01, n10 = int(m["control_only_right"]), int(m["arm_only_right"])
    p = binomtest(n10, n01 + n10, 0.5, alternative="greater").pvalue if n01 + n10 else 1.0
    alpha = float(reg["test"]["alpha"])
    return {"comparison": comparison, "registration_written_utc": reg["written_utc"],
            "hypothesis": reg["hypothesis"], "alpha": alpha,
            "n_paired": m["n_paired"], "control_p": m["control_p"], "arm_p": m["arm_p"],
            "delta": m["arm_p"] - m["control_p"], "control_only_right": n01,
            "arm_only_right": n10, "p_one_sided": float(p),
            "verdict": "PASS" if p < alpha else "FAIL"}


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verdict", type=Path, default=None,
                    help="path to the --loto payload; prints the single registered comparison")
    ap.add_argument("--registration", type=Path, default=Path("results/retest_registration.json"))
    ap.add_argument("--comparison", default="retest_vs_tau0.040")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--json-out", type=Path,
                    default=DEFAULT_RESULTS_DIR / "retest_dev_frontier.json")
    args = ap.parse_args()

    if args.verdict:
        v = verdict(args.verdict, args.registration, args.comparison)
        print(json.dumps(v, indent=2))
        (args.verdict.parent / "retest_verdict.json").write_text(json.dumps(v, indent=2),
                                                                 encoding="utf-8")
        return
    dev, _test = split_sequences(args.data_dir, args.out_dir)
    logger.info("DEV %d sequences (TEST is not touched by this module)", len(dev))
    bundles = load_bundles(args.data_dir, args.out_dir, dev, votes_subdir=VOTES_SUBDIR,
                           cache_subdir=CACHE_SUBDIR)
    payload = {"dev": dev, "config": args.config.name, **sweep(bundles, SolverConfig.load(
        args.config))}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("selected: %s", payload["selected"])
    logger.info("wrote %s", args.json_out)


if __name__ == "__main__":
    main()
