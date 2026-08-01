"""Spot-check: the 78-moment carrier factorisation with Stage-2 solver names (report only).

Reuses :mod:`tools.gta_carrier` verbatim -- same connector partition, same v1 frozen operating
point, same leakage controls -- and swaps only the name set: instead of the unanimous OCR-name
propagation inside a merge group, every tracklet takes the identity the Stage-2 MILP assigned it
(:mod:`tools.identity_match`, run first so the name parquets exist).

This is **not a gate**. `results/GTA_LINK_STAGE1.md` A4 already established that the label set
carries 23 usable naming moments and cannot resolve any intervention of realistic size; the numbers
here are reported with Wilson intervals and nothing is claimed from them.

CLI::

    python -m tools.identity_match          # first: write the solver name sets
    python -m tools.identity_carrier
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from functools import partial
from pathlib import Path

import pandas as pd

import tools.carrier_attribution_probe as probe
import tools.carrier_constrained as cc
import tools.gta_carrier as gc
from core import registry
from generator.identity_solve import SOLVER_VERSION
from tools.carrier_attribution_probe import PROBE_MATCHES
from tools.identity_match import LINEUP, PARTITION, PERCROP_FLOOR, percrop_reads

logger = logging.getLogger("identity_carrier")

#: Where :mod:`tools.identity_match` writes its per-match name sets.
NAMES = Path("outputs/identity/solver/{match}_solver_names.parquet")


def solver_namer(match_id: str, _merged: dict, names_dir: Path = NAMES.parent
                 ) -> tuple[dict[tuple[str, int], str], dict]:
    """Load the Stage-2 name set in :func:`tools.gta_carrier.propagate_names`' return shape."""
    df = pd.read_parquet(names_dir / NAMES.name.format(match=match_id))
    names = {(str(c), int(t)): str(p)
             for c, t, p in zip(df["chunk"], df["track_id"], df["player_name"])}
    return names, {"n_named_before": 0, "n_named_after": len(names),
                   "n_names_propagated": len(names), "n_groups_name_disagree": 0}


def percrop_namer(match_id: str, merged: dict, floor: str = PERCROP_FLOOR, variant: str = ""
                  ) -> tuple[dict[tuple[str, int], str], dict]:
    """Stage-1 naming (connector + unanimous propagation) on **per-crop** OCR evidence.

    Isolates evidence from inference: this arm changes only the read set the Stage-1 name
    propagation consumes -- close-up anchors -> per-crop reads -- and keeps the greedy rule the
    on-record 0.609 was measured with. A tracklet's dominant read number is looked up in the
    lineup, restricted to the tracklet's majority team and the half's availability.

    Args:
        match_id: Registry match id.
        merged: ``{(chunk, track_id): merged_id}`` from :func:`tools.gta_carrier.merge_match`.
        floor: Frozen per-crop aggregation rule (``results/ocr_density_rule.json``).
        variant: Per-crop cache variant, i.e. crop geometry (``tools.ocr_match.percrop_dir``).
            ``""`` is ``crop_scale = 1.0``; ``"_w125"`` is the widened crop.
    """
    lu = pd.read_parquet(str(LINEUP).format(match=match_id))
    shirt: dict[tuple[str, int, int], str] = {}
    for r in lu.itertuples():
        for half in ("h1", "h2"):
            if bool(getattr(r, f"on_{half}")):
                shirt[(half, int(r.team), int(r.shirt))] = str(getattr(r, "name"))
    df = pd.read_parquet(registry.get(match_id).aligned)
    team_of = df[df["team"] >= 0].groupby(["chunk", "track_id"])["team"].agg(
        lambda s: Counter(s).most_common(1)[0][0]).to_dict()

    base: dict[tuple[str, int], str] = {}
    n_no_team = n_no_slot = 0
    for chunk, per_track in percrop_reads(match_id, floor, variant=variant).items():
        for tid, votes in per_track.items():
            team = team_of.get((str(chunk), int(tid)))
            if team is None:
                n_no_team += 1
                continue
            num = Counter(n for n, _c in votes).most_common(1)[0][0]
            player = shirt.get((str(chunk)[:2], int(team), int(num)))
            if player is None:
                n_no_slot += 1
                continue
            base[(str(chunk), int(tid))] = player
    names, stat = gc.propagate_names(match_id, merged, base=base)
    return names, {**stat, "n_reads_no_team": n_no_team, "n_reads_no_roster_slot": n_no_slot}


def club_map(match_id: str) -> dict[str, str]:
    """Player -> club name, from the lineup (the solver names players OCR never read)."""
    lu = pd.read_parquet(str(LINEUP).format(match=match_id))
    return {str(r["name"]): str(r["team_name"]) for _i, r in lu.iterrows()}


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval for ``k`` successes in ``n`` trials (pure)."""
    if n == 0:
        return (float("nan"), float("nan"))
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=Path("outputs/identity_carrier"))
    ap.add_argument("--json-out", type=Path, default=Path("results/identity_carrier.json"))
    ap.add_argument("--names-dir", type=Path, default=NAMES.parent,
                    help="directory holding <match>_solver_names.parquet")
    ap.add_argument("--arm", default="solver", choices=("solver", "percrop_names"),
                    help="solver = Stage-2 MILP names; percrop_names = Stage-1 propagation on "
                         "per-crop OCR evidence (isolates evidence from inference)")
    ap.add_argument("--percrop-variant", default="",
                    help="with --arm percrop_names: per-crop cache variant, e.g. _w125")
    args = ap.parse_args()

    src = Path("results/carrier_attr")
    preps = {m: cc.prepare(m) for m in PROBE_MATCHES}
    preds = pd.concat([cc.run_config(preps[m], gc.BASELINE_CFG) for m in PROBE_MATCHES],
                      ignore_index=True)
    arms = {"baseline": {"factors": gc.factorise(preds, preps), "repair": {}}}

    arm_dir = args.out_root / args.arm
    if args.arm == "solver":
        namer, clubs = partial(solver_namer, names_dir=args.names_dir), club_map
    else:
        namer, clubs = partial(percrop_namer, variant=args.percrop_variant), club_map
    repair = {m: gc.prepare_arm(m, arm_dir, PARTITION, namer=namer, club=clubs(m))
              for m in PROBE_MATCHES}
    probe.OUT_DIR, cc.OUT_DIR = arm_dir, arm_dir  # test-harness redirection only
    try:
        preps = {m: cc.prepare(m) for m in PROBE_MATCHES}
        preds = pd.concat([cc.run_config(preps[m], gc.BASELINE_CFG) for m in PROBE_MATCHES],
                          ignore_index=True)
        arms[args.arm] = {"factors": gc.factorise(preds, preps), "repair": repair}
    finally:
        probe.OUT_DIR, cc.OUT_DIR = src, src

    for name, arm in arms.items():
        f = arm["factors"]
        for key in ("gate_hit", "team", "naming"):
            lo, hi = wilson(f[key]["n"], f[key]["d"])
            f[key]["wilson95"] = [lo, hi]
        logger.info("%-9s gate %d/%d=%.3f  team %d/%d=%.3f  naming %d/%d=%.3f [%.3f-%.3f]  e2e %s",
                    name, f["gate_hit"]["n"], f["gate_hit"]["d"], f["gate_hit"]["p"],
                    f["team"]["n"], f["team"]["d"], f["team"]["p"],
                    f["naming"]["n"], f["naming"]["d"], f["naming"]["p"],
                    f["naming"]["wilson95"][0], f["naming"]["wilson95"][1],
                    f["end_to_end_precision"])
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(
        {"version": SOLVER_VERSION, "note": "spot-check, not a gate; 23 usable naming moments",
         "arms": arms}, indent=2), encoding="utf-8")
    logger.info("wrote %s", args.json_out)


if __name__ == "__main__":
    main()
