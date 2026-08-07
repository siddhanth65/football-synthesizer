"""Campaign v7 session V1: the pre-registered solver retune on the dense v6 evidence regime.

`results/EVIDENCE_DENSITY_LAW.md` predicts that the value of identity constraints rises with
evidence density; every shipped solver prior (`results/identity_solver_config_percrop.json`) was
fitted in the d ~ 0.09 PRTreID era, while the v6 chain now runs at d ~ 0.31 at 0.926 read precision.
This module runs the pre-registered coordinate sweep of `results/GSR_V7_V1.md` §1 over
`generator.identity_solve.SolverConfig`, end-to-end on DEV-20 through the v6 chain, paired against
the V0 control.

**Only the solver config moves.** The EIoU re-link, the densified votes, the evidence bundles and
the GTA connector arm are not functions of any solver knob, so they are computed once (they are
already warm from `results/GSR_V7_V0.md`) and shared by every arm. That shortcut is *verified, not
assumed*: the ``control`` arm runs the frozen config through this harness and must reproduce the V0
control's GS-HOTA exactly before any other arm is scored.

CLI::

    python -m tools.gsr_v7_solver --arms control          # the equivalence check (run this first)
    python -m tools.gsr_v7_solver --arms coordinate       # the 11 pre-registered coordinate arms
    python -m tools.gsr_v7_solver --arms p_correct=0.95,r_abstain=0.05   # an ad-hoc joint point
    python -m tools.gsr_v7_solver --report                # the sweep table from the saved JSON
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger("gsr_v7_solver")

#: The V0 control every arm is paired against.
CONTROL_PATH = Path("results/gsr_benchmark/gsr_v7_control_dev.json")
#: Where every arm's numbers accumulate (resumable: an arm already present is skipped).
SWEEP_PATH = Path("results/gsr_benchmark/gsr_v7_v1_sweep.json")
#: The frozen v6det + EIoU operating point; nothing here is a sweep knob.
TAU = 0.450
FLOOR = "0.80"
DET_VARIANT = "_v6det"

#: The pre-registered coordinate grid: knob -> the values other than the incumbent's.
GRID: dict[str, tuple[float, ...]] = {
    "p_correct": (0.926, 0.95),
    "r_abstain": (0.02, 0.05, 0.10),
    "pi_none": (0.5, 0.85),
    "topk": (5,),
    "app_gain": (5.0,),
    "sim_none": (0.90, 0.95),
}


def arm_id(over: dict) -> str:
    """Directory-safe name of one sweep point (``control`` when nothing is overridden)."""
    return "_".join(f"{k}{v:g}" for k, v in sorted(over.items())) or "control"


def chain_args(out_dir: Path) -> dict:
    """The frozen v6 chain arguments every arm shares (see :func:`tools.gsr_v6det.stage_arm`)."""
    import tools.gsr_eiou as eiou  # noqa: PLC0415

    return {
        "floor": FLOOR, "tau": TAU, "gate": True, "refit": False,
        "variant": "_v6" + DET_VARIANT + eiou.VARIANT,
        "positions_subdir": "positions_gate" + DET_VARIANT + eiou.VARIANT,
        "embedder": "clip" + DET_VARIANT + eiou.VARIANT,
    }


def gk_diagnostics(data_dir: Path, out_dir: Path, seqs: list[str], cfg, args: dict) -> dict:
    """Re-solve the (cached) bundles to count what the arm named, by role.

    `results/RETEST_ABSTENTION.md` §4.3 measured that ``r_abstain > 0`` zeroes goalkeeper naming on
    real matches. The GSR arm cannot lose a GK *roster slot* (``roster_self`` emits ``player`` slots
    only), so the observable analogue reported here is how many GK-role-dominant tracklets the
    solver names at all, and how many rows they carry.

    Args:
        data_dir: GSR ground-truth folder.
        out_dir: Pipeline output root.
        seqs: DEV sequences.
        cfg: The arm's :class:`SolverConfig`.
        args: :func:`chain_args` for this chain.

    Returns:
        ``{n_tracklets, n_named, gk_tracklets, gk_named, gk_rows_named, ...}``.
    """
    from eval.gsr_identity import load_bundles, solve_bundle_scored  # noqa: PLC0415
    from generator.gta_link import GtaParams  # noqa: PLC0415

    from tools.gsr_deleak import _team_maps, retarget  # noqa: PLC0415
    from tools.gsr_v4 import config_key, votes_dir  # noqa: PLC0415

    key = config_key(FLOOR, TAU, True, args["variant"], False)
    bundles = load_bundles(
        data_dir, out_dir, seqs, votes_subdir=votes_dir(out_dir, FLOOR, args["variant"]).name,
        cache_subdir=f"identity_bundles_v4_{key}", positions_subdir=args["positions_subdir"],
        params=GtaParams(tau=TAU, eps=0.30, min_samples=5, min_run=5, frame_stride=2),
        jersey_gate=True)
    maps = _team_maps(data_dir, out_dir, seqs, "free", args["positions_subdir"])
    tot = {"n_tracklets": 0, "n_named": 0, "n_rows_named": 0,
           "gk_tracklets": 0, "gk_named": 0, "gk_rows": 0, "gk_rows_named": 0}
    for name in seqs:
        b = retarget(bundles[name], roster="self", sides=maps[name])
        assign, _conf = solve_bundle_scored(b, cfg)
        for trk, a in zip(b["tracklets"], assign):
            is_gk = trk.role_frac.get("goalkeeper", 0.0) > 0.5  # noqa: PLR2004 - role majority
            tot["n_tracklets"] += 1
            tot["gk_tracklets"] += int(is_gk)
            tot["gk_rows"] += trk.n_rows if is_gk else 0
            if a is not None:
                tot["n_named"] += 1
                tot["n_rows_named"] += trk.n_rows
                tot["gk_named"] += int(is_gk)
                tot["gk_rows_named"] += trk.n_rows if is_gk else 0
    return tot


def run_arm_point(data_dir: Path, out_dir: Path, seqs: list[str], over: dict) -> dict:
    """Score one sweep point end-to-end on DEV-20 and return its payload (arm dir deleted after)."""
    import tools.gsr_eiou as eiou  # noqa: PLC0415
    import tools.gsr_v4 as v4  # noqa: PLC0415
    from tools.gsr_v4 import solve_v4_arm, solver_config  # noqa: PLC0415

    args = chain_args(out_dir)
    eiou.set_embedder(args["embedder"])
    name = arm_id(over)
    tag = f"v7v1_{name}"
    v4.SOLVER_OVERRIDE = dict(over)
    try:
        cfg = solver_config()
        assert all(getattr(cfg, k) == v for k, v in over.items()), (cfg, over)
        t0 = time.time()
        payload = solve_v4_arm(data_dir, out_dir, seqs, tag=tag, floor=FLOOR, tau=TAU, gate=True,
                               variant=args["variant"], refit=False,
                               positions_subdir=args["positions_subdir"])
        payload["seconds"] = round(time.time() - t0, 1)
        payload["roles"] = gk_diagnostics(data_dir, out_dir, seqs, cfg, args)
    finally:
        v4.SOLVER_OVERRIDE = {}  # always restore the frozen default
    payload["solver_override"] = dict(over)
    assert payload["roles"]["n_named"] == payload["pooled"]["tracklets_named"], payload["roles"]
    shutil.rmtree(out_dir / f"deleak_{tag}", ignore_errors=True)
    return payload


def _load_sweep() -> dict:
    """The accumulated sweep payload (empty skeleton when nothing has run yet)."""
    if SWEEP_PATH.exists():
        return json.loads(SWEEP_PATH.read_text(encoding="utf-8"))
    return {"registration": "results/GSR_V7_V1.md §1", "control_ref": str(CONTROL_PATH), "arms": {}}


def _save_sweep(blob: dict) -> None:
    """Persist the sweep payload after every arm (the sweep is resumable from it)."""
    SWEEP_PATH.parent.mkdir(parents=True, exist_ok=True)
    SWEEP_PATH.write_text(json.dumps(blob, indent=1, default=str), encoding="utf-8")


def concentration(per_seq_delta: dict[str, float]) -> float:
    """Share of an arm's NET gain carried by its two best sequences (pure).

    The pre-registered demotion guard: >= 0.80 means one or two lucky sequences, not a regime effect.
    Returns ``nan`` when the net gain is not positive (the guard does not apply).
    """
    total = sum(per_seq_delta.values())
    if total <= 0:
        return float("nan")
    top2 = sum(sorted(per_seq_delta.values(), reverse=True)[:2])
    return top2 / total


def report(blob: dict, control: dict) -> None:
    """ASCII sweep table (cp1252-safe), sorted by GS-HOTA delta."""
    base = control["gs_hota_per_seq"]
    rows = []
    for name, a in blob["arms"].items():
        d = {s: a["gs_hota_per_seq"][s] - base[s] for s in base}
        p = a.get("paired") or {}
        rows.append((a["gs_hota"]["GS-HOTA"] - control["gs_hota"]["GS-HOTA"], name, a, p,
                     concentration(d)))
    rows.sort(reverse=True)
    print(f"{'arm':<20}{'GS-HOTA':>10}{'delta':>8}{'DetA':>8}{'AssA':>8}{'help':>5}{'hurt':>5}"
          f"{'p':>8}{'conc':>7}{'named':>7}{'GKnm':>6}{'jersey':>8}")
    for delta, name, a, p, conc in rows:
        h = a["gs_hota"]
        r = a.get("roles", {})
        print(f"{name:<20}{h['GS-HOTA']:>10.4f}{delta:>+8.3f}{h['GS-DetA']:>8.3f}"
              f"{h['GS-AssA']:>8.3f}{p.get('helped', 0):>5}{p.get('hurt', 0):>5}"
              f"{p.get('wilcoxon_p', float('nan')):>8.3g}{conc:>7.2f}"
              f"{a['pooled']['tracklets_named']:>7}{r.get('gk_named', -1):>6}"
              f"{a['pooled']['jersey']:>8.4f}")


def parse_arms(spec: str) -> list[dict]:
    """Turn an ``--arms`` spec into the list of override dicts to run (pure)."""
    if spec == "control":
        return [{}]
    if spec == "coordinate":
        return [{k: v} for k, vals in GRID.items() for v in vals]
    out = []
    for point in spec.split(";"):
        over = {}
        for kv in point.split(","):
            k, v = kv.split("=")
            over[k.strip()] = int(v) if k.strip() == "topk" else float(v)
        out.append(over)
    return out


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gsr"))
    ap.add_argument("--arms", default="control",
                    help="'control', 'coordinate', or 'k=v,k=v;k=v' joint points")
    ap.add_argument("--report", action="store_true", help="print the table and exit")
    ap.add_argument("--force", action="store_true", help="re-run arms already in the sweep JSON")
    args = ap.parse_args()

    control = json.loads(CONTROL_PATH.read_text(encoding="utf-8"))["arm"]
    blob = _load_sweep()
    if args.report:
        report(blob, control)
        return

    from eval.gsr_identity import paired_stats, split_sequences  # noqa: PLC0415

    seqs, _t38 = split_sequences(args.data_dir, args.out_dir)
    for i, over in enumerate(parse_arms(args.arms)):
        name = arm_id(over)
        if name in blob["arms"] and not args.force:
            logger.info("[%d] %s already scored, skipped", i + 1, name)
            continue
        logger.info("[%d] arm %s", i + 1, name)
        payload = run_arm_point(args.data_dir, args.out_dir, seqs, over)
        payload["paired"] = paired_stats(control["gs_hota_per_seq"], payload["gs_hota_per_seq"],
                                         seqs)
        blob["arms"][name] = payload
        _save_sweep(blob)
        delta = payload["gs_hota"]["GS-HOTA"] - control["gs_hota"]["GS-HOTA"]
        print(f"{name}: GS-HOTA {payload['gs_hota']['GS-HOTA']:.4f} ({delta:+.4f}) "
              f"helped {payload['paired']['helped']}/{len(seqs)} "
              f"GK-named {payload['roles']['gk_named']}/{payload['roles']['gk_tracklets']} "
              f"({payload['seconds']:.0f}s)")
        if name == "control":
            ref = control["gs_hota"]["GS-HOTA"]
            got = payload["gs_hota"]["GS-HOTA"]
            print(f"EQUIVALENCE CHECK: {got!r} vs V0 {ref!r} -> "
                  f"{'EXACT' if got == ref else f'MISMATCH {got - ref:+.10f}'}")
    report(blob, control)
    print(f"wrote {SWEEP_PATH}")


if __name__ == "__main__":
    main()
