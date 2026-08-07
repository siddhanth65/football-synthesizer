"""Campaign v7 session V3: score the Dirichlet evidential reader through the frozen v6 chain.

The head itself is trained server-side on cached frozen-trunk features (``~/work/edl/*.py``); this
module owns everything downstream of the per-crop evidence, i.e. the two rungs that decide whether
it ships:

* ``--sweep`` (rung 2) -- tracklet aggregation on the SAME evidential rows, two families:
  the incumbent Koshkina **voting** machinery (:func:`generator.jersey_id.percrop_votes`, optionally
  with the evidential ``max_u`` / ``max_p_none`` crop filter) and **additive Dirichlet fusion**
  (:func:`generator.evidential_jersey.fuse_tracklet`). Graded by read density ``d`` and read
  precision against the GSR jersey GT, on the ``_v6det`` track ids. Fusion must beat voting on the
  same evidence or voting-on-evidential-reads is what ships.
* ``--arm`` (rung 3) -- ONE chosen rule end-to-end on DEV-20 through the v6 chain, everything but
  the per-crop evidence frozen, paired against ``results/gsr_benchmark/gsr_v7_control_dev.json``.
  Reports the **admissible-set** effect V1 predicts is the mechanism: self-roster slots per
  sequence and merged tracklets named, against the incumbent's 610 of 870.

The rule a rung-3 arm runs at is written to ``results/ocr_density_rule<variant>.json`` -- the file
:func:`tools.gsr_v4.rule_for` resolves -- so the vote cache re-keys itself (``votes_dir`` wipes on a
rule-stamp mismatch) and no new plumbing is needed.

CLI::

    python -m tools.gsr_v7_edl --sweep                     # rung 2 (CPU)
    python -m tools.gsr_v7_edl --arm '{"min_crop_conf":0.9,...}' --name edl_fuse   # rung 3 (CPU)
    python -m tools.gsr_v7_edl --demo
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from itertools import product
from pathlib import Path

logger = logging.getLogger("gsr_v7_edl")

#: The v6det chain this session rides (see tools/gsr_v6det.py and tools/gsr_v7_solver.py).
DET_VARIANT = "_v6det"
FLOOR = "0.80"
TAU = 0.450
#: The pinned DEV-20 control every rung-3 arm is paired against (results/GSR_V7_V0.md).
CONTROL = Path("results/gsr_benchmark/gsr_v7_control_dev.json")
SWEEP_PATH = Path("results/gsr_benchmark/gsr_v7_v3_sweep.json")
ARMS_PATH = Path("results/gsr_benchmark/gsr_v7_v3_arms.json")

#: The incumbent 0.80-floor rule, verbatim (results/ocr_density_rule.json). The evidential arms are
#: measured against THIS rule applied to THEIR evidence, so the aggregation is not a free variable.
INCUMBENT_RULE = {"min_crop_conf": 0.9, "min_votes": 3, "min_legibility": 0.5, "emit_all": False}


def chain_variant(reader: str) -> str:
    """The ``tools.gsr_v4`` evidence variant for one reader arm on the EIoU partition (pure)."""
    import tools.gsr_eiou as eiou  # noqa: PLC0415

    from tools.gsr_v6det import percrop_variant  # noqa: PLC0415

    return percrop_variant(reader) + eiou.VARIANT


# === Rung 2: tracklet aggregation on the evidential rows =========================================
def sweep_rules(*, legs: tuple[float, ...], confs: tuple[float, ...], votes: tuple[int, ...],
                us: tuple[float, ...], nones: tuple[float, ...]) -> list[dict]:
    """The pre-registered rung-2 grid: voting arms then fusion arms (pure).

    Every arm carries the same four incumbent keys plus the two evidential filters, so the voting
    control and the fusion arm differ in ``fuse`` alone at matched filter settings.
    """
    out: list[dict] = []
    for leg, conf, v, u, pn in product(legs, confs, votes, us, nones):
        out.append({"min_crop_conf": conf, "min_votes": v, "min_legibility": leg,
                    "emit_all": False, "max_u": u, "max_p_none": pn, "fuse": False})
    for leg, conf, v, u, pn in product(legs, confs, votes, us, nones):
        out.append({"min_crop_conf": conf, "min_votes": v, "min_legibility": leg,
                    "emit_all": False, "max_u": u, "max_p_none": pn, "fuse": True})
    return out


def rule_id(rule: dict) -> str:
    """Compact, injective name of one aggregation rule (pure)."""
    return (f"{'fuse' if rule.get('fuse') else 'vote'}_c{rule['min_crop_conf']:g}"
            f"_v{rule['min_votes']:g}_l{rule['min_legibility']:g}"
            f"_u{rule.get('max_u', 1.01):g}_n{rule.get('max_p_none', 1.01):g}")


def run_sweep(data_dir: Path, out_dir: Path, seqs: list[str], variants: dict[str, str],
              rules: list[dict]) -> dict:
    """Grade every rule on every evidence variant (CPU); returns the raw sweep payload."""
    from tools.ocr_density import load_gt_cache, load_percrop, measure  # noqa: PLC0415

    gt = load_gt_cache(data_dir, out_dir, seqs, "positions" + DET_VARIANT)
    rows: list[dict] = []
    for vname, variant in variants.items():
        per = load_percrop(out_dir, seqs, variant)
        if not per:
            logger.warning("no per-crop evidence for variant %r, skipped", variant)
            continue
        has_alpha = "alpha" in next(iter(per.values())).columns
        logger.info("variant %s (%s): %d sequences, %d crops, alpha=%s", vname, variant,
                    len(per), sum(len(v) for v in per.values()), has_alpha)
        for rule in rules:
            if not has_alpha and (rule.get("fuse") or rule.get("max_u", 1.01) < 1.0
                                  or rule.get("max_p_none", 1.01) < 1.0):
                continue  # an evidential rule is not defined on a pre-v7 frame
            m = measure(per, gt, rule)
            m["arm"], m["variant"] = vname, variant
            rows.append(m)
            logger.info("  %-40s d %.4f (%4d/%4d) prec %.4f (n=%d)", f"{vname}/{rule_id(rule)}",
                        m["d"], m["n_tracks_read"], m["n_tracks"], m["read_precision"],
                        m["n_reads_auditable"])
    return {"seqs": seqs, "variants": variants, "rows": rows}


# === Rung 3: end to end ==========================================================================
def write_rule(variant: str, rule: dict) -> Path:
    """Freeze one aggregation rule as this variant's own rule file (what ``rule_for`` resolves)."""
    from tools.ocr_density import rule_path  # noqa: PLC0415

    p = rule_path(variant)
    p.write_text(json.dumps({"version": "v7-v3", "variant": variant, "floors": [0.8],
                             "primary_floor": 0.8, "rules": {FLOOR: rule}, **rule}, indent=2),
                 encoding="utf-8")
    return p


def admissible_diagnostics(data_dir: Path, out_dir: Path, seqs: list[str], variant: str) -> dict:
    """Self-roster slots and named tracklets for one arm -- the V1 admissible-set prediction.

    ``results/GSR_V7_V1.md`` §2.7: 260 of 870 merged tracklets are unnamed not because the solver
    abstains but because the sequence's self-built roster offers them no slot. If the evidential
    reader converts, it converts HERE first: more reads -> more slots -> more named tracklets.
    """
    from generator.gta_link import GtaParams  # noqa: PLC0415

    from eval.gsr_identity import load_bundles, solve_bundle_scored  # noqa: PLC0415
    from tools.gsr_deleak import _team_maps, retarget  # noqa: PLC0415
    from tools.gsr_v4 import config_key, solver_config, votes_dir  # noqa: PLC0415

    pos_sub = "positions_gate" + DET_VARIANT + "_eiou"
    key = config_key(FLOOR, TAU, True, variant, False)
    bundles = load_bundles(
        data_dir, out_dir, seqs, votes_subdir=votes_dir(out_dir, FLOOR, variant).name,
        cache_subdir=f"identity_bundles_v4_{key}", positions_subdir=pos_sub,
        params=GtaParams(tau=TAU, eps=0.30, min_samples=5, min_run=5, frame_stride=2),
        jersey_gate=True)
    maps = _team_maps(data_dir, out_dir, seqs, "free", pos_sub)
    cfg = solver_config()
    tot = {"n_tracklets": 0, "n_named": 0, "slots": 0, "per_seq": {}}
    for name in seqs:
        b = retarget(bundles[name], roster="self", sides=maps[name])
        assign, _conf = solve_bundle_scored(b, cfg)
        named = sum(a is not None for a in assign)
        slots = len(b["identities"])
        tot["n_tracklets"] += len(b["tracklets"])
        tot["n_named"] += named
        tot["slots"] += slots
        tot["per_seq"][name] = {"tracklets": len(b["tracklets"]), "named": named, "slots": slots}
    tot["slots_per_seq"] = tot["slots"] / max(len(seqs), 1)
    return tot


def run_arm(data_dir: Path, out_dir: Path, seqs: list[str], *, reader: str, rule: dict,
            name: str, w_app: float = 0.5, keep_arm_dir: bool = False) -> dict:
    """One rung-3 arm: freeze the rule, run the v6 chain on this reader's evidence, score it.

    Args:
        data_dir: GSR ground-truth folder.
        out_dir: Pipeline output root (``outputs/gsr`` for DEV-20, ``outputs/gsr_srv`` for TEST-38).
        seqs: Sequences to score.
        reader: ``'v6'`` (control) or ``'edl'`` (the evidential head's evidence).
        rule: The aggregation rule, frozen into this variant's rule file.
        name: Arm name (also the arm directory stem).
        w_app: The EIoU appearance weight -- 0.5 is the v6 incumbent, 0.7 the V4s2 fallback point.
            Every other association parameter stays at the frozen v6 value.
        keep_arm_dir: Keep the materialised submission directory (needed for packaging).
    """
    import tools.gsr_eiou as eiou  # noqa: PLC0415
    from tools.gsr_v6det import percrop_variant  # noqa: PLC0415

    variant = chain_variant(reader)
    # LOAD-BEARING, and it cost a wasted arm: eiou.BOX_SUBDIR is a module constant defaulting to
    # the ORIGINAL detector's box cache. tools.gsr_v6det.stage_arm rebinds it per detector arm; a
    # caller that forgets silently re-links the v6det positions against the base detector's boxes
    # (both caches exist on disk, so nothing warns) -- the run completes, takes hours, and is wrong.
    eiou.BOX_SUBDIR = "detbox_cache" + DET_VARIANT
    missing = [s for s in seqs if not (out_dir / eiou.BOX_SUBDIR / f"{s}.npz").exists()]
    if missing:
        raise SystemExit(f"{eiou.BOX_SUBDIR}: no box cache for {missing}")
    write_rule(variant, rule)  # run_point's own clear_arm_caches then wipes the stale votes
    t0 = time.time()
    payload = eiou.run_point(
        data_dir, out_dir, seqs, eiou.EiouParams(e=0.3, rounds=1, w_app=w_app, app_max=0.30),
        embedder="clip" + DET_VARIANT, tau=TAU, tag=f"v7v3_{name}",
        percrop_variant=percrop_variant(reader),
        positions_subdir="positions_gate" + DET_VARIANT)
    payload["v7v3"] = {"reader": reader, "rule": rule, "variant": variant, "w_app": w_app,
                       "seconds": round(time.time() - t0, 1)}
    payload["admissible"] = admissible_diagnostics(data_dir, out_dir, seqs, variant)
    if not keep_arm_dir:
        shutil.rmtree(out_dir / f"deleak_v7v3_{name}", ignore_errors=True)
    return payload


def paired_vs_control(payload: dict, seqs: list[str], control: Path = CONTROL) -> dict:
    """Pair one arm's per-sequence GS-HOTA against a control payload, one- and two-sided.

    The v7 PUSH registration tests ``arm > control`` on a held-out split, so the one-sided
    signed-rank p is the one the gate reads; the two-sided p that every earlier session quoted is
    reported beside it so the two are never confused.
    """
    import numpy as np  # noqa: PLC0415
    from scipy.stats import wilcoxon  # noqa: PLC0415

    from eval.gsr_identity import paired_stats  # noqa: PLC0415

    ctrl = json.loads(control.read_text(encoding="utf-8"))["arm"]
    st = paired_stats(ctrl["gs_hota_per_seq"], payload["gs_hota_per_seq"], seqs)
    d = np.array([payload["gs_hota_per_seq"][n] - ctrl["gs_hota_per_seq"][n] for n in seqs])
    st["wilcoxon_p_onesided"] = (float(wilcoxon(d, alternative="greater").pvalue)
                                 if np.any(d != 0) else 1.0)
    st["control"] = str(control)
    st["control_gs_hota"] = ctrl["gs_hota"]["GS-HOTA"]
    st["arm_gs_hota"] = payload["gs_hota"]["GS-HOTA"]
    st["delta"] = st["arm_gs_hota"] - st["control_gs_hota"]
    return st


def concentration(per_seq_delta: dict[str, float]) -> float:
    """Share of an arm's NET gain carried by its two best sequences (the GSR_V5 §2.1 guard)."""
    net = sum(per_seq_delta.values())
    top2 = sum(sorted(per_seq_delta.values(), reverse=True)[:2])
    return float("inf") if net <= 0 else top2 / net


def _demo() -> None:
    """Self-check: rule ids are injective, the grid is the declared size, guards behave."""
    rules = sweep_rules(legs=(0.0, 0.5), confs=(0.5, 0.9), votes=(1, 3), us=(0.6, 1.01),
                        nones=(0.5, 1.01))
    assert len(rules) == 2 * 2 * 2 * 2 * 2 * 2, len(rules)
    assert len({rule_id(r) for r in rules}) == len(rules)
    assert rule_id(rules[0]).startswith("vote_") and rule_id(rules[-1]).startswith("fuse_")
    assert chain_variant("v6") == "_v6_v6det_eiou", chain_variant("v6")
    assert chain_variant("edl") == "_v7e_v6det_eiou", chain_variant("edl")
    assert abs(concentration({"a": 4.0, "b": 4.0, "c": 2.0}) - 0.8) < 1e-9
    assert concentration({"a": -1.0}) == float("inf")
    print("gsr_v7_edl self-check OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gsr"))
    ap.add_argument("--sweep", action="store_true", help="rung 2: the tracklet aggregation grid")
    ap.add_argument("--arm", default=None, help="rung 3: JSON aggregation rule to run end-to-end")
    ap.add_argument("--reader", default="edl", choices=("edl", "v6", "incumbent"))
    ap.add_argument("--name", default="edl")
    ap.add_argument("--split", default="dev", choices=("dev", "t38", "test"),
                    help="which sequences to score; t38/test need --out-dir on the server caches")
    ap.add_argument("--w-app", type=float, default=0.5,
                    help="EIoU appearance weight (0.5 = v6 incumbent, 0.7 = the V4s2 point)")
    ap.add_argument("--control", type=Path, default=None,
                    help="control payload to pair against (default: the pinned DEV-20 V0 control)")
    ap.add_argument("--arms-path", type=Path, default=None, help="where the arm payload is stored")
    ap.add_argument("--keep-arm-dir", action="store_true", help="keep the submission directory")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return

    from eval.gsr_identity import split_sequences  # noqa: PLC0415

    if args.split == "test":
        from tools.gsr_deleak import split_names  # noqa: PLC0415

        dev = split_names(args.data_dir, "test")
    else:
        d20, t38 = split_sequences(args.data_dir, Path("outputs/gsr"))
        dev = d20 if args.split == "dev" else t38
    control = args.control or CONTROL
    arms_path = args.arms_path or ARMS_PATH
    SWEEP_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        rules = sweep_rules(legs=(0.0, 0.5), confs=(0.5, 0.9, 0.99), votes=(1, 2, 3),
                            us=(0.4, 0.6, 0.8, 1.01), nones=(0.3, 0.5, 1.01))
        res = run_sweep(args.data_dir, args.out_dir, dev,
                        {"edl": "_v7e" + DET_VARIANT, "v6": "_v6" + DET_VARIANT}, rules)
        SWEEP_PATH.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        print(f"wrote {SWEEP_PATH} ({len(res['rows'])} graded points)")
        return

    if args.arm:
        rule = json.loads(args.arm)
        payload = run_arm(args.data_dir, args.out_dir, dev, reader=args.reader, rule=rule,
                          name=args.name, w_app=args.w_app, keep_arm_dir=args.keep_arm_dir)
        payload["paired"] = paired_vs_control(payload, dev, control)
        blob = (json.loads(arms_path.read_text(encoding="utf-8")) if arms_path.exists()
                else {"split": args.split, "seqs": dev, "arms": {}})
        blob["arms"][args.name] = payload
        arms_path.write_text(json.dumps(blob, indent=1, default=str), encoding="utf-8")
        p, a = payload["paired"], payload["admissible"]
        print(f"{args.name}: GS-HOTA {p['arm_gs_hota']:.4f} vs control {p['control_gs_hota']:.4f} "
              f"({p['delta']:+.4f}), helped {p['helped']}/{len(dev)}, "
              f"p1={p['wilcoxon_p_onesided']:.3g} p2={p['wilcoxon_p']:.3g}")
        print(f"  admissible: {a['n_named']}/{a['n_tracklets']} named, "
              f"{a['slots_per_seq']:.2f} slots/sequence")
        print(f"wrote {arms_path}")
        return
    ap.error("choose --sweep / --arm / --demo")


if __name__ == "__main__":
    main()
