"""Campaign v7 session V4 step 2: score an external association through the frozen v6 chain.

`results/GSR_V7_V4S1.md` measured the linking ceiling on the v6 stack -- +24.79 AssA for a perfect
linker, +11.89 for a merge-only one, 19 of 20 DEV sequences fragmentation-dominated -- and routed the
session to CAMELTrack (arXiv 2505.01257, Apache-2.0). CAMELTrack runs server-side on GPU and writes
one ``{seq}.npz`` remap per sequence (``old_tid`` / ``frame`` / ``new_tid``); this module runs the
**identical** CPU chain that produced the V0 control over that remap -- EIoU replaced, everything
else (connector tau 0.450, jersey votes, MILP solver, official scorer) rebuilt per arm.

The `results/GSR_V7_V3.md` section 2.7 landmine is handled here: ``tools.gsr_eiou.BOX_SUBDIR`` is
rebound to ``detbox_cache_v6det`` and its presence asserted, so a v6det arm can never be silently
matched against the base detector's boxes.

CLI::

    python -m tools.gsr_v7_camel --arm z0_bee24                 # score one remap directory
    python -m tools.gsr_v7_camel --arm z0_bee24 --diag          # + the V4s1 decomposition
    python -m tools.gsr_v7_camel --eiou-sweep 0.2:1:0.5:0.3     # the pre-registered fallback
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger("gsr_v7_camel")

#: The v6 arm's artifact suffix (tools.gsr_v6det.VARIANT).
V6DET = "_v6det"
#: The frozen downstream point.
TAU = 0.450
FLOOR = "0.80"
#: The pinned-stack DEV-20 control (results/GSR_V7_V0.md).
CONTROL = 51.808280888021784
CONTROL_JSON = Path("results/gsr_benchmark/gsr_v7_control_dev.json")


def dev20(data_dir: Path, out_dir: Path) -> list[str]:
    """The 20 DEV sequences every v7 gate is measured on."""
    from eval.gsr_identity import split_sequences  # noqa: PLC0415

    return split_sequences(data_dir, out_dir)[0]


def control_per_seq() -> dict[str, float]:
    """Per-sequence GS-HOTA of the pinned control, for the paired read."""
    rec = json.loads(CONTROL_JSON.read_text(encoding="utf-8"))
    return rec["arm"]["gs_hota_per_seq"]


def score_arm(data_dir: Path, out_dir: Path, results_dir: Path, seqs: list[str], name: str, *,
              remap_dir: Path) -> dict:
    """Run the frozen v6 chain over one precomputed association and score it (CPU)."""
    from eval.gsr_identity import paired_stats  # noqa: PLC0415

    import tools.gsr_eiou as eiou  # noqa: PLC0415

    eiou.BOX_SUBDIR = "detbox_cache" + V6DET
    if not (out_dir / eiou.BOX_SUBDIR).is_dir():
        raise SystemExit(f"missing box cache {out_dir / eiou.BOX_SUBDIR}")
    missing = [s for s in seqs if not (remap_dir / f"{s}.npz").exists()]
    if missing:
        raise SystemExit(f"remap missing for {len(missing)} sequences: {missing[:3]}")
    res = eiou.run_point(data_dir, out_dir, seqs, eiou.EiouParams(),
                         embedder="clip" + V6DET, tau=TAU, tag=f"v7camel_{name}",
                         percrop_variant="_v6" + V6DET,
                         positions_subdir="positions_gate" + V6DET, remap_dir=remap_dir)
    res["paired"] = paired_stats(control_per_seq(), res["gs_hota_per_seq"], seqs)
    res["arm_name"] = name
    results_dir.mkdir(parents=True, exist_ok=True)
    dest = results_dir / f"gsr_v7_v4s2_{name}.json"
    dest.write_text(json.dumps({"seqs": seqs, "arm": res}, indent=1, default=str),
                    encoding="utf-8")
    _print_row(name, res)
    print(f"wrote {dest}")
    return res


def score_eiou(data_dir: Path, out_dir: Path, results_dir: Path, seqs: list[str],
               grid: str) -> list[dict]:
    """The pre-registered fallback: an EIoU parameter retune on the same chain (CPU)."""
    from eval.gsr_identity import paired_stats  # noqa: PLC0415

    import tools.gsr_eiou as eiou  # noqa: PLC0415

    eiou.BOX_SUBDIR = "detbox_cache" + V6DET
    out = []
    for p in eiou._grid(grid):  # noqa: SLF001 - the module's own grid parser
        name = f"eiou_e{p.e:g}r{p.rounds}w{p.w_app:g}a{p.app_max:g}"
        res = eiou.run_point(data_dir, out_dir, seqs, p, embedder="clip" + V6DET, tau=TAU,
                             tag=f"v7camel_{name}", percrop_variant="_v6" + V6DET,
                             positions_subdir="positions_gate" + V6DET)
        res["paired"] = paired_stats(control_per_seq(), res["gs_hota_per_seq"], seqs)
        res["arm_name"] = name
        _print_row(name, res)
        out.append(res)
        (results_dir / f"gsr_v7_v4s2_{name}.json").write_text(
            json.dumps({"seqs": seqs, "arm": res}, indent=1, default=str), encoding="utf-8")
    return out


def _print_row(name: str, res: dict) -> None:
    """One ASCII result line (cp1252-safe)."""
    h, p = res["gs_hota"], res["paired"]
    print(f"{name:<22}{h['GS-HOTA']:>9.4f}{h['GS-HOTA'] - CONTROL:>+9.4f}{h['GS-DetA']:>9.4f}"
          f"{h['GS-AssA']:>9.4f}{h['IDF1']:>9.4f}{p['helped']:>5}/{20}"
          f"{p.get('wilcoxon_p', 1.0):>9.3g}")


def diagnose(data_dir: Path, out_dir: Path, results_dir: Path, seqs: list[str],
             name: str) -> dict:
    """The V4s1 decomposition on an arm's own partition: ceiling, fragmentation, purity."""
    import numpy as np  # noqa: PLC0415

    from tools.gsr_assoc_diag import partition_report  # noqa: PLC0415
    from tools.gsr_eiou import VARIANT, partition_purity  # noqa: PLC0415

    sub = "positions_gate" + V6DET + VARIANT
    rows, _profile = partition_report(data_dir, out_dir, seqs, positions_subdir=sub)
    keys = ("assa_hat", "assa_merge_only", "assa_perfect_link", "assa_perfect_det", "frag_per_gt",
            "n_tracks")
    means = {k: float(np.mean([r[k] for r in rows])) for k in keys if k in rows[0]}
    purity = partition_purity(data_dir, out_dir, seqs, sub)
    out = {"arm": name, "means": means, "purity": purity, "per_seq": rows}
    dest = results_dir / f"gsr_v7_v4s2_{name}_diag.json"
    dest.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"{name} partition: " + " ".join(f"{k} {v:.4f}" for k, v in means.items()))
    print(f"{name} purity: {purity}")
    print(f"wrote {dest}")
    return out


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gsr"))
    ap.add_argument("--results-dir", type=Path, default=Path("results/gsr_benchmark"))
    ap.add_argument("--remap-root", type=Path, default=Path("outputs/gsr/camel_remap"))
    ap.add_argument("--arm", default=None, help="remap subdirectory name under --remap-root")
    ap.add_argument("--eiou-sweep", default=None, help="fallback grid 'e:rounds:w_app:app_max,...'")
    ap.add_argument("--diag", action="store_true", help="also run the V4s1 decomposition")
    ap.add_argument("--diag-only", action="store_true")
    ap.add_argument("--seqs", default=None)
    args = ap.parse_args()

    seqs = args.seqs.split(",") if args.seqs else dev20(args.data_dir, args.out_dir)
    print(f"{'arm':<22}{'GS-HOTA':>9}{'delta':>9}{'DetA':>9}{'AssA':>9}{'IDF1':>9}"
          f"{'helped':>9}{'p':>9}")
    if args.arm and not args.diag_only:
        score_arm(args.data_dir, args.out_dir, args.results_dir, seqs, args.arm,
                  remap_dir=args.remap_root / args.arm)
    if args.eiou_sweep:
        score_eiou(args.data_dir, args.out_dir, args.results_dir, seqs, args.eiou_sweep)
    if args.arm and (args.diag or args.diag_only):
        diagnose(args.data_dir, args.out_dir, args.results_dir, seqs, args.arm)
    if not args.arm and not args.eiou_sweep:
        ap.error("choose --arm or --eiou-sweep")


if __name__ == "__main__":
    main()
