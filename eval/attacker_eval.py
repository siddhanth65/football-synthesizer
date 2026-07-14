"""Evaluate the rebuilt attacker heads against the OLD StatsBomb-360 baselines (the C3 headline).

Old 360 numbers to beat (from football-state-of-play): run RMSE 6.33 m / hit@3m 0.244 (no-move 6.46
m / 0.221); receiver top-1 0.41 / top-3 0.78 (nearest-teammate heuristic ~0.42 top-1). Report the
video-trained heads on the same metrics so the improvement is unambiguous.

Run: ``python -m eval.attacker_eval --positions outputs/chunk000_dense.parquet``
"""

from __future__ import annotations

import pandas as pd

from attacker.heads import train_receiver_head, train_run_head
from attacker.labels import receiver_labels
from attacker.tracks import DEFAULT_FPS, build_tracks
from fingerprint.structural_metrics import resolve_attack_directions

OLD_360 = {
    "run_rmse_m": 6.33,
    "run_hit3m": 0.244,
    "run_nomove_rmse_m": 6.46,
    "receiver_top1": 0.41,
    "receiver_top3": 0.78,
    "receiver_heuristic_top1": 0.42,
}


def evaluate(positions: pd.DataFrame, *, fps: float = DEFAULT_FPS, seed: int = 0) -> dict:
    """Run the full C3 attacker evaluation on a dense positions table.

    Returns ``{"run": <train_run_head dict>, "receiver": <train_receiver_head dict>, "old_360": OLD_360}``.
    """
    attack_dirs = resolve_attack_directions(positions)
    tracks = build_tracks(positions, fps=fps)
    run = train_run_head(run_df=None, positions=positions, attack_dirs=attack_dirs, fps=fps, seed=seed)
    recv = train_receiver_head(tracks, receiver_labels(tracks, fps=fps), attack_dirs=attack_dirs,
                               seed=seed)
    return {"run": run, "receiver": recv, "old_360": OLD_360}


def _fmt_run(run: dict) -> str:
    lines = [f"RUN HEAD  (n_train={run['n_train']}, n_test={run['n_test']}, "
             f"outliers dropped={run.get('outlier_frac', 0):.1%}, "
             f"true median displacement={run.get('true_med_disp_m', float('nan')):.2f} m)",
             f"  {'model':<14}{'RMSE m':>9}{'MAE m':>9}{'hit@3m':>9}{'med|d| m':>10}{'dir cos':>9}"]
    for name, m in run["metrics"].items():
        lines.append(f"  {name:<14}{m['rmse_m']:>9.2f}{m['mae_m']:>9.2f}{m['hit@3m']:>9.3f}"
                     f"{m['med_disp_m']:>10.2f}{m.get('dir_cos', float('nan')):>9.3f}")
    lines.append(f"  {'OLD-360 run':<14}{OLD_360['run_rmse_m']:>9.2f}{'':>9}{OLD_360['run_hit3m']:>9.3f}")
    g = run.get("gaussian")
    if g:
        lines.append(f"  learned Gaussian dist: sigma=({run['sigma_m'][0]:.2f}, {run['sigma_m'][1]:.2f}) m"
                     f"  NLL={g['nll']:.2f}  coverage 1s={g['cov_1sigma']:.2f} (ideal 0.47) "
                     f"2s={g['cov_2sigma']:.2f} (ideal 0.91)")
    return "\n".join(lines)


def _fmt_recv(recv: dict) -> str:
    head = f"RECEIVER HEAD  (pass events={recv.get('n_events', 0)}"
    if "n_test_events" in recv:
        head += f", test events={recv['n_test_events']}"
    head += ")"
    lines = [head]
    if recv.get("note"):
        lines.append(f"  note: {recv['note']}")
    for name, m in recv.get("metrics", {}).items():
        lines.append(f"  {name:<14}top1={m.get('top1', float('nan')):.3f}  top3={m.get('top3', float('nan')):.3f}")
    lines.append(f"  {'OLD-360':<14}top1={OLD_360['receiver_top1']:.3f}  top3={OLD_360['receiver_top3']:.3f}"
                 f"  (heuristic top1={OLD_360['receiver_heuristic_top1']:.2f})")
    return "\n".join(lines)


def main() -> None:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", required=True, help="dense positions parquet")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res = evaluate(pd.read_parquet(args.positions), fps=args.fps, seed=args.seed)
    print(_fmt_run(res["run"]))
    print()
    print(_fmt_recv(res["receiver"]))


if __name__ == "__main__":
    main()
