"""Run the possession analytics (#2) on a ball track: passing networks + PPDA, with pitch plots.

Reuses a linked ball track (``tools/ball_possession.py`` output) + the player positions to recompute
possession, then builds each team's passing network and the PPDA pressing proxy
(``fingerprint.possession_metrics``). Renders a passing-network plot per team and prints the metrics.

Run:
    python tools/possession_report.py --positions outputs/chunk000_dense.parquet \
        --ball outputs/ball_track_chunk000.parquet --team0 "Man United" --team1 "Man City" \
        --out results/possession
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.pitch import PITCH_LEN, PITCH_WID  # noqa: E402
_TEAM_COLORS = {0: "#e6194B", 1: "#4363d8"}


def _draw_pitch(ax) -> None:
    import matplotlib.patches as patches
    ax.add_patch(patches.Rectangle((0, 0), PITCH_LEN, PITCH_WID, ec="white", fc="#2d8a4e", lw=2))
    ax.plot([PITCH_LEN / 2, PITCH_LEN / 2], [0, PITCH_WID], color="white", lw=1)
    ax.add_patch(patches.Circle((PITCH_LEN / 2, PITCH_WID / 2), 9.15, ec="white", fc="none", lw=1))
    for x0 in (0, PITCH_LEN - 16.5):
        ax.add_patch(patches.Rectangle((x0, 13.84), 16.5, 40.32, ec="white", fc="none", lw=1))
    ax.set_xlim(-3, PITCH_LEN + 3)
    ax.set_ylim(-3, PITCH_WID + 3)
    ax.set_aspect("equal")
    ax.set_facecolor("#2d8a4e")
    ax.invert_yaxis()


def render_network(nodes: pd.DataFrame, edges: pd.DataFrame, *, team: int, name: str,
                   out: Path, labels: dict[int, str] | None = None) -> Path | None:
    """Draw a passing network: node size ~ involvement, edge width ~ pass count, on a pitch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if nodes.empty:
        return None
    pos = {int(r.track_id): (r.mean_x, r.mean_y) for r in nodes.itertuples(index=False)}
    fig, ax = plt.subplots(figsize=(11, 7.5))
    _draw_pitch(ax)
    emax = float(edges["count"].max()) if not edges.empty else 1.0
    for e in edges.itertuples(index=False):
        if int(e.from_id) in pos and int(e.to_id) in pos:
            (x0, y0), (x1, y1) = pos[int(e.from_id)], pos[int(e.to_id)]
            ax.plot([x0, x1], [y0, y1], "-", color=_TEAM_COLORS.get(team, "#888"),
                    lw=1 + 4 * e.count / emax, alpha=0.5, zorder=2)
    tmax = max(int(nodes["touches"].max()), 1)
    for r in nodes.itertuples(index=False):
        ax.scatter(r.mean_x, r.mean_y, s=120 + 600 * r.touches / tmax,
                   c=_TEAM_COLORS.get(team, "#888"), edgecolors="white", lw=1.5, zorder=3)
        tag = labels.get(int(r.track_id), str(int(r.track_id))) if labels else str(int(r.track_id))
        tag = tag.split(":")[-1]  # show just the role (drop the team prefix)
        ax.text(r.mean_x, r.mean_y, tag, color="white", fontsize=8,
                ha="center", va="center", zorder=4)
    ax.set_title(f"{name} passing network — node ~ involvement, edge ~ pass count")
    fig.patch.set_facecolor("#1a1a2e")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--ball", required=True, help="linked ball track parquet (frame,x,y[,observed])")
    ap.add_argument("--team0", default="team 0")
    ap.add_argument("--team1", default="team 1")
    ap.add_argument("--out", default="results/possession")
    ap.add_argument("--roles", default=None, help="roles parquet (fingerprint.roles) -> 11-node networks")
    ap.add_argument("--chunk", default=None, help="restrict the role map to this chunk")
    ap.add_argument("--smooth", action="store_true", help="Viterbi-smooth possession instead of debounce")
    args = ap.parse_args()

    from fingerprint.possession_metrics import (
        extract_passes,
        network_metrics,
        passing_network,
        ppda,
    )
    from fingerprint.structural_metrics import resolve_attack_directions
    from generator.ball import assign_possession

    pos = pd.read_parquet(args.positions)
    players = pos[pos["role"].isin(["player", "goalkeeper"])]
    ball = pd.read_parquet(args.ball)
    names = {0: args.team0, 1: args.team1}

    possession = assign_possession(ball, players, smooth=args.smooth)
    directions = resolve_attack_directions(players)
    labels: dict[int, str] = {}
    if args.roles:
        from fingerprint.roles import relabel_to_roles  # noqa: PLC0415

        roles = pd.read_parquet(args.roles)
        possession, players, labels = relabel_to_roles(possession, players, roles, chunk=args.chunk)
        print(f"role-collapsed: {len(labels)} stable role slots "
              f"({possession['team'].nunique()} teams)")
    passes = extract_passes(possession, players)
    print(f"possession frames: {len(possession)}   passes: {len(passes)}   "
          f"directions: {directions}")

    out = Path(args.out)
    for t in sorted(x for x in players["team"].unique() if int(x) >= 0):
        nodes, edges = passing_network(passes, players, team=int(t))
        m = network_metrics(nodes, edges)
        share = (possession["team"] == t).mean()
        print(f"\n[{names.get(int(t), t)}]  possession {share:.1%}  passes {m['n_passes']}  "
              f"players {m['n_players']}  top_connector {m['top_connector']}")
        print(f"   shape: compactness {m['compactness_m']:.1f}m  width {m['width_m']:.1f}m  "
              f"depth {m['depth_m']:.1f}m")
        top = m["top_connector"]
        if labels and top is not None:
            print(f"   top connector: {labels.get(int(top), top).split(':')[-1]}")
        p = render_network(nodes, edges, team=int(t), name=names.get(int(t), f"team {t}"),
                           out=out / f"passnet_team{int(t)}.png", labels=labels)
        if p:
            print(f"   network -> {p}")

    pp = ppda(possession, players, directions=directions)
    print("\nPPDA (passes allowed per defensive action — lower = more intense pressing):")
    for r in pp.itertuples(index=False):
        print(f"  {names.get(int(r.team), r.team)} pressing: passes_allowed {r.passes_allowed}  "
              f"pressures {r.pressures}  PPDA {r.ppda:.1f}")
    out.mkdir(parents=True, exist_ok=True)
    pp.to_parquet(out / "ppda.parquet", index=False)
    passes.to_parquet(out / "passes.parquet", index=False)


if __name__ == "__main__":
    main()
