"""Tendency predictor: Tier A marginal P(y|z_T) -> Tier B matchup P(y|z_T, z_O).

Outputs **distributions** (not point predictions) over FIFA-EFI-aligned tendencies (defensive-line
height, build-up height, width, attacking-third share, wing share, ...).

* **Tier A (built)** -- the marginal "how this team plays": the empirical distribution of each tendency
  across the team's observed chunks/matches, summarised as mean + an 80% interval (10th-90th pct). With a
  single match the interval is the *within-match* spread (a lower bound on true predictive uncertainty;
  it widens honestly as more matches are ingested).
* **Tier B (TODO, data-gated)** -- the opponent-conditioned matchup ``P(y | z_T, z_O)``. Needs many
  team-matches to learn the opponent term; with 2 matches it cannot be fit, so it is left explicit.

The headline test stays: **does Tier B beat Tier A** once enough matches exist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from synthesizer.features import TENDENCIES, team_tendency_samples


def marginal_distribution(samples: pd.DataFrame, *, interval: float = 0.8) -> dict[str, dict]:
    """Tier-A distribution per tendency from per-chunk samples: mean / std / [lo, hi] interval / n.

    ``interval`` is the central probability mass of the empirical interval (0.8 -> 10th..90th pct).
    """
    lo_q, hi_q = (1 - interval) / 2 * 100, (1 + interval) / 2 * 100
    out: dict[str, dict] = {}
    for t in TENDENCIES:
        if t not in samples.columns:
            continue
        x = samples[t].dropna().to_numpy()
        if len(x) == 0:
            continue
        out[t] = {"mean": float(x.mean()), "std": float(x.std()),
                  "lo": float(np.percentile(x, lo_q)), "hi": float(np.percentile(x, hi_q)),
                  "n": int(len(x))}
    return out


def predict_tendencies(positions: pd.DataFrame, team: int, *, opponent: int | None = None,
                       tier: str = "A", interval: float = 0.8) -> dict[str, dict]:
    """Predict tendency distributions for ``team`` from match ``positions``.

    Args:
        positions: colour-anchored match positions (``chunk`` column).
        team: team label (0/1) to forecast.
        opponent: reserved for Tier B (opponent-conditioned); ignored for Tier A.
        tier: ``"A"`` marginal (built). ``"B"`` raises -- it needs many matches (data-gated).
        interval: central mass of the reported prediction interval.

    Returns:
        ``{tendency: {mean, std, lo, hi, n}}``.
    """
    if tier.upper() == "B":
        raise NotImplementedError(
            "Tier B (opponent-conditioned) needs many team-matches to learn the opponent term; "
            "we have too few. Ingest more matches, then fit P(y | z_T, z_O).")
    samples = team_tendency_samples(positions, team)
    if samples.empty:
        return {}
    return marginal_distribution(samples, interval=interval)


def main() -> None:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", required=True, help="colour-anchored match positions parquet")
    ap.add_argument("--team", type=int, default=0)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    dist = predict_tendencies(pd.read_parquet(args.positions), args.team, tier="A")
    print(f"Tier-A tendency forecast for {args.name or f'team {args.team}'} "
          f"(mean [80% interval], n chunks):\n")
    for t, d in dist.items():
        print(f"  {t:<22} {d['mean']:6.2f}  [{d['lo']:6.2f}, {d['hi']:6.2f}]  (n={d['n']})")


if __name__ == "__main__":
    main()
