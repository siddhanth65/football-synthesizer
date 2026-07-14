"""C5 opponent-conditioned predictor, pooled across every processed team-match.

Each match yields TWO observations (each team's shape + the opponent's defensive posture), so the 4
processed matches give 8 team-matches -- far more than France's 3 alone. All features are CV-derived (no
reliance on FIFA PMSRs, which only exist for the France games), so the model extends to any match.

We forecast a **vector** of tendencies (attacking-third share, line height, width, wing focus) as a
function of the **opponent's defensive depth** (opponent CV line height; lower = deeper block). Tier-B =
this pooled opponent-conditioned fit; Tier-A = the team-average baseline. Headline test (leave-one-match-
out): per tendency, does conditioning on the opponent beat the average? Honest: 8 obs is small -- a
validated signal + framework, not significance; tightens as matches are ingested.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.registry import matches
from fingerprint.roles import assign_roles
from generator.impute import line_estimates
from synthesizer.predict import predict_tendencies

# Team tendencies we forecast (the tactical shape a side adopts).
TARGETS = ("attacking_third_share", "def_line_height", "width", "wing_share")
OPP_FEATURE = "def_line_height"   # opponent's defensive-line height (deep = low)
OBS_PATH = "outputs/c5_observations.parquet"


def build_observations() -> pd.DataFrame:
    """One row per (team, opponent): team's tendency vector + opponent's defensive depth (all CV).

    The line-height dimensions (``def_line_height`` target + ``opp_def_depth`` feature) use the **P2
    de-biased line** (``generator.impute.line_estimates``, validated to ~5 m vs FIFA), not the raw
    censoring-inflated 20th-percentile line — so the opponent model is trained on the corrected geometry.
    Width / wing / attacking-third come from the position-only tendencies unchanged.
    """
    rows = []
    for m in matches(processed_only=True):
        pos = m.load_aligned()
        td = {t: predict_tendencies(pos, t, tier="A") for t in (0, 1)}
        if not (td[0] and td[1]):
            continue
        le = line_estimates(pos, assign_roles(pos))
        dline = {t: float(le[le["team"] == t]["line_debiased"].mean()) for t in (0, 1)}
        for t, o in [(0, 1), (1, 0)]:
            row = {"match": m.id, "team": m.teams[t], "opponent": m.teams[o],
                   "opp_def_depth": dline[o]}
            row.update({k: td[t][k]["mean"] for k in TARGETS if k != "def_line_height"})
            row["def_line_height"] = dline[t]
            rows.append(row)
    return pd.DataFrame(rows)


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, resid_std) for y ~ x (falls back to mean if x degenerate)."""
    if len(np.unique(x)) < 2:
        return 0.0, float(np.mean(y)), float(np.std(y))
    s, i = np.polyfit(x, y, 1)
    return float(s), float(i), float(np.std(y - (i + s * x)))


def backtest(obs: pd.DataFrame) -> pd.DataFrame:
    """Per tendency: leave-one-match-out Tier-A (mean) vs Tier-B (opp-conditioned) MAE + skill."""
    rows = []
    for tgt in TARGETS:
        a_err, b_err = [], []
        for mid in obs["match"].unique():
            tr, te = obs[obs["match"] != mid], obs[obs["match"] == mid]
            a_pred = float(tr[tgt].mean())
            s, i, _ = _fit_line(tr["opp_def_depth"].to_numpy(), tr[tgt].to_numpy())
            for r in te.itertuples(index=False):
                a_err.append(abs(a_pred - getattr(r, tgt)))
                b_err.append(abs((i + s * r.opp_def_depth) - getattr(r, tgt)))
        a, b = float(np.mean(a_err)), float(np.mean(b_err))
        rows.append({"tendency": tgt, "tier_a_mae": a, "tier_b_mae": b,
                     "skill": (a - b) / a if a else 0.0})
    return pd.DataFrame(rows)


def fit(obs: pd.DataFrame) -> dict:
    """Fit the pooled opponent model for every tendency: ``{tendency: (slope, intercept, resid_std)}``."""
    return {tgt: _fit_line(obs["opp_def_depth"].to_numpy(), obs[tgt].to_numpy()) for tgt in TARGETS}


def load_model() -> dict:
    """Fit from cached observations (fast)."""
    return fit(pd.read_parquet(OBS_PATH))


def forecast(opp_def_depth: float, model: dict | None = None) -> dict:
    """Forecast a team's full tendency vector vs an opponent of the given defensive depth (+/- 1 sigma)."""
    model = model or load_model()
    out = {}
    for tgt, (s, i, sd) in model.items():
        mu = i + s * opp_def_depth
        out[tgt] = {"mean": mu, "lo": mu - sd, "hi": mu + sd}
    return out


def main() -> None:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cached", action="store_true", help="reuse outputs/c5_observations.parquet")
    args = ap.parse_args()
    obs = (pd.read_parquet(OBS_PATH) if args.cached and Path(OBS_PATH).exists() else build_observations())
    obs.to_parquet(OBS_PATH, index=False)
    pd.set_option("display.width", 180)
    print("=== C5 observations (team tendency vector vs opponent defensive depth) ===\n")
    print(obs.round(3).to_string(index=False))
    bt = backtest(obs)
    print(f"\n=== per-tendency leave-one-match-out (n={len(obs)} team-matches) ===")
    print(f"  {'tendency':<24}{'Tier-A':>9}{'Tier-B':>9}{'skill':>8}")
    for r in bt.itertuples(index=False):
        print(f"  {r.tendency:<24}{r.tier_a_mae:>9.3f}{r.tier_b_mae:>9.3f}{r.skill:>7.0%}"
              f"{'  B wins' if r.skill > 0 else ''}")
    won = int((bt["skill"] > 0).sum())
    print(f"\n  Tier B beats Tier A on {won}/{len(bt)} tendencies.")
    model = fit(obs)
    print("\n=== demo: a team vs opponents of different depth (attacking_third_share) ===")
    # depths span the de-biased opp-line range (~14-36 m); extrapolating past it is not meaningful.
    for d, lbl in [(16, "deep low-block"), (25, "balanced"), (35, "high line")]:
        f = forecast(d, model)["attacking_third_share"]
        print(f"  vs {lbl:<16} (line {d}m) -> att3rd {f['mean']:.2f} [{f['lo']:.2f}, {f['hi']:.2f}]")
    _plot(obs, model, bt)


def _plot(obs: pd.DataFrame, model: dict, bt: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(5 * len(TARGETS), 4.6))
    skill = dict(zip(bt["tendency"], bt["skill"]))
    for ax, tgt in zip(axes, TARGETS):
        for team, g in obs.groupby("team"):
            ax.scatter(g["opp_def_depth"], g[tgt], s=70, label=team, zorder=3)
        xs = np.linspace(obs["opp_def_depth"].min() - 3, obs["opp_def_depth"].max() + 3, 30)
        s, i, sd = model[tgt]
        ax.plot(xs, i + s * xs, "--", color="#333")
        ax.fill_between(xs, i + s * xs - sd, i + s * xs + sd, color="#333", alpha=0.1)
        ax.set_title(f"{tgt}\nLOMO skill {skill[tgt]:+.0%}", fontsize=10, fontweight="bold")
        ax.set_xlabel("opp line height (m)"); ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("C5 opponent-conditioned matchup forecast — tendency vs opponent defensive depth "
                 f"(n={len(obs)} team-matches)", fontsize=12, fontweight="bold")
    out = Path("results/c5_opponent_model.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
