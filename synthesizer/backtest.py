"""DEPRECATED — use :mod:`synthesizer.opponent_model` (the pooled, registry-driven C5 backtest).

This was the first France-only (n=3) leave-one-match-out backtest on the FIFA ``opp_low_block`` feature.
Its verdict (Tier B loses at n=3, skill -106%..-172%) was an overfitting artifact of fitting a line
through 2 points per fold; `opponent_model.py` pools every team-match (n=8, CV features both sides) and
Tier B wins (+43%/+40% on the vertical tendencies). Kept only as the historical record of that negative
result; do not extend it (docs/PROJECT_AUDIT_2026-07.md section 3.3).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

TENDENCIES = ("france_line", "france_att3rd", "france_wing")
OPP_FEATURE = "opp_low_block"


def tier_b_predict(train: pd.DataFrame, tendency: str, opp_value: float,
                   feature: str = OPP_FEATURE) -> float:
    """Tier-B point prediction: linear fit of ``tendency`` on the opponent feature, evaluated at ``opp_value``."""
    if train[feature].nunique() < 2:
        return float(train[tendency].mean())
    b, a = np.polyfit(train[feature].to_numpy(float), train[tendency].to_numpy(float), 1)
    return float(a + b * opp_value)


def backtest(profile: pd.DataFrame, *, feature: str = OPP_FEATURE) -> pd.DataFrame:
    """Leave-one-match-out: per tendency, Tier-A vs Tier-B held-out absolute error + skill.

    Returns ``tendency, tier_a_mae, tier_b_mae, skill`` where ``skill = (A-B)/A`` (>0 means Tier B helps).
    """
    warnings.warn("synthesizer.backtest is deprecated; use synthesizer.opponent_model.backtest",
                  DeprecationWarning, stacklevel=2)
    rows = []
    for tend in TENDENCIES:
        if tend not in profile.columns:
            continue
        a_errs, b_errs = [], []
        for i in range(len(profile)):
            test = profile.iloc[i]
            train = profile.drop(profile.index[i])
            a_pred = float(train[tend].mean())
            b_pred = tier_b_predict(train, tend, float(test[feature]), feature=feature)
            a_errs.append(abs(a_pred - test[tend]))
            b_errs.append(abs(b_pred - test[tend]))
        a_mae, b_mae = float(np.mean(a_errs)), float(np.mean(b_errs))
        rows.append({"tendency": tend, "tier_a_mae": a_mae, "tier_b_mae": b_mae,
                     "skill": (a_mae - b_mae) / a_mae if a_mae else 0.0})
    return pd.DataFrame(rows)


def main() -> None:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="outputs/france_profile.parquet")
    args = ap.parse_args()
    df = pd.read_parquet(args.profile)
    res = backtest(df)
    print("C5 leave-one-match-out backtest — does opponent-conditioning (Tier B) beat team-average (Tier A)?\n")
    print(f"  (n={len(df)} France matches; opponent feature = {OPP_FEATURE})\n")
    print(f"  {'tendency':<16}{'Tier-A MAE':>12}{'Tier-B MAE':>12}{'skill':>9}")
    for r in res.itertuples(index=False):
        flag = "  Tier B wins" if r.skill > 0 else "  (Tier A)"
        print(f"  {r.tendency:<16}{r.tier_a_mae:>12.2f}{r.tier_b_mae:>12.2f}{r.skill:>8.0%}{flag}")
    won = int((res["skill"] > 0).sum())
    print(f"\n  Tier B beats Tier A on {won}/{len(res)} tendencies "
          f"(directional — n={len(df)} matches; firms up with more data).")


if __name__ == "__main__":
    main()
