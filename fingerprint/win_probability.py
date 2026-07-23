"""Bayesian-style in-game win probability, BASE SUBSET (Robberechts, Van Haaren & Davis, KDD'21).

The KDD'21 model conditions win probability on the game state; its full feature set needs Opta-grade
located events (attacking passes, xT, chance quality, duel strength) that our 26.4%-usable-geometry
broadcast cannot supply. We implement the paper's **base subset** only -- the covariates that are
undisputed and that we (or open data) can measure exactly:

    minutes_remaining, score_diff, elo_diff (clubelo strength prior), red_diff, yellow_diff, is_home

and predict the three-way match outcome (win / draw / loss) from a team's perspective. The paper shows
this simplified model still beats naive baselines (its RPS 0.138 vs 0.134 full); the honesty trail
requires it be labelled *base subset* and pass a calibration check (ECE, reliability) before use.

Implementation choice (ponytail): PyMC is overkill and a gradient-boosted 3-class classifier is
WORSE here -- it fits the elo_diff dimension non-monotonically (a stronger team getting a lower win
prob) because the per-minute snapshots are dominated by the score_diff signal, so kickoff prediction
is unreliable. A **multinomial logistic regression over engineered game-state features** reproduces the
base subset, extrapolates monotonically and sensibly (kickoff home P(win) at equal strength ~= the
0.41 home-win base rate; a 1-goal lead with 5 min left ~0.84), keeps proper 3-class probabilities, and
is the same model family the paper uses. The engineered design adds the two interactions raw
covariates miss: ``score_diff / sqrt(minutes_remaining + 1)`` (a lead is worth more with less time)
and ``score_diff^2`` (distance from a draw). Every in-game minute of every training match is one
snapshot labelled with the eventual result; both perspectives (home and away) of each snapshot are
emitted so the model is symmetric and learns home advantage through ``is_home``. Whether it is "good
enough" is decided by the held-out ECE, not this docstring -- see :func:`expected_calibration_error`
and ``tools/build_win_probability.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["minutes_remaining", "score_diff", "elo_diff", "red_diff", "yellow_diff", "is_home"]
RESULTS = ("loss", "draw", "win")   # ordinal, but used as plain 3-class labels
FULL_TIME = 90


def _design(feats: pd.DataFrame) -> np.ndarray:
    """Engineered design matrix from the raw :data:`FEATURES` table (adds the WP interactions).

    Columns: ``minutes_remaining, score_diff, score_diff/sqrt(mr+1), score_diff^2, elo_diff/100,
    red_diff, yellow_diff, is_home``.
    """
    mr = feats["minutes_remaining"].to_numpy(float)
    sd = feats["score_diff"].to_numpy(float)
    return np.column_stack([mr, sd, sd / np.sqrt(mr + 1.0), sd * sd,
                            feats["elo_diff"].to_numpy(float) / 100.0,
                            feats["red_diff"].to_numpy(float), feats["yellow_diff"].to_numpy(float),
                            feats["is_home"].to_numpy(float)])


def _running(events: list[tuple[int, int]], upto_min: int) -> int:
    """Signed count (home - away) of ``(minute, side)`` events strictly before ``upto_min``.

    ``side`` is ``+1`` for the home team, ``-1`` for the away team.
    """
    return sum(side for mn, side in events if mn < upto_min)


def match_snapshots(goals: list[tuple[int, int]], reds: list[tuple[int, int]],
                    yellows: list[tuple[int, int]], *, elo_home: float, elo_away: float,
                    home_result: str | None = None, step_min: int = 1) -> pd.DataFrame:
    """Per-minute game-state snapshots for one match, BOTH perspectives (home and away rows).

    Each event list is ``(minute, side)`` with ``side`` in ``{+1 home, -1 away}``. A snapshot at
    minute ``t`` reflects only events with ``minute < t`` (state at the start of that minute) and
    ``minutes_remaining = FULL_TIME - t``. The home-perspective row uses ``score_diff = home - away``
    etc.; the away-perspective row negates the diffs and flips ``is_home`` -- so one match contributes
    a symmetric pair per minute and the fitted model is perspective-agnostic.

    Args:
        goals: goal events ``(minute, +1|-1)``.
        reds: red-card events (includes second yellows) ``(minute, +1|-1)``.
        yellows: yellow-card events ``(minute, +1|-1)``.
        elo_home: home clubelo Elo at the match date.
        elo_away: away clubelo Elo at the match date.
        home_result: ``"win"/"draw"/"loss"`` from the home side (for training); ``None`` for inference.
        step_min: snapshot spacing in minutes.

    Returns:
        Rows with :data:`FEATURES` (+ ``minute`` and, if ``home_result`` given, ``result``). Two rows
        (home, away perspective) per snapshot minute when training; the away ``result`` is the mirror.
    """
    elo_gap = float(elo_home - elo_away)
    away_result = {"win": "loss", "loss": "win", "draw": "draw"}.get(home_result or "")
    rows: list[dict] = []
    for t in range(0, FULL_TIME + 1, step_min):
        sd = _running(goals, t)
        rd = _running(reds, t)
        yd = _running(yellows, t)
        mr = FULL_TIME - t
        home = {"minute": t, "minutes_remaining": mr, "score_diff": sd, "elo_diff": elo_gap,
                "red_diff": rd, "yellow_diff": yd, "is_home": 1}
        away = {"minute": t, "minutes_remaining": mr, "score_diff": -sd, "elo_diff": -elo_gap,
                "red_diff": -rd, "yellow_diff": -yd, "is_home": 0}
        if home_result is not None:
            home["result"] = home_result
            away["result"] = away_result
        rows.append(home)
        rows.append(away)
    return pd.DataFrame(rows)


def perspective_features(goals: list[tuple[int, int]], reds: list[tuple[int, int]],
                         yellows: list[tuple[int, int]], *, elo_for: float, elo_against: float,
                         is_home: int, step_min: int = 1) -> pd.DataFrame:
    """Per-minute :data:`FEATURES` timeline from ONE team's perspective (for WP inference).

    Events are ``(minute, side)`` with ``side = +1`` for the perspective team and ``-1`` for the
    opponent; the diffs are already this-team-minus-opponent. Unlike :func:`match_snapshots` this
    emits a single row per minute and takes ``is_home`` explicitly (so a team playing away still gets
    ``is_home = 0``).

    Args:
        goals: goal events ``(minute, +1 this team / -1 opponent)``.
        reds: red-card events, same sign convention.
        yellows: yellow-card events, same sign convention.
        elo_for: the perspective team's Elo at the match date.
        elo_against: the opponent's Elo.
        is_home: 1 if the perspective team is at home, else 0.
        step_min: snapshot spacing.

    Returns:
        Rows with ``minute`` + :data:`FEATURES` (no ``result``).
    """
    gap = float(elo_for - elo_against)
    rows = [{"minute": t, "minutes_remaining": FULL_TIME - t, "score_diff": _running(goals, t),
             "elo_diff": gap, "red_diff": _running(reds, t), "yellow_diff": _running(yellows, t),
             "is_home": is_home}
            for t in range(0, FULL_TIME + 1, step_min)]
    return pd.DataFrame(rows)


def fit(train: pd.DataFrame, *, seed: int = 0):
    """Fit the 3-class game-state -> outcome model on stacked snapshots (``FEATURES`` + ``result``).

    Returns a fitted scikit-learn pipeline (standardise + multinomial logistic regression) whose
    ``classes_`` gives the label order for :func:`win_probability`. ``seed`` is unused (the solver is
    deterministic) but kept for a stable call signature.
    """
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.pipeline import make_pipeline  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    del seed
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    clf.fit(_design(train), train["result"].to_numpy())
    return clf


def win_probability(model, feats: pd.DataFrame) -> np.ndarray:
    """P(win) for each row of a ``FEATURES`` table (the WP(t) series when rows are a match timeline)."""
    proba = model.predict_proba(_design(feats))
    win_col = list(model.classes_).index("win")
    return proba[:, win_col]


def outcome_proba(model, feats: pd.DataFrame) -> pd.DataFrame:
    """Full ``loss/draw/win`` probabilities for a ``FEATURES`` table (columns in :data:`RESULTS`)."""
    proba = model.predict_proba(_design(feats))
    cols = {c: proba[:, i] for i, c in enumerate(model.classes_)}
    return pd.DataFrame({r: cols.get(r, np.zeros(len(feats))) for r in RESULTS})


def expected_calibration_error(y_win: np.ndarray, p_win: np.ndarray, *, n_bins: int = 10
                               ) -> tuple[float, pd.DataFrame]:
    """ECE of the win-probability + the reliability table (equal-width bins on ``p_win``).

    Args:
        y_win: binary outcome (1 = the perspective team won).
        p_win: predicted P(win).
        n_bins: number of equal-width probability bins.

    Returns:
        ``(ece, reliability)`` -- ``ece`` is the sample-weighted mean ``|confidence - accuracy|``;
        ``reliability`` has ``bin_lo, bin_hi, n, mean_p, frac_win`` per non-empty bin.
    """
    y_win = np.asarray(y_win, float)
    p_win = np.asarray(p_win, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p_win, edges[1:-1]), 0, n_bins - 1)
    rows: list[dict] = []
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        mean_p = float(p_win[mask].mean())
        frac = float(y_win[mask].mean())
        ece += n * abs(mean_p - frac)
        rows.append({"bin_lo": round(edges[b], 2), "bin_hi": round(edges[b + 1], 2), "n": n,
                     "mean_p": round(mean_p, 4), "frac_win": round(frac, 4)})
    return ece / len(p_win), pd.DataFrame(rows)


def _demo() -> None:
    """Self-check: snapshots are symmetric, and the model calibrates on a synthetic corpus."""
    # symmetry: away perspective mirrors home.
    snap = match_snapshots([(20, 1)], [], [(30, -1)], elo_home=1800, elo_away=1700,
                           home_result="win")
    h = snap[snap["is_home"] == 1]
    a = snap[snap["is_home"] == 0]
    assert (h["score_diff"].to_numpy() == -a["score_diff"].to_numpy()).all()
    assert (a["result"] == "loss").all() and (h["result"] == "win").all()
    # state at minute 25 (goal at 20) is +1 for home.
    assert int(h[h["minute"] == 25]["score_diff"].iloc[0]) == 1

    # synthetic corpus: home strength + score lead should drive a calibrated win prob.
    rng = np.random.default_rng(0)
    parts = []
    for _ in range(400):
        eh, ea = rng.normal(1750, 90), rng.normal(1750, 90)
        # crude ground-truth: goals ~ strength; simulate a final score
        lam_h = 1.4 * np.exp((eh - ea) / 400.0)
        lam_a = 1.4 * np.exp((ea - eh) / 400.0)
        gh, ga = rng.poisson(lam_h), rng.poisson(lam_a)
        res = "win" if gh > ga else ("loss" if gh < ga else "draw")
        gmins = sorted(rng.integers(1, 90, gh + ga))
        goals = []
        hleft, aleft = gh, ga
        for mn in gmins:                       # deal goals to sides in a fixed but mixed order
            if hleft and (not aleft or rng.random() < hleft / (hleft + aleft)):
                goals.append((int(mn), 1))
                hleft -= 1
            else:
                goals.append((int(mn), -1))
                aleft -= 1
        parts.append(match_snapshots(goals, [], [], elo_home=eh, elo_away=ea, home_result=res))
    data = pd.concat(parts, ignore_index=True)
    tr, te = data.iloc[: len(data) * 4 // 5], data.iloc[len(data) * 4 // 5:]
    model = fit(tr)
    p = win_probability(model, te)
    ece, rel = expected_calibration_error((te["result"] == "win").to_numpy(), p)
    assert 0.0 <= p.min() and p.max() <= 1.0
    assert ece < 0.08, (ece, rel)
    # monotonicity: a stronger team never has a lower kickoff win prob.
    grid = pd.DataFrame({"minutes_remaining": [90, 90], "score_diff": [0, 0], "elo_diff": [-150, 150],
                         "red_diff": [0, 0], "yellow_diff": [0, 0], "is_home": [1, 1]})
    lo, hi = win_probability(model, grid)
    assert hi > lo, (lo, hi)
    print(f"win_probability self-check OK (synthetic held-out ECE={ece:.3f}, n_bins={len(rel)}, "
          f"elo monotone {lo:.2f}<{hi:.2f})")


if __name__ == "__main__":
    _demo()
