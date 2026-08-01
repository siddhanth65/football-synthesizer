"""Score our from-pixels pipeline against the FOOTPASS VAL labels (PCBAS attribution).

The official SN-PCBAS baseline (Macro-F1 46.41) consumes **ground-truth game state** -- it is handed
the tracks and asks only "which action, by whom". This scorer asks the honest end-to-end question:
at each labelled event frame, does a pipeline that started from pixels name the acting player's
``(team, jersey)`` correctly?

PRIMARY metric -- *attribution-given-event*. The event frames are given (this is not action
spotting); the pipeline must produce ``(team, shirt)`` for the actor at that instant. Coverage is
over **all** events, precision over answered ones, and the dial is the pipeline's own confidence --
the exact convention of ``results/EVIDENCE_DENSITY_LAW.md`` §3, so the numbers are comparable to the
simulator's frontier and to ``results/OCR_DENSIFICATION.md`` §6.

Two things this scorer refuses to hide:

* **The off-screen ceiling.** ``EVIDENCE_DENSITY_LAW.md`` §1 measured 18.5% of PCBAS events as
  committed by a player carrying no broadcast ROI at that frame. Every split is reported separately
  and the VAL-specific rate is recomputed here rather than assumed.
* **The team-id mapping.** FOOTPASS teams are anonymised 1/2; our kit anchor emits 0/1. Both
  mappings are scored and both are printed; the reported one is the argmax, which is a 1-bit
  calibration off the rosters (an input), not a per-event fit.

Prediction schema (parquet or DataFrame), one row per answered *frame*::

    game        str   "game_18"
    frame       int   GLOBAL video frame index (chunk offset already added -- see
                      tools/footpass_prep.chunk_offsets)
    pred_team   int   pipeline team id (0/1); -1 or NA = no team
    pred_shirt  int   jersey number; <= 0 or NA = no answer
    conf        float confidence used as the coverage/precision dial (higher = surer)

CLI::

    python -m eval.footpass_score --preds outputs/footpass/preds.parquet
    python -m eval.footpass_score --selftest
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tools.evidence_sim import coverage_at, frontier

logger = logging.getLogger("footpass_score")

#: Digest root written by ``tools/footpass_digest.py`` (gitignored).
DIGEST_DIR = Path("data/footpass/digest")
#: The three VAL games (the only split whose labels we hold).
VAL_GAMES = ("game_18", "game_24", "game_47")
#: Halves per game, in the digest key convention.
HALVES = ("H1", "H2")
#: Our extraction samples every 5th frame, so an event frame is at most 2 away from a sampled one.
DEFAULT_TOL = 2
#: Precision floors reported for every split.
FLOORS = (0.85, 0.60)
#: Version stamp on the emitted JSON (bump when the metric definition changes).
SCORER_VERSION = "footpass-score-1.0"


def load_events(games: tuple[str, ...] = VAL_GAMES, *, digest_dir: Path = DIGEST_DIR
                ) -> pd.DataFrame:
    """Ground-truth PCBAS events of the given games, with actor visibility resolved.

    Args:
        games: FOOTPASS game ids (``game_18`` ...).
        digest_dir: Where ``tools/footpass_digest.py`` wrote ``val_<game>_<half>.npz``.

    Returns:
        One row per labelled event: ``game, half, frame, team, shirt, role_id, action, on_screen``.
        ``frame`` is the global (whole-file) video frame index, ``team`` is the FOOTPASS team id
        (1/2), ``on_screen`` is True when the actor carried a broadcast ROI at that exact frame.

    Raises:
        FileNotFoundError: if a digest is missing (run ``python -m tools.footpass_digest --split VAL``).
    """
    rows: list[pd.DataFrame] = []
    for game in games:
        for half in HALVES:
            path = digest_dir / f"val_{game}_{half}.npz"
            if not path.exists():
                raise FileNotFoundError(f"{path} -- run: python -m tools.footpass_digest --split VAL")
            d = np.load(path)
            ev, runs = d["events"], d["runs"]
            on = _on_screen(ev[:, 0], ev[:, 1], runs)
            rows.append(pd.DataFrame({
                "game": game, "half": half.lower(), "frame": ev[:, 0].astype(int),
                "team": d["team"][ev[:, 1]].astype(int), "shirt": d["shirt"][ev[:, 1]].astype(int),
                "role_id": d["role"][ev[:, 1]].astype(int), "action": ev[:, 2].astype(int),
                "on_screen": on,
            }))
    return pd.concat(rows, ignore_index=True).sort_values(["game", "frame"]).reset_index(drop=True)


def _on_screen(frames: np.ndarray, players: np.ndarray, runs: np.ndarray) -> np.ndarray:
    """True where player ``players[i]`` had a broadcast ROI at frame ``frames[i]`` (pure).

    Args:
        frames: Event frame indices.
        players: Player index into the digest roster, same length.
        runs: ``(n, 3)`` digest visibility runs ``(player index, start frame, end frame)``, inclusive.

    Returns:
        Boolean array, same length as ``frames``.
    """
    out = np.zeros(frames.size, bool)
    by_player: dict[int, np.ndarray] = {}
    for p in np.unique(runs[:, 0]) if runs.size else []:
        by_player[int(p)] = runs[runs[:, 0] == p][:, 1:]
    for i, (f, p) in enumerate(zip(frames, players)):
        r = by_player.get(int(p))
        if r is not None:
            out[i] = bool(np.any((r[:, 0] <= f) & (f <= r[:, 1])))
    return out


def match_predictions(events: pd.DataFrame, preds: pd.DataFrame, *, tol: int = DEFAULT_TOL
                      ) -> pd.DataFrame:
    """Attach the nearest prediction (within ``tol`` frames) to every event.

    Our extraction runs at ``sample_every=5``, so the pipeline has no row on most exact event
    frames. The nearest sampled frame inside ``tol`` is used; an event with none stays **unanswered**
    and still counts in the coverage denominator, which is the honest treatment.

    Args:
        events: :func:`load_events` output.
        preds: Prediction table (see module docstring).
        tol: Maximum |frame| distance accepted, in source frames.

    Returns:
        ``events`` plus ``pred_team, pred_shirt, conf, dframe``; NaN/-1 where nothing matched.
    """
    out = events.copy()
    out["pred_team"] = -1
    out["pred_shirt"] = -1
    out["conf"] = np.nan
    out["dframe"] = np.nan
    if preds.empty:
        return out
    p = preds.dropna(subset=["frame"]).sort_values(["game", "frame"])
    for game, ev_g in out.groupby("game"):
        pg = p[p["game"] == game]
        if pg.empty:
            continue
        pf = pg["frame"].to_numpy(np.int64)
        idx = np.searchsorted(pf, ev_g["frame"].to_numpy(np.int64))
        lo = np.clip(idx - 1, 0, pf.size - 1)
        hi = np.clip(idx, 0, pf.size - 1)
        ef = ev_g["frame"].to_numpy(np.int64)
        pick = np.where(np.abs(pf[lo] - ef) <= np.abs(pf[hi] - ef), lo, hi)
        d = np.abs(pf[pick] - ef)
        ok = d <= tol
        rows = ev_g.index.to_numpy()[ok]
        sel = pg.iloc[pick[ok]]
        out.loc[rows, "pred_team"] = sel["pred_team"].fillna(-1).to_numpy(int)
        out.loc[rows, "pred_shirt"] = sel["pred_shirt"].fillna(-1).to_numpy(int)
        out.loc[rows, "conf"] = sel["conf"].to_numpy(float)
        out.loc[rows, "dframe"] = d[ok]
    return out


def resolve_team_map(matched: pd.DataFrame) -> tuple[dict[int, int], dict[str, int]]:
    """Pick the pipeline-team -> FOOTPASS-team mapping whose TEAM label agrees with more events.

    Resolved on the team column only (not the jersey), because the kit-colour anchor is far more
    accurate than the jersey reader and would otherwise be graded through it. One bit, resolved once
    per scoring run. Both arms are returned so the choice is auditable and so a near-tie -- which
    would mean the kit anchor is not separating the teams at all -- is visible.

    Args:
        matched: :func:`match_predictions` output.

    Returns:
        ``(mapping, {"identity": n_correct, "flipped": n_correct})`` where ``mapping`` sends
        pipeline team 0/1 to FOOTPASS team 1/2.
    """
    ans = matched[(matched["pred_shirt"] > 0) & (matched["pred_team"] >= 0)]
    ident = {0: 1, 1: 2}
    flip = {0: 2, 1: 1}
    hits = {}
    for name, m in (("identity", ident), ("flipped", flip)):
        hits[name] = int((ans["pred_team"].map(m).fillna(-1) == ans["team"]).sum())
    return (ident if hits["identity"] >= hits["flipped"] else flip), hits


def _split_metrics(sub: pd.DataFrame, mapping: dict[int, int], n_points: int = 60) -> dict:
    """Frontier + coverage-at-floor for one slice of events (pure)."""
    answered = ((sub["pred_shirt"] > 0) & (sub["pred_team"] >= 0)).to_numpy()
    correct = (answered
               & (sub["pred_shirt"].to_numpy() == sub["shirt"].to_numpy())
               & (sub["pred_team"].map(mapping).fillna(-1).to_numpy() == sub["team"].to_numpy()))
    conf = sub["conf"].fillna(-1.0).to_numpy(float)
    n = int(len(sub))
    front = frontier(correct, answered, conf, n, n_points=n_points)
    n_ans = int(answered.sum())
    return {
        "n_events": n,
        "n_answered": n_ans,
        "answer_rate": n_ans / n if n else float("nan"),
        "n_correct": int(correct.sum()),
        "full_coverage": {"coverage": n_ans / n if n else float("nan"),
                          "precision": float(correct.sum() / n_ans) if n_ans else float("nan")},
        **{f"coverage_at_{f}": coverage_at(front, f) for f in FLOORS},
        "frontier": [{"threshold": t, "coverage": c, "precision": p} for t, c, p in front],
    }


def score(events: pd.DataFrame, preds: pd.DataFrame, *, tol: int = DEFAULT_TOL,
          mapping: dict[int, int] | None = None) -> dict:
    """Full PRIMARY metric: attribution-given-event, split by actor visibility.

    Args:
        events: :func:`load_events` output.
        preds: Prediction table (see module docstring).
        tol: Frame tolerance when attaching a prediction to an event frame.
        mapping: Force a pipeline-team -> FOOTPASS-team map; ``None`` resolves it from the data.

    Returns:
        Nested metrics dict: ``overall`` / ``on_screen`` / ``off_screen`` / per-game / per-action,
        plus the resolved team map and the measured on-screen ceiling.
    """
    matched = match_predictions(events, preds, tol=tol)
    resolved, hits = (mapping, {}) if mapping else resolve_team_map(matched)
    n = len(matched)
    out = {
        "version": SCORER_VERSION,
        "n_events": n,
        "frame_tolerance": tol,
        "team_map": {str(k): v for k, v in resolved.items()},
        "team_map_hits": hits,
        "on_screen_rate": float(matched["on_screen"].mean()) if n else float("nan"),
        "n_events_with_a_prediction_frame": int(matched["conf"].notna().sum()),
        "median_frame_offset_used": float(matched["dframe"].median(skipna=True)) if n else None,
        "overall": _split_metrics(matched, resolved),
        "on_screen": _split_metrics(matched[matched["on_screen"]], resolved),
        "off_screen": _split_metrics(matched[~matched["on_screen"]], resolved),
        "per_game": {g: _split_metrics(s, resolved, n_points=30)
                     for g, s in matched.groupby("game")},
        "per_action": {int(a): _split_metrics(s, resolved, n_points=20)
                       for a, s in matched.groupby("action")},
    }
    return out


def summary_lines(res: dict) -> list[str]:
    """ASCII-safe one-screen summary of a :func:`score` result."""
    lines = [f"FOOTPASS VAL attribution-given-event ({res['version']})",
             f"  events {res['n_events']}  actor on screen {100 * res['on_screen_rate']:.1f}%"
             f"  team map {res['team_map']} hits {res['team_map_hits']}",
             f"  events with a prediction frame within +-{res['frame_tolerance']}: "
             f"{res['n_events_with_a_prediction_frame']}"]
    for split in ("overall", "on_screen", "off_screen"):
        m = res[split]
        fc = m["full_coverage"]
        lines.append(
            f"  {split:10s} n={m['n_events']:5d}  cov@0.85 {m['coverage_at_0.85']:.4f}  "
            f"cov@0.60 {m['coverage_at_0.6']:.4f}  "
            f"full-cov {fc['coverage']:.4f} @ prec {fc['precision']:.4f}")
    return lines


def _selftest() -> None:
    """Deterministic mini-fixture: the scorer's arithmetic, end to end, with no real data."""
    ev = pd.DataFrame({
        "game": ["g"] * 8, "half": ["h1"] * 8, "frame": [0, 10, 20, 30, 40, 50, 60, 70],
        "team": [1, 1, 2, 2, 1, 2, 1, 2], "shirt": [7, 9, 3, 4, 7, 3, 9, 4],
        "role_id": [10] * 8, "action": [2] * 8,
        "on_screen": [True, True, True, True, True, True, False, False],
    })
    # pipeline team 0 == FOOTPASS team 1 (identity map); 6 answers, 4 of them right.
    pr = pd.DataFrame({
        "game": ["g"] * 6, "frame": [1, 11, 21, 31, 41, 61],
        "pred_team": [0, 0, 1, 1, 0, 0], "pred_shirt": [7, 9, 3, 99, 99, 9],
        "conf": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
    })
    m = match_predictions(ev, pr, tol=2)
    assert int((m["pred_shirt"] > 0).sum()) == 6, m["pred_shirt"].tolist()
    assert m.loc[5, "pred_shirt"] == -1  # frame 50 has no prediction within +-2
    mapping, hits = resolve_team_map(m)
    # the map is resolved on TEAM agreement alone: all 6 answers carry the right team, 4 the right shirt.
    assert mapping == {0: 1, 1: 2} and hits == {"identity": 6, "flipped": 0}, (mapping, hits)

    res = score(ev, pr, tol=2)
    o = res["overall"]
    assert o["n_events"] == 8 and o["n_answered"] == 6 and o["n_correct"] == 4, o
    assert abs(o["full_coverage"]["coverage"] - 6 / 8) < 1e-9
    assert abs(o["full_coverage"]["precision"] - 4 / 6) < 1e-9
    # top-3 answers by confidence are all correct -> precision 1.0 at coverage 3/8.
    assert abs(o["coverage_at_0.85"] - 3 / 8) < 1e-9, o["coverage_at_0.85"]
    # cumulative precision by confidence: 1, 1, 1, .75, .60, .667 -> the 0.60 floor holds to the end.
    assert abs(o["coverage_at_0.6"] - 6 / 8) < 1e-9, o["coverage_at_0.6"]
    assert res["off_screen"]["n_events"] == 2 and res["on_screen"]["n_events"] == 6
    assert abs(res["on_screen_rate"] - 0.75) < 1e-9

    # a flipped pipeline must be detected, and score identically once mapped.
    pr_f = pr.assign(pred_team=1 - pr["pred_team"])
    mp, hits_f = resolve_team_map(match_predictions(ev, pr_f, tol=2))
    assert mp == {0: 2, 1: 1} and hits_f == {"identity": 0, "flipped": 6}, (mp, hits_f)
    assert score(ev, pr_f, tol=2)["overall"]["n_correct"] == 4

    # tolerance is respected: at tol=0 only the exact-frame prediction survives (none here).
    assert int((match_predictions(ev, pr, tol=0)["pred_shirt"] > 0).sum()) == 0
    print("footpass_score selftest ok")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", type=Path, help="prediction parquet (see module docstring)")
    ap.add_argument("--games", default=",".join(VAL_GAMES))
    ap.add_argument("--tol", type=int, default=DEFAULT_TOL)
    ap.add_argument("--out", type=Path, default=Path("results/footpass/val_attribution.json"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.preds:
        raise SystemExit("--preds is required (or --selftest)")
    events = load_events(tuple(args.games.split(",")))
    res = score(events, pd.read_parquet(args.preds), tol=args.tol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n".join(summary_lines(res)))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
