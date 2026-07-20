"""Score-state segmentation (Plan B-5): label every tracked frame level / chasing / leading.

Turns the VALIDATED goal timeline into a per-frame Man Utd score state, then slices the existing
style-fingerprint primitives (phase profile, counter-press) by that state. No new science here --
this module only assembles validated pieces:

* **Boundary times** are the E2E-Spot goal peaks (``results/action_spotting_probe/<match>/
  summary.json``, the top-N peaks that match the oracle per-half count). The Sofascore season dict
  caches per-half goal *counts*, not incident minutes, so the exact boundary *times* come from those
  validated spots. Across the six matches the E2E per-half count matches the Sofascore split
  directly, with one correction: Tottenham emitted a fourth H2 peak (score 0.338) above the true
  1H1/2H2 split -- the known replay false positive -- which is dropped (see :data:`GOALS`).
* **Goal ownership** (which team scored) comes from the Sofascore per-half score deltas
  (``homeScore/awayScore`` ``period1``/``period2``). The two same-half Brighton second-half goals are
  disambiguated by the final scoreline: the 90+' goal is the winning team's (the known Joao Pedro
  stoppage-time winner) -- so the earlier H2 goal is Man Utd's equaliser. Tottenham (0-3) and
  Southampton (0-3 Man Utd win) are single-team scorelines; Palace (0-0) has no goals, so Man Utd is
  level the whole match.

Frame clock: each parquet frame sits in chunk ``hX_chunk_NNN`` at within-chunk 25 fps index
``frame``; its half-relative second is ``chunk_offset[hX][NNN] + frame / 25``. Offsets are the E2E
score-npz durations (``score_len / 2 fps``) -- the SAME clock the goal peaks were measured in, so a
frame time and a goal time are directly comparable (verified: parquet ``max_frame / 25`` equals E2E
``score_len / 2`` per chunk to < 0.3 s).

Every number this module feeds is still the tracking-native fingerprint under the same ball-gap
ceilings; score state only re-buckets frames that were already computed and validated.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from fingerprint import style_fingerprint as sf

E2E_DIR = Path("results/action_spotting_probe")
GRID_FPS = 25.0
STOPPAGE_S = 45 * 60          # half-relative seconds past regulation -> stoppage-time flag
MANU = "Man Utd"
_CHUNK_NUM = re.compile(r"chunk(\d+)\.npz$")

# Validated goal timeline per match: (half, half-relative t_s seconds, scorer). ``scorer`` is
# 'manu' or 'opp'. Provenance in the module docstring (validated E2E spots + Sofascore per-half
# deltas). Times are the ``t_s`` of the top-N goal peaks in each summary.json that match the oracle
# per-half count -- liverpool 2H1/1H2 (all Liverpool), brighton 1H1/2H2, fulham 0H1/1H2,
# tottenham 1H1/2H2 (all Tottenham), southampton 2H1/1H2 (all Man Utd), palace 0-0 (no goals).
# tottenham: E2E emitted 4 peaks >0.3 (1H1/3H2) but the Sofascore split is 1H1/2H2; the extra H2
# peak (h2 t_s=1692.5, score 0.338 -- the lowest of the three H2 peaks) is the known replay FP and
# is dropped, keeping the two high-confidence H2 goals (0.963, 0.907) + the ~3' H1 opener.
GOALS: dict[str, list[tuple[str, float, str]]] = {
    "manutd_liverpool": [("h1", 2066.0, "opp"), ("h1", 2518.0, "opp"), ("h2", 739.0, "opp")],
    "brighton_manutd": [("h1", 1895.5, "opp"), ("h2", 794.5, "manu"), ("h2", 2881.0, "opp")],
    "manutd_fulham": [("h2", 2512.5, "manu")],
    "manutd_tottenham": [("h1", 162.5, "opp"), ("h2", 210.5, "opp"), ("h2", 2005.5, "opp")],
    "southampton_manutd": [("h1", 2103.5, "manu"), ("h1", 2462.5, "manu"), ("h2", 3069.5, "manu")],
    "palace_manutd": [],  # 0-0 draw: Man Utd level the whole match (Sofascore 12436962).
}


def manu_index(match) -> int:
    """Team index (0/1) of Man Utd in ``match.teams`` (the ``team`` column's convention)."""
    return match.teams.index(MANU)


def chunk_offsets(match_id: str) -> dict[str, float]:
    """Half-relative start second of every chunk, from the E2E score-npz durations.

    Reads ``results/action_spotting_probe/<match_id>/scores_<half>_chunkNNN.npz`` in order and
    accumulates ``score_len / fps`` per half. This is the exact clock the goal peaks were measured
    in, so frame times and goal times share one origin.

    Args:
        match_id: registry match id.

    Returns:
        ``{chunk_key: start_offset_s}`` for every chunk with an E2E score file (e.g.
        ``{"h1_chunk_000": 0.0, "h1_chunk_001": 600.0, ...}``).
    """
    out: dict[str, float] = {}
    base = E2E_DIR / match_id
    for half in ("h1", "h2"):
        off = 0.0
        for p in sorted(base.glob(f"scores_{half}_chunk*.npz")):
            mo = _CHUNK_NUM.search(p.name)
            if mo is None:
                continue
            z = np.load(p)
            out[f"{half}_chunk_{int(mo.group(1)):03d}"] = off
            off += z["scores"].shape[0] / float(z["fps"])
    return out


def score_state_at(match_id: str, half: str, t_s: float) -> dict:
    """Man Utd score state at a half-relative time (pure; the unit-tested segmentation seam).

    Args:
        match_id: registry match id (keys :data:`GOALS`).
        half: ``"h1"`` or ``"h2"``.
        t_s: half-relative seconds within ``half``.

    Returns:
        ``{"manu": int, "opp": int, "scoreline": "m-o", "state": level|chasing|leading,
        "stoppage": bool}`` -- ``scoreline`` and ``state`` are from Man Utd's perspective.
    """
    manu = opp = 0
    for gh, gt, who in GOALS.get(match_id, []):
        counted = (gh == "h1" and half == "h2") or (gh == half and gt <= t_s)
        if not counted:
            continue
        if who == "manu":
            manu += 1
        else:
            opp += 1
    diff = manu - opp
    state = "level" if diff == 0 else ("leading" if diff > 0 else "chasing")
    return {"manu": manu, "opp": opp, "scoreline": f"{manu}-{opp}", "state": state,
            "stoppage": t_s > STOPPAGE_S}


def annotate(match_id: str, df: pd.DataFrame,
             offsets: dict[str, float] | None = None) -> pd.DataFrame:
    """Add ``half, t_s, manu, opp, scoreline, state, stoppage`` columns to a ``chunk``+``frame`` table.

    Args:
        match_id: registry match id.
        df: any frame table with ``chunk`` and ``frame`` columns.
        offsets: chunk->start-second map; computed from :func:`chunk_offsets` when ``None``.

    Returns:
        A copy of ``df`` with the score-state columns appended.
    """
    if offsets is None:
        offsets = chunk_offsets(match_id)
    half = df["chunk"].str[:2]
    t_s = df["chunk"].map(offsets).fillna(0.0) + df["frame"] / GRID_FPS
    recs = [score_state_at(match_id, h, float(t)) for h, t in zip(half, t_s)]
    add = pd.DataFrame(recs, index=df.index)
    out = df.copy()
    out["half"] = half.to_numpy()
    out["t_s"] = t_s.to_numpy()
    for c in ("manu", "opp", "scoreline", "state", "stoppage"):
        out[c] = add[c].to_numpy()
    return out


def timeline(match) -> pd.DataFrame:
    """Per-unique-frame score-state timeline for one match (``chunk, frame, half, t_s, ...``)."""
    df = match.load_aligned()[["chunk", "frame"]].drop_duplicates()
    return annotate(match.id, df).sort_values(["half", "t_s"]).reset_index(drop=True)


def segments(match) -> pd.DataFrame:
    """Contiguous score-state segments per half: ``half, state, scoreline, t0_s, t1_s, stoppage``.

    Collapses the goal timeline into the level/chasing/leading spans (with the running scoreline)
    for one match -- the score-state story spine. Boundaries are the validated goal times; the half
    end is the last observed frame time in that half.
    """
    tl = timeline(match)
    rows: list[dict] = []
    for half in ("h1", "h2"):
        sub = tl[tl["half"] == half]
        if sub.empty:
            continue
        t_end = float(sub["t_s"].max())
        bounds = [0.0] + sorted(gt for gh, gt, _ in GOALS.get(match.id, []) if gh == half)
        for i, t0 in enumerate(bounds):
            t1 = bounds[i + 1] if i + 1 < len(bounds) else t_end
            mid = (t0 + t1) / 2.0
            st = score_state_at(match.id, half, mid)
            rows.append({"half": half, "state": st["state"], "scoreline": st["scoreline"],
                         "t0_s": round(t0, 1), "t1_s": round(t1, 1),
                         "stoppage": t1 > STOPPAGE_S})
    return pd.DataFrame(rows, columns=["half", "state", "scoreline", "t0_s", "t1_s", "stoppage"])


def segment_phase(match, *, by: str = "state", transition_s: float = sf.TRANSITION_S
                  ) -> pd.DataFrame:
    """Phase profile sliced by score state: ``team, <by>, phase, frames`` + shape metrics.

    Args:
        match: registry match.
        by: bucketing column -- ``"state"`` (level/chasing/leading) or ``"scoreline"`` (0-0, 0-1...).
        transition_s: transition window passed through to the phase labeller.

    Returns:
        Mean block/width/compactness/depth per ``(team, <by>, phase)`` with a ``frames`` count.
    """
    tab = sf.phase_frame_table(match, transition_s=transition_s)
    if tab.empty:
        return pd.DataFrame(columns=["team", by, "phase", "frames", *sf.PHASE_METRIC_COLS])
    ann = annotate(match.id, tab)
    keys = ["team", by, "phase"]
    agg = ann.groupby(keys)[list(sf.PHASE_METRIC_COLS)].mean()
    agg.insert(0, "frames", ann.groupby(keys).size())
    return agg.reset_index()


def segment_counterpress(match, *, by: str = "state", transition_s: float = sf.TRANSITION_S,
                         press_radius_m: float = sf.PRESS_RADIUS_M) -> pd.DataFrame:
    """Counter-press sliced by score state: ``team, <by>, losses_outside_third, ...`` per bucket.

    Args:
        match: registry match.
        by: bucketing column (``"state"`` or ``"scoreline"``).
        transition_s: counter-press window.
        press_radius_m: pressure radius.

    Returns:
        ``team, <by>, losses_outside_third, counterpress_frac, regain_5s_frac`` per bucket
        (the ball-gap ``poss_frames``/``turnovers`` totals are match-level and not re-split here --
        ``losses_outside_third`` is the per-bucket evaluable base).
    """
    losses, _poss, _turn = sf.turnover_press_table(
        match, transition_s=transition_s, press_radius_m=press_radius_m)
    cols = ["team", by, "losses_outside_third", "counterpress_frac", "regain_5s_frac"]
    if losses.empty:
        return pd.DataFrame(columns=cols)
    ann = annotate(match.id, losses)
    g = ann.groupby(["team", by])
    out = g.agg(losses_outside_third=("pressed", "size"),
                counterpress_frac=("pressed", "mean"),
                regain_5s_frac=("regained", "mean")).reset_index()
    return out[cols]


def _demo() -> None:
    """Self-check on the pure score-state seam (no match data)."""
    # Liverpool: Man Utd chases from the 34:26 (2066 s) opener; level before it, chasing after.
    assert score_state_at("manutd_liverpool", "h1", 2000.0)["state"] == "level"
    s = score_state_at("manutd_liverpool", "h1", 2100.0)
    assert s["state"] == "chasing" and s["scoreline"] == "0-1", s
    assert score_state_at("manutd_liverpool", "h2", 100.0)["scoreline"] == "0-2"   # carries H1 goals
    assert score_state_at("manutd_liverpool", "h2", 800.0)["scoreline"] == "0-3"
    # Fulham: level until the 87' winner (2512.5 s in H2), then Man Utd lead 1-0.
    assert score_state_at("manutd_fulham", "h2", 2000.0)["state"] == "level"
    lead = score_state_at("manutd_fulham", "h2", 2600.0)
    assert lead["state"] == "leading" and lead["scoreline"] == "1-0", lead
    # Brighton: 0-0, then chasing 0-1, then level 1-1 (Man Utd equaliser 794.5 s H2), then chasing 1-2.
    assert score_state_at("brighton_manutd", "h1", 100.0)["state"] == "level"
    assert score_state_at("brighton_manutd", "h2", 500.0)["scoreline"] == "0-1"
    assert score_state_at("brighton_manutd", "h2", 1000.0)["state"] == "level"      # equalised
    end = score_state_at("brighton_manutd", "h2", 2900.0)
    assert end["state"] == "chasing" and end["scoreline"] == "1-2" and end["stoppage"], end
    # Tottenham 0-3: chasing from the ~3' opener (162.5 s H1), 0-3 by the late H2 goal (2005.5 s).
    assert score_state_at("manutd_tottenham", "h1", 100.0)["state"] == "level"
    tot = score_state_at("manutd_tottenham", "h1", 200.0)
    assert tot["state"] == "chasing" and tot["scoreline"] == "0-1", tot
    assert score_state_at("manutd_tottenham", "h2", 300.0)["scoreline"] == "0-2"
    assert score_state_at("manutd_tottenham", "h2", 2100.0)["scoreline"] == "0-3"
    # Southampton (Man Utd 0-3 win): leading from the 1st H1 goal (2103.5 s), 3-0 by the H2 goal.
    assert score_state_at("southampton_manutd", "h1", 2000.0)["state"] == "level"
    sou = score_state_at("southampton_manutd", "h1", 2200.0)
    assert sou["state"] == "leading" and sou["scoreline"] == "1-0", sou
    assert score_state_at("southampton_manutd", "h1", 2500.0)["scoreline"] == "2-0"
    assert score_state_at("southampton_manutd", "h2", 3100.0)["scoreline"] == "3-0"
    # Palace 0-0: level for the whole match.
    assert score_state_at("palace_manutd", "h2", 2600.0)["state"] == "level"
    print("score_state self-check OK")


if __name__ == "__main__":
    _demo()
