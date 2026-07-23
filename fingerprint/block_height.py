"""Defensive block geometry: line height, vertical compactness, ball-to-block, low/mid/high class.

Phase-1 of the style-analysis v2 architecture (results/STYLE_RESEARCH_METHODS.md Axis 2 #2): the single
most-requested claim we cannot make from event counts alone -- "defends in a low block at X m vs presses
high". It runs on **trusted-geometry frames only** (the same ``calib_error_m`` gate report/facts uses) and
reuses the validated de-biased line-height convention (``generator.impute.line_estimates`` -- the
16.4->7.2 m held-out line work), so the block line is on the same scale as the shipped defensive line.

Out-of-possession is resolved from the **Viterbi possession smoother** (``generator.ball.assign_possession``
``smooth=True``): the team NOT in possession each frame is the defending team, and its own de-biased line is
its block. Each out-of-possession spell (contiguous frames a team defends within a chunk) is classified
low / mid / high by its median line height against declared thresholds on the de-biased (own-goal=0) scale.

Two honesty numbers ship with every block figure, per the research honesty trail:

* **coverage** -- the fraction of out-of-possession *frames* that have usable geometry (>=3 gated
  defenders visible to fix a line). The block is only measured on the frames the broadcast gives us.
* **broadcast bias** -- the broadcast frames the block preferentially during forward passes, so the frames
  with usable geometry are not a random sample of defensive time. We measure it directly: the mean ball
  advancement (in the defending team's own-goal=0 frame) on usable-geometry frames minus on
  missing-geometry frames. A positive number means the block is measured when the ball is further forward
  than typical -- i.e. the sampled block reads deeper/lower than the true average.

**Gate 1 (hand-annotated line-height validation) is not done here** -- every summary carries
``gate1_status="pending"``. No block-height claim is validated until a human annotates block lines on a
sample of frames; treat the classes as descriptive, not certified.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.registry import Match, get, matches
from fingerprint.roles import assign_roles
from fingerprint.structural_metrics import (
    attacking_coord,
    resolve_attack_directions,
    resolve_attack_directions_from_ball,
)
from fingerprint.theory_metrics import complete_directions
from generator.ball import assign_possession
from generator.impute import _outfield, line_estimates

# Trusted-geometry gate: identical to report/facts.CALIB_MAX_M so block frames match the fact store.
CALIB_MAX_M = 1.0

# Low/mid/high thresholds on the DE-BIASED line (own goal = 0 m, attacked goal = 105 m). These are the
# TRUE-scale block heights (the de-bias already removes the ~11 m broadcast inflation that
# fingerprint.phase_metrics carries on the raw visible line), so the thresholds are the un-inflated
# equivalents of that module's HIGH_PRESS_MIN_LINE (50 m true) / MID_BLOCK_MIN_LINE (33 m true).
LOW_BLOCK_MAX_M = 33.0    # median line below this over a spell => low block
HIGH_BLOCK_MIN_M = 50.0   # median line at/above this => high block/press; between the two => mid block
MIN_SPELL_FRAMES = 5      # ignore possession flickers shorter than this when classifying spells


def _directions(pos: pd.DataFrame, ball: pd.DataFrame) -> dict[int, int]:
    """Attacking directions for one chunk: ball-based first (robust on follow-play), keeper fallback."""
    dirs = resolve_attack_directions_from_ball(pos, ball)
    if len(dirs) < 2:
        dirs = resolve_attack_directions(pos)
    return complete_directions(dirs)


def block_frames(match: Match) -> pd.DataFrame:
    """Per out-of-possession frame: the defending team's block geometry on trusted frames.

    Iterates the match's linked-ball chunks (possession needs the ball). For each frame a team is out of
    possession (from the Viterbi smoother), records that team's de-biased line height, its block vertical
    spread, the ball's advancement in the team's own frame, the ball-to-block gap, and whether the frame
    had usable geometry. Missing-geometry out-of-possession frames are kept (``usable=False``, geometry
    columns NaN) so coverage and the broadcast-bias delta can both be computed downstream.

    Returns:
        ``chunk, frame, def_team, line_m, vspread_m, ball_adv_m, ball_to_block_m, usable``. Empty if the
        match has no linked-ball chunks or no resolvable directions.
    """
    aligned = match.load_aligned()
    roles = assign_roles(aligned)
    rows: list[dict] = []
    for ck, ball_path in match.ball_chunks():
        ball = pd.read_parquet(ball_path)
        # Possession + the out-of-possession timeline are built on ALL detected positions (ungated), so
        # the coverage denominator includes the frames the block geometry is missing on (deep defenders
        # off-screen / high calib error). Block geometry itself uses only the trusted-geometry subset.
        pos_all = aligned[aligned["chunk"] == ck].dropna(subset=["pitch_x", "pitch_y"])
        pos = pos_all[pos_all["calib_error_m"] <= CALIB_MAX_M]
        if pos_all.empty or ball.empty:
            continue
        dirs = _directions(pos_all, ball)
        if len(dirs) < 2:
            continue
        poss = assign_possession(ball, pos_all, smooth=True)
        if poss.empty:
            continue
        # de-biased line per (frame, team) for THIS chunk (same convention as the shipped def line)
        le = line_estimates(pos, roles) if not pos.empty else pd.DataFrame()
        line_of = {(int(r.frame), int(r.team)): float(r.line_debiased)
                   for r in le.itertuples(index=False)} if not le.empty else {}
        # per-(frame,team) vertical spread of the visible outfield block (std along attacking-x)
        vspread_of = _vspread(pos, dirs)
        ball_xy = {int(r.frame): (float(r.x), float(r.y)) for r in ball.itertuples(index=False)}
        teams = [t for t in dirs if int(t) >= 0]
        for r in poss.itertuples(index=False):
            fr, hold = int(r.frame), int(r.team)
            deft = next((t for t in teams if t != hold), None)
            if deft is None or fr not in ball_xy:
                continue
            bx, _ = ball_xy[fr]
            ball_adv = float(attacking_coord(np.array([bx]), dirs[deft])[0])  # own-goal=0 for defender
            line = line_of.get((fr, deft))
            vsp = vspread_of.get((fr, deft))
            usable = line is not None
            rows.append({
                "chunk": ck, "frame": fr, "def_team": deft,
                "line_m": line if usable else np.nan,
                "vspread_m": vsp if vsp is not None else np.nan,
                "ball_adv_m": ball_adv,
                "ball_to_block_m": (ball_adv - line) if usable else np.nan,
                "usable": usable})
    return pd.DataFrame(rows, columns=["chunk", "frame", "def_team", "line_m", "vspread_m",
                                       "ball_adv_m", "ball_to_block_m", "usable"])


def _vspread(pos: pd.DataFrame, dirs: dict[int, int]) -> dict[tuple[int, int], float]:
    """``(frame, team) -> std of the team's visible outfield along its attacking-x`` (block depth)."""
    out: dict[tuple[int, int], float] = {}
    of = _outfield(pos)
    for (fr, team), g in of.groupby(["frame", "team"]):
        team = int(team)
        if team not in dirs or len(g) < 3:
            continue
        ax = attacking_coord(g["pitch_x"].to_numpy(), dirs[team])
        out[(int(fr), team)] = float(np.std(ax))
    return out


def _classify(line_m: float) -> str:
    """Low / mid / high block from a de-biased median line height."""
    if line_m < LOW_BLOCK_MAX_M:
        return "low"
    if line_m >= HIGH_BLOCK_MIN_M:
        return "high"
    return "mid"


def block_spells(frames: pd.DataFrame) -> pd.DataFrame:
    """Contiguous out-of-possession spells per team, each classified low/mid/high by its median line.

    A spell is a run of consecutive sampled frames with the same ``def_team`` within a chunk (a possession
    flip or a chunk boundary ends it). Only spells with at least ``MIN_SPELL_FRAMES`` *usable* frames get a
    class; the median usable line height defines it.

    Returns:
        ``chunk, def_team, frame_start, frame_end, n_frames, n_usable, median_line_m, block_class``.
        Empty if no usable spells.
    """
    if frames.empty:
        return pd.DataFrame(columns=["chunk", "def_team", "frame_start", "frame_end", "n_frames",
                                     "n_usable", "median_line_m", "block_class"])
    rows: list[dict] = []
    for ck, g in frames.groupby("chunk"):
        g = g.sort_values("frame")
        # a new spell starts when the defending team changes or a frame gap opens
        new = (g["def_team"].to_numpy() != np.roll(g["def_team"].to_numpy(), 1))
        new[0] = True
        spell_id = np.cumsum(new)
        for _, sg in g.assign(_s=spell_id).groupby("_s"):
            usable = sg[sg["usable"]]
            if len(usable) < MIN_SPELL_FRAMES:
                continue
            med = float(usable["line_m"].median())
            rows.append({
                "chunk": ck, "def_team": int(sg["def_team"].iloc[0]),
                "frame_start": int(sg["frame"].iloc[0]), "frame_end": int(sg["frame"].iloc[-1]),
                "n_frames": int(len(sg)), "n_usable": int(len(usable)),
                "median_line_m": med, "block_class": _classify(med)})
    return pd.DataFrame(rows)


def block_summary(match: Match) -> dict:
    """Per-team block-geometry summary for one match, with coverage + broadcast-bias honesty numbers.

    Returns a dict keyed by team NAME with, per team: usable-frame geometry means (line, vertical spread,
    ball-to-block), the low/mid/high spell-share (weighted by usable frames), out-of-possession coverage,
    and the broadcast-bias delta (ball advancement on usable minus missing-geometry frames). Also carries
    ``gate1_status="pending"`` -- block heights are not human-validated here.
    """
    frames = block_frames(match)
    spells = block_spells(frames)
    name = {t: match.teams[t] for t in (0, 1)}
    out: dict = {
        "match": match.id, "metrics_version_note": "block_height v1 (unversioned; Gate 1 pending)",
        "gate1_status": "pending",
        # Coverage denominator = out-of-possession frames with a Viterbi possession fix (ball tracked).
        # This is bounded above by ball-track coverage (32-52% of match time), so a high coverage means
        # geometry is dense WITHIN observed defensive moments, not that all defensive time is seen.
        "coverage_denominator": "oop frames with a ball/possession fix (bounded by 32-52% ball coverage)",
        "teams": {}}
    for t in (0, 1):
        tf = frames[frames["def_team"] == t]
        if tf.empty:
            continue
        usable = tf[tf["usable"]]
        missing = tf[~tf["usable"]]
        n_oop = int(len(tf))
        coverage = float(usable.shape[0] / n_oop) if n_oop else float("nan")
        bias = (float(usable["ball_adv_m"].mean() - missing["ball_adv_m"].mean())
                if len(usable) and len(missing) else float("nan"))
        ts = spells[spells["def_team"] == t]
        cls_frames = {c: int(ts[ts["block_class"] == c]["n_usable"].sum()) for c in ("low", "mid", "high")}
        tot = sum(cls_frames.values()) or 1
        out["teams"][name[t]] = {
            "oop_frames": n_oop,
            "usable_frames": int(len(usable)),
            "missing_frames": int(len(missing)),
            "coverage": round(coverage, 3),
            "broadcast_bias_m": round(bias, 2) if np.isfinite(bias) else None,
            "mean_line_m": round(float(usable["line_m"].mean()), 1) if len(usable) else None,
            "median_line_m": round(float(usable["line_m"].median()), 1) if len(usable) else None,
            "mean_vspread_m": round(float(usable["vspread_m"].mean()), 1)
            if usable["vspread_m"].notna().any() else None,
            "mean_ball_to_block_m": round(float(usable["ball_to_block_m"].mean()), 1)
            if len(usable) else None,
            "n_spells": int(len(ts)),
            "block_class_share": {c: round(cls_frames[c] / tot, 3) for c in ("low", "mid", "high")},
            "modal_block": max(("low", "mid", "high"), key=lambda c: cls_frames[c]) if tot > 1 else None,
        }
    return out


def main() -> None:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", default="all", help="registered match id, or 'all' (processed, ball-linked)")
    args = ap.parse_args()
    ids = ([m.id for m in matches(processed_only=True)] if args.match == "all" else [args.match])
    for mid in ids:
        m = get(mid)
        if not m.ball_chunks():
            print(f"[block] {mid}: no linked-ball chunks, skipping")
            continue
        s = block_summary(m)
        print(f"[block] {mid} (gate1={s['gate1_status']})")
        for team, v in s["teams"].items():
            bc = v["block_class_share"]
            print(f"  {team:14s} line={v['mean_line_m']}m vspread={v['mean_vspread_m']}m "
                  f"b2b={v['mean_ball_to_block_m']}m cov={v['coverage']} bias={v['broadcast_bias_m']}m "
                  f"low/mid/high={bc['low']}/{bc['mid']}/{bc['high']} modal={v['modal_block']}")


if __name__ == "__main__":
    main()
