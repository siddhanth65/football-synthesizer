"""Phase split of the team metrics -- coarse (in/out possession) and fine-grained (ball-driven, FIFA-style).

Splitting the per-frame structural metrics by phase gives the FIFA-style contrast the overall fingerprint
flattens: **line height & width in possession vs the defensive block** (EFI pp.6-7 vs 27-28), and, with a
real ball track, the finer **build-up / progression / final-third** (in possession) and **high-press /
mid-block / low-block** (out of possession) split (EFI pp.4, 27-28).

Two entry points:

* :func:`phase_split_fingerprint` -- coarse, **no ball needed**. Possession is carried forward from the
  sparse ball-carrier (``is_actor``) detections; only ``{in_poss, out_poss}``. Carry-forward can
  misattribute across replays/turnovers.
* :func:`ball_phase_fingerprint` -- fine-grained, **needs a linked ball track** (the fine-tuned detector
  -> ``tools/ball_possession.py``). The ball's attacking-x buckets the in-possession phase; the team's own
  defensive-line height buckets the out-of-possession block. This is the split FIFA reports per phase.

Direction is resolved per chunk from the keeper. Teams should be colour-anchored for a multi-chunk match.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fingerprint.structural_metrics import (
    PITCH_LEN,
    attacking_coord,
    compute_metrics_table,
    resolve_attack_directions,
)

_PHASE_COLS = ("buildup_height", "def_line_height", "width", "compactness")

# Ball-driven in-possession phase = where the ball is, in the holding team's attacking-x (0=own goal).
# Thresholds are PARTIAL-VIEW corrected: broadcast shows ~6/11 players so the visible defensive line
# reads ~+11 m too high (deep defenders off-frame; measured vs FIFA). Build-up = own-half possession
# (FIFA lumps unopposed+opposed build-up there); block thresholds carry the +11 m line offset so a real
# mid-block (~38 m) that we *see* at ~49 m is still bucketed as mid-block, not high press.
PARTIAL_VIEW_LINE_OFFSET = 11.0   # measured inflation of the visible line vs the true line (m)
BUILDUP_MAX_X = 50.0       # own half = build-up (was 35; FIFA build-up spans most own-half possession)
PROGRESSION_MAX_X = 70.0   # middle third: progression; beyond = final third
# Out-of-possession block = where the defending team's own (visible) line sits, in *its* attacking-x.
HIGH_PRESS_MIN_LINE = 50.0 + PARTIAL_VIEW_LINE_OFFSET   # 61 m visible ~= true high line past halfway
MID_BLOCK_MIN_LINE = 33.0 + PARTIAL_VIEW_LINE_OFFSET    # 44-61 m visible = mid block; below = low block


def possession_by_frame(positions: pd.DataFrame) -> pd.DataFrame:
    """``(chunk, frame) -> poss_team`` by carrying the ball-carrier's team forward within each chunk."""
    groups = positions.groupby("chunk") if "chunk" in positions.columns else [("_", positions)]
    rows = []
    for ck, g in groups:
        actors = g[g["is_actor"] == True].groupby("frame")["team"].first().sort_index()  # noqa: E712
        if actors.empty:
            continue
        frames = np.sort(g["frame"].unique())
        idx = np.searchsorted(actors.index.to_numpy(), frames, side="right") - 1  # last carrier <= frame
        at = actors.to_numpy()
        for f, i in zip(frames, idx):
            if i >= 0:
                rows.append({"chunk": ck, "frame": int(f), "poss_team": int(at[i])})
    return pd.DataFrame(rows, columns=["chunk", "frame", "poss_team"])


def phase_split_fingerprint(positions: pd.DataFrame) -> pd.DataFrame:
    """Per ``(team, phase)`` mean structural metrics, ``phase`` in ``{in_poss, out_poss}``.

    Teams should already be globally consistent (colour-anchored) for a multi-chunk match.
    """
    has_chunk = "chunk" in positions.columns
    groups = positions.groupby("chunk") if has_chunk else [("_", positions)]
    poss = possession_by_frame(positions)
    parts = []
    for ck, g in groups:
        mt = compute_metrics_table(g, attack_dirs=resolve_attack_directions(g))
        if mt.empty:
            continue
        p = poss[poss["chunk"] == ck].set_index("frame")["poss_team"] if not poss.empty else pd.Series(dtype=int)
        mt = mt.assign(poss_team=mt["frame"].map(p)).dropna(subset=["poss_team"])
        parts.append(mt)
    if not parts:
        return pd.DataFrame()
    allm = pd.concat(parts, ignore_index=True)
    allm["phase"] = np.where(allm["team"] == allm["poss_team"].astype(int), "in_poss", "out_poss")
    agg = allm.groupby(["team", "phase"])[list(_PHASE_COLS)].mean()
    agg.insert(0, "frames", allm.groupby(["team", "phase"]).size())
    return agg.reset_index()


def ball_phase_by_frame(ball: pd.DataFrame, possession: pd.DataFrame, players: pd.DataFrame, *,
                        directions: dict[int, int] | None = None) -> pd.DataFrame:
    """Classify each possession frame into a FIFA-style phase from the ball location + block height.

    *In possession* (the holding team): bucket by the **ball's attacking-x** (the holder's direction) ->
    ``build_up`` / ``progression`` / ``final_third``. *Out of possession* (the other team, same frame):
    bucket by **that team's own defensive-line height** -> ``high_press`` / ``mid_block`` / ``low_block``.

    Args:
        ball: linked ball track ``frame, x, y`` (pitch metres).
        possession: ``frame, carrier, team`` from :func:`generator.ball.assign_possession`.
        players: positions ``frame, team, pitch_x, pitch_y, is_keeper, role`` (one chunk).
        directions: attacking directions per team; auto-resolved from keepers if ``None``.

    Returns:
        ``frame, team, phase, in_possession`` -- two rows per possession frame (holder + defender) when
        both teams have a resolved direction and on-pitch players.
    """
    cols = ["frame", "team", "phase", "in_possession"]
    if ball.empty or possession.empty:
        return pd.DataFrame(columns=cols)
    directions = directions or resolve_attack_directions(players)
    ball_xy = {int(r.frame): (float(r.x), float(r.y)) for r in ball.itertuples(index=False)}
    pl = players.dropna(subset=["pitch_x", "pitch_y"])
    by_frame = {fr: g for fr, g in pl.groupby("frame")}
    teams = [t for t in directions if int(t) >= 0]
    rows = []
    for r in possession.itertuples(index=False):
        fr, hold = int(r.frame), int(r.team)
        if hold not in directions or fr not in ball_xy:
            continue
        bx, _ = ball_xy[fr]
        ball_ax = float(attacking_coord(np.array([bx]), directions[hold])[0])
        phase = ("build_up" if ball_ax <= BUILDUP_MAX_X
                 else "progression" if ball_ax <= PROGRESSION_MAX_X else "final_third")
        rows.append({"frame": fr, "team": hold, "phase": phase, "in_possession": True})
        # The defending team, same frame: classify its block by its own line height.
        deft = next((t for t in teams if t != hold), None)
        g = by_frame.get(fr)
        if deft is None or g is None:
            continue
        d = g[(g["team"] == deft)]
        if "is_keeper" in d.columns:
            d = d[~d["is_keeper"].astype(bool)]
        if len(d) < 3:
            continue
        line = float(np.percentile(attacking_coord(d["pitch_x"].to_numpy(), directions[deft]), 20))
        block = ("high_press" if line >= HIGH_PRESS_MIN_LINE
                 else "mid_block" if line >= MID_BLOCK_MIN_LINE else "low_block")
        rows.append({"frame": fr, "team": deft, "phase": block, "in_possession": False})
    return pd.DataFrame(rows, columns=cols)


def ball_phase_fingerprint(ball: pd.DataFrame, possession: pd.DataFrame,
                           players: pd.DataFrame) -> pd.DataFrame:
    """Per ``(team, phase)`` mean structural metrics, using a real ball track for the phase split.

    Joins the ball-driven phase label onto the per-frame structural metrics so each FIFA phase
    (build_up / progression / final_third / high_press / mid_block / low_block) gets its own
    width / line height / compactness -- the per-phase team shape FIFA prints.
    """
    phases = ball_phase_by_frame(ball, possession, players)
    if phases.empty:
        return pd.DataFrame()
    mt = compute_metrics_table(players, attack_dirs=resolve_attack_directions(players))
    if mt.empty:
        return pd.DataFrame()
    merged = mt.merge(phases, on=["frame", "team"], how="inner")
    if merged.empty:
        return pd.DataFrame()
    agg = merged.groupby(["team", "in_possession", "phase"])[list(_PHASE_COLS)].mean()
    agg.insert(0, "frames", merged.groupby(["team", "in_possession", "phase"]).size())
    return agg.reset_index().sort_values(["team", "in_possession", "phase"]).reset_index(drop=True)


def main() -> None:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", required=True, help="colour-anchored match positions parquet")
    ap.add_argument("--ball", default=None, help="linked ball track parquet (enables fine-grained phases)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    positions = pd.read_parquet(args.positions)
    pd.set_option("display.width", 200)
    if args.ball:
        from generator.ball import assign_possession  # noqa: PLC0415

        ball = pd.read_parquet(args.ball)
        poss = assign_possession(ball, positions)
        z = ball_phase_fingerprint(ball, poss, positions)
        print("Ball-driven phase split (FIFA-style build-up/progression/final-third + block):\n")
    else:
        z = phase_split_fingerprint(positions)
        print("In/out-of-possession split (line height & block by phase):\n")
    print(z.round(2).to_string(index=False))
    if args.out:
        z.to_parquet(args.out, index=False)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
