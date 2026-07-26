"""Evidence engine for the Manchester United EPL 2024-25 identity profile.

Computes the four position-and-territory layers that the profile
(``results/MANUTD_IDENTITY_PROFILE.md``) is allowed to stand on, and writes that profile. Every
layer is POSITIONAL: where the ball travels, where players stand, how the block is shaped. No
per-player event counts (attribution is 1-4% of truth,
``results/PLAYER_ANALYSIS_v2.md``) and no cross-team rate comparisons (58-90 matches needed,
``results/W1B_WINDOW_AND_SAMPLING.md``).

Layers:

1. **Ball progression** (:func:`ball_layer`) -- the linked ball track under the Viterbi possession
   smoother, oriented so Man Utd attack ``+x``. Per match: post-link ball coverage, lateral-channel
   and vertical-zone occupancy of United-possession ball samples, forward metres gained per channel,
   and final-third entries per channel. Cross-checked against the validated PASS ledger where one
   exists (4 matches).
2. **Player territory** (:func:`player_layer`) -- reuses
   :func:`fingerprint.player_profiles.oriented_positions` on the 9 PRTreID identity matches, pooled
   per player per manager era: mean position, spread, lateral-channel occupancy, deepest/highest
   percentiles. Roles are territorial labels, never volume.
3. **Block shape and its variance** (:func:`block_layer`) -- :mod:`fingerprint.block_height` per
   match, plus the new quantity: how much the block MOVES. Between-match SD of line and of vertical
   spread, within-match spell-to-spell SD, leg-to-leg swing against the same opponent, and a WP-band
   split on the 6 matches with a win-probability series.
4. **Manager era** (:func:`era_layer`) -- the block/ball/territory numbers split ten Hag vs Amorim.
   Venue and manager flip together in every pair, so the confound is printed with every number.

Orientation is resolved per chunk (ball-based, keeper fallback) then forced to a per-half majority
with the half-time end swap, because per-chunk resolution flips on 5-10% of chunks and a flipped
chunk mirrors left and right. The disagreement rate is reported as a first-class honesty number.

Run (CPU, ~4 min for 12 matches)::

    python -m tools.manutd_identity            # compute + write the profile
    python -m tools.manutd_identity --cached   # re-render prose from the cached evidence
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.pitch import PITCH_LEN, PITCH_WID
from core.registry import Match, get, matches
from fingerprint import block_height as bh
from fingerprint import player_profiles as pp
from fingerprint import score_state as ss
from fingerprint.structural_metrics import resolve_attack_directions
from generator.ball import assign_possession
from tools.run_game_state_v2 import WP_DIR, band_of, match_minute

MANU = "Man Utd"
EVIDENCE_PATH = Path("outputs/manutd_identity/evidence.json")
PROFILE_PATH = Path("results/MANUTD_IDENTITY_PROFILE.md")

GRID_FPS = 25.0
CHANNELS = ("left", "central", "right")
ZONES = ("own third", "middle third", "final third")
# Lateral channel edges on the 68 m width, in Man Utd's attacking frame. Channel naming is
# ANCHORED EMPIRICALLY, not assumed: see player_layer's `channel_anchor` check (known left-sided
# and right-sided players must land on opposite sides of halfway, or the profile abstains on
# left/right language). y_or < 22.67 is the LEFT channel under the anchor that check confirms.
CHAN_EDGES = (PITCH_WID / 3.0, 2.0 * PITCH_WID / 3.0)
ZONE_EDGES = (PITCH_LEN / 3.0, 2.0 * PITCH_LEN / 3.0)
FINAL_THIRD_X = ZONE_EDGES[1]

KEEPER_SEP_M = 40.0  # the two keepers must be this far apart before a chunk may vote on direction
MAX_STEP_M = 20.0   # ponytail: per-sample ball displacement cap; a >20 m jump in 0.2 s is a link error
MAX_GAP_FR = 15     # consecutive ball samples must be within 0.6 s (3 grid steps) to chain
MIN_PLAYER_FRAMES = 25       # a pooled per-era player position needs this many named frames
MIN_HEADLINE_FRAMES = 150    # a player may only carry a headline claim above this frame count
# Known-sided United players, used to ANCHOR (not assume) the lateral sign of the pitch model.
LEFT_ANCHORS = ("Alejandro Garnacho", "Marcus Rashford", "Luke Shaw", "Tyrell Malacia",
                "Lisandro Martínez")
RIGHT_ANCHORS = ("Diogo Dalot", "Amad Diallo", "Noussair Mazraoui", "Antony")
ERAS = ("ten_hag", "amorim")


# ==================================================================================================
# Orientation (shared by every layer that needs left/right)
# ==================================================================================================
def manu_matches() -> list[Match]:
    """Processed, ball-linked Manchester United matches, in date order."""
    ms = [m for m in matches(processed_only=True)
          if MANU in m.teams and m.date and m.ball_chunks()]
    return sorted(ms, key=lambda m: m.date or "")


def manu_dirs(match: Match) -> tuple[dict[str, int], dict]:
    """Man Utd's attacking sign per half, from cleanly-separated keeper pairs + the half-time swap.

    Direction is the load-bearing convention for every left/right and own-half/final-third claim
    here, and BOTH shipped per-chunk resolvers are unreliable on this corpus:

    * :func:`resolve_attack_directions_from_ball` is systematically **inverted** -- its 4 m
      nearest-player carrier proxy picks the *defending* team in a crowded defensive third, so the
      team defending the ``x=105`` goal reads as the team attacking it. Measured against the
      Sofascore ``D<M<F`` position oracle it flips the sign of the rank correlation in 5/5 matches
      tested (``+0.93 -> -0.93`` on ``manutd_brighton``).
    * :func:`resolve_attack_directions` (keeper-based) is correct in principle but noisy per chunk:
      on follow-play footage both teams' keeper-tagged rows cluster at the on-screen goal. On
      ``tottenham_manutd`` h1 it resolves 3 of 6 chunks backwards against the goal-direction
      evidence in ``results/PAIR_ANALYSIS_v1.md``.

    So a chunk only votes when the two keepers **separate cleanly** (medians at least
    :data:`KEEPER_SEP_M` apart, the same test the pair-analysis mapping adjudication used), the
    votes are summed per half weighted by keeper-row support, and the half-time end swap is
    enforced because a real match cannot attack the same goal in both halves.

    Args:
        match: registry match.

    Returns:
        ``({chunk_key: +1/-1 for Man Utd}, diagnostics)``; diagnostics carry how many chunks
        supplied a clean vote and whether the half swap had to be forced.
    """
    mi = match.teams.index(MANU)
    aligned = match.load_aligned()
    is_gk = (aligned["role"] == "goalkeeper") | aligned.get("is_keeper", False)
    kp = aligned[is_gk].dropna(subset=["pitch_x"])
    votes: dict[str, tuple[int, float]] = {}
    for ck, g in kp.groupby("chunk"):
        med = g.groupby("team")["pitch_x"].median()
        cnt = g.groupby("team").size()
        pair = [t for t in med.index if int(t) >= 0]
        if len(pair) != 2 or abs(med[pair[0]] - med[pair[1]]) < KEEPER_SEP_M:
            continue
        if mi not in pair:
            continue
        sign = 1 if float(med[mi]) < PITCH_LEN / 2 else -1   # keeper at own goal -> attack away
        votes[str(ck)] = (sign, float(min(cnt[pair[0]], cnt[pair[1]])))
    half_score = {h: sum(s * w for ck, (s, w) in votes.items() if ck.startswith(h))
                  for h in ("h1", "h2")}
    forced = half_score["h1"] * half_score["h2"] >= 0
    sign = {h: (1 if half_score[h] >= 0 else -1) for h in ("h1", "h2")}
    if sign["h1"] == sign["h2"]:                       # impossible: force the weaker half to swap
        weak = min(("h1", "h2"), key=lambda h: abs(half_score[h]))
        sign[weak] = -sign[weak]
    chunks = {str(c) for c in aligned["chunk"].unique()}
    out = {ck: sign[ck[:2]] for ck in chunks if ck[:2] in sign}
    diag = {"n_chunks": len(chunks), "n_clean_votes": len(votes),
            "n_chunk_votes_overruled": sum(1 for ck, (s, _) in votes.items() if s != out[ck]),
            "half_swap_forced": bool(forced)}
    return out, diag


def _bin(values: np.ndarray, edges: tuple[float, float], names: tuple[str, ...]) -> np.ndarray:
    """Bin values into three named buckets by two edges."""
    idx = np.digitize(values, edges)
    return np.asarray(names, dtype=object)[np.clip(idx, 0, 2)]


# ==================================================================================================
# Layer 1 -- ball progression
# ==================================================================================================
def ball_frames(match: Match) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Possession ball samples oriented to the possessing team attacking ``+x``, plus coverage.

    The opponent frame is built the same way (mirrored) so the corpus has a within-corpus baseline:
    without it, "United funnel the ball infield" cannot be told apart from "football funnels the
    ball infield". It is a baseline, NOT a team-vs-team rate comparison.

    Args:
        match: registry match.

    Returns:
        ``(manu_frames, opp_frames, coverage)`` with ``chunk, frame, x_or, y_or, channel, zone``.
    """
    mi = match.teams.index(MANU)
    dirs, diag = manu_dirs(match)
    aligned = match.load_aligned()
    n_aligned = int(aligned.drop_duplicates(["chunk", "frame"]).shape[0])
    rows: list[pd.DataFrame] = []
    orows: list[pd.DataFrame] = []
    n_ball = n_poss = n_manu = 0
    for ck, path in match.ball_chunks():
        if ck not in dirs:
            continue
        ball = pd.read_parquet(path)
        pos = aligned[aligned["chunk"] == ck].dropna(subset=["pitch_x", "pitch_y"])
        if pos.empty or ball.empty:
            continue
        n_ball += len(ball)
        poss = assign_possession(ball, pos, smooth=True)
        if poss.empty:
            continue
        n_poss += len(poss)
        for sgn, holder, sink in ((dirs[ck], mi, rows), (-dirs[ck], 1 - mi, orows)):
            held = poss[poss["team"] == holder]
            if holder == mi:
                n_manu += len(held)
            b = ball[ball["frame"].isin(set(held["frame"].astype(int)))]
            if b.empty:
                continue
            x_or = b["x"].to_numpy(float) if sgn > 0 else PITCH_LEN - b["x"].to_numpy(float)
            y_or = b["y"].to_numpy(float) if sgn > 0 else PITCH_WID - b["y"].to_numpy(float)
            sink.append(pd.DataFrame({
                "chunk": ck, "frame": b["frame"].to_numpy(int), "x_or": x_or, "y_or": y_or,
                "channel": _bin(y_or, CHAN_EDGES, CHANNELS),
                "zone": _bin(x_or, ZONE_EDGES, ZONES)}))
    cols = ["chunk", "frame", "x_or", "y_or", "channel", "zone"]
    frames = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)
    opp = pd.concat(orows, ignore_index=True) if orows else pd.DataFrame(columns=cols)
    cov = {"aligned_frames": n_aligned, "ball_rows": n_ball,
           "post_link_coverage": round(n_ball / n_aligned, 3) if n_aligned else None,
           "possession_fixed": n_poss, "manu_possession_samples": n_manu,
           "manu_possession_share": round(n_manu / n_poss, 3) if n_poss else None,
           "orientation": diag}
    return frames, opp, cov


def progression(frames: pd.DataFrame) -> dict:
    """Channel/zone occupancy, forward metres per channel, and final-third entries per channel.

    Forward metres chain consecutive United-possession ball samples inside one chunk when they are
    within :data:`MAX_GAP_FR` frames and the advance is a physically plausible
    ``0 < dx <= MAX_STEP_M``; the metres are credited to the channel the ball started the step in.
    A final-third entry is a step crossing the 70 m line, credited to the channel it crossed in.

    Args:
        frames: :func:`ball_frames` output.

    Returns:
        Occupancy counts/shares, forward metres and entry counts per channel, and the step count.
    """
    if frames.empty:
        return {}
    occ = frames["channel"].value_counts().reindex(CHANNELS).fillna(0).astype(int)
    zone = frames["zone"].value_counts().reindex(ZONES).fillna(0).astype(int)
    fwd = dict.fromkeys(CHANNELS, 0.0)
    entries = dict.fromkeys(CHANNELS, 0)
    n_steps = 0
    for _ck, g in frames.groupby("chunk"):
        g = g.sort_values("frame")
        fr = g["frame"].to_numpy(int)
        x, y = g["x_or"].to_numpy(float), g["y_or"].to_numpy(float)
        chan = g["channel"].to_numpy(object)
        ok = (np.diff(fr) <= MAX_GAP_FR)
        dx = np.diff(x)
        step = ok & (dx > 0) & (dx <= MAX_STEP_M)
        n_steps += int(step.sum())
        for i in np.flatnonzero(step):
            fwd[chan[i]] += float(dx[i])
        cross = ok & (x[:-1] < FINAL_THIRD_X) & (x[1:] >= FINAL_THIRD_X) & (dx <= MAX_STEP_M)
        for i in np.flatnonzero(cross):
            entries[_bin(y[i + 1:i + 2], CHAN_EDGES, CHANNELS)[0]] += 1
    tot_f = sum(fwd.values()) or 1.0
    tot_e = sum(entries.values()) or 1
    grid = {z: {c: int(((frames["zone"] == z) & (frames["channel"] == c)).sum())
                for c in CHANNELS} for z in ZONES}
    return {
        "n_samples": int(len(frames)),
        "occupancy": {c: int(occ[c]) for c in CHANNELS},
        "occupancy_share": {c: round(float(occ[c] / occ.sum()), 3) for c in CHANNELS},
        "zone_share": {z: round(float(zone[z] / zone.sum()), 3) for z in ZONES},
        "grid": grid,
        "mean_x_or": round(float(frames["x_or"].mean()), 1),
        "forward_m": {c: round(fwd[c], 1) for c in CHANNELS},
        "forward_share": {c: round(fwd[c] / tot_f, 3) for c in CHANNELS},
        "n_steps": n_steps,
        "entries": {c: entries[c] for c in CHANNELS},
        "entry_share": {c: round(entries[c] / tot_e, 3) for c in CHANNELS},
        "n_entries": int(tot_e if sum(entries.values()) else 0),
    }


def pass_channel_check(match: Match, dirs: dict[str, int]) -> dict | None:
    """Validated-PASS origin channel split for one match, or None when no ledger exists.

    The ledger (``outputs/<id>/ledger.parquet``) carries the frozen-threshold PASS events whose
    per-team counts sit within 0.97-1.09x Sofascore. Each Man Utd PASS is placed on the pitch via
    the linked ball position at its grid frame; this is a coverage-limited CROSS-CHECK on the ball
    track's channel split, not an independent measurement.

    Args:
        match: registry match.
        dirs: ``{chunk: +1/-1}`` Man Utd attacking sign from :func:`manu_dirs`.

    Returns:
        ``{n, share: {channel: float}}`` or ``None``.
    """
    path = match.aligned.parent.parent / "ledger.parquet"
    if not path.exists():
        return None
    mi = match.teams.index(MANU)
    led = pd.read_parquet(path)
    led = led[(led["class"] == "PASS") & (led["team"] == mi)]
    if led.empty:
        return None
    counts = dict.fromkeys(CHANNELS, 0)
    for ck, g in led.groupby("chunk"):
        if ck not in dirs:
            continue
        bp = [p for c, p in match.ball_chunks() if c == ck]
        if not bp:
            continue
        ball = pd.read_parquet(bp[0]).set_index("frame")
        sgn = dirs[str(ck)]
        for fi in g["frame_index"].to_numpy(int):
            gf = int(round(fi / 5)) * 5
            near = ball.index[np.argmin(np.abs(ball.index.to_numpy() - gf))] if len(ball) else None
            if near is None or abs(int(near) - gf) > MAX_GAP_FR:
                continue
            yv = float(ball.loc[near, "y"])
            y_or = yv if sgn > 0 else PITCH_WID - yv
            counts[_bin(np.array([y_or]), CHAN_EDGES, CHANNELS)[0]] += 1
    n = sum(counts.values())
    if not n:
        return None
    return {"n": n, "share": {c: round(counts[c] / n, 3) for c in CHANNELS}}


def ball_layer() -> dict:
    """Ball-progression evidence for every Man Utd match + pooled and per-era aggregates."""
    per_match: dict[str, dict] = {}
    opp_recs: list[dict] = []
    for m in manu_matches():
        print(f"[ball] {m.id} ...")
        dirs, _ = manu_dirs(m)
        fr, opp, cov = ball_frames(m)
        rec = {"manager": m.manager, "date": m.date, "opponent": m.teams[1 - m.teams.index(MANU)],
               "venue": "H" if m.home_team == MANU else "A", "coverage": cov, **progression(fr)}
        rec["pass_check"] = pass_channel_check(m, dirs)
        per_match[m.id] = rec
        if len(opp):
            opp_recs.append({"match": m.id, "opponent": rec["opponent"], **progression(opp)})
    return {"per_match": per_match,
            "pooled": _pool_ball(per_match.values()),
            "opponents_pooled": _pool_ball(opp_recs),
            "opponents_per_match": opp_recs,
            "funnel_test": {"manutd": _funnel_test(list(per_match.values())),
                            "opponents": _funnel_test(opp_recs)},
            "side_test": {"manutd": _side_test(list(per_match.values())),
                          "opponents": _side_test(opp_recs)},
            "by_era": {e: _pool_ball([r for r in per_match.values() if r["manager"] == e])
                       for e in ERAS}}


def _zone_channel_share(rec: dict, zone: str, channel: str) -> float:
    """One cell of a match's 3x3 grid, as a share of that zone's ball samples."""
    row = rec["grid"][zone]
    return row[channel] / (sum(row.values()) or 1)


def _sign_p(n_pos: int, n: int) -> float:
    """Two-sided exact sign-test p-value for ``n_pos`` positives out of ``n`` (p0 = 0.5)."""
    from math import comb  # noqa: PLC0415

    k = min(n_pos, n - n_pos)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def _funnel_test(recs: list[dict]) -> dict:
    """Per-match sign test: is the ball LESS central in the final third than in the own third?

    Pooled ball samples are 0.2 s apart and heavily autocorrelated, so a pooled share has no honest
    n. The match is the independent unit, so the claim is tested as a paired per-match sign test.
    """
    recs = [r for r in recs if r.get("grid")]
    diffs = [round(_zone_channel_share(r, "own third", "central")
                   - _zone_channel_share(r, "final third", "central"), 3) for r in recs]
    n_pos = sum(1 for d in diffs if d > 0)
    return {"n": len(diffs), "n_more_central_at_the_back": n_pos,
            "median_drop": round(float(np.median(diffs)), 3),
            "p_sign": _sign_p(n_pos, len(diffs)), "per_match": diffs}


def _side_test(recs: list[dict]) -> dict:
    """Per-match sign test on a left-vs-right tilt of final-third ball occupancy."""
    recs = [r for r in recs if r.get("grid")]
    diffs = [round(_zone_channel_share(r, "final third", "left")
                   - _zone_channel_share(r, "final third", "right"), 3) for r in recs]
    n_pos = sum(1 for d in diffs if d > 0)
    return {"n": len(diffs), "n_left_heavy": n_pos,
            "median_tilt": round(float(np.median(diffs)), 3),
            "p_sign": _sign_p(n_pos, len(diffs)), "per_match": diffs}


def _pool_ball(recs) -> dict:
    """Pool per-match ball profiles by summing counts and metres (not by averaging shares)."""
    recs = [r for r in recs if r.get("occupancy")]
    if not recs:
        return {}
    occ = {c: sum(r["occupancy"][c] for r in recs) for c in CHANNELS}
    fwd = {c: round(sum(r["forward_m"][c] for r in recs), 1) for c in CHANNELS}
    ent = {c: sum(r["entries"][c] for r in recs) for c in CHANNELS}
    zon = {z: sum(r["zone_share"][z] * r["n_samples"] for r in recs) for z in ZONES}
    grid = {z: {c: sum(r["grid"][z][c] for r in recs) for c in CHANNELS} for z in ZONES}
    n_s, n_f, n_e, n_z = sum(occ.values()), sum(fwd.values()), sum(ent.values()), sum(zon.values())
    return {
        "n_matches": len(recs), "n_samples": n_s, "n_entries": n_e,
        "grid": grid,
        "grid_share": {z: {c: round(grid[z][c] / (sum(grid[z].values()) or 1), 3)
                           for c in CHANNELS} for z in ZONES},
        "occupancy_share": {c: round(occ[c] / n_s, 3) for c in CHANNELS},
        "forward_share": {c: round(fwd[c] / (n_f or 1), 3) for c in CHANNELS},
        "forward_m": fwd,
        "entry_share": {c: round(ent[c] / (n_e or 1), 3) for c in CHANNELS},
        "entries": ent,
        "zone_share": {z: round(zon[z] / (n_z or 1), 3) for z in ZONES},
        "lr_ratio_occupancy": round(occ["left"] / occ["right"], 2) if occ["right"] else None,
        "lr_ratio_entries": round(ent["left"] / ent["right"], 2) if ent["right"] else None,
    }


# ==================================================================================================
# Layer 2 -- player territory
# ==================================================================================================
def player_layer() -> dict:
    """Per named Man Utd player, pooled territory per manager era, plus the left/right anchor check.

    Positions come from :func:`fingerprint.player_profiles.oriented_positions` (PRTreID identity
    matches only, 9 of 12) but are RE-ORIENTED with :func:`manu_dirs` so left/right is the
    half-majority-corrected frame the ball layer uses. Volume is never reported.

    Returns:
        ``{"players": [...], "channel_anchor": {...}, "n_matches": int}``.
    """
    recs: dict[tuple[str, str], list[pd.DataFrame]] = {}
    n_matches = 0
    for m, named in pp.prtreid_matches():
        if MANU not in m.teams or not m.manager:
            continue
        n_matches += 1
        print(f"[players] {m.id} ...")
        dirs, _ = manu_dirs(m)
        pos = pp.oriented_positions(m, named)
        # Team membership comes from the PRTreID roster, NOT the anchored team int: the colour
        # anchor mislabels tracks often enough that filtering on team_int leaks opponents in.
        roster = {n: g["team"].mode().iloc[0] for n, g in named.groupby("player_name")}
        manu_players = {p for p, t in roster.items() if t == MANU}
        pos = pos[pos["player"].isin(manu_players)]
        if pos.empty:
            continue
        # pp.oriented_positions already flipped by its own per-chunk resolve; undo where our
        # half-majority sign disagrees, so every match shares one left/right convention.
        raw = _pp_chunk_signs(m, named)
        for ck, g in pos.groupby("chunk"):
            if ck not in dirs or ck not in raw:
                continue
            g = g.copy()
            if raw[ck] != dirs[ck]:
                g["x"], g["y"] = PITCH_LEN - g["x"], PITCH_WID - g["y"]
            for player, gp in g.groupby("player"):
                recs.setdefault((str(player), m.manager), []).append(gp[["x", "y"]])
    rows = []
    for (player, era), parts in recs.items():
        g = pd.concat(parts, ignore_index=True)
        if len(g) < MIN_PLAYER_FRAMES:
            continue
        chan = pd.Series(_bin(g["y"].to_numpy(float), CHAN_EDGES, CHANNELS))
        share = chan.value_counts(normalize=True).reindex(CHANNELS).fillna(0.0)
        zone = pd.Series(_bin(g["x"].to_numpy(float), ZONE_EDGES, ZONES))
        zshare = zone.value_counts(normalize=True).reindex(ZONES).fillna(0.0)
        rows.append({
            "player": player, "era": era, "frames": int(len(g)),
            "x": round(float(g["x"].mean()), 1), "y": round(float(g["y"].mean()), 1),
            "spread_x": round(float(g["x"].std(ddof=0)), 1),
            "spread_y": round(float(g["y"].std(ddof=0)), 1),
            "x_p10": round(float(g["x"].quantile(0.10)), 1),
            "x_p90": round(float(g["x"].quantile(0.90)), 1),
            **{f"ch_{c}": round(float(share[c]), 3) for c in CHANNELS},
            **{f"z_{z.split()[0]}": round(float(zshare[z]), 3) for z in ZONES},
        })
    players = sorted(rows, key=lambda r: (-r["frames"],))
    return {"players": players, "n_matches": n_matches,
            "channel_anchor": _channel_anchor(players)}


def _pp_chunk_signs(match: Match, named: pd.DataFrame) -> dict[str, int]:
    """Man Utd sign that :func:`player_profiles.oriented_positions` used, per chunk (keeper-based)."""
    mi = match.teams.index(MANU)
    df = match.load_aligned()
    df = df[df["role"].isin(["player", "goalkeeper"])].dropna(subset=["pitch_x", "pitch_y"])
    out: dict[str, int] = {}
    for ck, g in df.groupby("chunk"):
        adir = resolve_attack_directions(g)
        if adir.get(mi) is not None:
            out[str(ck)] = int(adir[mi])
    return out


def _channel_anchor(players: list[dict]) -> dict:
    """Anchor the left/right channel labels on known left- and right-sided United players.

    The lateral sign of the pitch model is a convention, not a measurement, so it is checked rather
    than assumed: pooled mean ``y`` of known left-sided players must sit BELOW the pooled mean of
    known right-sided players for ``y < 22.67 m`` to be called the left channel.

    Args:
        players: :func:`player_layer` rows.

    Returns:
        ``{left_mean_y, right_mean_y, left_names, right_names, passes}``.
    """
    def pooled(names: tuple[str, ...]) -> tuple[float | None, list[str]]:
        rs = [r for r in players if r["player"] in names]
        if not rs:
            return None, []
        w = sum(r["frames"] for r in rs)
        return round(sum(r["y"] * r["frames"] for r in rs) / w, 1), sorted({r["player"] for r in rs})

    ly, ln = pooled(LEFT_ANCHORS)
    ry, rn = pooled(RIGHT_ANCHORS)
    checked = [(r, r["player"] in LEFT_ANCHORS) for r in players
               if r["frames"] >= 100 and r["player"] in LEFT_ANCHORS + RIGHT_ANCHORS]
    ok = sum(1 for r, is_left in checked
             if (r["ch_left"] > r["ch_right"]) == is_left)
    return {"left_mean_y": ly, "right_mean_y": ry, "left_names": ln, "right_names": rn,
            "passes": bool(ly is not None and ry is not None and ly < ry),
            "n_side_checked": len(checked), "n_side_correct": ok}


# ==================================================================================================
# Layer 3 -- block shape and how much it moves
# ==================================================================================================
def block_layer() -> dict:
    """Per-match block geometry + the variance quantities that make "rigid" a testable word."""
    per_match: dict[str, dict] = {}
    opp: list[dict] = []
    for m in manu_matches():
        print(f"[block] {m.id} ...")
        frames = bh.block_frames(m)
        spells = bh.block_spells(frames)
        mi = m.teams.index(MANU)
        tf = frames[frames["def_team"] == mi]
        usable = tf[tf["usable"]]
        ts = spells[spells["def_team"] == mi]
        rec = {
            "manager": m.manager, "date": m.date,
            "opponent": m.teams[1 - mi], "venue": "H" if m.home_team == MANU else "A",
            "oop_frames": int(len(tf)), "usable_frames": int(len(usable)),
            "coverage": round(len(usable) / len(tf), 3) if len(tf) else None,
            "mean_line_m": round(float(usable["line_m"].mean()), 1) if len(usable) else None,
            "mean_vspread_m": round(float(usable["vspread_m"].mean()), 2)
            if usable["vspread_m"].notna().any() else None,
            "n_spells": int(len(ts)),
            "spell_line_sd_m": round(float(ts["median_line_m"].std(ddof=1)), 2)
            if len(ts) > 1 else None,
            "spell_line_iqr_m": round(float(ts["median_line_m"].quantile(0.75)
                                            - ts["median_line_m"].quantile(0.25)), 2)
            if len(ts) > 1 else None,
        }
        rec["wp_bands"] = _wp_block(m, frames, mi)
        per_match[m.id] = rec
        ou = frames[(frames["def_team"] == 1 - mi) & frames["usable"]]
        if len(ou):
            opp.append({"match": m.id, "team": m.teams[1 - mi],
                        "mean_line_m": round(float(ou["line_m"].mean()), 1),
                        "mean_vspread_m": round(float(ou["vspread_m"].mean()), 2)
                        if ou["vspread_m"].notna().any() else None})
    out = {"per_match": per_match, "opponents": opp,
           "opponent_spread": _spread([r["mean_vspread_m"] for r in opp]),
           "opponent_line_spread": _spread([r["mean_line_m"] for r in opp]),
           **_block_variance(per_match)}
    out["variance_tests"] = {
        "line": _var_test(out["line"], out["opponent_line_spread"]),
        "vspread": _var_test(out["vspread"], out["opponent_spread"])}
    return out


def _var_test(a: dict, b: dict) -> dict:
    """Two-sided F-test that United's between-match variance differs from the opponent set's.

    The opponent set pools six different clubs, so it carries between-team variance United's does
    not -- which makes it a conservative comparator, not a like-for-like one. Reported so the
    "more/less variable than the opposition" reading is never asserted without its p-value.
    """
    from scipy import stats  # noqa: PLC0415

    f = (a["sd"] / b["sd"]) ** 2
    d1, d2 = a["n"] - 1, b["n"] - 1
    p = 2 * min(float(stats.f.sf(f, d1, d2)), float(stats.f.cdf(f, d1, d2)))
    return {"F": round(f, 2), "df": [d1, d2], "p": round(min(1.0, p), 3)}


def _spread(vals: list) -> dict:
    """Mean / SD / CV / range of a list of per-match values (Nones dropped)."""
    v = np.asarray([x for x in vals if x is not None], float)
    return {"mean": round(float(v.mean()), 2), "sd": round(float(v.std(ddof=1)), 2),
            "cv": round(float(v.std(ddof=1) / v.mean()), 3),
            "min": round(float(v.min()), 2), "max": round(float(v.max()), 2), "n": int(v.size)}


def _wp_block(match: Match, frames: pd.DataFrame, mi: int) -> dict | None:
    """Man Utd block line / vertical spread split by win-probability band, or None with no WP series."""
    wp_path = WP_DIR / f"{match.id}.parquet"
    if not wp_path.exists() or frames.empty:
        return None
    wp = pd.read_parquet(wp_path)
    by_min = dict(zip(wp["minute"].to_numpy(int), wp["wp_win"].to_numpy(float)))
    offs = ss.chunk_offsets(match.id)
    f = frames[(frames["def_team"] == mi) & frames["usable"]].copy()
    if f.empty:
        return None
    half = f["chunk"].str[:2]
    t_s = f["chunk"].map(offs).fillna(0.0) + f["frame"] / GRID_FPS
    f["band"] = [band_of(by_min.get(match_minute(h, t), 0.5)) for h, t in zip(half, t_s)]
    out = {}
    for band, g in f.groupby("band"):
        out[str(band)] = {"frames": int(len(g)),
                          "line_m": round(float(g["line_m"].mean()), 1),
                          "vspread_m": round(float(g["vspread_m"].mean()), 2)
                          if g["vspread_m"].notna().any() else None}
    return out


def _block_variance(per_match: dict[str, dict]) -> dict:
    """Between-match SD/CV of line and spread, leg-to-leg swing per opponent, pooled WP bands."""
    df = pd.DataFrame(per_match.values())
    out: dict = {"n_matches": int(len(df))}
    for col, key in (("mean_line_m", "line"), ("mean_vspread_m", "vspread")):
        v = df[col].dropna().to_numpy(float)
        out[key] = {"mean": round(float(v.mean()), 2), "sd": round(float(v.std(ddof=1)), 2),
                    "cv": round(float(v.std(ddof=1) / v.mean()), 3),
                    "min": round(float(v.min()), 2), "max": round(float(v.max()), 2), "n": int(v.size)}
    out["within_match_spell_line_sd_m"] = round(
        float(df["spell_line_sd_m"].dropna().mean()), 2)
    pairs = []
    for opp, g in df.groupby("opponent"):
        if len(g) == 2:
            pairs.append({"opponent": str(opp),
                          "line_gap_m": round(abs(float(np.diff(g["mean_line_m"].to_numpy())[0])), 1),
                          "vspread_gap_m": round(
                              abs(float(np.diff(g["mean_vspread_m"].to_numpy())[0])), 2)})
    out["pairs"] = pairs
    out["pair_mean_line_gap_m"] = round(float(np.mean([p["line_gap_m"] for p in pairs])), 2)
    out["pair_mean_vspread_gap_m"] = round(float(np.mean([p["vspread_gap_m"] for p in pairs])), 2)
    bands: dict[str, dict[str, list]] = {}
    for rec in per_match.values():
        for band, v in (rec.get("wp_bands") or {}).items():
            b = bands.setdefault(band, {"frames": 0, "line": [], "vspread": [], "w": []})
            b["frames"] += v["frames"]
            b["line"].append(v["line_m"] * v["frames"])
            b["w"].append(v["frames"])
            if v["vspread_m"] is not None:
                b["vspread"].append(v["vspread_m"] * v["frames"])
    out["wp_pooled"] = {
        b: {"frames": v["frames"],
            "line_m": round(sum(v["line"]) / v["frames"], 1),
            "vspread_m": round(sum(v["vspread"]) / v["frames"], 2) if v["vspread"] else None}
        for b, v in bands.items()}
    out["n_wp_matches"] = sum(1 for r in per_match.values() if r.get("wp_bands"))
    return out


# ==================================================================================================
# Layer 4 -- manager era
# ==================================================================================================
def era_layer(ball: dict, block: dict, players: dict) -> dict:
    """Era aggregates for block, ball channel profile, and player advance, with the confound stated."""
    out: dict = {"confound": "venue and manager flip together in all 6 pairs; n=6 per era"}
    for era in ERAS:
        bm = [r for r in block["per_match"].values() if r["manager"] == era]
        line = [r["mean_line_m"] for r in bm if r["mean_line_m"] is not None]
        vsp = [r["mean_vspread_m"] for r in bm if r["mean_vspread_m"] is not None]
        out[era] = {
            "n_matches": len(bm),
            "block_line_m": round(float(np.mean(line)), 1),
            "block_line_sd_m": round(float(np.std(line, ddof=1)), 2),
            "block_vspread_m": round(float(np.mean(vsp)), 2),
            "block_vspread_sd_m": round(float(np.std(vsp, ddof=1)), 2),
            "ball": ball["by_era"][era],
        }
    both = {}
    for r in players["players"]:
        both.setdefault(r["player"], {})[r["era"]] = r
    dx = {p: round(v["amorim"]["x"] - v["ten_hag"]["x"], 1)
          for p, v in both.items() if len(v) == 2}
    out["player_dx"] = dict(sorted(dx.items(), key=lambda kv: -kv[1]))
    out["n_both_era"] = len(dx)
    out["n_higher_under_amorim"] = sum(1 for v in dx.values() if v > 0)
    return out


# ==================================================================================================
# Evidence assembly + profile rendering
# ==================================================================================================
def build_evidence() -> dict:
    """Run all four layers and cache the result to :data:`EVIDENCE_PATH`."""
    ball = ball_layer()
    block = block_layer()
    players = player_layer()
    ev = {"ball": ball, "block": block, "players": players,
          "era": era_layer(ball, block, players)}
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(ev, indent=1), encoding="utf-8")
    return ev


def load_evidence() -> dict:
    """Cached evidence, or a fresh build when the cache is absent."""
    if EVIDENCE_PATH.exists():
        return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    return build_evidence()


def pass_agreement(ball: dict) -> dict:
    """Mean absolute gap between the ball-track channel split and the validated-PASS channel split."""
    gaps, n_pass, ids = [], 0, []
    for mid, rec in ball["per_match"].items():
        pc = rec.get("pass_check")
        if not pc:
            continue
        ids.append(mid)
        n_pass += pc["n"]
        gaps += [abs(pc["share"][c] - rec["occupancy_share"][c]) for c in CHANNELS]
    return {"n_matches": len(ids), "matches": ids, "n_passes": n_pass,
            "mean_abs_gap": round(float(np.mean(gaps)), 3) if gaps else None,
            "max_abs_gap": round(float(np.max(gaps)), 3) if gaps else None}


def _pct(x: float) -> str:
    """Share as a whole-number percentage string."""
    return f"{x * 100:.1f}%"


def write_profile(ev: dict) -> Path:
    """Render ``results/MANUTD_IDENTITY_PROFILE.md`` from the cached evidence."""
    from tools.manutd_identity_text import render  # noqa: PLC0415

    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(render(ev), encoding="utf-8")
    return PROFILE_PATH


def main() -> None:
    """CLI: compute the evidence (or reuse the cache) and write the identity profile."""
    ap = argparse.ArgumentParser(description="Man Utd 2024-25 identity profile evidence + render.")
    ap.add_argument("--cached", action="store_true",
                    help="reuse outputs/manutd_identity/evidence.json")
    args = ap.parse_args()
    ev = load_evidence() if args.cached else build_evidence()
    path = write_profile(ev)
    print(f"[identity] wrote {path}")


def demo() -> None:
    """Self-check on the pure seams: binning, half-swap enforcement, and pooling arithmetic."""
    assert list(_bin(np.array([1.0, 34.0, 67.0]), CHAN_EDGES, CHANNELS)) == list(CHANNELS)
    assert list(_bin(np.array([1.0, 52.0, 104.0]), ZONE_EDGES, ZONES)) == list(ZONES)
    pooled = _pool_ball([
        {"occupancy": {"left": 10, "central": 20, "right": 5}, "n_samples": 35,
         "forward_m": {"left": 1.0, "central": 2.0, "right": 1.0},
         "entries": {"left": 2, "central": 1, "right": 1},
         "grid": {z: dict.fromkeys(CHANNELS, 1) for z in ZONES},
         "zone_share": {z: 1 / 3 for z in ZONES}},
    ])
    assert pooled["occupancy_share"] == {"left": 0.286, "central": 0.571, "right": 0.143}, pooled
    assert pooled["lr_ratio_occupancy"] == 2.0
    assert pooled["entry_share"]["left"] == 0.5
    assert _sign_p(11, 12) == 0.0063 and _sign_p(6, 12) == 1.0
    assert _var_test({"sd": 4.25, "n": 12}, {"sd": 3.09, "n": 12})["F"] == 1.89
    m = get("manutd_brighton")
    dirs, diag = manu_dirs(m)
    h1 = {s for ck, s in dirs.items() if ck.startswith("h1")}
    h2 = {s for ck, s in dirs.items() if ck.startswith("h2")}
    assert len(h1) == 1 and len(h2) == 1 and h1 != h2, (h1, h2)
    print(f"demo ok: orientation forced to one sign per half, {diag}")


if __name__ == "__main__":
    main()
