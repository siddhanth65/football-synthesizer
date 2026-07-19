"""Tracking-native style fingerprint v1 (Plan B-4): OT embedding, phase profile, counter-press.

Three deterministic, ball-gap-tolerant primitives that turn per-frame player positions (plus the
proximity possession proxy) into a team-match style signature. All pure numpy/pandas/scipy/sklearn;
no learned model, no pass stream.

1. **OT sliced-Wasserstein embedding** (Baouan et al. 2025, arXiv:2501.10299). Each accepted
   tactical frame is the outfield players as a uniform discrete measure; we embed it by projecting on
   K fixed directions, taking Q quantiles per direction and concatenating -- an L2 vector whose
   squared distance approximates the sliced-W2 between two frames (:func:`sw_embed`). Attack direction
   is normalised per chunk (the team always attacks +x) so both halves are comparable; an optional
   centred variant subtracts the centroid to isolate *shape* from field position. Per team-match we
   k-means the frame embeddings into prototypes; the distance between two team-matches is the
   Wasserstein distance between their prototype distributions (:func:`w2_distance`).

2. **Phase segmentation v1** (rule-based, mirrors the verified DEC 4-phase design). From the Viterbi
   possession track (:func:`generator.ball.assign_possession`, ``smooth=True``) each frame is labelled
   per team ``{in_poss, out_poss, trans_pos, trans_neg}`` -- ``trans_*`` within 5 s of a possession
   flip (:func:`phase_by_frame`). Each phase gets block height, width, compactness and centroid depth.

3. **Counter-press primitive** (StatsBomb defs): a pressure is a defender within 4.57 m (5 yd) of the
   ball carrier; a counter-press is a pressure within 5 s of losing the ball, counted only for
   possessions lost **outside the losing team's defending third** (:func:`counterpress`). We report the
   counter-press fraction, the 5 s regain rate, and the evaluable share (possession is only defined
   where the ball links, so most of the match is unobserved -- that share is stated, not hidden).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance_nd

from core.pitch import PITCH_LEN, PITCH_WID
from fingerprint.structural_metrics import (
    compute_metrics_table,
    resolve_attack_directions,
)
from fingerprint.transitions import detect_turnovers
from generator.ball import assign_possession

# --- 1. OT sliced-Wasserstein embedding -----------------------------------------------------------
SW_K = 24            # projection directions (evenly spaced in [0, pi))
SW_Q = 12            # quantile levels per direction
N_PROTOTYPES = 64    # k-means prototypes per team-match (in the 50-100 range the plan asks for)
MIN_OUTFIELD = 4     # a frame needs at least this many outfield points to be a measure


def sw_directions(k: int = SW_K) -> np.ndarray:
    """``k`` unit projection directions evenly spaced over ``[0, pi)`` (shape ``(k, 2)``)."""
    ang = np.linspace(0.0, np.pi, k, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def sw_embed(points: np.ndarray, directions: np.ndarray, *, n_quantiles: int = SW_Q,
             center: bool = False) -> np.ndarray:
    """Sliced-Wasserstein embedding of a point cloud (uniform discrete measure).

    Projects the points onto each direction, takes ``n_quantiles`` quantiles of each projection and
    concatenates, scaled so that ``||phi(A) - phi(B)||^2`` approximates the sliced-W2^2 between the two
    measures. Sorting the projections makes the embedding invariant to the point ordering; centring
    makes it invariant to a global translation (isolating shape from field position).

    Args:
        points: ``(n, 2)`` player positions in pitch metres (already attack-oriented if desired).
        directions: ``(k, 2)`` unit projection directions (from :func:`sw_directions`).
        n_quantiles: quantile levels per direction.
        center: subtract the centroid before projecting (translation-invariant shape embedding).

    Returns:
        A ``k * n_quantiles`` embedding vector.
    """
    p = np.asarray(points, float)
    if center:
        p = p - p.mean(axis=0)
    proj = p @ directions.T                                   # (n, k)
    levels = np.linspace(0.0, 1.0, n_quantiles)
    q = np.quantile(proj, levels, axis=0)                     # (n_quantiles, k)
    scale = 1.0 / np.sqrt(directions.shape[0] * n_quantiles)  # -> distances in metres
    return (q * scale).ravel(order="F")


def _orient(px: np.ndarray, py: np.ndarray, attack_dir: int) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a team's points 180 deg when it attacks -x, so every frame attacks +x (chirality kept)."""
    if attack_dir > 0:
        return px, py
    return PITCH_LEN - px, PITCH_WID - py


def match_embeddings(match, *, k: int = SW_K, n_quantiles: int = SW_Q,
                     center: bool = False) -> dict[int, np.ndarray]:
    """Per-team stack of frame embeddings for one match (attack-normalised per chunk).

    Iterates the aligned parquet chunk by chunk, resolves each chunk's attack directions from the
    keepers, and embeds every ``(frame, team)`` outfield cloud with >= :data:`MIN_OUTFIELD` players.
    Frames whose team has no resolved direction are skipped (they cannot be attack-normalised).

    Args:
        match: a :class:`core.registry.Match`.
        k: number of SW projection directions.
        n_quantiles: quantile levels per direction.
        center: use the centred (shape-only) embedding variant.

    Returns:
        ``{team_index: (n_frames, k * n_quantiles) embeddings}``.
    """
    df = match.load_aligned()
    players = df[df["role"].isin(["player", "goalkeeper"])].dropna(subset=["pitch_x", "pitch_y"])
    dirs = sw_directions(k)
    out: dict[int, list[np.ndarray]] = {}
    chunks = players.groupby("chunk") if "chunk" in players.columns else [("_", players)]
    for _ck, g in chunks:
        adir = resolve_attack_directions(g)
        for (_fr, team), fg in g.groupby(["frame", "team"]):
            t = int(team)
            if t < 0 or len(fg) < MIN_OUTFIELD or adir.get(t) is None:
                continue
            px, py = _orient(fg["pitch_x"].to_numpy(), fg["pitch_y"].to_numpy(), adir[t])
            out.setdefault(t, []).append(
                sw_embed(np.column_stack([px, py]), dirs, n_quantiles=n_quantiles, center=center))
    return {t: np.array(v) for t, v in out.items() if v}


def prototypes(emb: np.ndarray, *, n_clusters: int = N_PROTOTYPES,
               seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """K-means prototypes of a frame-embedding stack: ``(centers, weights)`` (weights = cluster share)."""
    from sklearn.cluster import KMeans  # noqa: PLC0415

    n = min(n_clusters, len(emb))
    km = KMeans(n_clusters=n, n_init=4, random_state=seed).fit(emb)
    counts = np.bincount(km.labels_, minlength=n).astype(float)
    return km.cluster_centers_, counts / counts.sum()


def w2_distance(centers_a: np.ndarray, w_a: np.ndarray, centers_b: np.ndarray,
                w_b: np.ndarray) -> float:
    """Wasserstein (EMD, Euclidean ground metric) between two weighted prototype distributions."""
    return float(wasserstein_distance_nd(centers_a, centers_b, w_a, w_b))


def distance_matrix(sides: dict[str, np.ndarray], *, n_clusters: int = N_PROTOTYPES,
                    seed: int = 0) -> pd.DataFrame:
    """Symmetric side-vs-side embedding-distance matrix (prototype Wasserstein), labelled by side."""
    names = list(sides)
    protos = {nm: prototypes(sides[nm], n_clusters=n_clusters, seed=seed) for nm in names}
    d = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            d[i, j] = d[j, i] = w2_distance(*protos[a], *protos[b])
    return pd.DataFrame(d, index=names, columns=names)


# --- 2. Phase segmentation v1 ---------------------------------------------------------------------
PHASES = ("in_poss", "out_poss", "trans_pos", "trans_neg")
TRANSITION_S = 5.0
PHASE_METRIC_COLS = ("def_line_height", "width", "compactness", "buildup_height")


def phase_by_frame(possession: pd.DataFrame, *, teams: list[int], fps: float,
                   transition_s: float = TRANSITION_S) -> pd.DataFrame:
    """Label each possession frame per team ``{in_poss, out_poss, trans_pos, trans_neg}``.

    ``trans_pos`` = the team that just won the ball, within ``transition_s`` of the flip; ``trans_neg``
    = the team that just lost it, same window. Outside a transition window the holder is ``in_poss``
    and the other team ``out_poss``. Built from the (Viterbi-smoothed) possession sequence only.

    Args:
        possession: ``frame, carrier, team`` (one chunk), sorted or not.
        teams: the (two) team indices to label.
        fps: native frames per second, for the seconds->frames window.
        transition_s: transition half-life window in seconds after a flip.

    Returns:
        ``frame, team, phase`` -- two rows (both teams) per possession frame.
    """
    cols = ["frame", "team", "phase"]
    if possession.empty:
        return pd.DataFrame(columns=cols)
    p = possession.sort_values("frame").reset_index(drop=True)
    fr = p["frame"].to_numpy(int)
    hold = p["team"].to_numpy(int)
    win = round(transition_s * fps)
    # For each possession sample, distance (in frames) since the last flip, and who won/lost it.
    since = np.full(len(p), np.iinfo(np.int64).max)
    won = np.full(len(p), -1)
    lost = np.full(len(p), -1)
    flip_frame, flip_won, flip_lost = None, -1, -1
    for i in range(len(p)):
        if i > 0 and hold[i] != hold[i - 1]:
            flip_frame, flip_won, flip_lost = fr[i], hold[i], hold[i - 1]
        if flip_frame is not None:
            since[i] = fr[i] - flip_frame
            won[i], lost[i] = flip_won, flip_lost
    rows = []
    for i in range(len(p)):
        in_trans = since[i] <= win
        for t in teams:
            if in_trans and t == won[i]:
                phase = "trans_pos"
            elif in_trans and t == lost[i]:
                phase = "trans_neg"
            elif t == hold[i]:
                phase = "in_poss"
            else:
                phase = "out_poss"
            rows.append({"frame": int(fr[i]), "team": int(t), "phase": phase})
    return pd.DataFrame(rows, columns=cols)


def phase_frame_table(match, *, transition_s: float = TRANSITION_S) -> pd.DataFrame:
    """Per-frame phase label + shape metrics for one match, keyed by chunk (segmentation seam).

    Same computation as :func:`phase_profile` but returns the *un-aggregated* per-frame rows
    (``chunk, frame, team, phase`` + :data:`PHASE_METRIC_COLS`) so downstream code (e.g.
    :mod:`fingerprint.score_state`) can regroup them by score state instead of team alone.
    """
    df = match.load_aligned()
    players = df[df["role"].isin(["player", "goalkeeper"])]
    parts = []
    for ck, path in match.ball_chunks():
        g = players[players["chunk"] == ck]
        if g.empty:
            continue
        adir = resolve_attack_directions(g)
        teams = [t for t in adir if int(t) >= 0]
        if len(teams) < 2:
            continue
        ball = pd.read_parquet(path)
        poss = assign_possession(ball, g, smooth=True)
        phases = phase_by_frame(poss, teams=teams, fps=match.chunk_fps(ck),
                                transition_s=transition_s)
        if phases.empty:
            continue
        mt = compute_metrics_table(g, attack_dirs=adir)
        part = mt.merge(phases, on=["frame", "team"], how="inner")
        part["chunk"] = ck
        parts.append(part)
    cols = ["chunk", "frame", "team", "phase", *PHASE_METRIC_COLS]
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)[cols]


def phase_profile(match, *, transition_s: float = TRANSITION_S) -> pd.DataFrame:
    """Per ``(team, phase)`` mean shape metrics for one match (block height, width, compactness, depth).

    Joins the rule-based phase labels onto the per-frame structural metrics, per chunk, and averages.
    Returns ``team, phase, frames`` + :data:`PHASE_METRIC_COLS`.
    """
    allm = phase_frame_table(match, transition_s=transition_s)
    if allm.empty:
        return pd.DataFrame(columns=["team", "phase", "frames", *PHASE_METRIC_COLS])
    agg = allm.groupby(["team", "phase"])[list(PHASE_METRIC_COLS)].mean()
    agg.insert(0, "frames", allm.groupby(["team", "phase"]).size())
    return agg.reset_index()


# --- 3. Counter-press primitive (StatsBomb defs) --------------------------------------------------
PRESS_RADIUS_M = 4.57       # 5 yards -- StatsBomb pressure radius
DEF_THIRD_X = PITCH_LEN / 3  # a loss with attacking-x <= this (own third) is excluded


def turnover_press_table(match, *, transition_s: float = TRANSITION_S,
                         press_radius_m: float = PRESS_RADIUS_M
                         ) -> tuple[pd.DataFrame, int, int]:
    """Per outside-third loss: ``chunk, frame, team, pressed, regained`` + poss/turnover totals.

    The evaluable atoms behind :func:`counterpress`, kept per-event so they can be sliced by score
    state (:mod:`fingerprint.score_state`). ``team`` is the losing team; ``pressed`` is a
    counter-press pressure within ``transition_s``; ``regained`` is possession won back in the same
    window. Only losses outside the losing team's defending third are rows. Returns the table plus the
    two ball-gap honesty counters (total possession samples, total turnovers) as match-level totals.
    """
    df = match.load_aligned()
    players = df[df["role"].isin(["player", "goalkeeper"])].dropna(subset=["pitch_x", "pitch_y"])
    rows: list[dict] = []
    total_poss = 0
    total_turn = 0
    for ck, path in match.ball_chunks():
        g = players[players["chunk"] == ck]
        if g.empty:
            continue
        adir = resolve_attack_directions(g)
        ball = pd.read_parquet(path)
        poss = assign_possession(ball, g, smooth=True)
        total_poss += len(poss)
        if poss.empty:
            continue
        turnovers = detect_turnovers(poss, ball)
        total_turn += len(turnovers)
        fps = match.chunk_fps(ck)
        win = round(transition_s * fps)
        ballxy = {int(r.frame): (float(r.x), float(r.y))
                  for r in ball.sort_values("frame").itertuples(index=False)}
        ball_frames = np.array(sorted(ballxy))
        pbf = {fr: fg for fr, fg in g.groupby("frame")}
        poss_frames = np.sort(poss["frame"].to_numpy())
        poss_team = poss.set_index("frame")["team"].to_dict()
        for tv in turnovers.itertuples(index=False):
            lost, f = int(tv.lost_team), int(tv.frame)
            d = adir.get(lost)
            if d is None:
                continue
            ac = tv.x if d > 0 else (PITCH_LEN - tv.x)  # loss location in loser's attacking-x
            if ac <= DEF_THIRD_X:                       # lost in own third -> excluded
                continue
            wf = ball_frames[(ball_frames > f) & (ball_frames <= f + win)]
            pressed = False
            for w in wf:
                bx, by = ballxy[int(w)]
                fg = pbf.get(int(w))
                if fg is None:
                    continue
                dteam = fg[fg["team"] == lost]
                if len(dteam) and float(
                        np.hypot(dteam["pitch_x"] - bx, dteam["pitch_y"] - by).min()) <= press_radius_m:
                    pressed = True
                    break
            # regain within the window: losing team is the possessing team on any sample in (f, f+win]
            pf = poss_frames[(poss_frames > f) & (poss_frames <= f + win)]
            regained = any(poss_team.get(int(x)) == lost for x in pf)
            rows.append({"chunk": ck, "frame": f, "team": lost,
                         "pressed": bool(pressed), "regained": bool(regained)})
    losses = pd.DataFrame(rows, columns=["chunk", "frame", "team", "pressed", "regained"])
    return losses, total_poss, total_turn


def counterpress(match, *, transition_s: float = TRANSITION_S,
                 press_radius_m: float = PRESS_RADIUS_M) -> pd.DataFrame:
    """Per team-match counter-press fraction + 5 s regain rate for balls lost outside the own third.

    A turnover is a possession flip located by the ball. For the losing team, if the loss happened
    outside its defending third we ask: within ``transition_s`` after the loss, does a losing-team
    player get within ``press_radius_m`` of the ball (a counter-press pressure), and is possession
    regained? Only frames with a linked ball / known carrier are evaluable, so we also report how many
    turnovers were evaluable and the raw possession-frame count behind them.

    Returns:
        ``team, losses_outside_third, counterpress_frac, regain_5s_frac, poss_frames, turnovers``
        (one row per team). ``poss_frames`` / ``turnovers`` are the ball-gap honesty counters.
    """
    losses, total_poss, total_turn = turnover_press_table(
        match, transition_s=transition_s, press_radius_m=press_radius_m)
    rows = []
    for t in sorted(losses["team"].unique()):
        sub = losses[losses["team"] == t]
        n = len(sub)
        rows.append({
            "team": int(t), "losses_outside_third": n,
            "counterpress_frac": float(sub["pressed"].mean()) if n else float("nan"),
            "regain_5s_frac": float(sub["regained"].mean()) if n else float("nan"),
            "poss_frames": total_poss, "turnovers": total_turn})
    return pd.DataFrame(rows, columns=["team", "losses_outside_third", "counterpress_frac",
                                       "regain_5s_frac", "poss_frames", "turnovers"])


def _demo() -> None:
    """Self-check on the pure seams: SW invariances, phase rules, pressure filter."""
    rng = np.random.default_rng(0)
    dirs = sw_directions(8)
    pts = rng.uniform(0, 60, size=(10, 2))
    # permutation invariance
    e0 = sw_embed(pts, dirs)
    e1 = sw_embed(pts[rng.permutation(10)], dirs)
    assert np.allclose(e0, e1), "SW embedding not permutation-invariant"
    # translation invariance after centering
    ec0 = sw_embed(pts, dirs, center=True)
    ec1 = sw_embed(pts + np.array([12.0, -7.0]), dirs, center=True)
    assert np.allclose(ec0, ec1), "centred SW embedding not translation-invariant"
    # uncentred moves under translation
    assert not np.allclose(e0, sw_embed(pts + 5.0, dirs)), "uncentred embedding should move"
    # orient: attacking -x is a 180 deg rotation
    ox, oy = _orient(np.array([10.0]), np.array([20.0]), -1)
    assert abs(ox[0] - (PITCH_LEN - 10.0)) < 1e-9 and abs(oy[0] - (PITCH_WID - 20.0)) < 1e-9
    # phase rules: a single flip makes trans_pos / trans_neg inside the window
    poss = pd.DataFrame({"frame": [0, 10, 20, 200], "carrier": [1, 1, 2, 2],
                         "team": [0, 0, 1, 1]})
    ph = phase_by_frame(poss, teams=[0, 1], fps=25.0, transition_s=5.0)
    at20 = ph[ph["frame"] == 20].set_index("team")["phase"].to_dict()
    assert at20[1] == "trans_pos" and at20[0] == "trans_neg", at20
    at200 = ph[ph["frame"] == 200].set_index("team")["phase"].to_dict()
    assert at200[1] == "in_poss" and at200[0] == "out_poss", at200
    print("style_fingerprint self-check OK")


if __name__ == "__main__":
    _demo()
