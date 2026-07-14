"""C4 'Instinct': value-optimal off-ball runs via GAT counterfactuals vs actual runs (decision quality).

For a ball-carrier moment, an off-ball attacker's **optimal run** is the nearby target that most raises
that player's value when moved there. The value is a ``player_value(frame, idx) -> float`` callable, so
the same engine works with either signal:

* **receiver value** (recommended) -- the GAT's per-player next-receiver probability ("run to where you
  are the best passing option"); player-specific, so it actually responds to off-ball runs.
* **graph xT** -- team Dynamic-xT (ignores ``idx``); ball-dominated, so off-ball runs barely move it.

Two outputs: **off-ball value potential** (the optimal value-gain) and **decision quality** (cosine
between the optimal run and the player's *actual* run, matched in the next freeze-frame). Pure geometry
here; the GAT enters only through the ``player_value`` callable, so this module imports no torch.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from generator.contract import PITCH_LENGTH, PITCH_WIDTH, FreezeFrame, PlayerNode

RUN_RADIUS = 10.0  # candidate run radius on the 120x80 frame (~1.5 s at a sprint, in StatsBomb units)


def _unit(dx: float, dy: float) -> tuple[float, float]:
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else (0.0, 0.0)


def candidate_targets(x: float, y: float, *, radius: float = RUN_RADIUS, n_dirs: int = 8,
                      rings: tuple[float, ...] = (0.5, 1.0)) -> list[tuple[float, float]]:
    """Radial candidate run targets around ``(x, y)``, clipped to the pitch."""
    out = []
    for ring in rings:
        for k in range(n_dirs):
            a = 2 * math.pi * k / n_dirs
            out.append((float(np.clip(x + ring * radius * math.cos(a), 0.0, PITCH_LENGTH)),
                        float(np.clip(y + ring * radius * math.sin(a), 0.0, PITCH_WIDTH))))
    return out


def move_player(frame: FreezeFrame, idx: int, x: float, y: float) -> FreezeFrame:
    """A copy of ``frame`` with player ``idx`` relocated to ``(x, y)`` (everything else unchanged)."""
    ps = list(frame.players)
    p = ps[idx]
    ps[idx] = PlayerNode(x=float(x), y=float(y), is_teammate=p.is_teammate, is_actor=p.is_actor,
                         is_keeper=p.is_keeper, vx=p.vx, vy=p.vy, orientation=p.orientation,
                         observed=p.observed)
    return FreezeFrame(players=ps, substrate=frame.substrate, ball=frame.ball,
                       play_pattern=frame.play_pattern, from_counter=frame.from_counter,
                       trigger_time_s=frame.trigger_time_s)


def offball_attacker_indices(frame: FreezeFrame) -> list[int]:
    """Indices of attacking-team players who are not the ball-carrier or a keeper."""
    return [i for i, p in enumerate(frame.players)
            if p.is_teammate and not p.is_actor and not p.is_keeper]


def optimal_run(player_value, frame: FreezeFrame, idx: int, *, radius: float = RUN_RADIUS) -> dict:
    """Best counterfactual run for off-ball attacker ``idx`` by ``player_value(frame, idx)``.

    ``player_value`` returns the value of player ``idx`` in a (possibly counterfactual) frame -- e.g.
    its receiver probability. The optimal run is the candidate maximising it vs staying put.
    """
    base = float(player_value(frame, idx))
    px, py = frame.players[idx].x, frame.players[idx].y
    best_xy, best_v = (px, py), base
    for cx, cy in candidate_targets(px, py, radius=radius):
        v = float(player_value(move_player(frame, idx, cx, cy), idx))
        if v > best_v:
            best_v, best_xy = v, (cx, cy)
    return {"baseline": base, "best": best_v, "best_xy": best_xy,
            "value_gain": best_v - base, "opt_dir": _unit(best_xy[0] - px, best_xy[1] - py)}


def matched_actual_dir(frame_now: FreezeFrame, frame_future: FreezeFrame, idx: int) -> tuple | None:
    """Nearest same-team match of attacker ``idx`` into ``frame_future`` -> actual run unit direction."""
    p = frame_now.players[idx]
    cands = [(q.x, q.y) for q in frame_future.players if q.is_teammate and not q.is_keeper]
    if not cands:
        return None
    qx, qy = cands[int(np.argmin([math.hypot(p.x - cx, p.y - cy) for cx, cy in cands]))]
    return _unit(qx - p.x, qy - p.y)


def team_instinct(moments: list[dict], player_value, *, radius: float = RUN_RADIUS) -> pd.DataFrame:
    """Per-team off-ball value potential + decision quality over ball-carrier ``moments``.

    Each moment: ``{team, frame_now: FreezeFrame, frame_future: FreezeFrame | None}``;
    ``player_value(frame, idx) -> float`` scores a player (e.g. its receiver probability).
    """
    per: dict[int, dict[str, list]] = {}
    for mo in moments:
        fnow, ffut, team = mo["frame_now"], mo.get("frame_future"), int(mo["team"])
        for idx in offball_attacker_indices(fnow):
            o = optimal_run(player_value, fnow, idx, radius=radius)
            per.setdefault(team, {"gain": [], "cos": []})["gain"].append(o["value_gain"])
            if ffut is not None and (o["opt_dir"][0] or o["opt_dir"][1]):
                act = matched_actual_dir(fnow, ffut, idx)
                if act is not None:
                    per[team]["cos"].append(o["opt_dir"][0] * act[0] + o["opt_dir"][1] * act[1])
    rows = []
    for team, d in sorted(per.items()):
        rows.append({"team": team, "n_runs": len(d["gain"]),
                     "offball_value_potential": float(np.mean(d["gain"])) if d["gain"] else float("nan"),
                     "decision_quality_cos": float(np.mean(d["cos"])) if d["cos"] else float("nan")})
    return pd.DataFrame(rows)


def run_match_instinct(positions: "pd.DataFrame", model, *, horizon: int = 75,
                       n_moments: int | None = 40) -> tuple["pd.DataFrame", int]:
    """End-to-end per-team instinct using the GAT **receiver head** as the off-ball run value.

    Builds ball-carrier moments (with a +``horizon``-frame future), samples ``n_moments`` of them, and
    scores each off-ball attacker's optimal "run to get open" vs their actual run. Returns
    ``(per_team_table, n_moments_used)``. Lazy-imports the model bridge so the module stays torch-free.
    """
    from generator.sop_bridge import receiver_probs  # noqa: PLC0415
    from generator.to_frames import frames_from_positions  # noqa: PLC0415

    frames = {fr: fm for fr, fm in frames_from_positions(positions)}
    actor_team = positions[positions["is_actor"] == True].groupby("frame")["team"].first().to_dict()  # noqa: E712
    moments = [{"team": actor_team[fr], "frame_now": frames[fr], "frame_future": frames.get(fr + horizon)}
               for fr in sorted(frames) if fr in actor_team and frames.get(fr + horizon) is not None]
    if n_moments and len(moments) > n_moments:
        moments = [moments[i] for i in np.linspace(0, len(moments) - 1, n_moments).astype(int)]
    return team_instinct(moments, lambda fr, idx: float(receiver_probs(model, fr)[idx])), len(moments)
