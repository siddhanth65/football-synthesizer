"""Per-player run + receiver labels from dense tracks.

- **Run target**: a player's *displacement* at t + 1.5 s read directly off the persistent track -- no
  nearest-neighbour identity guessing, which is the noise that capped the old 360 run head at 6.33 m.
- **Receiver**: the actual next on-ball receiver, inferred from consecutive distinct ball-carriers
  (``is_actor``) on the same team within a short window (a completed pass). Optional EFI
  movement-to-receive / offers enrichment is layered on later via :mod:`ingest.fifa_efi`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from attacker.tracks import DEFAULT_FPS

RUN_HORIZON_S = 1.5
MAX_PASS_S = 3.0  # carrier -> next distinct same-team carrier within this is treated as a pass


def run_targets(tracks: pd.DataFrame, *, horizon_s: float = RUN_HORIZON_S,
                fps: float = DEFAULT_FPS) -> pd.DataFrame:
    """Attach the +``horizon_s`` displacement ``(dx, dy)`` to each track row that has a future sample.

    For each row, the same track's smoothed position ``horizon_s`` later (matched to the nearest
    sampled frame within one step) gives the label. Rows whose track does not extend that far are
    dropped -- there is no future to predict.

    Returns:
        The input rows that have a future, plus ``f_target, x_fut, y_fut, dx, dy``.
    """
    if tracks.empty:
        return tracks.assign(dx=pd.Series(dtype=float), dy=pd.Series(dtype=float))
    horizon = int(round(horizon_s * fps))
    step = int(np.median(np.diff(np.sort(tracks["frame"].unique())))) or 1
    out = []
    for _, g in tracks.sort_values("frame").groupby("track_id", sort=False):
        base = g.copy()
        base["f_target"] = base["frame"] + horizon
        fut = g[["frame", "x_s", "y_s"]].rename(
            columns={"frame": "f_fut", "x_s": "x_fut", "y_s": "y_fut"})
        m = pd.merge_asof(base.sort_values("f_target"), fut.sort_values("f_fut"),
                          left_on="f_target", right_on="f_fut", direction="nearest", tolerance=step)
        out.append(m.dropna(subset=["x_fut", "y_fut"]))
    res = pd.concat(out).reset_index(drop=True)
    res["dx"], res["dy"] = res["x_fut"] - res["x_s"], res["y_fut"] - res["y_s"]
    return res


def receiver_labels(tracks: pd.DataFrame, events: pd.DataFrame | None = None, *,
                    max_pass_s: float = MAX_PASS_S, fps: float = DEFAULT_FPS) -> pd.DataFrame:
    """Pass events ``(frame, carrier, receiver, team, dt_s)`` from consecutive same-team ball-carriers.

    A receiver label is created when the ball-carrier (``is_actor``) changes to a *different* track of
    the *same* team within ``max_pass_s`` (a completed pass). ``events`` is reserved for optional FIFA
    EFI enrichment (not required). Sparse where ball/carrier detection is sparse -- aggregate over the
    whole match for enough examples to train a learned receiver head.
    """
    if "is_actor" not in tracks.columns:
        return pd.DataFrame(columns=["frame", "carrier", "receiver", "team", "dt_s"])
    actors = tracks[tracks["is_actor"] == True].sort_values("frame")  # noqa: E712
    rows = []
    prev = None
    for r in actors.itertuples(index=False):
        if prev is not None:
            dt = (r.frame - prev.frame) / fps
            if r.track_id != prev.track_id and r.team == prev.team and 0 < dt <= max_pass_s:
                rows.append({"frame": int(prev.frame), "carrier": int(prev.track_id),
                             "receiver": int(r.track_id), "team": int(prev.team), "dt_s": float(dt)})
        prev = r
    return pd.DataFrame(rows, columns=["frame", "carrier", "receiver", "team", "dt_s"])
