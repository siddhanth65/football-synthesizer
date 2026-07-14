"""Per-player run + receiver heads, retrained on dense video tracks (C3).

The dense, ID-persistent tracks the generator now produces give what StatsBomb 360 could not: each
player's **true** position 1.5 s later and their **actual velocity**. That turns run prediction from a
near-null (the old 360 head barely beat "they don't move") into a real signal, and lets a receiver head
score the actual next ball-receiver.

Everything is **attack-direction normalised** (every team mapped to attack toward +x, goal at (105, 34))
using the keeper-derived directions from :func:`fingerprint.structural_metrics.resolve_attack_directions`,
so the heads are team-agnostic. Run head and receiver head are each compared against the baselines a
sparse-360 world was stuck with:

* run -- **no-move** (predict zero displacement) and **constant-velocity** (extrapolate current
  velocity; impossible without motion data),
* receiver -- **nearest-teammate**.

Pure scikit-learn (Ridge / logistic) -- deterministic, CPU, testable. The relational GAT backbone is
layered on later; this is the honest tabular baseline that proves motion data carries the signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from attacker.labels import RUN_HORIZON_S
from attacker.tracks import DEFAULT_FPS, SPEED_CAP_MS

ATTACK_GOAL = np.array([105.0, 34.0])  # all teams normalised to attack toward this goal
RUN_FEATURES = ("x", "y", "vx", "vy", "speed", "dist_goal", "cos_goal", "sin_goal")
RECV_FEATURES = ("dist_carrier", "fwd_carrier", "lat_carrier", "dist_goal", "ahead")


def _attack_dirs(positions: pd.DataFrame) -> dict[int, int]:
    from fingerprint.structural_metrics import resolve_attack_directions  # noqa: PLC0415

    return resolve_attack_directions(positions)


def _group_split(groups: np.ndarray, *, test_frac: float = 0.3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (train, test) masks splitting on whole groups (no track/event leaks across the split)."""
    uniq = np.array(sorted(pd.unique(groups)))
    np.random.default_rng(seed).shuffle(uniq)
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test = np.isin(groups, uniq[:n_test])
    return ~test, test


# --- run head ------------------------------------------------------------------------------------
def run_examples(run_df: pd.DataFrame, attack_dirs: dict[int, int]) -> pd.DataFrame:
    """Attack-normalised run-head feature/label table from :func:`attacker.labels.run_targets` output."""
    df = run_df.dropna(subset=["vx", "vy", "dx", "dy"]).copy()
    df = df[df["team"].map(lambda t: int(t) in attack_dirs)]
    if df.empty:
        return pd.DataFrame(columns=[*RUN_FEATURES, "dx", "dy", "track_id"])
    s = df["team"].map(lambda t: attack_dirs[int(t)]).to_numpy(float)  # +1 / -1
    x = np.where(s > 0, df["x_s"].to_numpy(), ATTACK_GOAL[0] - df["x_s"].to_numpy())
    y = df["y_s"].to_numpy()
    vx, vy = s * df["vx"].to_numpy(), df["vy"].to_numpy()
    to_goal = ATTACK_GOAL - np.column_stack([x, y])
    ang = np.arctan2(to_goal[:, 1], to_goal[:, 0])
    out = pd.DataFrame({
        "x": x, "y": y, "vx": vx, "vy": vy, "speed": np.hypot(vx, vy),
        "dist_goal": np.hypot(to_goal[:, 0], to_goal[:, 1]), "cos_goal": np.cos(ang),
        "sin_goal": np.sin(ang),
        "dx": s * df["dx"].to_numpy(), "dy": df["dy"].to_numpy(), "track_id": df["track_id"].to_numpy(),
        "team": df["team"].to_numpy(),
    })
    if "frame" in df.columns:  # passthrough for per-frame inspection/visualisation
        out["frame"] = df["frame"].to_numpy()
    if "chunk" in df.columns:  # keep the chunk key so the by-track split can stay chunk-local
        out["chunk"] = df["chunk"].to_numpy()
    return out


def _run_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Point-prediction metrics + magnitude/direction fidelity (so under-shooting runs is visible)."""
    err = np.hypot(*(pred - true).T)
    out = {"rmse_m": float(np.sqrt(np.mean(err ** 2))), "mae_m": float(np.mean(err)),
           "hit@3m": float(np.mean(err <= 3.0)), "med_disp_m": float(np.median(np.hypot(*pred.T)))}
    mv = np.hypot(*true.T) > 1.0  # direction only defined for players actually moving
    if mv.any():
        cos = (pred[mv] * true[mv]).sum(1) / (np.linalg.norm(pred[mv], axis=1)
                                              * np.linalg.norm(true[mv], axis=1) + 1e-9)
        out["dir_cos"] = float(np.median(cos))
    else:
        out["dir_cos"] = float("nan")
    return out


def _fit_speed_heading(feats: np.ndarray, target: np.ndarray, alpha: float):
    """Two Ridge models: heading (unit vector, fit on moving examples) and scalar speed (all)."""
    from sklearn.linear_model import Ridge  # noqa: PLC0415

    mag = np.hypot(target[:, 0], target[:, 1])
    moving = mag > 0.5
    unit = target[moving] / mag[moving, None]
    head = Ridge(alpha=alpha).fit(feats[moving], unit)
    speed = Ridge(alpha=alpha).fit(feats, mag)
    return head, speed


def _predict_speed_heading(head, speed, feats: np.ndarray) -> np.ndarray:
    """Combine predicted heading (renormalised to unit) with predicted (clipped >=0) speed."""
    h = head.predict(feats)
    n = np.linalg.norm(h, axis=1, keepdims=True)
    unit = h / np.where(n > 1e-9, n, 1.0)
    return np.clip(speed.predict(feats), 0.0, None)[:, None] * unit


def speed_heading_predictor(features: np.ndarray, target: np.ndarray, alpha: float = 1.0):
    """Fit the magnitude-faithful speed x heading run head; returns a ``features -> displacement`` fn."""
    head, speed = _fit_speed_heading(features, target, alpha)
    return lambda f: _predict_speed_heading(head, speed, f)


def _gaussian_scores(true: np.ndarray, mean: np.ndarray, sigma: tuple[float, float]) -> dict:
    """Per-axis diagonal-Gaussian NLL + 1/2-sigma box coverage (ideal ~0.47 / ~0.91 if calibrated)."""
    sig = np.array(sigma)
    var = sig ** 2
    nll = float(np.mean(0.5 * np.sum((true - mean) ** 2 / var + np.log(2 * np.pi * var), axis=1)))
    z = np.abs(true - mean) / sig
    return {"nll": nll, "cov_1sigma": float(np.mean(np.all(z <= 1.0, axis=1))),
            "cov_2sigma": float(np.mean(np.all(z <= 2.0, axis=1)))}


def train_run_head(positions: pd.DataFrame | None = None, *, run_df: pd.DataFrame | None = None,
                   attack_dirs: dict[int, int] | None = None, horizon_s: float = RUN_HORIZON_S,
                   fps: float = DEFAULT_FPS, alpha: float = 1.0, speed_cap_ms: float = SPEED_CAP_MS,
                   seed: int = 0) -> dict:
    """Fit per-player run heads and score them vs the baselines, exposing the RMSE/magnitude tradeoff.

    Pass ``positions`` (builds tracks + labels) or a precomputed ``run_df``. Displacement labels
    implying a sustained speed above ``speed_cap_ms`` are dropped as ID-switch / jitter outliers.

    Heads scored on the held-out (by-track) split: ``no_move``, ``const_vel``, ``learned`` (Ridge ->
    2D mean, RMSE-optimal but magnitude-shrinking) and ``speed_heading`` (Ridge speed x Ridge heading,
    which recovers realistic run length). Also returns the ``learned`` head's Gaussian distribution
    scores (NLL + sigma-coverage), the fitted models, example counts and the outlier-drop fraction.
    """
    from sklearn.linear_model import Ridge  # noqa: PLC0415

    if run_df is None:
        from attacker.labels import run_targets  # noqa: PLC0415
        from attacker.tracks import build_tracks  # noqa: PLC0415

        run_df = run_targets(build_tracks(positions, fps=fps), horizon_s=horizon_s, fps=fps)
    if attack_dirs is None:
        attack_dirs = _attack_dirs(positions if positions is not None else run_df)
    ex = run_examples(run_df, attack_dirs)
    n_raw = len(ex)
    ex = ex[np.hypot(ex["dx"], ex["dy"]) <= speed_cap_ms * horizon_s].reset_index(drop=True)
    dropped = 1.0 - len(ex) / n_raw if n_raw else 0.0
    if len(ex) < 10:
        raise ValueError(f"too few run examples ({len(ex)}) to train/evaluate")
    feats, target = ex[list(RUN_FEATURES)].to_numpy(), ex[["dx", "dy"]].to_numpy()
    if "chunk" in ex.columns:  # a physical track is (chunk, track_id) -- split on that, not the id
        split_groups = (ex["chunk"].astype(str) + ":" + ex["track_id"].astype(str)).to_numpy()
    else:
        split_groups = ex["track_id"].to_numpy()
    tr, te = _group_split(split_groups, seed=seed)
    model = Ridge(alpha=alpha).fit(feats[tr], target[tr])
    head, speed = _fit_speed_heading(feats[tr], target[tr], alpha)
    preds = {
        "no_move": np.zeros((int(te.sum()), 2)),
        "const_vel": ex.loc[te, ["vx", "vy"]].to_numpy() * horizon_s,
        "learned": model.predict(feats[te]),
        "speed_heading": _predict_speed_heading(head, speed, feats[te]),
    }
    resid = target[tr] - model.predict(feats[tr])
    sigma = (float(resid[:, 0].std()), float(resid[:, 1].std()))
    return {
        "metrics": {k: _run_metrics(p, target[te]) for k, p in preds.items()},
        "gaussian": _gaussian_scores(target[te], preds["learned"], sigma),
        "model": model, "speed_heading_models": (head, speed), "sigma_m": sigma,
        "n_train": int(tr.sum()), "n_test": int(te.sum()), "outlier_frac": float(dropped),
        "disp_rms_m": float(np.sqrt(np.mean(np.sum(target ** 2, axis=1)))),
        "true_med_disp_m": float(np.median(np.hypot(*target[te].T))),
    }


# --- receiver head -------------------------------------------------------------------------------
def receiver_candidates(tracks: pd.DataFrame, recv_df: pd.DataFrame,
                        attack_dirs: dict[int, int]) -> pd.DataFrame:
    """One row per (pass event, candidate teammate): geometry features + ``is_receiver`` label.

    Candidates are the carrier's same-team team-mates on the pitch at the pass frame (carrier excluded).
    Features are attack-normalised relative to the carrier (forward = toward the attacking goal).

    When both ``tracks`` and ``recv_df`` carry a ``chunk`` column the pass frame is resolved *within its
    chunk* -- otherwise the chunk-local frame index would collide across chunks and pull team-mate
    positions from the wrong moment (and a chunk-local carrier ``track_id`` from the wrong player).
    """
    rows = []
    has_chunk = "chunk" in tracks.columns and "chunk" in recv_df.columns
    gkey = ["chunk", "frame"] if has_chunk else "frame"
    by_frame = {k: g for k, g in tracks.groupby(gkey)}
    for ev_id, ev in enumerate(recv_df.itertuples(index=False)):
        g = by_frame.get((ev.chunk, ev.frame) if has_chunk else ev.frame)
        if g is None:
            continue
        s = attack_dirs.get(int(ev.team))
        if s is None:
            continue
        team_g = g[g["team"] == ev.team]
        carrier = team_g[team_g["track_id"] == ev.carrier]
        mates = team_g[team_g["track_id"] != ev.carrier]
        if carrier.empty or mates.empty:
            continue
        cx, cy = float(carrier["x_s"].iloc[0]), float(carrier["y_s"].iloc[0])
        mx, my = mates["x_s"].to_numpy(), mates["y_s"].to_numpy()
        fwd = s * (mx - cx)  # ahead of the carrier toward goal
        lat = my - cy
        gx = np.where(s > 0, mx, ATTACK_GOAL[0] - mx)
        for j, tid in enumerate(mates["track_id"].to_numpy()):
            rows.append({
                "event": ev_id, "frame": int(ev.frame), "team": int(ev.team), "candidate": int(tid),
                "is_receiver": int(tid == ev.receiver),
                "dist_carrier": float(np.hypot(fwd[j], lat[j])), "fwd_carrier": float(fwd[j]),
                "lat_carrier": float(abs(lat[j])),
                "dist_goal": float(np.hypot(ATTACK_GOAL[0] - gx[j], ATTACK_GOAL[1] - my[j])),
                "ahead": int(fwd[j] > 0),
            })
    return pd.DataFrame(rows, columns=["event", "frame", "team", "candidate", "is_receiver",
                                       *RECV_FEATURES])


def _topk_accuracy(cand: pd.DataFrame, score: np.ndarray, ks=(1, 3)) -> dict:
    """Per-event top-k accuracy: is the true receiver among the k highest-scored candidates?"""
    cand = cand.assign(_s=score)
    hits = {k: [] for k in ks}
    for _, g in cand.groupby("event"):
        order = g.sort_values("_s", ascending=False)
        true_rank = np.where(order["is_receiver"].to_numpy() == 1)[0]
        if len(true_rank) == 0:
            continue
        for k in ks:
            hits[k].append(bool(true_rank[0] < k))
    return {f"top{k}": (float(np.mean(v)) if v else float("nan")) for k, v in hits.items()}


def train_receiver_head(tracks: pd.DataFrame, recv_df: pd.DataFrame, *,
                        attack_dirs: dict[int, int] | None = None, seed: int = 0) -> dict:
    """Score candidate receivers (logistic on geometry) vs the nearest-team-mate baseline (top-1/3).

    Returns per-model top-k accuracy plus event counts. Receiver labels are sparse where ball detection
    is sparse; aggregate the whole match for a robust learned head.
    """
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    if attack_dirs is None:
        attack_dirs = _attack_dirs(tracks)
    cand = receiver_candidates(tracks, recv_df, attack_dirs)
    n_events = cand["event"].nunique()
    if n_events < 6:
        return {"n_events": int(n_events), "note": "too few pass events to train; baseline only",
                "metrics": {"nearest": _topk_accuracy(cand, -cand["dist_carrier"].to_numpy())}
                if len(cand) else {}}
    tr, te = _group_split(cand["event"].to_numpy(), seed=seed)
    model = LogisticRegression(max_iter=200).fit(cand.loc[tr, list(RECV_FEATURES)],
                                                 cand.loc[tr, "is_receiver"])
    learned = model.predict_proba(cand.loc[te, list(RECV_FEATURES)])[:, 1]
    te_cand = cand[te]
    return {
        "n_events": int(n_events), "n_test_events": int(te_cand["event"].nunique()),
        "metrics": {
            "nearest": _topk_accuracy(te_cand, -te_cand["dist_carrier"].to_numpy()),
            "learned": _topk_accuracy(te_cand, learned),
        },
        "model": model,
    }
