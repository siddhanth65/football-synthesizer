"""Ball tracking: detection seam + deterministic trajectory linking + possession.

Per-frame ball *detection* is the hard part (tiny, fast, occluded) and needs a motion-aware model — we
keep that as a swappable **seam** (:class:`BallDetector`), the intended concrete impl being **WASB**
(Widely Applicable Strong Baseline, multi-frame heatmap; external weights). Everything downstream is
deterministic and unit-tested here, and runs on *any* ball detections (including the sparse ones the
player detector already emits):

* :func:`link_ball` — turn noisy/multi/missing per-frame detections into one clean trajectory
  (greedy nearest-to-prediction + short-gap interpolation + a physical speed gate),
* :func:`assign_possession` — nearest-player-to-ball possession (with debounce),

so when WASB drops in, possession → phases → passes → line breaks follow (see ``docs/EFI_ALIGNMENT.md``).
Linking/possession work in **pitch metres** (project ball via the same homography as players).
"""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

# ImageNet normalisation WASB trained with (RGB, [0,1] then standardise).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
_DEFAULT_WASB_SRC = os.environ.get("FOOTBALL_WASB_PATH", str(Path.home() / "WASB-SBDT" / "src"))
_DEFAULT_WASB_WEIGHTS = os.environ.get(
    "FOOTBALL_WASB_WEIGHTS", str(Path.home() / "WASB-SBDT" / "pretrained_weights" / "wasb_soccer_best.pth.tar"))

BALL_MAX_SPEED_MS = 40.0     # a struck ball is fast; cap rejects detection teleports (m/s on the pitch)
MAX_INTERP_GAP = 8           # interpolate ball gaps up to this many samples; longer = out/occluded
POSSESSION_RADIUS_M = 2.0    # a player within this of the ball is the carrier
POSSESSION_DEBOUNCE = 3      # possession must persist this many samples before switching (legacy units)
POSSESSION_DEBOUNCE_S = 0.3  # ...or this long in SECONDS (fps-independent; see assign_possession)


class BallDetector(Protocol):
    """Per-frame ball detector. Concrete impl = WASB (external weights); the seam keeps it swappable."""

    def detect(self, video: str, frames: list[int]) -> pd.DataFrame:
        """Return ball candidates: columns ``frame, x, y, conf`` (image px); 0+ rows per frame."""
        ...


class WASBBallDetector:
    """WASB ball detector (Tarashima et al., BMVC 2023) — multi-frame HRNet heatmap.

    Reuses WASB's `HRNet` + pretrained soccer weights but reimplements a minimal (no-Hydra, CPU/GPU)
    forward: stack ``frames_in`` consecutive frames -> 9-ch input at 512x288, ImageNet-normalised RGB;
    the heatmap of the **last** frame in each window -> sigmoid -> threshold -> connected-components
    weighted centroid -> ball ``(x, y, conf)`` rescaled to the original frame. Per-frame detections feed
    :func:`link_ball`. Paths default to ``~/WASB-SBDT`` (override via ``$FOOTBALL_WASB_PATH`` /
    ``$FOOTBALL_WASB_WEIGHTS``).
    """

    def __init__(self, weights: str | None = None, src_dir: str | None = None, device: str | None = None,
                 score_threshold: float = 0.5, frames_in: int = 3, native_res: bool = True):
        self.weights = weights or _DEFAULT_WASB_WEIGHTS
        self.src_dir = src_dir or _DEFAULT_WASB_SRC
        self.device = device
        self.score_threshold = score_threshold
        self.frames_in = frames_in
        self.native_res = native_res  # feed full-res frames (ball ~2x bigger -> far stronger heatmaps)
        self._model = None
        self._inp_wh = (512, 288)

    def _load(self) -> None:
        import importlib.util  # noqa: PLC0415

        import torch  # noqa: PLC0415
        from omegaconf import OmegaConf  # noqa: PLC0415

        if not os.path.exists(self.weights):
            raise FileNotFoundError(f"WASB weights not found: {self.weights} (set $FOOTBALL_WASB_WEIGHTS)")
        spec = importlib.util.spec_from_file_location("wasb_hrnet",
                                                      os.path.join(self.src_dir, "models", "hrnet.py"))
        hrnet = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hrnet)  # standalone (hrnet.py has no relative imports)
        cfg = OmegaConf.load(os.path.join(self.src_dir, "configs", "model", "wasb.yaml"))
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = hrnet.HRNet(cfg)
        ckpt = torch.load(self.weights, map_location=self.device)
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        model.load_state_dict(sd)
        self._model = model.eval().to(self.device)
        self._inp_wh = (int(cfg["inp_width"]), int(cfg["inp_height"]))

    def _preprocess(self, bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        import cv2  # noqa: PLC0415

        rgb = cv2.cvtColor(cv2.resize(bgr, size), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return ((rgb - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)  # [3, H, W]

    def _peak(self, hm: np.ndarray) -> tuple[tuple[float, float] | None, float]:
        import cv2  # noqa: PLC0415

        if float(hm.max()) <= self.score_threshold:
            return None, 0.0
        _, th = cv2.threshold(hm, self.score_threshold, 1, cv2.THRESH_BINARY)
        n, labels = cv2.connectedComponents(th.astype(np.uint8))
        best, best_s = None, -1.0
        for m in range(1, n):
            ys, xs = np.where(labels == m)
            w = hm[ys, xs]
            s = float(w.sum())
            if s > best_s:
                best_s = s
                best = (float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum()))
        return best, best_s

    def detect(self, video: str, frames: list[int]) -> pd.DataFrame:
        """Ball ``frame, x, y, conf`` (image px) for each requested frame with a detection.

        ``frames`` should be a **contiguous native-fps** range — WASB needs consecutive frames for its
        motion cue (a window of ``frames_in`` ending at each frame).
        """
        import cv2  # noqa: PLC0415
        import torch  # noqa: PLC0415

        if self._model is None:
            self._load()
        want = {int(f) for f in frames}
        lo, hi = min(want), max(want)
        cap = cv2.VideoCapture(video)
        wo, ho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (wo - wo % 8, ho - ho % 8) if self.native_res else self._inp_wh  # HRNet needs /8 dims
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(lo - (self.frames_in - 1), 0))
        fr = max(lo - (self.frames_in - 1), 0)
        buf: deque = deque(maxlen=self.frames_in)
        sx, sy = wo / size[0], ho / size[1]
        rows = []
        while fr <= hi:
            ok, bgr = cap.read()
            if not ok:
                break
            buf.append(self._preprocess(bgr, size))
            if len(buf) == self.frames_in and fr in want:
                inp = torch.from_numpy(np.concatenate(list(buf), 0)[None]).to(self.device)
                with torch.no_grad():
                    hm = self._model(inp)[0].sigmoid()[0, -1].cpu().numpy()  # last frame's heatmap
                xy, score = self._peak(hm)
                if xy is not None:
                    rows.append({"frame": fr, "x": xy[0] * sx, "y": xy[1] * sy, "conf": score})
            fr += 1
        cap.release()
        return pd.DataFrame(rows, columns=["frame", "x", "y", "conf"])


def link_ball(detections: pd.DataFrame, *, fps: float = 50.0, max_speed_ms: float = BALL_MAX_SPEED_MS,
              max_gap: int = MAX_INTERP_GAP) -> pd.DataFrame:
    """Link per-frame ball candidates into one clean trajectory (pitch metres).

    Greedy forward pass: keep the candidate nearest the constant-velocity prediction that is within the
    physical speed gate; linearly interpolate gaps up to ``max_gap`` samples. ``detections`` has columns
    ``frame, x, y`` (+ optional ``conf``); 0+ rows per frame.

    Two guards keep the greedy pass from a runaway death (a single stale/jittery estimate rejecting
    every later frame, which collapsed post-link coverage to ~3-17%):

    * **Velocity clamp** -- the running velocity is capped at ``max_speed_ms``; jitter over a small dt
      can otherwise imply a super-physical speed (>40 m/s) that flings the prediction off-pitch.
    * **Re-seed after a long gap** -- once the ball has been unobserved for more than ``max_gap``
      samples there is no reliable prediction, so anchoring the gate on a stale constant-velocity
      extrapolation only runs away further; instead we drop the prediction and re-seed on the next
      candidate, starting a fresh segment.

    Returns one row per sampled frame in span: ``frame, x, y, observed`` (``observed=False`` =
    interpolated). Frames in gaps longer than ``max_gap`` are omitted.
    """
    if detections.empty:
        return pd.DataFrame(columns=["frame", "x", "y", "observed"])
    det = detections.dropna(subset=["x", "y"]).sort_values("frame")
    step = int(np.median(np.diff(np.sort(det["frame"].unique())))) or 1
    cand = {fr: g[["x", "y"]].to_numpy() for fr, g in det.groupby("frame")}
    frames = sorted(cand)
    picked: list[tuple[int, float, float]] = []
    pos = vel = None
    last_fr = None
    for fr in frames:
        pts = cand[fr]
        if pos is not None and (fr - last_fr) > max_gap * step:
            pos = vel = last_fr = None  # unobserved too long -> prediction is stale, re-seed
        if pos is None:
            xy = pts[0]  # seed (or re-seed) on this frame's first candidate
        else:
            dt = (fr - last_fr) / fps
            pred = pos + (vel if vel is not None else 0.0) * dt
            d = np.hypot(*(pts - pred).T)
            j = int(np.argmin(d))
            if d[j] > max_speed_ms * dt + 1e-6:  # physically impossible jump -> treat frame as a miss
                continue
            xy = pts[j]
        if pos is not None:
            dt = (fr - last_fr) / fps
            if dt > 0:
                v = (xy - pos) / dt
                speed = float(np.hypot(*v))
                vel = v * (max_speed_ms / speed) if speed > max_speed_ms else v
        pos, last_fr = xy, fr
        picked.append((fr, float(xy[0]), float(xy[1])))

    track = pd.DataFrame(picked, columns=["frame", "x", "y"]).assign(observed=True)
    return _interpolate_gaps(track, step=step, max_gap=max_gap)


def _interpolate_gaps(track: pd.DataFrame, *, step: int, max_gap: int) -> pd.DataFrame:
    """Linearly fill gaps up to ``max_gap`` samples between consecutive observed ball positions."""
    if len(track) < 2:
        return track
    rows = [track.iloc[0].to_dict()]
    for a, b in zip(track.iloc[:-1].itertuples(index=False), track.iloc[1:].itertuples(index=False)):
        n = (b.frame - a.frame) // step
        if 1 < n <= max_gap:
            for k in range(1, n):
                f = a.frame + k * step
                w = k / n
                rows.append({"frame": int(f), "x": a.x + w * (b.x - a.x),
                             "y": a.y + w * (b.y - a.y), "observed": False})
        rows.append(b._asdict())
    return pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)


SWITCH_PENALTY_M = 1.5  # Viterbi cost (pitch metres) charged for flipping the possessing team


def assign_possession(ball: pd.DataFrame, players: pd.DataFrame, *, radius_m: float = POSSESSION_RADIUS_M,
                      debounce: int = POSSESSION_DEBOUNCE, debounce_s: float | None = None,
                      fps: float | None = None, smooth: bool = False,
                      switch_penalty_m: float = SWITCH_PENALTY_M) -> pd.DataFrame:
    """Nearest-player-to-ball **territorial possession proxy** per frame.

    This is a *proximity* proxy, not event-based possession: the carrier is the nearest player to the
    projected ball (within ``radius_m``), so it conflates "near the ball" with "in control" and has no
    notion of touches/duels/aerials. Validate the *share* against a provider's figure, but read it as
    territorial control, not Opta possession.

    ``ball`` has ``frame, x, y`` (pitch m); ``players`` has ``frame, track_id, team, pitch_x, pitch_y``.
    Returns ``frame, carrier, team, dist_m`` for frames where a player is within ``radius_m``.

    Smoothing (pick one):
        * ``debounce`` (default) -- drop spells shorter than ``debounce`` samples (hysteresis).
        * ``smooth=True`` -- a Viterbi pass over the team sequence that charges ``switch_penalty_m`` per
          flip, so a closer opponent only steals possession when it clearly beats the carry-cost. More
          principled than debounce on noisy ball tracks; the carrier within the chosen team is then its
          nearest player.

    The debounce hysteresis may be given in **seconds** (``debounce_s`` + ``fps``) instead of a raw
    sample count, so the same physical window applies regardless of the source fps / sampling stride;
    it is converted to samples via the ball track's native stride (``debounce_s * fps / stride``). Note
    ``debounce`` only applies on the non-``smooth`` path; with ``smooth=True`` the Viterbi pass governs
    spell structure and these arguments are ignored.

    Args:
        ball: linked ball track ``frame, x, y`` (pitch metres).
        players: player positions ``frame, track_id, team, pitch_x, pitch_y`` (pitch metres).
        radius_m: max ball-to-player distance for a carrier.
        debounce: hysteresis in possession samples (legacy units; used when ``debounce_s`` is ``None``).
        debounce_s: hysteresis in **seconds**; requires ``fps`` and takes precedence when given.
        fps: native frames per second of this chunk (needed only with ``debounce_s``).
        smooth: use the Viterbi team smoother instead of debounce.
        switch_penalty_m: Viterbi cost (metres) per possessing-team flip.

    Raises:
        ValueError: if ``debounce_s`` is given without a positive ``fps``.
    """
    if debounce_s is not None:
        if fps is None or fps <= 0:
            raise ValueError("assign_possession: debounce_s requires a positive fps")
        frames = np.sort(ball["frame"].to_numpy())
        stride = int(np.median(np.diff(frames))) if len(frames) > 1 else 1
        stride = stride or 1
        debounce = max(1, round(debounce_s * fps / stride))
    pl = players.dropna(subset=["pitch_x", "pitch_y"])
    by_frame = {fr: g for fr, g in pl.groupby("frame")}
    raw = []
    for b in ball.itertuples(index=False):
        g = by_frame.get(b.frame)
        if g is None:
            continue
        d = np.hypot(g["pitch_x"].to_numpy() - b.x, g["pitch_y"].to_numpy() - b.y)
        j = int(np.argmin(d))
        if d[j] <= radius_m:
            r = g.iloc[j]
            raw.append({"frame": int(b.frame), "carrier": int(r["track_id"]), "team": int(r["team"]),
                        "dist_m": float(d[j]), "_g": g, "_d": d})
    if not raw:
        return pd.DataFrame(columns=["frame", "carrier", "team", "dist_m"])
    if smooth:
        return _viterbi_possession(raw, switch_penalty_m=switch_penalty_m)
    out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in raw],
                       columns=["frame", "carrier", "team", "dist_m"])
    return _debounce_team(out, debounce) if debounce > 1 else out


def _viterbi_possession(raw: list[dict], *, switch_penalty_m: float) -> pd.DataFrame:
    """Smooth the possessing-team sequence with a Viterbi DP, then pick each frame's nearest in-team player.

    Emission cost for assigning team ``t`` at a frame = the distance from the ball to ``t``'s nearest
    player (a far team is expensive); transition cost = ``switch_penalty_m`` when the team changes. The
    min-cost path resists single-frame flips driven by ball-projection jitter.
    """
    teams = sorted({int(t) for r in raw for t in r["_g"]["team"].unique() if int(t) >= 0})
    if not teams:
        cols = ["frame", "carrier", "team", "dist_m"]
        return pd.DataFrame([{k: r[k] for k in cols} for r in raw], columns=cols)
    big = 1e6
    # Per-frame emission cost per team = ball-to-nearest-player distance for that team (inf if absent).
    emit = []
    for r in raw:
        g, d = r["_g"], r["_d"]
        tn = g["team"].to_numpy()
        row = {}
        for t in teams:
            mask = tn == t
            row[t] = float(d[mask].min()) if mask.any() else big
        emit.append(row)
    n, k = len(raw), len(teams)
    cost = np.full((n, k), big)
    back = np.zeros((n, k), int)
    for ti, t in enumerate(teams):
        cost[0, ti] = emit[0][t]
    for i in range(1, n):
        for ti, t in enumerate(teams):
            trans = cost[i - 1] + np.array([0.0 if pj == ti else switch_penalty_m for pj in range(k)])
            back[i, ti] = int(np.argmin(trans))
            cost[i, ti] = emit[i][t] + trans[back[i, ti]]
    path = np.zeros(n, int)
    path[-1] = int(np.argmin(cost[-1]))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    rows = []
    for r, pj in zip(raw, path):
        t = teams[pj]
        g, d = r["_g"], r["_d"]
        mask = (g["team"].to_numpy() == t)
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        j = int(idx[np.argmin(d[mask])])
        pr = g.iloc[j]
        rows.append({"frame": int(r["frame"]), "carrier": int(pr["track_id"]), "team": int(t),
                     "dist_m": float(d[j])})
    return pd.DataFrame(rows, columns=["frame", "carrier", "team", "dist_m"])


def _debounce_team(poss: pd.DataFrame, n: int) -> pd.DataFrame:
    """Drop possession rows whose team holds for fewer than ``n`` consecutive possession samples."""
    if poss.empty:
        return poss
    team = poss["team"].to_numpy()
    keep = np.ones(len(team), bool)
    i = 0
    while i < len(team):
        j = i
        while j < len(team) and team[j] == team[i]:
            j += 1
        if j - i < n:
            keep[i:j] = False
        i = j
    return poss[keep].reset_index(drop=True)
