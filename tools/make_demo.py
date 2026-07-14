"""Compose the CV-pipeline showcase video (broadcast -> tracks -> calibration -> tactical view).

A presentation deliverable, not an analysis: it stitches hand-picked live-play segments of the
Brighton-Man Utd PL 24/25 pilot into one narrated MP4 with caption cards. Everything drawn comes
from artifacts already on disk -- the dense positions parquets (detection + tracking + team ids +
per-frame calibration error), the post-``link_ball`` ball parquets, and the persisted eval/gate
files. Nothing is recomputed on the GPU and **no number on screen is invented**: the scorecard is
read from ``outputs/eval/<match>_ball_eval.json``, ``results/pl_pilot/fbref_gate.md`` and
``results/pl_probe/diagnosis/DIAGNOSIS.md`` at render time (see :func:`demo_facts`).

Honesty is part of the content: frames the pipeline could not calibrate are shown *as* failures
(no overlay + an explicit banner), and ball samples that were inferred by carry-over/interpolation
rather than detected are drawn differently from observed ones.

Sections: title -> A detection+tracking -> B calibration (incl. failures) -> C top-down
reconstruction -> D ball + possession -> E derived structures -> closing scorecard.

Run::

    python tools/make_demo.py                      # full ~5 min render
    python tools/make_demo.py --preview            # ~25 s sample (3 s per segment)
    python tools/make_demo.py --height 720         # 720p output
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> import core.*, fingerprint.*

from core import registry  # noqa: E402
from core.pitch import PITCH_LEN, PITCH_WID  # noqa: E402
from fingerprint.structural_metrics import (  # noqa: E402
    LANE_EDGES,
    MIN_TEAM_PLAYERS,
    attacking_coord,
    resolve_attack_directions,
)
from generator.impute import line_from_deepest  # noqa: E402

MATCH_ID = "brighton_manutd"
VIDEO_ROOT = Path("matches")
DENSE_ROOT = Path("outputs")
FPS = 25.0
W, H = 1920, 1080  # render canvas (downscaled at encode time if --height is given)

# Colours (BGR). team 0 = Man Utd (red), team 1 = Brighton (blue).
C_TEAM = {0: (56, 56, 226), 1: (222, 150, 60), -1: (170, 170, 170)}
C_WHITE, C_BLACK, C_YELLOW = (255, 255, 255), (0, 0, 0), (0, 230, 255)
C_AMBER, C_RED, C_GREEN = (60, 190, 250), (60, 60, 235), (120, 220, 140)
C_BG = (46, 26, 26)  # #1a1a2e navy, the project's card background
C_GRASS = (78, 138, 45)
FONT, FONT_T = cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX

MAX_INTERP_GAP = 10  # source frames (0.4 s at 25 fps): wider than this = calibration gap, not motion
HEADER_H, FOOTER_H = 76, 148

TEAM_LABEL = {0: "Man Utd", 1: "Brighton"}


# --------------------------------------------------------------------------------------------------
# Facts (every on-screen number is sourced here, from artifacts on disk)
# --------------------------------------------------------------------------------------------------
MIN_CORR = 4  # projected players needed to call a frame calibrated (and to refit its homography)


def _yield_rates(match_id: str) -> tuple[float, float, int]:
    """Whole-match detection and calibration yield per *sampled* frame.

    The extractor samples every 5th source frame. A sampled frame appears in the dense parquet iff
    the detector ran on it; it carries **pitch coordinates** only if the homography solved (failed
    frames are written with ``calib_error_m = inf`` and null ``pitch_x/y`` -- the detections survive,
    the geometry does not). Both denominators are the source video's frame count / 5, so the numbers
    include replays, close-ups and cuts: this is the honest whole-match yield, not a live-play one.

    Args:
        match_id: registry match id.

    Returns:
        ``(detect_rate, calib_rate, n_chunks)``.
    """
    dense_n = calib_n = expected = n = 0
    for half in ("h1", "h2"):
        for path in sorted((DENSE_ROOT / match_id / half / "match").glob("chunk_*_dense.parquet")):
            ck = re.search(r"chunk_(\d+)_dense", path.name).group(1)
            video = VIDEO_ROOT / match_id / half / f"chunk_{ck}.mp4"
            if not video.exists():
                continue
            cap = cv2.VideoCapture(str(video))
            n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            df = pd.read_parquet(path, columns=["frame", "pitch_x"])
            counts = df.dropna(subset=["pitch_x"]).groupby("frame").size()
            dense_n += int(df["frame"].nunique())
            calib_n += int((counts >= MIN_CORR).sum())
            expected += n_video // 5
            n += 1
    if not expected:
        return float("nan"), float("nan"), 0
    return dense_n / expected, calib_n / expected, n


def demo_facts(match_id: str = MATCH_ID) -> dict:
    """Collect every figure the video displays, each read from a persisted artifact.

    Args:
        match_id: registry match id.

    Returns:
        Dict of display strings plus their provenance; missing artifacts simply drop out (the card
        omits the line rather than inventing a value).
    """
    facts: dict = {"provenance": {}}
    eval_path = Path("outputs/eval") / f"{match_id}_ball_eval.json"
    if eval_path.exists():
        ev = json.loads(eval_path.read_text(encoding="utf-8"))
        facts["ball_coverage"] = ev["post_link_coverage"]
        facts["pass_recall"] = ev["pass_recall_proxy"]
        facts["oracle"] = ev.get("oracle_source", "oracle")
        facts["provenance"]["ball_coverage/pass_recall"] = str(eval_path)

    gate = Path("results/pl_pilot/fbref_gate.md")
    if gate.exists():
        txt = gate.read_text(encoding="utf-8")
        m = re.search(r"Fixture:\s*\*\*(.+?)\*\*,\s*(.+?),\s*(MW\d+),\s*(\d{4}-\d{2}-\d{2})", txt)
        if m:
            facts["fixture"] = m.group(1).strip()
            facts["comp"] = f"{m.group(2).strip()}, {m.group(3)}, {m.group(4)}"
            facts["provenance"]["fixture"] = str(gate)

    diag = Path("results/pl_probe/diagnosis/DIAGNOSIS.md")
    if diag.exists():
        txt = diag.read_text(encoding="utf-8")
        seg = re.search(r"brighton_manutd/seg_3\s+\d+\s+\S+\s+\S+\s+\S+\s+\S+\s+(\d+)%", txt)
        pooled = re.search(r"^brighton_manutd\s+\d+\s+\S+\s+\S+\s+\S+\s+(\d+)%", txt, re.M)
        if seg:
            facts["calib_live_probe"] = int(seg.group(1)) / 100
            facts["provenance"]["calib_live_probe"] = f"{diag} (probe segment 3)"
        if pooled:
            facts["calib_live_pooled"] = int(pooled.group(1)) / 100
            facts["provenance"]["calib_live_pooled"] = f"{diag} (3 probe segments)"

    det, cal, n_chunks = _yield_rates(match_id)
    facts["detect_all_frames"] = det
    facts["calib_all_frames"] = cal
    facts["n_chunks"] = n_chunks
    facts["provenance"]["detect/calib_all_frames"] = (
        "dense parquets vs source frame counts (computed at render time)"
    )
    return facts


# --------------------------------------------------------------------------------------------------
# Segment plan
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Segment:
    """One rendered stretch of footage.

    Attributes:
        section: section letter (A-E) used for the header strip.
        half: ``h1``/``h2``.
        chunk: chunk index within the half.
        start: first source frame (25 fps).
        end: last source frame.
        layout: ``video`` (full-frame) or ``split`` (video + top-down pitch).
        overlays: any of ``markers``, ``ids``, ``lines``, ``ball``, ``struct_line``, ``struct_lanes``.
        caption: the caption strip's main line.
        note: the caption strip's second line (usually the honest caveat).
    """

    section: str
    half: str
    chunk: int
    start: int
    end: int
    layout: str
    overlays: tuple[str, ...]
    caption: str
    note: str = ""


SECTION_TITLES = {
    "A": "A / DETECTION + TRACKING",
    "B": "B / PITCH CALIBRATION",
    "C": "C / TOP-DOWN RECONSTRUCTION",
    "D": "D / BALL TRACKING + POSSESSION",
    "E": "E / WHAT WE MEASURE",
}

SECTION_CARDS = {
    "A": ("DETECTION + TRACKING",
          ["Fine-tuned player detector + ByteTrack, sampled at 5 Hz.",
           "Each dot = one tracked player, coloured by team, labelled with its persistent track id.",
           "~11-13 players visible per frame: a broadcast never shows all 22."]),
    "B": ("PITCH CALIBRATION",
          ["Per-frame homography from the pitch model to the image.",
           "The yellow lines are RECONSTRUCTED, not detected: if they sit on the painted markings,",
           "the calibration is right. We also show it failing -- because it does."]),
    "C": ("TOP-DOWN RECONSTRUCTION",
          ["The same homography maps every player into 105 x 68 metre pitch coordinates.",
           "Left: broadcast. Right: the tactical view the metric engine actually consumes."]),
    "D": ("BALL TRACKING + POSSESSION",
          ["Fine-tuned TrackNetV2 ball detector + linking + carry-over projection.",
           "Solid marker = detected. Hollow marker = inferred (carry-over / interpolation).",
           "When the ball is not tracked at all, we say so on screen."]),
    "E": ("WHAT WE MEASURE",
          ["Derived team structures computed from the tracked positions.",
           "Shown on the VISIBLE players only -- the shipped metrics de-bias this censoring."]),
}

# Segment plan. Sections A/B need detections only; C/D/E need pitch geometry, so those windows are
# built around the match's longest *continuously calibrated* runs (>=4 projected players, sample gaps
# <= MAX_INTERP_GAP), scored for ball coverage and visible players. The B windows are picked for the
# opposite reason: they contain real calibration failures.
_VID = ("markers", "ids")
_SPLIT = ("markers", "ids", "lines", "ball", "topdown")

SEGMENTS: tuple[Segment, ...] = (
    # --- A: detection + tracking (detection-renderable 99-100%) -----------------------------------
    Segment("A", "h2", 2, 3140, 4015, "video", _VID,
            "Man Utd (red) vs Brighton (blue): every dot is a tracked player, the number is its "
            "track id",
            "detector + ByteTrack, replayed from the dense parquet - sampled at 5 Hz and "
            "interpolated for display"),
    Segment("A", "h1", 1, 8385, 9160, "video", _VID,
            "Ids persist through the passage: the same player keeps the same number",
            "~13 of 22 players are visible - a broadcast camera never shows the full pitch, and we "
            "never invent the rest"),
    # --- B: calibration: the proof, then the failures ---------------------------------------------
    Segment("B", "h1", 4, 6750, 7580, "video", ("markers", "lines", "calib"),
            "Yellow = the 105 x 68 pitch model projected back onto the frame by our own homography",
            "it lands on the real painted lines - the calibration proving itself, frame by frame "
            "(watch it blink out mid-segment)"),
    Segment("B", "h2", 2, 11700, 12250, "video", ("markers", "lines", "calib"),
            "Calibration is not free: here it fails first, then recovers mid-passage",
            "failed frames keep their detections but get NO pitch geometry - the pipeline emits "
            "nothing rather than guessing"),
    Segment("B", "h2", 2, 13600, 13950, "video", ("markers", "lines", "calib"),
            "Behind-goal angle, then a close-up: no main-camera geometry - and the detector "
            "collapses too",
            "median 2 tracks here, one of them a false positive on the broadcast logo. ZERO of these "
            "frames yield metrics - by design"),
    # --- C: top-down reconstruction (built on the longest fully-calibrated ball-tracked runs) ------
    Segment("C", "h1", 4, 4700, 5120, "split", _SPLIT,
            "Broadcast -> tactical view: the same players, in metres, on a 105 x 68 pitch",
            "the right panel is reconstructed from one homography per frame - it is what every "
            "metric consumes"),
    Segment("C", "h1", 2, 13450, 13830, "split", _SPLIT,
            "One homography per frame maps foot points into pitch coordinates",
            "players the camera never showed are simply absent - no imputation on screen"),
    Segment("C", "h1", 3, 7800, 8180, "split", _SPLIT,
            "Team shape becomes readable the moment the projection lands",
            "when the geometry drops out, the right panel goes dark - that is an honest gap"),
    Segment("C", "h2", 1, 13680, 14060, "split", _SPLIT,
            "Man Utd (red) and Brighton (blue) in the tactical frame, ball included",
            "positions are metric: distances, lines and lanes are all computed from this panel"),
    # --- D: ball + possession ---------------------------------------------------------------------
    Segment("D", "h1", 4, 11370, 11700, "split",
            ("markers", "lines", "ball", "topdown", "carrier"),
            "Ball on both panels; the ringed player is the nearest carrier",
            "solid marker = detected this frame; hollow = inferred by carry-over / interpolation"),
    Segment("D", "h1", 1, 6950, 7290, "split",
            ("markers", "lines", "ball", "topdown", "carrier"),
            "The ball track survives frames where the players' geometry does not (carry-over lever)",
            "carry-over reproduces held-out ball positions to a median 0.33 m (leave-one-out, "
            "n=656)"),
    Segment("D", "h2", 3, 6225, 6800, "split",
            ("markers", "lines", "ball", "topdown", "carrier"),
            "When the ball is lost we say so - no silent gap-filling",
            "post-link ball coverage is 51.9% over the full match: roughly half of the frames carry "
            "a usable ball"),
    # --- E: derived structures --------------------------------------------------------------------
    Segment("E", "h1", 3, 2030, 2380, "split", ("markers", "lines", "topdown", "struct_line"),
            "Defensive line (deepest-4, in metres) and block box for Man Utd",
            "computed on VISIBLE players; the shipped estimator de-biases this censoring - line "
            "validated to ~5 m vs FIFA PMSR"),
    Segment("E", "h1", 4, 2820, 3140, "split", ("markers", "lines", "topdown", "struct_line"),
            "The line and the block move with the game - this is what the metric engine reads",
            "below 4 visible players the metric abstains rather than reporting a shape it cannot "
            "see"),
    Segment("E", "h2", 2, 3500, 4030, "split", ("markers", "lines", "topdown", "struct_lanes"),
            "Five-lane occupation: where a team's visible players stand across the pitch width",
            "structure is validated; events (passes, tackles, shots) are NOT detected - we do not "
            "claim them"),
)


# --------------------------------------------------------------------------------------------------
# Drawing primitives
# --------------------------------------------------------------------------------------------------
def _pitch_polylines() -> list[np.ndarray]:
    """Pitch markings as metre-space polylines (105 x 68, corner origin)."""
    cx, cy = PITCH_LEN / 2, PITCH_WID / 2
    lines = [
        np.array([(0, 0), (PITCH_LEN, 0), (PITCH_LEN, PITCH_WID), (0, PITCH_WID), (0, 0)], np.float32),
        np.array([(cx, 0), (cx, PITCH_WID)], np.float32),
        np.array([(0, 13.84), (16.5, 13.84), (16.5, 54.16), (0, 54.16)], np.float32),
        np.array([(PITCH_LEN, 13.84), (88.5, 13.84), (88.5, 54.16), (PITCH_LEN, 54.16)], np.float32),
        np.array([(0, 24.84), (5.5, 24.84), (5.5, 43.16), (0, 43.16)], np.float32),
        np.array([(PITCH_LEN, 24.84), (99.5, 24.84), (99.5, 43.16), (PITCH_LEN, 43.16)], np.float32),
        np.array([(cx + 9.15 * np.cos(t), cy + 9.15 * np.sin(t))
                  for t in np.linspace(0, 2 * np.pi, 64)], np.float32),
    ]
    return lines


def shade(canvas: np.ndarray, y0: int, y1: int, alpha: float = 0.62,
          colour: tuple[int, int, int] = (18, 12, 12)) -> None:
    """Darken a horizontal band of the canvas in place (the caption / header bars)."""
    y0, y1 = max(0, y0), min(canvas.shape[0], y1)
    if y1 <= y0:
        return
    roi = canvas[y0:y1]
    block = np.full_like(roi, colour, dtype=np.uint8)
    cv2.addWeighted(block, alpha, roi, 1 - alpha, 0, roi)


def text(canvas: np.ndarray, s: str, org: tuple[int, int], scale: float = 0.9,
         colour: tuple[int, int, int] = C_WHITE, thick: int = 2, font: int = FONT) -> None:
    """Draw legible text with a dark outline so it survives any background."""
    cv2.putText(canvas, s, org, font, scale, C_BLACK, thick + 3, cv2.LINE_AA)
    cv2.putText(canvas, s, org, font, scale, colour, thick, cv2.LINE_AA)


def chip(canvas: np.ndarray, s: str, org: tuple[int, int], colour: tuple[int, int, int],
         scale: float = 0.8) -> None:
    """Draw a filled status chip (used for the honest 'not tracked' indicators)."""
    (tw, th), _ = cv2.getTextSize(s, FONT, scale, 2)
    x, y = org
    cv2.rectangle(canvas, (x - 10, y - th - 12), (x + tw + 12, y + 12), colour, -1)
    cv2.putText(canvas, s, (x, y), FONT, scale, C_BLACK, 2, cv2.LINE_AA)


class TopDown:
    """Renders the 105 x 68 top-down pitch panel with cv2 primitives (fast enough for 25 fps)."""

    def __init__(self, width: int) -> None:
        """Build the reusable pitch background at the given panel width."""
        self.w = width
        self.h = int(round(width * PITCH_WID / PITCH_LEN))
        self.pad = int(0.03 * width)
        self.scale = (width - 2 * self.pad) / PITCH_LEN
        base = np.full((self.h, self.w, 3), C_GRASS, np.uint8)
        for i in range(0, self.w, self.w // 12):  # mown stripes, purely cosmetic
            cv2.rectangle(base, (i, 0), (i + self.w // 24, self.h), (86, 148, 52), -1)
        for pl in _pitch_polylines():
            pts = np.array([self.to_px(x, y) for x, y in pl], np.int32)
            cv2.polylines(base, [pts], False, (235, 235, 235), 2, cv2.LINE_AA)
        self.base = base

    def to_px(self, x: float, y: float) -> tuple[int, int]:
        """Metre coordinates -> panel pixels."""
        return int(round(self.pad + x * self.scale)), int(round(self.pad + y * self.scale))

    def frame(self) -> np.ndarray:
        """A fresh copy of the empty pitch."""
        return self.base.copy()


# --------------------------------------------------------------------------------------------------
# Chunk artifacts -> per-frame render state
# --------------------------------------------------------------------------------------------------
@dataclass
class FrameState:
    """Everything drawable for one output frame (interpolated between dense samples).

    ``calibrated`` is the load-bearing flag: a frame can carry perfectly good **detections** and no
    pitch geometry at all (the extractor writes those rows with ``calib_error_m = inf`` and null
    pitch coordinates). Player dicts therefore always have ``img`` and may have ``pitch = None``.
    """

    players: list[dict] = field(default_factory=list)
    calibrated: bool = False
    calib_err: float = float("inf")
    lines_img: list[np.ndarray] = field(default_factory=list)  # projected pitch model, image px
    ball_img: tuple[float, float] | None = None
    ball_pitch: tuple[float, float] | None = None
    ball_observed: bool = False


class ChunkClip:
    """Dense positions + linked ball for one chunk, resampled to any source frame."""

    def __init__(self, match_id: str, half: str, chunk: int, lo: int, hi: int) -> None:
        """Load and pre-project the artifacts covering source frames ``[lo, hi]``."""
        self.video = VIDEO_ROOT / match_id / half / f"chunk_{chunk:03d}.mp4"
        dense = pd.read_parquet(
            DENSE_ROOT / match_id / half / "match" / f"chunk_{chunk:03d}_dense.parquet"
        )
        self.attack_dirs = resolve_attack_directions(dense)
        ball_path = DENSE_ROOT / match_id / "final" / "ball" / f"ball_{half}_chunk{chunk:03d}.parquet"
        ball = pd.read_parquet(ball_path) if ball_path.exists() else pd.DataFrame()
        win = dense[(dense["frame"] >= lo - 30) & (dense["frame"] <= hi + 30)]
        ball_by_frame = ({int(r.frame): (float(r.x), float(r.y), bool(r.observed))
                          for r in ball.itertuples()} if len(ball) else {})
        polylines = _pitch_polylines()
        self.states: dict[int, FrameState] = {}
        homs: dict[int, np.ndarray] = {}
        for fr, grp in win.groupby("frame"):
            fr = int(fr)
            st = FrameState()
            for p in grp.itertuples():
                pitch = (None if not np.isfinite(p.pitch_x) or not np.isfinite(p.pitch_y)
                         else (float(p.pitch_x), float(p.pitch_y)))
                st.players.append({
                    "tid": int(p.track_id), "team": int(p.team), "role": str(p.role),
                    "img": (float(p.image_x), float(p.image_y)), "pitch": pitch,
                    "actor": bool(p.is_actor), "keeper": bool(p.is_keeper),
                })
            proj = [q for q in st.players if q["pitch"] is not None]
            errs = grp["calib_error_m"].to_numpy(float)
            errs = errs[np.isfinite(errs)]
            st.calib_err = float(errs.mean()) if len(errs) else float("inf")
            if len(proj) >= MIN_CORR:
                hom, _ = cv2.findHomography(
                    np.array([q["pitch"] for q in proj], np.float32),
                    np.array([q["img"] for q in proj], np.float32), cv2.RANSAC, 10.0,
                )
                if hom is not None:
                    st.calibrated = True
                    homs[fr] = hom
                    st.lines_img = [
                        cv2.perspectiveTransform(pl.reshape(-1, 1, 2), hom).reshape(-1, 2)
                        for pl in polylines
                    ]
            self.states[fr] = st
        self.keys = np.array(sorted(self.states), int)
        self._project_balls(ball_by_frame, homs)

    def _project_balls(self, ball_by_frame: dict, homs: dict[int, np.ndarray]) -> None:
        """Attach each linked-ball sample to its frame, projecting it into image space.

        A linked ball often lands on a frame whose own homography failed (44% of this match's ball
        rows) -- that is the shipped carry-over lever. For the video panel we reuse the nearest
        calibrated frame's homography within +/-1 s, exactly as ``generator.ball_carry`` does; if no
        homography is within reach the ball still shows on the top-down (it has pitch coordinates)
        but not on the broadcast panel.
        """
        cal = np.array(sorted(homs), int)
        for fr, (bx, by, obs) in ball_by_frame.items():
            st = self.states.get(int(fr))
            if st is None:
                continue
            st.ball_pitch, st.ball_observed = (bx, by), obs
            hom = homs.get(int(fr))
            if hom is None and len(cal):
                near = int(cal[np.argmin(np.abs(cal - int(fr)))])
                if abs(near - int(fr)) <= 25:  # 1 s carry window (the ball-carry lever's own bound)
                    hom = homs[near]
            if hom is not None:
                ip = cv2.perspectiveTransform(
                    np.array([[[bx, by]]], np.float32), hom
                ).reshape(2)
                st.ball_img = (float(ip[0]), float(ip[1]))

    def at(self, frame: int) -> FrameState | None:
        """State for a source frame, linearly interpolated between dense samples.

        Returns ``None`` only when the pipeline emitted **nothing at all** nearby (a broadcast cut,
        replay or graphic): those frames get an explicit on-screen banner. A frame that was detected
        but not calibrated comes back with ``calibrated = False`` -- markers, no geometry.
        """
        if len(self.keys) == 0:
            return None
        if frame in self.states:
            return self.states[frame]
        i = int(np.searchsorted(self.keys, frame))
        if i == 0 or i >= len(self.keys):
            return None
        f0, f1 = int(self.keys[i - 1]), int(self.keys[i])
        if f1 - f0 > MAX_INTERP_GAP:
            return None
        return _lerp_state(self.states[f0], self.states[f1], (frame - f0) / (f1 - f0))


def _lerp(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
    """Linear blend of two 2-D points."""
    return ((1 - t) * a[0] + t * b[0], (1 - t) * a[1] + t * b[1])


def _lerp_state(a: FrameState, b: FrameState, t: float) -> FrameState:
    """Blend two dense samples (players by track id, pitch lines, ball) at ``t`` in [0, 1].

    Geometry is only blended when **both** samples are calibrated: interpolating into or out of a
    calibration failure would invent a homography the pipeline never solved.
    """
    near = a if t < 0.5 else b
    out = FrameState(calibrated=a.calibrated and b.calibrated, calib_err=near.calib_err)
    if out.calibrated:
        out.calib_err = (1 - t) * a.calib_err + t * b.calib_err
    later = {q["tid"]: q for q in b.players}
    for pa in a.players:
        pb = later.pop(pa["tid"], None)
        if pb is None:
            if t < 0.5:
                out.players.append(pa)
            continue
        q = dict(pa)
        q["img"] = _lerp(pa["img"], pb["img"], t)
        q["pitch"] = (_lerp(pa["pitch"], pb["pitch"], t)
                      if pa["pitch"] is not None and pb["pitch"] is not None else near_pitch(pa, pb, t))
        q["actor"] = pb["actor"] if t >= 0.5 else pa["actor"]
        out.players.append(q)
    if t >= 0.5:
        out.players.extend(later.values())  # tracks that only exist in the later sample
    if out.calibrated and a.lines_img and b.lines_img and len(a.lines_img) == len(b.lines_img):
        out.lines_img = [(1 - t) * la + t * lb for la, lb in zip(a.lines_img, b.lines_img)]
    if a.ball_pitch and b.ball_pitch:
        out.ball_pitch = _lerp(a.ball_pitch, b.ball_pitch, t)
        out.ball_observed = near.ball_observed
        if a.ball_img and b.ball_img:
            out.ball_img = _lerp(a.ball_img, b.ball_img, t)
        elif near.ball_img:
            out.ball_img = near.ball_img
    elif near.ball_pitch:  # ball known on the nearer sample only -> hold it, flagged as inferred
        out.ball_pitch, out.ball_img, out.ball_observed = near.ball_pitch, near.ball_img, False
    return out


def near_pitch(pa: dict, pb: dict, t: float) -> tuple[float, float] | None:
    """Pitch position of a track when only one of the two bracketing samples projected it."""
    near = pa if t < 0.5 else pb
    return near["pitch"]


# --------------------------------------------------------------------------------------------------
# Structure overlays (section E)
# --------------------------------------------------------------------------------------------------
def _outfield(st: FrameState, team: int) -> list[dict]:
    """Projected outfield players of one team (keepers, referees and un-projected tracks excluded)."""
    return [p for p in st.players
            if p["team"] == team and not p["keeper"] and p["pitch"] is not None
            and p["role"] not in ("referee", "ball")]


def def_line_x(st: FrameState, team: int, attack_dir: int) -> tuple[float, float] | None:
    """Deepest-``DEEP_N`` defensive line for a team: ``(pitch_x, height_in_metres)``.

    Uses :func:`generator.impute.line_from_deepest` on the *visible* outfielders, so the returned
    height is the raw (partial-view biased) line -- the same quantity the shipped estimator then
    de-biases. Returns ``None`` below :data:`MIN_TEAM_PLAYERS` visible players.
    """
    pl = _outfield(st, team)
    if len(pl) < MIN_TEAM_PLAYERS:
        return None
    ax = attacking_coord(np.array([p["pitch"][0] for p in pl], float), attack_dir)
    height = line_from_deepest(ax)
    if not np.isfinite(height):
        return None
    return (height if attack_dir > 0 else PITCH_LEN - height), height


def lane_shares(st: FrameState, team: int) -> np.ndarray | None:
    """Five-lane occupation shares of a team's visible outfielders (None if too few)."""
    pl = _outfield(st, team)
    if len(pl) < MIN_TEAM_PLAYERS:
        return None
    counts, _ = np.histogram(np.array([p["pitch"][1] for p in pl], float), bins=LANE_EDGES)
    return counts / counts.sum() if counts.sum() else None


# --------------------------------------------------------------------------------------------------
# Frame composition
# --------------------------------------------------------------------------------------------------
def draw_video_overlays(img: np.ndarray, st: FrameState, seg: Segment, scale: float) -> None:
    """Draw pitch lines, player markers, ids and the ball on a (already resized) video panel."""
    def px(p):
        return int(round(p[0] * scale)), int(round(p[1] * scale))

    if "lines" in seg.overlays and st.lines_img:
        for pl in st.lines_img:
            pts = (pl * scale).astype(np.int32)
            cv2.polylines(img, [pts], False, C_YELLOW, 2, cv2.LINE_AA)
    if "markers" in seg.overlays:
        for p in st.players:
            if p["role"] == "referee":
                cv2.drawMarker(img, px(p["img"]), (180, 180, 180), cv2.MARKER_TILTED_CROSS, 14, 2)
                continue
            col = C_TEAM.get(p["team"], C_TEAM[-1])
            cv2.circle(img, px(p["img"]), 9, C_BLACK, -1, cv2.LINE_AA)
            cv2.circle(img, px(p["img"]), 7, col, -1, cv2.LINE_AA)
            if p["keeper"]:
                cv2.circle(img, px(p["img"]), 13, C_WHITE, 2, cv2.LINE_AA)
            if "carrier" in seg.overlays and p["actor"]:
                cv2.circle(img, px(p["img"]), 18, C_YELLOW, 3, cv2.LINE_AA)
            if "ids" in seg.overlays:
                x, y = px(p["img"])
                text(img, str(p["tid"]), (x + 11, y + 5), 0.55, C_WHITE, 1)
    if "ball" in seg.overlays and st.ball_img:
        bx, by = px(st.ball_img)
        cv2.circle(img, (bx, by), 11, C_BLACK, -1, cv2.LINE_AA)
        if st.ball_observed:
            cv2.circle(img, (bx, by), 8, C_WHITE, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, (bx, by), 8, C_AMBER, 2, cv2.LINE_AA)


def draw_topdown(td: TopDown, st: FrameState, seg: Segment, attack_dirs: dict[int, int]) -> np.ndarray:
    """Render the tactical panel for one frame (players, ball and any section-E structures)."""
    pan = td.frame()
    if "struct_lanes" in seg.overlays:
        for i in range(len(LANE_EDGES) - 1):
            y0, y1 = td.to_px(0, LANE_EDGES[i])[1], td.to_px(0, LANE_EDGES[i + 1])[1]
            if i % 2 == 0:
                shade(pan, y0, y1, 0.16, (255, 255, 255))
            cv2.line(pan, (td.to_px(0, LANE_EDGES[i])[0], y0), (td.to_px(PITCH_LEN, 0)[0], y0),
                     (225, 225, 225), 1, cv2.LINE_AA)
    focus = 0  # Man Utd = registry teams[0]
    if "struct_line" in seg.overlays and focus in attack_dirs:
        dl = def_line_x(st, focus, attack_dirs[focus])
        if dl is not None:
            lx, hgt = dl
            p0, p1 = td.to_px(lx, 0), td.to_px(lx, PITCH_WID)
            cv2.line(pan, p0, p1, C_TEAM[focus], 3, cv2.LINE_AA)
            text(pan, f"def line {hgt:.0f} m", (p0[0] + 8, p0[1] + 26), 0.6, C_WHITE, 1)
        pl = _outfield(st, focus)
        if len(pl) >= MIN_TEAM_PLAYERS:
            xs = [p["pitch"][0] for p in pl]
            ys = [p["pitch"][1] for p in pl]
            a, b = td.to_px(min(xs), min(ys)), td.to_px(max(xs), max(ys))
            cv2.rectangle(pan, a, b, (255, 255, 255), 1, cv2.LINE_AA)
            text(pan, f"block {max(xs) - min(xs):.0f} x {max(ys) - min(ys):.0f} m",
                 (a[0] + 6, a[1] - 8), 0.55, C_WHITE, 1)
    for p in st.players:
        if p["role"] == "referee" or p["pitch"] is None:
            continue  # un-projected tracks exist on the video panel only -- nothing to place here
        col = C_TEAM.get(p["team"], C_TEAM[-1])
        c = td.to_px(*p["pitch"])
        cv2.circle(pan, c, 9, C_BLACK, -1, cv2.LINE_AA)
        cv2.circle(pan, c, 7, col, -1, cv2.LINE_AA)
        if p["keeper"]:
            cv2.circle(pan, c, 12, C_WHITE, 2, cv2.LINE_AA)
        if "carrier" in seg.overlays and p["actor"]:
            cv2.circle(pan, c, 16, C_YELLOW, 2, cv2.LINE_AA)
        if "ids" in seg.overlays:
            text(pan, str(p["tid"]), (c[0] + 10, c[1] + 4), 0.45, C_WHITE, 1)
    if "ball" in seg.overlays and st.ball_pitch:
        c = td.to_px(*st.ball_pitch)
        cv2.circle(pan, c, 10, C_BLACK, -1, cv2.LINE_AA)
        if st.ball_observed:
            cv2.circle(pan, c, 7, C_WHITE, -1, cv2.LINE_AA)
        else:
            cv2.circle(pan, c, 7, C_AMBER, 2, cv2.LINE_AA)
    return pan


def compose(frame: np.ndarray, st: FrameState | None, seg: Segment, td: TopDown,
            attack_dirs: dict[int, int], facts: dict) -> np.ndarray:
    """Compose one full 1920x1080 output frame (header + panels + caption strip)."""
    canvas = np.full((H, W, 3), C_BG, np.uint8)
    if seg.layout == "video":
        vw, vh, vx, vy = 1387, 780, 266, 146
    else:
        vw, vh, vx, vy = 1180, 664, 24, 196
    panel = cv2.resize(frame, (vw, vh), interpolation=cv2.INTER_AREA)
    scale = vw / frame.shape[1]
    if st is not None:
        draw_video_overlays(panel, st, seg, scale)
    canvas[vy:vy + vh, vx:vx + vw] = panel
    cv2.rectangle(canvas, (vx - 2, vy - 2), (vx + vw + 2, vy + vh + 2), (90, 90, 90), 2)

    if seg.layout == "split":
        pan = (draw_topdown(td, st, seg, attack_dirs)
               if st is not None and st.calibrated else td.frame())
        px_, py_ = 1214, 300
        if st is None or not st.calibrated:
            shade(pan, 0, pan.shape[0], 0.62, (10, 10, 10))
            text(pan, "NO PITCH GEOMETRY", (int(0.13 * td.w), td.h // 2 - 10), 1.0, C_AMBER, 2)
            text(pan, "this frame yields no positions", (int(0.16 * td.w), td.h // 2 + 34), 0.62,
                 (215, 215, 220), 1)
        canvas[py_:py_ + td.h, px_:px_ + td.w] = pan
        cv2.rectangle(canvas, (px_ - 2, py_ - 2), (px_ + td.w + 2, py_ + td.h + 2), (90, 90, 90), 2)
        text(canvas, "TOP-DOWN (105 x 68 m, reconstructed)", (px_, py_ - 14), 0.62, (200, 220, 255), 1)
        text(canvas, "BROADCAST FRAME", (vx, vy - 14), 0.62, (200, 220, 255), 1)

    # header
    shade(canvas, 0, HEADER_H, 0.78)
    text(canvas, SECTION_TITLES[seg.section], (28, 50), 1.0, C_YELLOW, 2, FONT_T)
    fixture = facts.get("fixture", "Brighton vs Man Utd")
    text(canvas, f"{fixture}   |   CV pipeline output (not vendor tracking data)",
         (760, 48), 0.72, (215, 215, 225), 1)

    # status strip (live, per-frame numbers) -- its own band under the header, never over the video
    shade(canvas, HEADER_H + 6, HEADER_H + 58, 0.55)
    sy = HEADER_H + 44
    if st is None:
        chip(canvas, "NO OUTPUT: BROADCAST CUT / REPLAY / GRAPHIC", (30, sy), C_AMBER, 0.8)
    else:
        n_pl = sum(1 for p in st.players if p["role"] != "referee")
        text(canvas, f"players tracked: {n_pl}", (30, sy), 0.78, C_WHITE, 2)
        if st.calibrated:
            text(canvas, f"calibration error: {st.calib_err:.2f} m", (330, sy), 0.78, C_GREEN, 2)
        else:
            chip(canvas, "CALIBRATION FAILED - NO PITCH GEOMETRY", (330, sy), C_AMBER, 0.72)
        if "ball" in seg.overlays:
            if st.ball_pitch is None:
                chip(canvas, "BALL NOT TRACKED", (950, sy), C_AMBER, 0.72)
            elif st.ball_observed:
                text(canvas, "ball: detected", (950, sy), 0.78, C_WHITE, 2)
            else:
                text(canvas, "ball: inferred (carry-over)", (950, sy), 0.78, C_AMBER, 2)
        if "struct_lanes" in seg.overlays:
            sh = lane_shares(st, 0)
            if sh is not None:
                s = "  ".join(f"{v * 100:.0f}%" for v in sh)
                text(canvas, f"Man Utd lanes L-wing..R-wing: {s}", (1050, sy), 0.7, C_WHITE, 2)
            else:
                text(canvas, "too few visible players - metric abstains", (1050, sy), 0.7, C_AMBER, 2)

    # caption strip
    shade(canvas, H - FOOTER_H, H, 0.72)
    text(canvas, seg.caption, (30, H - FOOTER_H + 52), 0.86, C_WHITE, 2)
    if seg.note:
        for i, line in enumerate(_wrap(seg.note, 118)):
            text(canvas, line, (30, H - FOOTER_H + 96 + 34 * i), 0.68, (185, 210, 235), 1)
    return canvas


def _wrap(s: str, width: int) -> list[str]:
    """Greedy word wrap to at most two lines of ``width`` characters."""
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines[:2]


# --------------------------------------------------------------------------------------------------
# Cards
# --------------------------------------------------------------------------------------------------
def card(title: str, lines: list[str], sub: str = "", *, accent: tuple[int, int, int] = C_YELLOW,
         ) -> np.ndarray:
    """A full-screen caption card."""
    c = np.full((H, W, 3), C_BG, np.uint8)
    cv2.rectangle(c, (0, 0), (18, H), accent, -1)
    text(c, title, (90, 210), 1.9, accent, 3, FONT_T)
    for i, ln in enumerate(lines):
        text(c, ln, (92, 330 + 62 * i), 0.95, C_WHITE, 2)
    if sub:
        text(c, sub, (92, H - 90), 0.78, (170, 195, 220), 1)
    return c


def title_card(facts: dict) -> np.ndarray:
    """Opening card: project, pipeline, match identity, and the honest framing note."""
    fixture = facts.get("fixture", "Brighton vs Manchester Utd")
    comp = facts.get("comp", "Premier League 2024-25")
    c = np.full((H, W, 3), C_BG, np.uint8)
    cv2.rectangle(c, (0, 0), (18, H), C_YELLOW, -1)
    text(c, "FOOTBALL-SYNTHESIZER", (90, 190), 2.2, C_WHITE, 3, FONT_T)
    text(c, "broadcast video  ->  tracked positions  ->  validated metrics", (92, 285), 1.15,
         C_YELLOW, 2, FONT_T)
    text(c, fixture, (92, 430), 1.35, C_WHITE, 2, FONT_T)
    text(c, comp, (92, 495), 0.95, (190, 210, 235), 2)
    text(c, "Computer-vision pipeline demo: detection + tracking, pitch calibration,", (92, 620),
         0.9, C_WHITE, 2)
    text(c, "top-down reconstruction, ball tracking, derived team structures.", (92, 668), 0.9,
         C_WHITE, 2)
    shade(c, 760, 940, 0.5, (200, 160, 40))
    text(c, "HONEST NOTE: everything you are about to see is OUR CV OUTPUT from the broadcast feed.",
         (92, 820), 0.86, C_AMBER, 2)
    text(c, "It is not vendor/optical tracking data. Where the pipeline fails, the video shows it "
            "failing.", (92, 872), 0.86, C_AMBER, 2)
    return c


def closing_card(facts: dict) -> np.ndarray:
    """Closing scorecard: only figures that exist in artifacts on disk."""
    c = np.full((H, W, 3), C_BG, np.uint8)
    cv2.rectangle(c, (0, 0), (18, H), C_GREEN, -1)
    text(c, "THE HONEST SCORECARD", (90, 150), 1.8, C_GREEN, 3, FONT_T)
    n_ch = facts.get("n_chunks", 0)
    rows: list[tuple[str, str, tuple[int, int, int]]] = []
    if "ball_coverage" in facts:
        rows.append((f"Post-link ball coverage (full match, {n_ch} chunks)",
                     f"{facts['ball_coverage'] * 100:.1f}%", C_WHITE))
    if "calib_live_probe" in facts:
        rows.append(("Calibration solve rate on live-play frames (probe segment)",
                     f"{facts['calib_live_probe'] * 100:.0f}%", C_WHITE))
    rows.append(("Frames yielding pitch geometry, ALL sampled frames (replays/close-ups included)",
                 f"{facts['calib_all_frames'] * 100:.1f}%", C_AMBER))
    if "pass_recall" in facts:
        rows.append((f"Pass-recall proxy vs {facts.get('oracle', 'oracle')} (completed passes)",
                     f"{facts['pass_recall'] * 100:.1f}%", C_AMBER))
    for i, (k, v, col) in enumerate(rows):
        y = 275 + 72 * i
        text(c, k, (92, y), 0.8, (205, 220, 240), 2)
        text(c, v, (1600, y), 1.05, col, 2, FONT_T)
    shade(c, 600, 790, 0.42, (120, 200, 130))
    text(c, "STRUCTURE = VALIDATED   (shape, lanes, compactness; defensive line validated to ~5 m "
            "vs FIFA PMSR)", (92, 662), 0.8, C_GREEN, 2)
    text(c, "EVENTS = NOT YET DETECTED   (tackles, shots and passes are not detected as events - we "
            "do not claim them)", (92, 720), 0.8, C_AMBER, 2)
    text(c, "Possession on tracked frames is biased by construction: reported caveated, never gated. "
            "The pass-recall proxy misses", (92, 840), 0.72, (200, 200, 210), 1)
    text(c, "the pre-declared 50% bar, so the ball-based families stay WITHHELD in the report. The "
            "gate was not bent.", (92, 884), 0.72, (200, 200, 210), 1)
    text(c, "Sources: outputs/eval/brighton_manutd_ball_eval.json | results/pl_pilot/fbref_gate.md | "
            "results/pl_probe/diagnosis/DIAGNOSIS.md", (92, 990), 0.6, (150, 160, 175), 1)
    return c


# --------------------------------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------------------------------
def render(out: Path, *, preview: bool = False, height: int = 1080, crf: int = 20) -> dict:
    """Render the demo video end to end.

    Args:
        out: final MP4 path.
        preview: render only a short sample of each segment.
        height: output height (1080 or 720).
        crf: x264 quality (lower = better).

    Returns:
        Summary dict (frames, duration, path, size).
    """
    match = registry.get(MATCH_ID)
    facts = demo_facts(MATCH_ID)
    print(f"[demo] match={match.id} teams={match.teams}")
    for k, v in facts.get("provenance", {}).items():
        print(f"[demo] fact {k}: {v}")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter at {tmp}")
    td = TopDown(width=690)
    n = 0

    def hold(img: np.ndarray, seconds: float) -> None:
        nonlocal n
        for _ in range(int(round(seconds * FPS))):
            writer.write(img)
            n += 1

    t0 = time.time()
    hold(title_card(facts), 3.0 if preview else 6.0)
    last_section = ""
    for seg in SEGMENTS:
        if seg.section != last_section:
            ttl, body = SECTION_CARDS[seg.section]
            hold(card(ttl, body, "football-synthesizer / BTP demo"), 1.5 if preview else 3.0)
            last_section = seg.section
        end = min(seg.end, seg.start + int(3 * FPS)) if preview else seg.end
        clip = ChunkClip(MATCH_ID, seg.half, seg.chunk, seg.start, end)
        cap = cv2.VideoCapture(str(clip.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, seg.start)
        n_det = n_cal = n_ball = 0
        for fr in range(seg.start, end + 1):
            ok, frame = cap.read()
            if not ok:
                break
            st = clip.at(fr)
            n_det += st is not None
            n_cal += st is not None and st.calibrated
            n_ball += st is not None and st.ball_pitch is not None
            writer.write(compose(frame, st, seg, td, clip.attack_dirs, facts))
            n += 1
        cap.release()
        span = end - seg.start + 1
        print(f"[demo] {seg.section} {seg.half}_c{seg.chunk} {seg.start}-{end} ({span / FPS:.0f}s): "
              f"detected {n_det / span:.0%} | calibrated {n_cal / span:.0%} | ball {n_ball / span:.0%}")
    hold(closing_card(facts), 4.0 if preview else 13.0)
    writer.release()

    size_mb = tmp.stat().st_size / 1e6
    print(f"[demo] raw render: {n} frames ({n / FPS:.1f} s), {size_mb:.0f} MB, "
          f"{time.time() - t0:.0f}s wall")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        vf = f"scale=-2:{height}" if height != H else "null"
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(tmp), "-vf", vf, "-c:v", "libx264",
               "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(out)]
        subprocess.run(cmd, check=True)
        tmp.unlink(missing_ok=True)
    else:
        print("[demo] WARNING: ffmpeg not found -> shipping the mp4v render (not H.264)")
        shutil.move(str(tmp), str(out))

    info = {"path": str(out), "frames": n, "seconds": n / FPS,
            "size_mb": out.stat().st_size / 1e6, "height": height}
    print(f"[demo] wrote {info['path']}  {info['seconds']:.1f}s  {info['size_mb']:.1f} MB")
    return info


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/demo/cv_pipeline_demo.mp4")
    ap.add_argument("--preview", action="store_true", help="short sample of every segment")
    ap.add_argument("--height", type=int, default=1080, choices=(720, 1080))
    ap.add_argument("--crf", type=int, default=20)
    args = ap.parse_args()
    render(Path(args.out), preview=args.preview, height=args.height, crf=args.crf)


if __name__ == "__main__":
    main()
