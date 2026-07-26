"""Tactical clip generator: annotated broadcast + top-down board, with uncertainty drawn.

The analyst-breakdown deliverable. For one passage of play it renders a two-panel video:

* **Left** -- the broadcast frame with every tracked player marked, team-coloured and labelled by
  role band (GK / DEF / MID / ATT, from the player's median position along its own team's attack
  direction over the passage). A **name** is drawn only where identity is gated (see
  :data:`MIN_ANCHORS`); everything else stays anonymous behind its band.
* **Right** -- the 105 x 68 m tactical board: observed players as solid markers, the linked ball,
  the attack arrows, and -- the point of the whole thing -- the players we **cannot** see drawn as
  calibrated uncertainty, never as confident dots.

Three visual states, and they are the argument:

===============  ================================================================================
state            how it is drawn
===============  ================================================================================
observed         solid team-coloured disc; this frame projected it through a fitted homography
imputed          hollow diamond inside two nested ellipses = the frozen B4 v1 model's calibrated
                 50% / 90% predictive regions (``synthesizer.imputation_v1.emit`` -> ``source``
                 ``"v1"`` or ``"anchor"``). Tight blob at short occlusion, wide smear at long.
abstained        faded ghost at the last-seen position with "last seen N s ago" -- the model
                 declined to assert (region wider than the frozen SGR threshold ``R_MAX``)
===============  ================================================================================

Honesty constraints enforced in code, not in prose:

* the imputation model is the **frozen** B4 v1 (Metrica-trained; ``results/B4_MODEL_V1.md``) with
  the frozen CALIB conformal multipliers and the adopted P2 defer-to-anchor policy
  (``results/B4_ABSTENTION_POLICY.md``). Nothing here refits anything.
* it has **no ground truth on broadcast footage** -- ``results/B4_TRANSFER_M3.md`` calls the
  application to our own tracks a smell test. The banner on every frame says so.
* a passage below :data:`MIN_PLAYERS` gated players/frame or :data:`MIN_BALL_COV` ball coverage is
  **refused**, with the numbers that failed.

Run::

    python tools/tactical_clip.py --match manutd_liverpool --shortlist
    python tools/tactical_clip.py --match manutd_liverpool --half h1 --start 612 --end 626
    python tools/tactical_clip.py --match manutd_liverpool --auto 1
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

import make_demo as md  # noqa: E402  (reuse the demo's drawing machinery, not a copy of it)
from core import registry  # noqa: E402
from core.pitch import ATT_THIRD_X, PITCH_LEN, PITCH_WID  # noqa: E402
from fingerprint.block_height import (  # noqa: E402
    HIGH_BLOCK_MIN_M,
    LOW_BLOCK_MAX_M,
    _classify,
    block_frames,
)
from fingerprint.structural_metrics import (  # noqa: E402
    attacking_coord,
    resolve_attack_directions,
    resolve_attack_directions_from_ball,
)
from fingerprint.theory_metrics import complete_directions  # noqa: E402

OUT_ROOT = Path("results/tactical_clips")
CACHE_DIR = "data/imputation/cache"
IDENTITY_ROOT = Path("outputs/identity")
CALIB_MAX_M = 1.0  # trusted-geometry gate; identical to report/facts and fingerprint.block_height

# ---- frozen B4 v1 constants (results/B4_MODEL_V1_runlog.md + B4_ABSTENTION_runlog.md) ------------
# Per-horizon-bucket split-conformal multipliers on the quantile-head half-widths. Buckets are
# synthesizer.imputation.BIN_LABELS = 0-1s / 1-3s / 3-5s / 5-10s / 10-30s / 30s+.
K50_V1 = np.array([0.523, 0.758, 0.756, 0.791, 0.853, 0.914])
K90_V1 = np.array([1.335, 1.603, 1.583, 1.630, 1.816, 1.953])
K50_ANC = np.array([0.431, 0.827, 0.918, 0.985, 1.307, 1.639])
K90_ANC = np.array([1.351, 1.792, 1.864, 1.995, 2.699, 2.871])
R_MAX = 30.4339                      # frozen SGR threshold (layer B), metres
SKILL_BUCKETS = (1, 2, 3, 4, 5)      # buckets where v1 beat the anchor on CALIB; 0-1s defers

# ---- render / selection knobs -------------------------------------------------------------------
FPS = 25.0
W, H = md.W, md.H
MIN_PLAYERS = 12.0      # mean gated players/frame below which a passage is refused
MIN_BALL_COV = 0.60     # linked-ball coverage below which a passage is refused
MIN_FRAME_COV = 0.85    # fraction of a passage's frames that must yield trusted geometry
MIN_POSS_FRAC = 0.60    # modal possession share below which the passage is called "contested"
MIN_ANCHORS = 3         # close-up name reads agreeing on a track fragment before we print a name
MAX_IMPUTE_S = 20.0     # never draw an imputed player hidden longer than this (re-id churn)
SQUAD = 11              # observed + imputed markers per team are capped at a legal XI
DUP_M = 6.0             # a ghost this close to a live same-team track is that track, re-identified
WIN_S = 14.0            # shortlist window length
STRIDE_S = 7.0          # shortlist window stride

C_TEAM = md.C_TEAM
C_WHITE, C_BLACK, C_YELLOW = md.C_WHITE, md.C_BLACK, md.C_YELLOW
C_AMBER, C_RED, C_GREEN = md.C_AMBER, md.C_RED, md.C_GREEN
C_BG, FONT, FONT_T = md.C_BG, md.FONT, md.FONT_T
BAND_ORDER = ("GK", "DEF", "MID", "ATT")


def _ascii(s: str) -> str:
    """Strip accents so cv2's Hershey fonts (and the cp1252 console) can render a name."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def _clock(t_s: float) -> str:
    """Seconds into a half -> ``MM:SS``."""
    return f"{int(t_s) // 60:02d}:{int(t_s) % 60:02d}"


def block_class(line_m: float) -> str | None:
    """Low/mid/high class of a de-biased block line, or ``None`` when the line is off the pitch.

    ``generator.impute.line_estimates`` de-biases the visible defensive line, and on a passage
    where the broadcast shows too little of the block that correction can push the estimate behind
    its own goal line (a real read of -7 m turned up in ``fulham_manutd``). Such a line is not a
    "low block", it is a failed measurement, and the caption must say nothing rather than something
    wrong. Gate 1 for block height is still pending (``fingerprint.block_height``), so even an
    in-range class is descriptive, not certified.

    Args:
        line_m: Median de-biased line height over the passage (own goal = 0 m).

    Returns:
        ``"low" | "mid" | "high"``, or ``None`` if the line is not on the pitch.
    """
    if not np.isfinite(line_m) or not 0.0 <= line_m <= PITCH_LEN:
        return None
    return _classify(line_m)


def phase_of(own_m: float, poss_frac: float, poss_team: int) -> tuple[str, int | None]:
    """Phase label from ball advancement, suppressed when possession itself is unstable.

    ``generator.ball.assign_possession`` is a *proximity* proxy: the nearest player to the ball
    owns it. In a crowded box -- a corner, a scramble -- the modal owner flips every few frames and
    a phase label built on it ("Man Utd build-up") is worse than no label. When the modal team
    holds less than :data:`MIN_POSS_FRAC` of the passage's possession frames we say so instead.

    Args:
        own_m: Ball advancement with the possessing team's own goal at 0 m.
        poss_frac: Share of the passage's possession frames held by the modal team.
        poss_team: The modal possessing team id.

    Returns:
        Tuple ``(phase label, possessing team or None)``.
    """
    if poss_frac < MIN_POSS_FRAC:
        return "contested (possession flips)", None
    if own_m < PITCH_LEN / 3:
        return "build-up", poss_team
    return ("progression" if own_m < ATT_THIRD_X else "attack"), poss_team


# --------------------------------------------------------------------------------------------------
# Time <-> chunk mapping
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ChunkIndex:
    """Cumulative time index over one half's chunk videos.

    The registry stores chunks, not a match clock; every chunk's true duration is read from its own
    container so a short final chunk cannot silently shift the clock.
    """

    half: str
    keys: tuple[str, ...]
    starts: tuple[float, ...]
    durations: tuple[float, ...]
    fps: tuple[float, ...]

    @property
    def total_s(self) -> float:
        """Length of the half, in seconds of chunked video."""
        return sum(self.durations)

    def locate(self, t_s: float) -> tuple[int, int]:
        """Half-seconds -> ``(chunk position, source frame)``, clamped inside the chunk.

        The frame is clamped to the chunk's own last index, not just to the half: rounding at the
        very end of a chunk otherwise yields one frame past the file and ``cap.read()`` fails.
        """
        t = float(np.clip(t_s, 0.0, self.total_s - 1e-3))
        i = int(np.searchsorted(np.asarray(self.starts), t, side="right") - 1)
        i = int(np.clip(i, 0, len(self.keys) - 1))
        last = max(int(self.durations[i] * self.fps[i]) - 1, 0)
        return i, min(int(round((t - self.starts[i]) * self.fps[i])), last)

    def to_half_s(self, i: int, frame: int) -> float:
        """``(chunk position, source frame)`` -> seconds into the half."""
        return self.starts[i] + frame / self.fps[i]


def chunk_index(match: registry.Match, half: str) -> ChunkIndex:
    """Build the cumulative chunk-time index for one half of a match."""
    keys = [k for k, _ in match.ball_chunks() if k.startswith(half)]
    if not keys:
        raise SystemExit(f"no linked-ball chunks for {match.id} {half}")
    durs, fps = [], []
    for k in keys:
        video = registry.VIDEO_ROOT / match.id / half / f"chunk_{k.split('_')[-1]}.mp4"
        cap = cv2.VideoCapture(str(video))
        n, f = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        f = f if f and f > 1.0 else match.chunk_fps(k)
        durs.append((n / f) if n > 0 else 600.0)
        fps.append(f)
    starts = np.concatenate([[0.0], np.cumsum(durs)[:-1]])
    return ChunkIndex(half, tuple(keys), tuple(starts.tolist()), tuple(durs), tuple(fps))


# --------------------------------------------------------------------------------------------------
# Passage assembly
# --------------------------------------------------------------------------------------------------
@dataclass
class Mark:
    """One player marker for one render frame."""

    tid: int
    team: int
    band: str
    name: str | None
    img: tuple[float, float] | None
    pitch: tuple[float, float] | None
    keeper: bool


@dataclass
class Ghost:
    """One un-observed player: the frozen v1's emitted position plus its calibrated regions."""

    tid: int
    team: int
    band: str
    pos: tuple[float, float]
    semi50: tuple[float, float]
    semi90: tuple[float, float]
    source: str          # "v1" | "anchor" | "abstained"
    tsls: float


@dataclass
class Situation:
    """The caption strip: what the detectors say about this passage."""

    poss_team: int | None = None
    poss_frac: float = 0.0
    phase: str = "unknown"
    block_class: str | None = None
    block_line_m: float = float("nan")
    ball_zone: str = "unknown"
    events: list[tuple[float, str, str]] = field(default_factory=list)  # (half_s, class, team)


@dataclass
class Quality:
    """Per-passage tracking-quality numbers -- the refusal test and the shortlist score."""

    players: float = 0.0
    both_teams: float = 0.0
    ball_cov: float = 0.0
    frame_cov: float = 0.0

    def refusal(self) -> str | None:
        """Why this passage must not be rendered, or ``None`` if it may be.

        Every rate here is measured against the number of samples the passage *should* have had
        (its length at the source sampling rate), not against the samples that survived the gate --
        a window that lost 40% of its frames to a replay cut must not score as if it were clean.
        """
        if self.frame_cov < MIN_FRAME_COV:
            return (f"broken passage: only {self.frame_cov:.0%} of its frames yield trusted "
                    f"geometry (need >= {MIN_FRAME_COV:.0%}) -- replay, cut or calibration failure")
        if self.players < MIN_PLAYERS:
            return (f"tracking too sparse: {self.players:.1f} gated players/frame "
                    f"(need >= {MIN_PLAYERS:.0f})")
        if self.ball_cov < MIN_BALL_COV:
            return (f"ball not tracked enough: {self.ball_cov:.0%} of frames "
                    f"(need >= {MIN_BALL_COV:.0%})")
        if self.both_teams < 0.5:
            return (f"only one team on screen: both teams >= 4 players on {self.both_teams:.0%} "
                    "of frames (need >= 50%)")
        return None


def load_names(match: registry.Match) -> dict[tuple[str, int], str]:
    """``(chunk, track_id) -> player name`` for the tracks whose identity clears the gate.

    Two gates, both cheap and both necessary given the 35% per-moment attribution precision
    (``results/CARRIER_CONSTRAINED_v2.md``): the track fragment needs at least
    :data:`MIN_ANCHORS` agreeing close-up name reads, and the name must appear on the match's own
    team sheet. Everything that fails stays anonymous behind its role band.

    Args:
        match: Registry match.

    Returns:
        Mapping from ``(chunk, track_id)`` to a printable ASCII name; empty when the match has no
        identity artifacts.
    """
    named = IDENTITY_ROOT / f"{match.id}_named_tracks_both2_prtreid.parquet"
    lineup = IDENTITY_ROOT / f"{match.id}_lineup_assign.parquet"
    if not named.exists():
        return {}
    nt = pd.read_parquet(named)
    nt = nt[nt["n_anchors"] >= MIN_ANCHORS]
    if lineup.exists():
        sheet = set(pd.read_parquet(lineup)["name"].astype(str))
        nt = nt[nt["player_name"].astype(str).isin(sheet)]
    return {(str(r.chunk), int(r.track_id)): _ascii(str(r.player_name)) for r in nt.itertuples()}


def role_bands(win: pd.DataFrame, dirs: dict[int, int]) -> dict[int, str]:
    """``track_id -> GK/DEF/MID/ATT`` from the track's median attacking-x over the passage.

    A band, not a position: it is the third of the pitch the track occupied, measured toward its own
    team's attacking goal, so it survives the camera panning. Per passage (not per frame) so the
    label does not flicker while a player jogs across a third boundary.

    Args:
        win: Gated position rows for the passage.
        dirs: ``{team: +1/-1}`` attacking directions.

    Returns:
        Band per track id; tracks of a team with no resolved direction are omitted.
    """
    out: dict[int, str] = {}
    for tid, g in win.groupby("track_id"):
        team = int(g["team"].iloc[0])
        if team not in dirs:
            continue
        if bool(g["is_keeper"].mean() >= 0.5) or str(g["role"].iloc[0]) == "goalkeeper":
            out[int(tid)] = "GK"
            continue
        ax = float(np.median(attacking_coord(g["pitch_x"].to_numpy(float), dirs[team])))
        out[int(tid)] = "DEF" if ax < PITCH_LEN / 3 else ("MID" if ax < ATT_THIRD_X else "ATT")
    return out


class Passage:
    """Everything drawable for one passage: frames, marks, ball, situation, quality, ghosts."""

    def __init__(self, match: registry.Match, half: str, start_s: float, end_s: float,
                 *, impute: bool = True) -> None:
        """Assemble a passage from the persisted artifacts (nothing is recomputed on the GPU).

        Args:
            match: Registry match.
            half: ``"h1"`` or ``"h2"``.
            start_s: Passage start, seconds into the half.
            end_s: Passage end, seconds into the half.
            impute: Run the frozen v1 imputation (the slow part; off for the shortlist).
        """
        self.match, self.half = match, half
        self.idx = chunk_index(match, half)
        ci, f_lo = self.idx.locate(start_s)
        cj, f_hi = self.idx.locate(end_s)
        if cj != ci:  # a passage never straddles a chunk cut -- clip it to the first chunk
            f_hi = int(self.idx.durations[ci] * self.idx.fps[ci]) - 1
        self.ci, self.f_lo, self.f_hi = ci, f_lo, max(f_hi, f_lo + 1)
        self.chunk = self.idx.keys[ci]
        self.fps = self.idx.fps[ci]
        self.start_s = self.idx.to_half_s(ci, self.f_lo)
        self.end_s = self.idx.to_half_s(ci, self.f_hi)
        self.video = registry.VIDEO_ROOT / match.id / half / f"chunk_{self.chunk.split('_')[-1]}.mp4"

        aligned = match.load_aligned()
        pos_chunk = aligned[aligned["chunk"] == self.chunk].dropna(subset=["pitch_x", "pitch_y"])
        self.ball = pd.read_parquet(dict(match.ball_chunks())[self.chunk])
        self.dirs = complete_directions(
            resolve_attack_directions_from_ball(pos_chunk, self.ball)
            or resolve_attack_directions(pos_chunk)
        )
        self.gated = pos_chunk[(pos_chunk["calib_error_m"] <= CALIB_MAX_M)
                               & pos_chunk["role"].isin(["player", "goalkeeper"])]
        fr_all = np.sort(pos_chunk["frame"].unique())
        self.step = int(np.median(np.diff(fr_all))) if fr_all.size > 1 else 1
        self.win = self.gated[(self.gated["frame"] >= self.f_lo)
                              & (self.gated["frame"] <= self.f_hi)]
        self.bands = role_bands(self.win, self.dirs)
        self.names = load_names(match)
        self.states = self._states()
        self.keys = np.array(sorted(self.states), int)
        self.quality = self._quality()
        self.situation = self._situation()
        self.ghosts: dict[int, list[Ghost]] = {}
        self.impute_note = "imputation off"
        if impute:
            self.ghosts, self.impute_note = self._impute()

    # -- per-frame state ---------------------------------------------------------------------
    def _states(self) -> dict[int, md.FrameState]:
        """Build a :class:`make_demo.FrameState` per sampled frame, with a refitted homography."""
        polylines = md._pitch_polylines()
        pad = int(round(2.0 * self.fps))  # a little context so interpolation has both endpoints
        sub = self.gated[(self.gated["frame"] >= self.f_lo - pad)
                         & (self.gated["frame"] <= self.f_hi + pad)]
        by_frame = {int(f): g for f, g in sub.groupby("frame")}
        states: dict[int, md.FrameState] = {}
        homs: dict[int, np.ndarray] = {}
        for fr, grp in by_frame.items():
            st = md.FrameState()
            for p in grp.itertuples():
                st.players.append({
                    "tid": int(p.track_id), "team": int(p.team), "role": str(p.role),
                    "img": (float(p.image_x), float(p.image_y)),
                    "pitch": (float(p.pitch_x), float(p.pitch_y)),
                    "actor": bool(p.is_actor), "keeper": bool(p.is_keeper),
                })
            st.calib_err = float(grp["calib_error_m"].mean())
            if len(st.players) >= md.MIN_CORR:
                hom, _ = cv2.findHomography(
                    np.array([q["pitch"] for q in st.players], np.float32),
                    np.array([q["img"] for q in st.players], np.float32), cv2.RANSAC, 10.0)
                if hom is not None:
                    st.calibrated = True
                    homs[fr] = hom
                    st.lines_img = [cv2.perspectiveTransform(pl.reshape(-1, 1, 2), hom).reshape(-1, 2)
                                    for pl in polylines]
            states[fr] = st
        cal = np.array(sorted(homs), int)
        for b in self.ball.itertuples(index=False):
            st = states.get(int(b.frame))
            if st is None:
                continue
            st.ball_pitch, st.ball_observed = (float(b.x), float(b.y)), bool(b.observed)
            hom = homs.get(int(b.frame))
            if hom is None and len(cal):
                near = int(cal[np.argmin(np.abs(cal - int(b.frame)))])
                if abs(near - int(b.frame)) <= self.fps:  # the shipped 1 s ball-carry window
                    hom = homs[near]
            if hom is not None:
                ip = cv2.perspectiveTransform(np.array([[[b.x, b.y]]], np.float32), hom).reshape(2)
                st.ball_img = (float(ip[0]), float(ip[1]))
        return states

    def at(self, frame: int) -> md.FrameState | None:
        """Interpolated state for a source frame (``None`` on a broadcast cut / replay)."""
        if len(self.keys) == 0:
            return None
        if frame in self.states:
            return self.states[frame]
        i = int(np.searchsorted(self.keys, frame))
        if i == 0 or i >= len(self.keys):
            return None
        f0, f1 = int(self.keys[i - 1]), int(self.keys[i])
        if f1 - f0 > md.MAX_INTERP_GAP:
            return None
        return md._lerp_state(self.states[f0], self.states[f1], (frame - f0) / (f1 - f0))

    def marks(self, st: md.FrameState) -> list[Mark]:
        """Turn a frame state into drawable marks (band + gated name attached)."""
        out = []
        for p in st.players:
            tid = int(p["tid"])
            out.append(Mark(
                tid=tid, team=int(p["team"]), band=self.bands.get(tid, "?"),
                name=self.names.get((self.chunk, tid)), img=p["img"], pitch=p["pitch"],
                keeper=bool(p["keeper"])))
        return out

    # -- quality + situation -----------------------------------------------------------------
    def _quality(self) -> Quality:
        """Measure the four gate numbers, all against the frames the passage *should* have."""
        expect = max(int((self.f_hi - self.f_lo) // self.step) + 1, 1)
        sampled = sorted(f for f in self.states if self.f_lo <= f <= self.f_hi
                         and self.states[f].calibrated)
        if not sampled:
            return Quality()
        counts, both = [], 0
        for f in sampled:
            st = self.states[f]
            counts.append(len(st.players))
            n0 = sum(1 for p in st.players if p["team"] == 0)
            both += int(n0 >= 4 and len(st.players) - n0 >= 4)
        b = self.ball
        n_ball = int(((b["frame"] >= self.f_lo) & (b["frame"] <= self.f_hi)).sum())
        return Quality(players=float(np.mean(counts)), both_teams=both / expect,
                       ball_cov=min(n_ball / expect, 1.0), frame_cov=len(sampled) / expect)

    def _situation(self, blocks: pd.DataFrame | None = None) -> Situation:
        """Label the passage from the shipped detectors (possession, block class, zone, events)."""
        sit = Situation()
        bf = block_frames(self.match) if blocks is None else blocks
        if not bf.empty:
            w = bf[(bf["chunk"] == self.chunk) & (bf["frame"] >= self.f_lo)
                   & (bf["frame"] <= self.f_hi)]
            if not w.empty:
                deft = int(w["def_team"].mode().iloc[0])
                sit.poss_frac = float((w["def_team"] == deft).mean())
                usable = w[w["usable"]]
                if len(usable) >= 3:
                    line = float(usable["line_m"].median())
                    sit.block_class = block_class(line)
                    sit.block_line_m = line if sit.block_class else float("nan")
                adv = float(w["ball_adv_m"].median())  # defender's own-goal = 0
                own = PITCH_LEN - adv                  # -> possessing team's own-goal = 0
                sit.ball_zone = ("own third" if own < PITCH_LEN / 3
                                 else "middle third" if own < ATT_THIRD_X else "final third")
                sit.phase, sit.poss_team = phase_of(own, sit.poss_frac, 1 - deft)
        ledger = Path("outputs") / self.match.id / "ledger.parquet"
        if ledger.exists():
            lg = pd.read_parquet(ledger)
            lg = lg[(lg["chunk"] == self.chunk) & (lg["frame_index"] >= self.f_lo)
                    & (lg["frame_index"] <= self.f_hi)]
            sit.events = [(self.idx.to_half_s(self.ci, int(r.frame_index)), str(r["class"]),
                           "" if pd.isna(r.team_name) else str(r.team_name))
                          for _, r in lg.iterrows()]
        return sit

    # -- imputation --------------------------------------------------------------------------
    def _impute(self) -> tuple[dict[int, list[Ghost]], str]:
        """Run the FROZEN B4 v1 over this passage's chunk and collect per-frame ghosts.

        The heads, the anchor, the conformal multipliers and the SGR threshold are all frozen
        artifacts of ``results/B4_MODEL_V1.md``; the emit policy is the adopted P2 defer-to-anchor
        of ``results/B4_ABSTENTION_POLICY.md``. Nothing is fitted here.

        Returns:
            ``({source frame: [Ghost, ...]}, note)`` -- the note is the on-screen provenance line.
        """
        from synthesizer.imputation import collect_samples, slot_prediction, visible_centroid
        from synthesizer.imputation_features import build_features, bucket_of, derive_fields
        from synthesizer.imputation_v1 import b7_position, emit, sample_signs, v1_prediction
        from tools.imputation_b4_external import (
            BAR_HALFLIFE_S, depth_rank, fit_frozen_v1, load_ours, remap_slot_coeffs,
        )
        from tools.imputation_b4_transfer import Tee

        out = Tee()
        heads, coeffs, tau, w7, g1 = fit_frozen_v1(CACHE_DIR, out)
        train_order = depth_rank(g1["truth"], g1["visible"], g1["period"], g1["ranges"])
        bundle = load_ours(self.match.id, 0, chunks=[self.chunk])
        ext_order = depth_rank(bundle["truth"], bundle["visible"], bundle["period"],
                               bundle["ranges"])
        coeffs_ext = remap_slot_coeffs(coeffs, train_order, ext_order)
        fields = derive_fields(bundle["truth"], bundle["visible"], bundle["ball"], bundle["cam"],
                               bundle["period"], bundle["fps"], bundle["ranges"], coeffs_ext,
                               BAR_HALFLIFE_S)
        samples = collect_samples(
            bundle["truth"], bundle["visible"], bundle["period"], bundle["fps"], bundle["vel"],
            slot_prediction(visible_centroid(bundle["truth"], bundle["visible"], bundle["ranges"]),
                            bundle["cam"], bundle["ranges"], coeffs_ext),
            fields={"b6": np.asarray(fields["b6"])})
        anchor = b7_position(samples, tau, w7)
        mu, halfw = v1_prediction(heads, build_features(fields, samples, anchor), anchor,
                                  sample_signs(fields, samples))
        bkt = bucket_of(samples["tsls"])
        geo = np.sqrt(halfw[:, 0] * halfw[:, 1])
        ems = emit(samples, mu, anchor, K90_V1[bkt] * geo, bkt, SKILL_BUCKETS, R_MAX,
                   K90_ANC[bkt] * geo)

        off, f0, step = bundle["frame_map"][1]
        slot_track = bundle["slot_track"][1]
        team_of = dict(self.win.groupby("track_id")["team"].first().astype(int))
        src_frame = f0 + (samples["frame"].astype(int) - off) * step
        keep = ((src_frame >= self.f_lo) & (src_frame <= self.f_hi)
                & (samples["tsls"] <= MAX_IMPUTE_S))
        ghosts: dict[int, list[Ghost]] = {}
        for i in np.flatnonzero(keep):
            tid = slot_track.get(int(samples["slot_id"][i]))
            if tid is None or tid not in team_of:
                continue
            b = int(bkt[i])
            v1 = ems["source"][i] == "v1"
            k50 = (K50_V1 if v1 else K50_ANC)[b]
            k90 = (K90_V1 if v1 else K90_ANC)[b]
            ghosts.setdefault(int(src_frame[i]), []).append(Ghost(
                tid=tid, team=int(team_of[tid]), band=self.bands.get(tid, "?"),
                pos=(float(ems["pos"][i][0]), float(ems["pos"][i][1])),
                semi50=(k50 * float(halfw[i, 0]), k50 * float(halfw[i, 1])),
                semi90=(k90 * float(halfw[i, 0]), k90 * float(halfw[i, 1])),
                source=str(ems["source"][i]), tsls=float(samples["tsls"][i])))
        n = sum(len(v) for v in ghosts.values())
        note = (f"frozen B4 v1 (Metrica-trained, no broadcast truth): {n} imputed samples over "
                f"{len(ghosts)} frames")
        return ghosts, note

    def ghosts_at(self, frame: int) -> list[Ghost]:
        """Ghosts for the nearest sampled frame, filtered by the two things our tracks get wrong.

        1. **Re-identification duplicates.** A "hidden" slot is often a live player who simply
           picked up a new track id (``results/B4_TRANSFER_M3.md`` calls this out explicitly). Any
           ghost within :data:`DUP_M` of an observed same-team player is dropped: we are not
           entitled to draw a second player where we can already see one.
        2. **Squad arithmetic.** A team has eleven players, so if the broadcast shows eight we can
           be missing at most three. After de-duplication each team's ghosts are capped at that
           remainder, most-recently-seen first -- otherwise fragmentation puts a crowd on the board
           that does not exist.
        """
        if not self.ghosts:
            return []
        keys = np.array(sorted(self.ghosts), int)
        near = int(keys[np.argmin(np.abs(keys - frame))])
        if abs(near - frame) > md.MAX_INTERP_GAP:
            return []
        st = self.at(frame) or self.at(near)
        live: dict[int, list[tuple[float, float]]] = {0: [], 1: []}
        if st is not None:
            for p in st.players:
                if p["pitch"] is not None:
                    live.setdefault(int(p["team"]), []).append(p["pitch"])
        out: list[Ghost] = []
        for team in (0, 1):
            pts = np.array(live.get(team, []), float).reshape(-1, 2)
            room = max(SQUAD - len(pts), 0)
            cand = [g for g in self.ghosts[near] if g.team == team]
            if len(pts):
                cand = [g for g in cand
                        if np.linalg.norm(pts - np.array(g.pos), axis=1).min() > DUP_M]
            out.extend(sorted(cand, key=lambda g: g.tsls)[:room])
        return out


# --------------------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------------------
def _ellipse(canvas: np.ndarray, td: md.TopDown, centre: tuple[float, float],
             semi: tuple[float, float], colour: tuple[int, int, int], alpha: float) -> None:
    """Alpha-blend a metre-space axis-aligned ellipse onto the tactical board."""
    cx, cy = td.to_px(*centre)
    ax = max(int(round(semi[0] * td.scale)), 2)
    ay = max(int(round(semi[1] * td.scale)), 2)
    layer = canvas.copy()
    cv2.ellipse(layer, (cx, cy), (ax, ay), 0, 0, 360, colour, -1, cv2.LINE_AA)
    cv2.addWeighted(layer, alpha, canvas, 1 - alpha, 0, canvas)
    cv2.ellipse(canvas, (cx, cy), (ax, ay), 0, 0, 360, colour, 1, cv2.LINE_AA)


def draw_broadcast(img: np.ndarray, marks: list[Mark], st: md.FrameState, scale: float) -> None:
    """Player markers + role bands + gated names + the ball on the (already resized) video panel."""
    for pl in st.lines_img:
        cv2.polylines(img, [(pl * scale).astype(np.int32)], False, (70, 190, 200), 1, cv2.LINE_AA)
    for m in sorted(marks, key=lambda q: q.img[1] if q.img else 0):
        if m.img is None:
            continue
        x, y = int(round(m.img[0] * scale)), int(round(m.img[1] * scale))
        col = C_TEAM.get(m.team, C_TEAM[-1])
        cv2.ellipse(img, (x, y + 4), (13, 5), 0, 0, 360, C_BLACK, 3, cv2.LINE_AA)
        cv2.ellipse(img, (x, y + 4), (13, 5), 0, 0, 360, col, 2, cv2.LINE_AA)
        cv2.circle(img, (x, y - 22), 8, C_BLACK, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y - 22), 6, col, -1, cv2.LINE_AA)
        label = m.name if m.name else m.band
        colour = C_YELLOW if m.name else C_WHITE
        (tw, _), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        md.text(img, label, (x - tw // 2, y - 32), 0.5, colour, 1)
    if st.ball_img:
        bx, by = int(round(st.ball_img[0] * scale)), int(round(st.ball_img[1] * scale))
        cv2.circle(img, (bx, by), 10, C_BLACK, -1, cv2.LINE_AA)
        cv2.circle(img, (bx, by), 7, C_WHITE if st.ball_observed else C_AMBER,
                   -1 if st.ball_observed else 2, cv2.LINE_AA)


def draw_board(td: md.TopDown, marks: list[Mark], ghosts: list[Ghost], st: md.FrameState,
               dirs: dict[int, int], teams: tuple[str, str]) -> np.ndarray:
    """The tactical board: attack arrows, uncertainty regions, ghosts, observed players, ball."""
    pan = td.frame()
    for team, sgn in dirs.items():
        if team not in (0, 1):
            continue
        y = 6 if team == 0 else PITCH_WID - 6
        x0, x1 = (PITCH_LEN / 2 - 14, PITCH_LEN / 2 + 14)[::sgn if sgn > 0 else -1]
        cv2.arrowedLine(pan, td.to_px(x0, y), td.to_px(x1, y), C_TEAM[team], 3, cv2.LINE_AA,
                        tipLength=0.25)
        md.text(pan, f"{teams[team]} attack", (td.to_px(min(x0, x1), y)[0], td.to_px(0, y)[1] - 10),
                0.45, C_WHITE, 1)
    for g in ghosts:                                   # uncertainty first, under everything
        col = C_TEAM.get(g.team, C_TEAM[-1])
        if g.source == "abstained":
            continue
        _ellipse(pan, td, g.pos, g.semi90, col, 0.22)
        _ellipse(pan, td, g.pos, g.semi50, col, 0.34)
    for g in ghosts:
        col = C_TEAM.get(g.team, C_TEAM[-1])
        c = td.to_px(*g.pos)
        if g.source == "abstained":
            cv2.drawMarker(pan, c, (150, 150, 150), cv2.MARKER_TILTED_CROSS, 12, 1, cv2.LINE_AA)
            md.text(pan, f"? {g.tsls:.0f}s", (c[0] + 8, c[1] + 4), 0.4, (200, 200, 200), 1)
            continue
        cv2.drawMarker(pan, c, C_BLACK, cv2.MARKER_DIAMOND, 13, 3, cv2.LINE_AA)
        cv2.drawMarker(pan, c, col, cv2.MARKER_DIAMOND, 11, 2, cv2.LINE_AA)
        md.text(pan, f"{g.tsls:.0f}s", (c[0] + 9, c[1] + 4), 0.4, (235, 235, 240), 1)
    for m in marks:
        if m.pitch is None:
            continue
        col = C_TEAM.get(m.team, C_TEAM[-1])
        c = td.to_px(*m.pitch)
        cv2.circle(pan, c, 9, C_BLACK, -1, cv2.LINE_AA)
        cv2.circle(pan, c, 7, col, -1, cv2.LINE_AA)
        if m.keeper:
            cv2.circle(pan, c, 12, C_WHITE, 1, cv2.LINE_AA)
        if m.name:  # the board stays uncluttered: bands live on the broadcast panel
            md.text(pan, m.name.split()[-1], (c[0] + 10, c[1] + 4), 0.42, C_YELLOW, 1)
    if st.ball_pitch:
        c = td.to_px(*st.ball_pitch)
        cv2.circle(pan, c, 9, C_BLACK, -1, cv2.LINE_AA)
        cv2.circle(pan, c, 6, C_WHITE if st.ball_observed else C_AMBER,
                   -1 if st.ball_observed else 2, cv2.LINE_AA)
    return pan


LEGEND = (
    ("obs", "OBSERVED", "detected + projected this frame"),
    ("imp", "IMPUTED", "frozen B4 v1: 50% / 90% region"),
    ("abs", "ABSTAINED", "model declined; last-seen ghost"),
    ("ball", "BALL", "solid = detected, ring = carried"),
)


def draw_legend(canvas: np.ndarray, x: int, y: int, w: int, note: str) -> None:
    """The three-state key -- the visual argument, spelled out on every frame."""
    md.shade(canvas, y, y + 176, 0.55, (12, 12, 20))
    cv2.rectangle(canvas, (x, y), (x + w, y + 176), (90, 90, 100), 1)
    md.text(canvas, "WHAT THE MARKERS MEAN", (x + 14, y + 26), 0.56, C_YELLOW, 1)
    for i, (kind, name, desc) in enumerate(LEGEND):
        cy = y + 56 + i * 30
        cx = x + 26
        if kind == "obs":
            cv2.circle(canvas, (cx, cy - 4), 7, C_TEAM[0], -1, cv2.LINE_AA)
        elif kind == "imp":
            cv2.ellipse(canvas, (cx, cy - 4), (16, 8), 0, 0, 360, C_TEAM[0], 1, cv2.LINE_AA)
            cv2.drawMarker(canvas, (cx, cy - 4), C_TEAM[0], cv2.MARKER_DIAMOND, 11, 2, cv2.LINE_AA)
        elif kind == "abs":
            cv2.drawMarker(canvas, (cx, cy - 4), (150, 150, 150), cv2.MARKER_TILTED_CROSS, 12, 1,
                           cv2.LINE_AA)
        else:
            cv2.circle(canvas, (cx, cy - 4), 6, C_WHITE, -1, cv2.LINE_AA)
        md.text(canvas, name, (x + 52, cy), 0.5, C_WHITE, 1)
        md.text(canvas, desc, (x + 158, cy), 0.46, (205, 205, 215), 1)
    md.text(canvas, note[:64], (x + 14, y + 168), 0.42, (185, 190, 210), 1)


def compose(frame: np.ndarray, st: md.FrameState | None, pas: Passage, td: md.TopDown,
            fr: int) -> np.ndarray:
    """Compose one 1920x1080 output frame (header, situation strip, both panels, legend)."""
    half_s = pas.idx.to_half_s(pas.ci, fr)
    canvas = np.full((H, W, 3), C_BG, np.uint8)
    vw, vh, vx, vy = 1164, 655, 24, 214
    panel = cv2.resize(frame, (vw, vh), interpolation=cv2.INTER_AREA)
    scale = vw / frame.shape[1]
    marks = pas.marks(st) if st is not None else []
    if st is not None:
        draw_broadcast(panel, marks, st, scale)
    canvas[vy:vy + vh, vx:vx + vw] = panel
    cv2.rectangle(canvas, (vx - 2, vy - 2), (vx + vw + 2, vy + vh + 2), (90, 90, 90), 2)

    px_, py_ = 1206, 214
    ghosts = pas.ghosts_at(fr)
    pan = (draw_board(td, marks, ghosts, st, pas.dirs, pas.match.teams)
           if st is not None and st.calibrated else td.frame())
    if st is None or not st.calibrated:
        md.shade(pan, 0, pan.shape[0], 0.62, (10, 10, 10))
        md.text(pan, "NO PITCH GEOMETRY", (int(0.15 * td.w), td.h // 2), 0.9, C_AMBER, 2)
    canvas[py_:py_ + td.h, px_:px_ + td.w] = pan
    cv2.rectangle(canvas, (px_ - 2, py_ - 2), (px_ + td.w + 2, py_ + td.h + 2), (90, 90, 90), 2)
    md.text(canvas, "BROADCAST (tracked, role-banded)", (vx, vy - 12), 0.58, (200, 220, 255), 1)
    md.text(canvas, "TACTICS BOARD 105 x 68 m", (px_, py_ - 12), 0.58, (200, 220, 255), 1)

    # header
    md.shade(canvas, 0, 84, 0.80)
    md.text(canvas, "TACTICAL PASSAGE", (26, 52), 0.95, C_YELLOW, 2, FONT_T)
    fx = f"{pas.match.teams[0]} vs {pas.match.teams[1]}"
    md.text(canvas, f"{fx}   {pas.half.upper()} {_clock(half_s)}   (CV pipeline output, not vendor "
            "tracking)", (430, 50), 0.66, (215, 215, 225), 1)

    # situation strip
    sit = pas.situation
    md.shade(canvas, 92, 156, 0.55)
    cx = 26
    poss = (f"{pas.match.teams[sit.poss_team]} ({sit.poss_frac:.0%})"
            if sit.poss_team is not None else "contested")
    for label, colour in (
        (f"POSSESSION: {poss}",
         C_TEAM.get(sit.poss_team, C_TEAM[-1]) if sit.poss_team is not None else (150, 150, 150)),
        (f"PHASE: {sit.phase}", (170, 200, 235)),
        (f"BALL: {sit.ball_zone}", (170, 200, 235)),
        (f"OPP BLOCK: {sit.block_class or 'not measurable'}"
         + ("" if not np.isfinite(sit.block_line_m) else f" ({sit.block_line_m:.0f} m)"),
         C_GREEN if sit.block_class == "high" else (200, 200, 210)),
    ):
        md.chip(canvas, label, (cx, 134), colour, 0.56)
        (tw, _), _ = cv2.getTextSize(label, FONT, 0.56, 2)
        cx += tw + 34
    live = [e for e in sit.events if abs(e[0] - half_s) <= 0.6]
    if live:
        md.chip(canvas, f"EVENT: {live[0][1]} {live[0][2]}".strip(), (cx, 134), C_AMBER, 0.56)

    # per-frame status + legend
    q = pas.quality
    md.shade(canvas, vy + vh + 8, vy + vh + 62, 0.55)
    sy = vy + vh + 46
    if st is None:
        md.chip(canvas, "NO OUTPUT: BROADCAST CUT / REPLAY / GRAPHIC", (30, sy), C_AMBER, 0.7)
    else:
        n_obs = len(marks)
        md.text(canvas, f"observed {n_obs}   imputed {len(ghosts)}   calib {st.calib_err:.2f} m"
                f"   passage: {q.players:.1f} players/frame, ball {q.ball_cov:.0%}",
                (30, sy), 0.62, C_WHITE, 1)
    draw_legend(canvas, px_, py_ + td.h + 26, td.w, pas.impute_note)
    md.text(canvas, "Names shown only where >= 3 close-up reads agree; otherwise the role band.",
            (px_, py_ + td.h + 224), 0.46, (185, 190, 210), 1)
    md.text(canvas, "Imputed positions have no broadcast ground truth (smell test only).",
            (px_, py_ + td.h + 250), 0.46, (185, 190, 210), 1)
    return canvas


# --------------------------------------------------------------------------------------------------
# Shortlist
# --------------------------------------------------------------------------------------------------
def shortlist(match: registry.Match, *, halves: tuple[str, ...] = ("h1", "h2"),
              win_s: float = WIN_S, stride_s: float = STRIDE_S, top: int = 10) -> pd.DataFrame:
    """Rank candidate passages by tracking quality, ball coverage and situation interest.

    Scoring (all terms in [0, 1], summed with the weights below). Every rate is measured against
    the frames the window *should* contain, so a window half-eaten by a replay cannot win on the
    quality of its surviving half:

    * ``0.30`` tracking -- mean gated players/frame, normalised at 18 (a full visible XI + keepers)
    * ``0.25`` ball -- fraction of the window's expected samples with a post-``link_ball`` sample
    * ``0.20`` frame coverage -- expected samples that actually yield trusted geometry
    * ``0.05`` both teams on screen
    * ``0.10`` identity -- at least one name-gated track is on the pitch in this window
    * ``0.10`` interest -- the Carrick-video theme: a team building from its own third against an
      opponent block classed ``mid`` or ``high``

    Args:
        match: Registry match.
        halves: Halves to scan.
        win_s: Window length in seconds.
        stride_s: Window stride in seconds.
        top: Rows to return.

    Returns:
        Ranked frame with ``match, half, start_s, end_s, clock, players, ball_cov, frame_cov,
        both_teams, named, poss_team, phase, block, score, why``.
    """
    aligned = match.load_aligned()
    gated = aligned[(aligned["calib_error_m"] <= CALIB_MAX_M) & aligned["pitch_x"].notna()
                    & aligned["role"].isin(["player", "goalkeeper"])]
    bf = block_frames(match)
    names = load_names(match)
    rows: list[dict] = []
    for half in halves:
        try:
            idx = chunk_index(match, half)
        except SystemExit:
            continue
        balls = dict(match.ball_chunks())
        for i, ck in enumerate(idx.keys):
            g = gated[gated["chunk"] == ck]
            if g.empty:
                continue
            named_tids = {t for (c, t) in names if c == ck}
            per_frame = g.groupby("frame").agg(
                n=("track_id", "size"), n0=("team", lambda s: int((s == 0).sum())),
                named=("track_id", lambda s: int(s.isin(named_tids).any())))
            frames = per_frame.index.to_numpy()
            all_fr = np.sort(aligned.loc[aligned["chunk"] == ck, "frame"].unique())
            sample_step = int(np.median(np.diff(all_fr))) if all_fr.size > 1 else 1
            ball_fr = pd.read_parquet(balls[ck])["frame"].to_numpy()
            bfc = bf[bf["chunk"] == ck]
            step = int(round(win_s * idx.fps[i]))
            expect = max(step // sample_step + 1, 1)
            for lo in range(0, int(idx.durations[i] * idx.fps[i]) - step, int(stride_s * idx.fps[i])):
                hi = lo + step
                sel = (frames >= lo) & (frames <= hi)
                fcov = float(min(sel.sum() / expect, 1.0))
                if fcov < MIN_FRAME_COV:
                    continue
                n = per_frame["n"].to_numpy()[sel]
                n0 = per_frame["n0"].to_numpy()[sel]
                players = float(n.mean())
                both = float(np.sum((n0 >= 4) & (n - n0 >= 4)) / expect)
                named = int(per_frame["named"].to_numpy()[sel].sum())
                cov = float(min(((ball_fr >= lo) & (ball_fr <= hi)).sum() / expect, 1.0))
                bw = bfc[(bfc["frame"] >= lo) & (bfc["frame"] <= hi)]
                poss, phase, blk = None, "unknown", None
                if not bw.empty:
                    deft = int(bw["def_team"].mode().iloc[0])
                    own = PITCH_LEN - float(bw["ball_adv_m"].median())
                    phase, poss = phase_of(own, float((bw["def_team"] == deft).mean()), 1 - deft)
                    ub = bw[bw["usable"]]
                    if len(ub) >= 3:
                        blk = block_class(float(ub["line_m"].median()))
                interest = float(phase == "build-up" and blk in ("mid", "high"))
                score = (0.30 * min(players / 18.0, 1.0) + 0.25 * cov + 0.20 * fcov + 0.05 * both
                         + 0.10 * float(named > 0) + 0.10 * interest)
                why = []
                if interest:
                    why.append(f"build-up vs {blk} block")
                elif blk:
                    why.append(f"{phase} vs {blk} block")
                why.append(f"{players:.0f} players/frame, ball {cov:.0%}")
                if named:
                    why.append(f"named track on {named} frames")
                rows.append({
                    "match": match.id,
                    "half": half, "start_s": round(idx.to_half_s(i, lo), 1),
                    "end_s": round(idx.to_half_s(i, hi), 1),
                    "clock": _clock(idx.to_half_s(i, lo)), "players": round(players, 1),
                    "ball_cov": round(cov, 2), "frame_cov": round(fcov, 2),
                    "both_teams": round(both, 2), "named": named,
                    "poss_team": None if poss is None else match.teams[poss], "phase": phase,
                    "block": blk, "score": round(score, 3), "why": "; ".join(why)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[(df["players"] >= MIN_PLAYERS) & (df["ball_cov"] >= MIN_BALL_COV)]
    df = df.sort_values("score", ascending=False)
    # de-overlap: keep the best window, drop anything it overlaps, repeat
    kept: list[dict] = []
    for r in df.to_dict("records"):
        if all(r["match"] != k["match"] or r["half"] != k["half"] or r["end_s"] <= k["start_s"]
               or r["start_s"] >= k["end_s"] for k in kept):
            kept.append(r)
        if len(kept) >= top:
            break
    return pd.DataFrame(kept)


def shortlist_many(match_ids: list[str], *, top: int = 12, per_match: int = 4) -> pd.DataFrame:
    """Shortlist over several matches, keeping at most ``per_match`` windows from any one fixture.

    Args:
        match_ids: Registry ids to scan; unprocessed ids are skipped with a note.
        top: Length of the merged shortlist.
        per_match: Cap on windows contributed by a single match, so one clean fixture cannot
            fill the whole list.

    Returns:
        The merged, ranked shortlist.
    """
    parts = []
    for mid in match_ids:
        m = registry.get(mid)
        if not m.processed:
            print(f"[clip] skip {mid}: no aligned parquet")
            continue
        try:
            sl = shortlist(m, top=per_match)
        except (SystemExit, FileNotFoundError, KeyError) as exc:  # noqa: PERF203
            print(f"[clip] skip {mid}: {exc}")
            continue
        print(f"[clip] {mid}: {len(sl)} candidate passage(s)")
        if not sl.empty:
            parts.append(sl)
    if not parts:
        return pd.DataFrame()
    return (pd.concat(parts, ignore_index=True)
            .sort_values("score", ascending=False).head(top).reset_index(drop=True))


# --------------------------------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------------------------------
def render(match: registry.Match, half: str, start_s: float, end_s: float, *, out_dir: Path,
           height: int = 1080, crf: int = 20, impute: bool = True) -> dict:
    """Render one passage to mp4 + a contact-sheet png, refusing poor-quality passages.

    Args:
        match: Registry match.
        half: ``"h1"`` or ``"h2"``.
        start_s: Passage start, seconds into the half.
        end_s: Passage end, seconds into the half.
        out_dir: Output directory.
        height: Output height.
        crf: x264 quality.
        impute: Run the frozen v1 imputation panel.

    Returns:
        Summary dict; ``{"refused": reason}`` when the passage fails the quality gate.
    """
    t0 = time.time()
    pas = Passage(match, half, start_s, end_s, impute=impute)
    q = pas.quality
    print(f"[clip] {match.id} {half} {_clock(pas.start_s)}-{_clock(pas.end_s)} "
          f"({pas.chunk} f{pas.f_lo}-{pas.f_hi})")
    print(f"[clip] quality: {q.players:.1f} players/frame, ball {q.ball_cov:.0%}, "
          f"both teams {q.both_teams:.0%}, frames with geometry {q.frame_cov:.0%}")
    reason = q.refusal()
    if reason:
        print(f"[clip] REFUSED: {reason}")
        return {"refused": reason, "half": half, "start_s": pas.start_s, "end_s": pas.end_s}
    print(f"[clip] situation: possession={pas.situation.poss_team} "
          f"({pas.situation.poss_frac:.0%} of possession frames) phase={pas.situation.phase} "
          f"block={pas.situation.block_class} events={len(pas.situation.events)}")
    print(f"[clip] {pas.impute_note}")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{match.id}_{half}_{int(pas.start_s):04d}s"
    mp4, png = out_dir / f"{stem}.mp4", out_dir / f"{stem}_sheet.png"
    tmp = mp4.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter at {tmp}")
    td = md.TopDown(width=690)
    cap = cv2.VideoCapture(str(pas.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, pas.f_lo)
    sheet_at = np.linspace(pas.f_lo, pas.f_hi, 6).astype(int)
    sheet: list[np.ndarray] = []
    n = 0
    for fr in range(pas.f_lo, pas.f_hi + 1):
        ok, frame = cap.read()
        if not ok:
            break
        canvas = compose(frame, pas.at(fr), pas, td, fr)
        writer.write(canvas)
        if fr in sheet_at:
            sheet.append(canvas)
        n += 1
    cap.release()
    writer.release()
    if sheet:
        cols = [cv2.resize(s, (W // 2, H // 2), interpolation=cv2.INTER_AREA) for s in sheet[:6]]
        while len(cols) < 6:
            cols.append(np.full_like(cols[0], C_BG, np.uint8))
        cv2.imwrite(str(png), np.vstack([np.hstack(cols[i:i + 2]) for i in (0, 2, 4)]))

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        vf = f"scale=-2:{height}" if height != H else "null"
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(tmp), "-vf", vf,
                        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)
        tmp.unlink(missing_ok=True)
    else:
        print("[clip] WARNING: ffmpeg not found -> shipping the mp4v render")
        shutil.move(str(tmp), str(mp4))
    info = {"mp4": str(mp4), "png": str(png), "frames": n, "seconds": n / FPS,
            "size_mb": round(mp4.stat().st_size / 1e6, 2), "half": half,
            "start_s": round(pas.start_s, 1), "end_s": round(pas.end_s, 1),
            "players": round(q.players, 1), "ball_cov": round(q.ball_cov, 2),
            "wall_s": round(time.time() - t0, 1)}
    print(f"[clip] wrote {mp4} ({info['seconds']:.1f} s, {info['size_mb']} MB) and {png} "
          f"in {info['wall_s']} s")
    return info


def _spread(sl: pd.DataFrame, n: int) -> list[dict]:
    """Pick ``n`` shortlist rows that show different things, not ``n`` copies of the best one.

    Walks the ranked list taking the best window of each unseen ``(phase, opponent block class)``
    situation type first, then falls back to plain rank once every type has been used.

    Args:
        sl: Ranked shortlist.
        n: How many passages to render.

    Returns:
        The chosen rows, best first.
    """
    if n <= 0 or sl.empty:
        return []
    rows = sl.to_dict("records")
    chosen, seen = [], set()
    for r in rows:
        key = (r["phase"], r["block"])
        if key not in seen:
            seen.add(key)
            chosen.append(r)
        if len(chosen) >= n:
            return chosen
    for r in rows:
        if r not in chosen:
            chosen.append(r)
        if len(chosen) >= n:
            break
    return chosen


def _self_check() -> None:
    """Assert the geometry/labelling seams that the renderer depends on."""
    idx = ChunkIndex("h1", ("h1_chunk_000", "h1_chunk_001"), (0.0, 600.0), (600.0, 540.0),
                     (25.0, 25.0))
    assert idx.locate(0.0) == (0, 0)
    assert idx.locate(599.9) == (0, 14998), idx.locate(599.9)
    assert idx.locate(600.0) == (1, 0), idx.locate(600.0)
    assert abs(idx.to_half_s(1, 250) - 610.0) < 1e-9
    assert idx.locate(1e9)[0] == 1  # clamped, never off the end
    win = pd.DataFrame({
        "track_id": [1, 1, 2, 2, 3, 3, 4, 4],
        "team": [0, 0, 0, 0, 1, 1, 1, 1],
        "role": ["goalkeeper", "goalkeeper", "player", "player", "player", "player",
                 "player", "player"],
        "is_keeper": [True, True, False, False, False, False, False, False],
        "pitch_x": [5.0, 6.0, 20.0, 22.0, 20.0, 22.0, 95.0, 96.0],
    })
    bands = role_bands(win, {0: 1, 1: -1})
    assert bands == {1: "GK", 2: "DEF", 3: "ATT", 4: "DEF"}, bands
    # attack direction must flip the band: team 1 attacks x=0, so x=21 is its ATTacking third.
    assert role_bands(win, {0: 1, 1: 1})[3] == "DEF"
    ok = {"ball_cov": 1.0, "both_teams": 1.0, "frame_cov": 1.0}
    assert Quality(players=5, **ok).refusal().startswith("tracking")
    assert Quality(players=20, **{**ok, "ball_cov": 0.1}).refusal().startswith("ball")
    assert Quality(players=20, **{**ok, "both_teams": 0.1}).refusal().startswith("only one team")
    assert Quality(players=20, **{**ok, "frame_cov": 0.5}).refusal().startswith("broken passage")
    assert Quality(players=20, **ok).refusal() is None
    # frozen conformal tables: 90% regions must be wider than 50% in every bucket, both sources.
    assert (K90_V1 > K50_V1).all() and (K90_ANC > K50_ANC).all()
    assert LOW_BLOCK_MAX_M < HIGH_BLOCK_MIN_M
    # --auto must show variety, not four copies of the same situation type.
    sl = pd.DataFrame([
        {"match": "a", "phase": "build-up", "block": "high", "score": 0.9},
        {"match": "a", "phase": "build-up", "block": "high", "score": 0.8},
        {"match": "a", "phase": "attack", "block": "low", "score": 0.7},
        {"match": "b", "phase": "build-up", "block": "high", "score": 0.6},
    ])
    picked = _spread(sl, 3)  # 0.8 and 0.6 repeat 0.9's situation type -> skipped until the fallback
    assert [p["score"] for p in picked] == [0.9, 0.7, 0.8], picked
    assert len(_spread(sl, 4)) == 4 and _spread(sl, 0) == []
    print("tactical_clip self-check OK (chunk clock, role bands, refusal gate, region ordering)")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", default="manutd_liverpool",
                    help="registry id; 'all' or a comma-list scans several (shortlist/auto only)")
    ap.add_argument("--half", default="h1", choices=("h1", "h2"))
    ap.add_argument("--start", type=float, help="seconds into the half")
    ap.add_argument("--end", type=float, help="seconds into the half")
    ap.add_argument("--shortlist", action="store_true", help="rank candidate passages, render none")
    ap.add_argument("--auto", type=int, default=0, help="render the top N shortlisted passages")
    ap.add_argument("--top", type=int, default=10, help="shortlist length")
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--height", type=int, default=1080, choices=(720, 1080))
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--no-impute", action="store_true", help="skip the uncertainty panel")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        _self_check()
        return
    out_dir = Path(args.out)
    ids = ([m.id for m in registry.matches(processed_only=True)] if args.match == "all"
           else [s.strip() for s in args.match.split(",") if s.strip()])

    if args.shortlist or args.auto:
        sl = shortlist_many(ids, top=max(args.top, args.auto))
        if sl.empty:
            print("[clip] no passage clears the quality gate")
            return
        print("\n[clip] shortlist")
        print(sl.to_string(index=False))
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = ids[0] if len(ids) == 1 else "all"
        sl.to_csv(out_dir / f"{tag}_shortlist.csv", index=False, encoding="utf-8")
        print(f"[clip] wrote {out_dir / f'{tag}_shortlist.csv'}")
        for r in _spread(sl, args.auto):
            render(registry.get(r["match"]), r["half"], r["start_s"], r["end_s"], out_dir=out_dir,
                   height=args.height, crf=args.crf, impute=not args.no_impute)
        return

    if len(ids) != 1:
        raise SystemExit("a single --match is required to render one passage")
    match = registry.get(ids[0])
    if not match.processed:
        raise SystemExit(f"{match.id} has no aligned parquet at {match.aligned}")
    if args.start is None or args.end is None:
        raise SystemExit("give --start/--end (seconds into the half), or --shortlist / --auto N")
    render(match, args.half, args.start, args.end, out_dir=out_dir, height=args.height,
           crf=args.crf, impute=not args.no_impute)


if __name__ == "__main__":
    main()
