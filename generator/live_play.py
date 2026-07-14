"""Live-play / broadcast-content filter (Phase-B B1.2).

Classifies each *sampled* broadcast frame into a coarse shot type so downstream reporting can quote
**live-play-conditional denominators** and ingestion can skip wasted compute on non-tactical content.

Honestly re-priced (see ``STATUS.md`` pose-carry + calibration-correction entries): this filter
**cannot create geometry**. The ~63% of detection-frames that yield no player pitch coordinates are
overwhelmingly close-ups (a broadcast simply is not showing a tactical view most of the time), and a
pose cannot be conjured from a face-cam. What the filter *does* buy is (a) honest denominators -- a
yield/coverage/recall number reported against the frames where the measurement was even possible, not
diluted by replays and graphics -- and (b) a cheap pre-filter that removes ~half the frames from the
heavy pipeline. Both are reporting/compute wins, NOT a data fix.

Signals are read from artifacts the pipeline already produces (the dense positions parquet +
calibration error), so classification is CPU-only and needs no video re-decode:

* ``n_det``     -- detected players+GK on the frame (a close-up cannot frame eight players; the
  football-trained YOLO is additionally blind to close-up figures, so tight shots collapse to a
  handful of -- or zero -- detections; STATUS: geometry-less frames carry a median of 4 detected
  players vs 11 on good frames).
* ``x_spread``  -- horizontal image span of the detections (rejects a cluster of bench/crowd bodies
  masquerading as a wide shot).
* ``present``   -- whether the frame appears in the dense parquet at all. Frames with zero detections
  are dropped at extraction, so a full-screen graphic / hard cut / black frame / extreme close-up the
  detector cannot see is recoverable only as an *absent* grid frame -> the ``graphic`` bucket
  therefore conflates full-screen graphics with undetectable content (stated loudly, not hidden).

The classifier is a transparent, pre-committed rule set (no trained model, no tuning on any gate
outcome). ``classify_frame`` is pure and unit-tested on synthetic rows; the pandas helpers wrap it.
"""

from __future__ import annotations

import pandas as pd

# === Shot-type labels ============================================================================
SHOT_LIVE_WIDE = "live_wide"          # wide tactical main-camera view -- the geometry-yielding class
SHOT_CLOSE_UP = "close_up"            # few large figures (player/manager/celebration face-cam)
SHOT_REPLAY_OTHER = "replay_or_other"  # medium shots / replays of action -- ambiguous middle
SHOT_GRAPHIC = "graphic"              # zero-detection frame: full-screen graphic / cut / blank
SHOT_TYPES = (SHOT_LIVE_WIDE, SHOT_CLOSE_UP, SHOT_REPLAY_OTHER, SHOT_GRAPHIC)

# === PRE-COMMITTED thresholds (fixed BEFORE any recall/gate measurement; provenance in comments) ==
# A close-up cannot frame eight spread-out players, so >=8 detections over a wide span is the wide
# tactical view. Mirrors tools/pl_probe_diagnose (LIVE_MIN_DET=8, LIVE_MIN_XSPREAD=800) -- the proxy
# that recovered the WC-level >=6-corr yield on live PL play -- and is calibration-INDEPENDENT so the
# geometry yield measured on this class is not circular.
LIVE_MIN_DET = 8
LIVE_MIN_XSPREAD = 800.0     # px on the 1920-wide broadcast frame
# Geometry-less frames carry a median of 4 detected players (STATUS pose-carry anatomy); the
# football-YOLO caps close-up box height ~157 px and misses most tight-shot bodies. So <=4 detections
# is a close-up rather than a partially-occluded wide shot.
CLOSEUP_MAX_DET = 4
CALIB_GATE_M = 2.0           # PnLCalib reprojection acceptance (m); mirrors the WC/PL probes
MIN_CORR = 6                 # player correspondences a frame needs to place the ball (norway-killer)
FRAME_WIDTH_PX = 1920        # broadcast width the x-spread threshold is scaled to
PLAYER_ROLES = ("player", "goalkeeper")


# === Pure classifier =============================================================================
def classify_frame(n_det: int, x_spread: float, *, present: bool = True) -> str:
    """Classify one sampled frame from its detection count and horizontal spread.

    Ordered, transparent rules (first match wins):

    1. no detections (``present=False`` or ``n_det == 0``)      -> :data:`SHOT_GRAPHIC`
    2. ``n_det >= LIVE_MIN_DET`` and wide ``x_spread``          -> :data:`SHOT_LIVE_WIDE`
    3. ``n_det <= CLOSEUP_MAX_DET``                             -> :data:`SHOT_CLOSE_UP`
    4. otherwise (medium body count, or clustered)             -> :data:`SHOT_REPLAY_OTHER`

    Args:
        n_det: detected players+GK on the frame.
        x_spread: horizontal image span (px) of those detections.
        present: whether the frame appears in the dense parquet (has >= 1 detection).

    Returns:
        One of :data:`SHOT_TYPES`.
    """
    if not present or n_det <= 0:
        return SHOT_GRAPHIC
    if n_det >= LIVE_MIN_DET and x_spread >= LIVE_MIN_XSPREAD:
        return SHOT_LIVE_WIDE
    if n_det <= CLOSEUP_MAX_DET:
        return SHOT_CLOSE_UP
    return SHOT_REPLAY_OTHER


# === Per-frame signal extraction from the dense parquet ==========================================
def per_frame_signals(dense: pd.DataFrame) -> pd.DataFrame:
    """Reduce a dense positions parquet to one classification row per detected frame.

    Args:
        dense: dense positions table (one row per detection per sampled frame), with columns
            ``frame, role, image_x, image_y, pitch_x, pitch_y, calib_error_m``.

    Returns:
        A frame-indexed table with ``n_det`` (players+GK), ``n_pitch`` (those carrying valid pitch
        coords), ``x_spread``/``y_spread`` (image span), ``calib_err``, ``calibrated``
        (``<= CALIB_GATE_M``) and ``ge6`` (``>= MIN_CORR`` pitch correspondences). Only frames with
        at least one detection appear (zero-detection frames are absent from ``dense``).
    """
    players = dense[dense["role"].isin(PLAYER_ROLES)]
    g = players.groupby("frame")
    n_det = g.size()
    n_pitch = (
        players.dropna(subset=["pitch_x", "pitch_y"]).groupby("frame").size()
        .reindex(n_det.index, fill_value=0)
    )
    x_spread = g["image_x"].agg(lambda s: float(s.max() - s.min()))
    y_spread = g["image_y"].agg(lambda s: float(s.max() - s.min()))
    calib = dense.groupby("frame")["calib_error_m"].first().reindex(n_det.index)
    tab = pd.DataFrame(
        {"n_det": n_det, "n_pitch": n_pitch, "x_spread": x_spread, "y_spread": y_spread,
         "calib_err": calib}
    )
    tab["calibrated"] = tab["calib_err"] <= CALIB_GATE_M
    tab["ge6"] = tab["n_pitch"] >= MIN_CORR
    return tab


def classify_signals(signals: pd.DataFrame, grid_frames: pd.Index | None = None) -> pd.DataFrame:
    """Attach a ``shot_type`` column to a :func:`per_frame_signals` table.

    If ``grid_frames`` is given, frames on the sampling grid that are absent from ``signals`` (zero
    detections at extraction) are added as :data:`SHOT_GRAPHIC` rows, so the returned table spans the
    full sampled grid and per-class fractions use an honest denominator.

    Args:
        signals: output of :func:`per_frame_signals`.
        grid_frames: the full sampling grid (frame indices); optional.

    Returns:
        ``signals`` with a ``shot_type`` column, reindexed onto ``grid_frames`` when supplied.
    """
    out = signals.copy()
    out["shot_type"] = [
        classify_frame(int(r.n_det), float(r.x_spread), present=True)
        for r in out.itertuples(index=False)
    ]
    if grid_frames is not None:
        out = out.reindex(pd.Index(grid_frames, name=signals.index.name))
        out["shot_type"] = out["shot_type"].where(out["n_det"].notna(), SHOT_GRAPHIC)
        out["n_det"] = out["n_det"].fillna(0)
    return out


def classify_dense(dense: pd.DataFrame, grid_frames: pd.Index | None = None) -> pd.DataFrame:
    """Classify every frame of a dense parquet (convenience wrapper).

    Args:
        dense: dense positions table.
        grid_frames: full sampling grid; when given, absent frames become :data:`SHOT_GRAPHIC`.

    Returns:
        Frame-indexed table with signals plus a ``shot_type`` column.
    """
    return classify_signals(per_frame_signals(dense), grid_frames)


def class_fractions(labels: pd.Series) -> dict[str, float]:
    """Fraction of frames in each of :data:`SHOT_TYPES` (missing classes report 0.0)."""
    n = len(labels)
    counts = labels.value_counts()
    return {t: (float(counts.get(t, 0)) / n if n else 0.0) for t in SHOT_TYPES}
