"""Find the wide tactical-camera segments a freeze-frame model needs (>= ~10 players visible).

Broadcast football alternates between **wide** tactical shots (most of both teams on screen, spread
across the pitch) and **tight** shots (close-ups, corners, replays) where only a handful of players
are visible. The relational model only consumes wide frames (the >= ``MIN_PLAYERS`` gate in
:mod:`generator.to_frames`), so feeding it a tight segment yields nothing.

This module scores each sampled frame for "wideness" from detections alone (cheap -- no calibration):
many players **and** large horizontal spread. The pure scorer is tested; :func:`scan_wide_segments`
is the lazy CV driver used to pick where to run the (expensive) calibrated extract.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

MIN_PLAYERS_WIDE = 10  # mirror the freeze-frame wide-shot gate


def wide_shot_score(boxes_xyxy: np.ndarray, frame_width: float) -> float:
    """Score a frame's "wideness" from player boxes: ``n_players * horizontal_spread_fraction``.

    A wide tactical shot has many players spread across the frame width (high score); a tight/close
    shot has few players clustered together (low score). Returns 0 for too-few detections.

    Args:
        boxes_xyxy: ``(N, 4)`` player boxes ``[x1, y1, x2, y2]`` in image pixels.
        frame_width: Frame width in pixels (to normalise the spread).

    Returns:
        A non-negative wideness score.
    """
    boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    n = len(boxes)
    if n < MIN_PLAYERS_WIDE or frame_width <= 0:
        return 0.0
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    spread = float(cx.std()) / float(frame_width)
    return n * spread


def scan_wide_segments(
    source: str,
    detector,
    *,
    start: int = 0,
    stop: int | None = None,
    step: int = 150,
    min_score: float = 4.0,
) -> list[tuple[int, float, int]]:
    """Sample a video and return ``(frame_index, score, n_players)`` for wide-enough frames.

    Args:
        source: Video path.
        detector: Object with ``detect(frame_rgb) -> (boxes_xyxy, confs, ball_xy)`` (e.g.
            :class:`generator.extract._CocoDetector`).
        start: First frame to sample.
        stop: Last frame to sample (exclusive); ``None`` = to the end.
        step: Sampling stride in frames.
        min_score: Keep frames scoring at least this.

    Returns:
        ``(frame, score, n_players)`` for kept frames, sorted by score descending.
    """
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = total if stop is None else min(stop, total)
    out: list[tuple[int, float, int]] = []
    for fr in range(start, end, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, frame = cap.read()
        if not ok:
            continue
        boxes, _, _, _ = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        score = wide_shot_score(boxes, width)
        if score >= min_score:
            out.append((fr, score, len(boxes)))
    cap.release()
    out.sort(key=lambda r: r[1], reverse=True)
    return out


def best_wide_segment(scan: list[tuple[int, float, int]]) -> int | None:
    """Return the frame index of the highest-scoring wide candidate, or ``None`` if the scan is empty."""
    return scan[0][0] if scan else None


def iter_wide_frames(scan: list[tuple[int, float, int]]) -> Iterator[int]:
    """Yield candidate frame indices (score order)."""
    for fr, _, _ in scan:
        yield fr
