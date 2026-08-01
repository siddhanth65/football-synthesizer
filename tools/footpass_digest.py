"""Digest FOOTPASS tactical HDF5 into compact per-half arrays for the evidence-density experiment.

The raw dataset (``data/footpass/raw/tactical_data_{TRAIN,VAL}.zip``) is dense per-frame per-player
tracking: 177 M rows, 8.8 GB uncompressed for TRAIN alone. Every sweep in
``tools/evidence_sim.py`` needs only four things per (game, half):

* the roster -- ``(player_id, shirt_number, role_id, team, first_frame, last_frame)``;
* the **visibility runs** -- maximal contiguous frame intervals in which a player carries a
  broadcast ROI box. ``roi_*`` is ``NaN`` exactly when the player is off-screen (the tracking
  ``x, y`` is always present), so on-screen-ness is a clean binary signal;
* the action events -- rows with ``class != 0``;
* a coarse position grid, used to place the ball proxy and to find the tracklet nearest it when a
  commentary mention is bound.

Writing those to ``.npz`` shrinks a half from ~90 MB of float32 to ~2 MB, so the whole corpus fits
in memory during a sweep and the 8.8 GB extraction can be deleted immediately after.

CLI::

    python -m tools.footpass_digest --split VAL      # extracts the zip, digests, deletes the h5
    python -m tools.footpass_digest --split TRAIN --keep-h5
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path

import numpy as np

logger = logging.getLogger("footpass_digest")

#: Root of the downloaded FOOTPASS annotations (gitignored; see ``data/footpass/README.md``).
RAW_DIR = Path("data/footpass/raw")
#: Where digests land (gitignored, ~2 MB per half).
DIGEST_DIR = Path("data/footpass/digest")
#: Scratch for the extracted HDF5.
WORK_DIR = Path("data/footpass/work")
#: Broadcast frame rate of the FOOTPASS videos (verified by ffprobe, see the dataset README).
FPS = 25.0
#: Frame stride of the stored position grid (2.5 Hz -- the commentary lag has a 4 s IQR).
POS_STRIDE = 10

#: Column order of the TRAIN/VAL arrays, from ``tactical_data_format.txt``.
COLS = ("frame", "player_id", "left_to_right", "shirt_number", "role_id", "x", "y",
        "speed_x", "speed_y", "roi_x", "roi_y", "roi_width", "roi_height", "class")


def visibility_runs(frames: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Maximal contiguous on-screen intervals of one player (pure).

    Args:
        frames: Strictly increasing frame indices of the player's rows.
        visible: Boolean, same length, True where the player carries a broadcast ROI.

    Returns:
        ``(n, 2)`` int array of inclusive ``(start_frame, end_frame)`` intervals. A gap in
        ``frames`` breaks a run even if both sides are visible.
    """
    idx = np.flatnonzero(visible)
    if idx.size == 0:
        return np.zeros((0, 2), np.int64)
    fv = frames[idx]
    brk = np.flatnonzero(np.diff(fv) != 1)
    starts = np.concatenate(([0], brk + 1))
    ends = np.concatenate((brk, [fv.size - 1]))
    return np.stack([fv[starts], fv[ends]], axis=1)


def digest_half(arr: np.ndarray) -> dict[str, np.ndarray]:
    """Turn one ``(rows, 14)`` half-array into the compact arrays the simulator consumes.

    Args:
        arr: Raw FOOTPASS half array (float32, column order :data:`COLS`).

    Returns:
        Dict of arrays -- ``player_id, shirt, role, team, window`` (roster, one row per player),
        ``runs`` (``(n, 3)``: player index, start frame, end frame), ``events`` (``(n, 3)``: frame,
        player index, class), ``pos`` (``(n_steps, n_players, 2)`` float32, NaN before/after a
        player's window), ``grid`` (``(frame0, stride)``).
    """
    frame = arr[:, 0].astype(np.int64)
    pid = arr[:, 1].astype(np.int64)
    visible = ~np.isnan(arr[:, 9])
    players = np.unique(pid)
    slot = {int(p): k for k, p in enumerate(players)}

    shirt = np.zeros(players.size, np.int64)
    role = np.zeros(players.size, np.int64)
    window = np.zeros((players.size, 2), np.int64)
    runs: list[np.ndarray] = []
    order = np.argsort(pid, kind="stable")
    bounds = np.searchsorted(pid[order], players, side="left").tolist() + [pid.size]
    f0, f1 = int(frame.min()), int(frame.max())
    steps = np.arange(f0, f1 + 1, POS_STRIDE)
    pos = np.full((steps.size, players.size, 2), np.nan, np.float32)

    for k, p in enumerate(players):
        sel = order[bounds[k]:bounds[k + 1]]
        sel = sel[np.argsort(frame[sel], kind="stable")]
        fp = frame[sel]
        shirt[k] = int(np.bincount(arr[sel, 3].astype(np.int64)).argmax())
        role[k] = int(np.bincount(arr[sel, 4].astype(np.int64)).argmax())
        window[k] = (fp[0], fp[-1])
        r = visibility_runs(fp, visible[sel])
        if r.size:
            runs.append(np.column_stack([np.full(r.shape[0], k), r]))
        hit = np.isin(fp, steps)
        pos[np.searchsorted(steps, fp[hit]), k] = arr[sel][hit][:, 5:7]

    ev = np.flatnonzero(arr[:, 13] != 0)
    events = np.column_stack([frame[ev],
                              np.array([slot[int(v)] for v in pid[ev]], np.int64),
                              arr[ev, 13].astype(np.int64)])
    events = events[np.argsort(events[:, 0], kind="stable")]
    return {
        "player_id": players, "shirt": shirt, "role": role, "team": players // 100,
        "window": window,
        "runs": np.concatenate(runs) if runs else np.zeros((0, 3), np.int64),
        "events": events, "pos": pos, "grid": np.array([f0, POS_STRIDE], np.int64),
    }


def run(split: str, *, keep_h5: bool = False, limit: int | None = None) -> None:
    """Extract one split's zip, digest every half into ``data/footpass/digest``, drop the HDF5.

    Args:
        split: ``"TRAIN"`` or ``"VAL"``.
        keep_h5: Leave the extracted HDF5 on disk (8.8 GB for TRAIN).
        limit: Digest only the first ``limit`` half-datasets (smoke tests).
    """
    import h5py  # noqa: PLC0415  (only the digest needs it)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    DIGEST_DIR.mkdir(parents=True, exist_ok=True)
    zpath = RAW_DIR / f"tactical_data_{split}.zip"
    with zipfile.ZipFile(zpath) as z:
        name = z.namelist()[0]
        h5path = WORK_DIR / name
        if not h5path.exists():
            logger.info("extracting %s -> %s", zpath, h5path)
            z.extractall(WORK_DIR)
    try:
        with h5py.File(h5path, "r") as f:
            keys = sorted(f.keys())[:limit]
            for i, key in enumerate(keys):
                out = DIGEST_DIR / f"{split.lower()}_{key}.npz"
                if out.exists():
                    continue
                d = digest_half(f[key][:])
                np.savez_compressed(out, split=split, key=key, **d)
                logger.info("[%d/%d] %s: %d players, %d runs, %d events -> %.1f MB",
                            i + 1, len(keys), key, d["player_id"].size, d["runs"].shape[0],
                            d["events"].shape[0], out.stat().st_size / 1e6)
    finally:
        if not keep_h5 and h5path.exists():
            h5path.unlink()
            logger.info("deleted %s", h5path)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["TRAIN", "VAL"], required=True)
    ap.add_argument("--keep-h5", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.split, keep_h5=args.keep_h5, limit=args.limit)


if __name__ == "__main__":
    main()
