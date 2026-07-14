"""Batch the full CV pipeline over every chunk of a match -> a whole-match fingerprint input.

Runs :func:`generator.extract.extract_positions` + :func:`generator.tactical_frames.build_tactical_table`
over each ``chunk_*.mp4`` in a directory, writing a dense + tactical parquet **per chunk as it finishes**
(so the run is resumable with ``--skip-existing`` and partial output survives an interruption), then a
concatenated ``match_dense.parquet`` / ``match_tactical.parquet`` (with a ``chunk`` column) for the
fingerprint engine. The PnLCalib calibrator is built **once** and reused across chunks (weights load
only once); a failing chunk is logged and skipped, not fatal.

Calibration defaults to the **temporal lever** (``--calib-period 25``) so a full 11-chunk match is
tractable in one background run; pass ``--per-frame`` for the slower, most-accurate per-frame mode.

Run (background):
    python -m tools.batch_match --chunks-dir "<...>/cv-football/chunks" --out-dir outputs/match
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator.extract import extract_positions  # noqa: E402
from generator.tactical_frames import build_tactical_table  # noqa: E402

logger = logging.getLogger("batch_match")


def _summarise(stem: str, dense: pd.DataFrame, tac: pd.DataFrame) -> str:
    accepted = int(tac["accepted"].sum()) if len(tac) else 0
    players = dense[dense["role"].isin(["player", "goalkeeper"])]
    ppf = players.dropna(subset=["pitch_x"]).groupby("frame").size().mean() if len(players) else 0.0
    err = dense["calib_error_m"].replace([float("inf")], pd.NA).dropna()
    return (f"{stem}: {dense['frame'].nunique()} frames, {accepted} accepted tactical, "
            f"{ppf:.1f} players/frame, calib {err.mean():.2f} m" if len(err)
            else f"{stem}: {dense['frame'].nunique()} frames, {accepted} accepted tactical")


def run_batch(chunks_dir: Path, out_dir: Path, *, sample_every: int, calib_period: int,
              calib_drift: float, detector: str, tracker: str, skip_existing: bool,
              limit: int | None, max_frames: int | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = sorted(chunks_dir.glob("chunk_*.mp4"))[: limit or None]
    if not chunks:
        raise SystemExit(f"no chunk_*.mp4 under {chunks_dir}")
    logger.info("batch: %d chunks, calib_period=%d (%s), out=%s", len(chunks), calib_period,
                "per-frame" if calib_period == 1 else "temporal", out_dir)

    calibrator = None
    if calib_period >= 0:  # build the heavy HRNet calibrator once, share across chunks
        from generator.calibrate import PnLCalibCalibrator  # noqa: PLC0415

        calibrator = PnLCalibCalibrator()  # device=None -> auto CUDA

    dense_all: list[pd.DataFrame] = []
    tac_all: list[pd.DataFrame] = []
    for i, chunk in enumerate(chunks):
        stem = chunk.stem  # e.g. "chunk_000"
        dense_out, tac_out = out_dir / f"{stem}_dense.parquet", out_dir / f"{stem}_tactical.parquet"
        t0 = time.time()
        try:
            if skip_existing and dense_out.exists() and tac_out.exists():
                dense, tac = pd.read_parquet(dense_out), pd.read_parquet(tac_out)
                logger.info("[%d/%d] skip (exists) %s", i + 1, len(chunks), _summarise(stem, dense, tac))
            else:
                dense = extract_positions(
                    chunk, dense_out, sample_every=sample_every, calibrator=calibrator,
                    detector_name=detector, tracker_name=tracker, max_frames=max_frames,
                    calib_period=calib_period, calib_drift=calib_drift,
                )
                tac = build_tactical_table(dense)
                tac.to_parquet(tac_out, index=False)
                logger.info("[%d/%d] %s (%.0fs)", i + 1, len(chunks),
                            _summarise(stem, dense, tac), time.time() - t0)
            dense_all.append(dense.assign(chunk=stem))
            tac_all.append(tac.assign(chunk=stem))
        except Exception:  # noqa: BLE001 - one bad chunk must not kill the match batch
            logger.exception("[%d/%d] FAILED on %s -- skipping", i + 1, len(chunks), stem)
        finally:
            # extract_positions builds a fresh detector per chunk; free its VRAM or it accumulates -> OOM
            try:
                import gc  # noqa: PLC0415

                import torch  # noqa: PLC0415

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    if dense_all:  # concatenated whole-match tables for the fingerprint engine
        match_dense, match_tac = out_dir.parent / "match_dense.parquet", out_dir.parent / "match_tactical.parquet"
        pd.concat(dense_all, ignore_index=True).to_parquet(match_dense, index=False)
        pd.concat(tac_all, ignore_index=True).to_parquet(match_tac, index=False)
        n_acc = sum(int(t["accepted"].sum()) for t in tac_all if len(t))
        logger.info("DONE: %d/%d chunks ok, %d accepted tactical frames -> %s",
                    len(dense_all), len(chunks), n_acc, match_dense)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks-dir", required=True, help="directory of chunk_*.mp4")
    ap.add_argument("--out-dir", default="outputs/match", help="per-chunk parquet output dir")
    ap.add_argument("--sample-every", type=int, default=5)
    ap.add_argument("--calib-period", type=int, default=25, help="temporal calib period (1 = per-frame)")
    ap.add_argument("--per-frame", action="store_true", help="per-frame calibration (overrides period)")
    ap.add_argument("--calib-drift", type=float, default=2.0)
    ap.add_argument("--detector", default="football")
    ap.add_argument("--tracker", default="bytetrack")
    ap.add_argument("--skip-existing", action="store_true", help="reuse already-written chunk parquets")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N chunks (testing)")
    ap.add_argument("--max-frames", type=int, default=None, help="cap frames per chunk (smoke testing)")
    args = ap.parse_args()
    run_batch(
        Path(args.chunks_dir), Path(args.out_dir),
        sample_every=args.sample_every, calib_period=1 if args.per_frame else args.calib_period,
        calib_drift=args.calib_drift, detector=args.detector, tracker=args.tracker,
        skip_existing=args.skip_existing, limit=args.limit, max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
