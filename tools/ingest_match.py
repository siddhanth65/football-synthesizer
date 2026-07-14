"""Ingest one match end-to-end: video chunks -> positions -> colour-anchored teams -> facet fingerprint.

The pipeline for scaling to many matches (roadmap #5). Drop a match's chunks at
``<chunks-dir>/chunk_*.mp4`` and this runs: per-chunk CV extraction (detection + tracking + PnLCalib),
cross-chunk jersey-colour team anchoring, and the per-team facet fingerprint (optionally enriched with
the GAT relational reads + pitch control + synchrony). Outputs land under ``outputs/matches/<name>/``.

Each ingested match yields one ``facets.parquet`` (its style fingerprint). Once several exist, the
style-distance / matchup machinery and the C5 synthesizer have real data to work with.

Run:
    python tools/ingest_match.py --name france_senegal \
        --chunks-dir "<...>/france_senegal/chunks" --enrich
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("ingest_match")


def ingest(name: str, chunks_dir: Path, *, out_root: Path = Path("outputs/matches"),
           calib_period: int = 25, enrich: bool = False) -> Path:
    """Run the full pipeline for one match; returns the facet-fingerprint parquet path."""
    from fingerprint.facet_metrics import match_facet_fingerprint
    from generator.team_anchor import anchor_teams
    from tools.batch_match import run_batch

    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)

    logger.info("[%s] 1/3 CV extraction over chunks ...", name)
    # run_batch writes per-chunk parquets in out/chunks and the combined match_dense at its parent (out).
    run_batch(chunks_dir, out / "chunks", sample_every=5, calib_period=calib_period, calib_drift=2.0,
              detector="football", tracker="bytetrack", skip_existing=True)
    dense = pd.read_parquet(out / "match_dense.parquet")

    logger.info("[%s] 2/3 jersey-colour team anchoring ...", name)
    video_map = {c: str(chunks_dir / f"{c}.mp4") for c in dense["chunk"].unique()}
    anchored = anchor_teams(dense, video_map)
    anchored.to_parquet(out / "anchored_dense.parquet", index=False)

    logger.info("[%s] 3/3 facet fingerprint%s ...", name, " (+ GAT/pitch-control)" if enrich else "")
    rel = space = sync = None
    if enrich:
        from fingerprint.pitch_control import space_control_metrics
        from fingerprint.style_metrics import team_synchrony
        from generator.sop_bridge import per_team_relational
        rel, space, sync = (per_team_relational(anchored), space_control_metrics(anchored, sample=400),
                            team_synchrony(anchored))
    fp = match_facet_fingerprint(anchored, relational=rel, space=space, synchrony=sync)
    from core.pitch import METRICS_VERSION
    fp["metrics_version"] = METRICS_VERSION  # stamp: which metric definitions produced this artifact
    fp_path = out / "facets.parquet"
    fp.to_parquet(fp_path, index=False)
    logger.info("[%s] done -> %s", name, fp_path)
    return fp_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="match identifier (e.g. france_senegal)")
    ap.add_argument("--chunks-dir", required=True, help="dir of chunk_*.mp4 for this match")
    ap.add_argument("--enrich", action="store_true", help="add GAT relational + pitch control + synchrony")
    args = ap.parse_args()
    ingest(args.name, Path(args.chunks_dir), enrich=args.enrich)


if __name__ == "__main__":
    main()
