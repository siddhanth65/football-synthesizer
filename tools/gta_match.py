"""Run the GTA-Link tracklet repair on a real broadcast match and report fragmentation.

The GSR benchmark (:mod:`eval.gsr_gta`) measures GTA against external ground truth. This tool asks
the question that matters for the attribution roadmap: on our own broadcast footage, does the repair
reduce **tracklets per named-player-half** -- the fragmentation that forces the naming factor in
``results/CARRIER_CONSTRAINED_v2.md`` down to 0.609?

Two seams:

* ``--build-cache`` (GPU): per-detection PRTreID embeddings for every person row of the match.
  Crops come from the chunk videos via :func:`generator.team_anchor.estimate_player_box` (the same
  foot-point box the identity chain uses), so no re-detection is needed.
* the report (CPU): split + connect per chunk (and optionally across chunks within a half), then

  1. **fragmentation** -- distinct tracklets per ``(half, player)`` before vs after, using the
     OCR-named tracks (``outputs/identity/<match>_named_tracks_both2_prtreid.parquet``);
  2. **merge precision on real footage** -- any merged group holding two *different* named players
     is a proven wrong merge. The anchor reads behind those names are 98.6% verified, so this is a
     real (if partial) precision audit with no new labelling.

Paths come from :mod:`core.registry`; artifacts are written next to the match's aligned parquet.

CLI::

    python -m tools.gta_match --match fulham_manutd --build-cache
    python -m tools.gta_match --match fulham_manutd --tau 0.04
    python -m tools.gta_match --match fulham_manutd --tau 0.04 --cross-chunk
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from core import registry
from generator.gta_link import (
    GTA_VERSION,
    GtaParams,
    apply_splits,
    connect,
    mean_embeddings,
    split_detection_embeddings,
    split_tracklets,
)
from generator.track_relink import summarize_fragments

logger = logging.getLogger("gta_match")

#: Named-track artifact produced by ``tools/wire_anchors.py`` (the B2 Stage-2c identity chain).
NAMED_TRACKS = Path("outputs/identity/{match}_named_tracks_both2_prtreid.parquet")
#: Minimum crop height (px) worth embedding -- below this PRTreID sees noise.
MIN_BOX_H = 24
#: Frames a chunk is padded by when chunks are laid onto one half-wide timeline.
CHUNK_SPAN = 20_000


def cache_dir(match_id: str) -> Path:
    """Per-detection embedding cache directory, derived from the registry (never hardcoded)."""
    return Path(registry.get(match_id).aligned).parent / "gta"


def video_for(match_id: str, chunk: str) -> Path:
    """Resolve a chunk key (``h1_chunk_003``) to its source video via the registry root."""
    return registry.VIDEO_ROOT / match_id / chunk[:2] / f"chunk_{chunk.split('_chunk_')[1]}.mp4"


# === GPU stage ===================================================================================
def chunk_detection_embeddings(
    video: Path, chunk_df: pd.DataFrame, embedder, *, stride: int, batch: int = 64,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Embed every ``stride``-th sampled frame's person crops of one chunk (GPU/IO).

    Args:
        video: The chunk's source video.
        chunk_df: Person rows of this chunk (needs ``frame, track_id, image_x, image_y``).
        embedder: An embedder exposing ``embed(crops) -> (N, D)`` L2-normalised features.
        stride: Take every Nth *sampled* frame present in ``chunk_df``.
        batch: Crops accumulated before one forward pass.

    Returns:
        ``{track_id: (frames, embeddings)}`` with ``frames`` ascending.
    """
    import cv2  # noqa: PLC0415

    from generator.team_anchor import estimate_player_box  # noqa: PLC0415

    cap = cv2.VideoCapture(str(video))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    frames = np.sort(chunk_df["frame"].astype(int).unique())[::max(stride, 1)]
    by_frame = {int(f): g for f, g in chunk_df.groupby("frame")}
    per_tid: dict[int, list[tuple[int, np.ndarray]]] = {}
    crops: list[np.ndarray] = []
    owners: list[tuple[int, int]] = []

    def flush() -> None:
        if not crops:
            return
        feats = embedder.embed(crops)
        for (tid, fr), f in zip(owners, feats):
            per_tid.setdefault(tid, []).append((fr, f))
        crops.clear()
        owners.clear()

    for frame_idx in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for r in by_frame[int(frame_idx)].itertuples():
            x1, y1, x2, y2 = estimate_player_box(float(r.image_x), float(r.image_y), fh, fw)
            if y2 - y1 < MIN_BOX_H or x2 - x1 < 8:  # noqa: PLR2004
                continue
            crop = rgb[y1:y2, x1:x2]
            if crop.size:
                crops.append(crop)
                owners.append((int(r.track_id), int(frame_idx)))
        if len(crops) >= batch:
            flush()
    flush()
    cap.release()
    return {tid: (np.array([f for f, _ in v], int), np.stack([e for _, e in v]))
            for tid, v in per_tid.items() if v}


def build_cache(match_id: str, stride: int) -> None:
    """GPU stage: build one ``.npz`` of per-detection embeddings per chunk (resumable)."""
    import time  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from tools.prtreid_probe import PrtreidEmbedder  # noqa: PLC0415

    df = pd.read_parquet(registry.get(match_id).aligned)
    people = df[(df["role"] != "ball") & np.isfinite(df["image_x"]) & np.isfinite(df["image_y"])]
    out_dir = cache_dir(match_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = sorted(people["chunk"].unique())
    cold = [c for c in chunks if not (out_dir / f"detemb_{c}.npz").exists()]
    logger.info("%s: %d chunks, %d cold, stride=%d", match_id, len(chunks), len(cold), stride)
    if not cold:
        return
    embedder = PrtreidEmbedder(device="cuda" if torch.cuda.is_available() else "cpu")
    for i, chunk in enumerate(cold):
        video = video_for(match_id, chunk)
        if not video.exists():
            logger.warning("missing video %s -- skipping %s", video, chunk)
            continue
        t0 = time.time()
        det = chunk_detection_embeddings(
            video, people[people["chunk"] == chunk], embedder, stride=stride)
        if not det:
            continue
        tids = np.concatenate([np.full(len(f), t, int) for t, (f, _) in sorted(det.items())])
        np.savez_compressed(
            out_dir / f"detemb_{chunk}.npz", version=np.array([GTA_VERSION]), track_ids=tids,
            frames=np.concatenate([f for _t, (f, _e) in sorted(det.items())]),
            embeddings=np.concatenate([e for _t, (_f, e) in sorted(det.items())]))
        logger.info("[%d/%d] %s: %d tracks, %d crops (%.0fs)", i + 1, len(cold), chunk,
                    len(det), sum(len(f) for f, _ in det.values()), time.time() - t0)


def load_chunk_cache(match_id: str, chunk: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load one chunk's per-detection embedding cache -> ``{track_id: (frames, embeddings)}``."""
    z = np.load(cache_dir(match_id) / f"detemb_{chunk}.npz")
    tids, frames, embs = z["track_ids"].astype(int), z["frames"].astype(int), z["embeddings"]
    order = np.lexsort((frames, tids))
    tids, frames, embs = tids[order], frames[order], embs[order]
    cuts = np.flatnonzero(np.diff(tids)) + 1
    return {int(t[0]): (f, e) for t, f, e in
            zip(np.split(tids, cuts), np.split(frames, cuts), np.split(embs, cuts))}


# === CPU repair + report =========================================================================
def repair_group(
    df: pd.DataFrame, det: dict, params: GtaParams, *, do_split: bool = True,
) -> tuple[dict[tuple[int, int], int], dict[int, int], dict]:
    """Split + connect one independent group of tracklets -> ``(row_lookup, remap, stats)``."""
    splits: dict[int, list[tuple[int, int, int]]] = {}
    sstat = {"n_tracks_embedded": len(det), "n_tracks_split": 0, "n_new_subtracks": 0}
    if do_split:
        splits, sstat = split_tracklets(df, det, params)
    sdf, lookup = apply_splits(df, splits)
    embs, counts = mean_embeddings(split_detection_embeddings(det, splits))
    fragments = summarize_fragments(sdf)
    remap = connect(fragments, embs, counts, params)
    return lookup, remap, {**sstat, "n_fragments_before": len(fragments),
                           "n_fragments_after": len(set(remap.values())),
                           "n_merges": len(fragments) - len(set(remap.values()))}


def repair_match(
    match_id: str, params: GtaParams, *, cross_chunk: bool, do_split: bool = True,
) -> tuple[dict[tuple[str, int], set[str]], dict]:
    """Repair per chunk (or per half, when ``cross_chunk``) -> ``({(chunk, tid): {final}}, stats)``.

    A tracklet maps to a *set* of final ids because the splitter may have cut it, so a split shows up
    in the fragmentation number as the increase it is rather than being silently collapsed.

    With ``cross_chunk`` the chunks of a half are laid onto one timeline (chunk index x
    :data:`CHUNK_SPAN` frames) and track ids are made globally unique, so a player who leaves the
    frame at a chunk boundary can be reconnected. The motion constraint is effectively free across
    such a gap, which is stated here because it makes cross-chunk merges rest on appearance alone.
    """
    df = pd.read_parquet(registry.get(match_id).aligned)
    people = df[df["role"] != "ball"].copy()
    final: dict[tuple[str, int], int] = {}
    stats: dict[str, dict] = {}
    chunks = sorted(people["chunk"].unique())
    groups = ({h: [c for c in chunks if c.startswith(h)] for h in ("h1", "h2")} if cross_chunk
              else {c: [c] for c in chunks})
    for gname, members in groups.items():
        parts, det = [], {}
        for k, chunk in enumerate(members):
            cache = cache_dir(match_id) / f"detemb_{chunk}.npz"
            if not cache.exists():
                logger.warning("no cache for %s -- skipped", chunk)
                continue
            sub = people[people["chunk"] == chunk].copy()
            offset = k * CHUNK_SPAN
            sub["frame"] = sub["frame"].astype(int) + offset
            sub["track_id"] = sub["track_id"].astype(int) + k * 1_000_000
            sub["src_chunk"] = chunk
            sub["src_track"] = sub["track_id"].astype(int) - k * 1_000_000
            parts.append(sub)
            for tid, (frames, embs) in load_chunk_cache(match_id, chunk).items():
                det[tid + k * 1_000_000] = (frames + offset, embs)
        if not parts:
            continue
        gdf = pd.concat(parts, ignore_index=True)
        lookup, remap, st = repair_group(gdf, det, params, do_split=do_split)
        stats[gname] = st
        for chunk, src, tid, frame in zip(gdf["src_chunk"], gdf["src_track"],
                                          gdf["track_id"], gdf["frame"]):
            sub_id = lookup.get((int(tid), int(frame)), int(tid))
            final.setdefault((str(chunk), int(src)), set()).add(
                f"{gname}:{int(remap.get(sub_id, sub_id))}")
        logger.info("%s: %d tracklets split, frags %d -> %d (%d merges)", gname,
                    st["n_tracks_split"], st["n_fragments_before"], st["n_fragments_after"],
                    st["n_merges"])
    return final, stats


def fragmentation_report(match_id: str, final: dict[tuple[str, int], set[str]]) -> dict:
    """Tracklets per named-player-half before/after + a merge audit against the OCR names.

    ``before`` counts the distinct OCR-named tracklets a ``(half, player)`` owns; ``after`` counts
    the distinct final ids those tracklets land on, so a split raises the count and a merge lowers
    it. The merge audit is the converse view: a final id carrying two *different* named players is a
    proven wrong merge (the anchor reads behind the names are 98.6% verified).
    """
    named = pd.read_parquet(str(NAMED_TRACKS).format(match=match_id))
    named["half"] = named["chunk"].str[:2]
    ids = [final.get((c, int(t)), set()) for c, t in zip(named["chunk"], named["track_id"])]
    named["n_final"] = [len(s) for s in ids]
    mapped = named[named["n_final"] > 0]
    before = named.groupby(["half", "player_name"]).size()
    after_sets: dict[tuple[str, str], set[str]] = {}
    owners: dict[str, set[str]] = {}
    for (half, player), s in zip(zip(named["half"], named["player_name"]), ids):
        after_sets.setdefault((half, player), set()).update(s)
        for fid in s:
            owners.setdefault(fid, set()).add(player)
    after = pd.Series({k: len(v) for k, v in after_sets.items() if v})
    multi = {fid: names for fid, names in owners.items()
             if sum(1 for s in ids if fid in s) > 1}
    mixed = {fid: sorted(names) for fid, names in multi.items() if len(names) > 1}
    return {
        "n_named_tracks": int(len(named)),
        "n_named_tracks_mapped": int(len(mapped)),
        "n_player_halves": int(len(before)),
        "tracklets_per_named_player_half_before": float(before.mean()),
        "tracklets_per_named_player_half_after": float(after.mean()) if len(after) else None,
        "median_before": float(before.median()),
        "median_after": float(after.median()) if len(after) else None,
        "max_before": int(before.max()), "max_after": int(after.max()) if len(after) else None,
        "n_named_tracks_split": int((named["n_final"] > 1).sum()),
        "n_multi_track_groups": len(multi),
        "n_groups_mixing_two_named_players": len(mixed),
        "merge_precision_named": float(1.0 - len(mixed) / max(len(multi), 1)),
        "mixed_groups": dict(list(mixed.items())[:10]),
        "worst_before": {f"{h}|{p}": int(v) for (h, p), v in
                         before.sort_values(ascending=False).head(5).items()},
    }


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", default="fulham_manutd")
    ap.add_argument("--tau", type=float, default=0.04)
    ap.add_argument("--eps", type=float, default=0.30)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--min-run", type=int, default=5)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--cross-chunk", action="store_true")
    ap.add_argument("--no-split", action="store_true", help="connector only")
    ap.add_argument("--build-cache", action="store_true", help="GPU: per-detection embeddings")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    if args.build_cache:
        build_cache(args.match, args.stride)
        return
    params = GtaParams(tau=args.tau, eps=args.eps, min_samples=args.min_samples,
                       min_run=args.min_run, frame_stride=args.stride)
    final, stats = repair_match(args.match, params, cross_chunk=args.cross_chunk,
                            do_split=not args.no_split)
    rep = fragmentation_report(args.match, final)
    print(f"\n=== GTA on {args.match} (tau={args.tau}, eps={args.eps}, "
          f"cross_chunk={args.cross_chunk}) ===")
    print(f"tracklets split                 : {sum(s['n_tracks_split'] for s in stats.values())}")
    print(f"fragments                       : "
          f"{sum(s['n_fragments_before'] for s in stats.values())} -> "
          f"{sum(s['n_fragments_after'] for s in stats.values())}")
    print(f"named tracks mapped             : {rep['n_named_tracks_mapped']}/"
          f"{rep['n_named_tracks']} over {rep['n_player_halves']} player-halves")
    print(f"tracklets per named-player-half : "
          f"{rep['tracklets_per_named_player_half_before']:.3f} -> "
          f"{rep['tracklets_per_named_player_half_after']:.3f} "
          f"(median {rep['median_before']:.1f} -> {rep['median_after']:.1f}, "
          f"max {rep['max_before']} -> {rep['max_after']})")
    print(f"merged groups holding 2+ names  : {rep['n_groups_mixing_two_named_players']}"
          f"/{rep['n_multi_track_groups']} "
          f"(named-merge precision {100 * rep['merge_precision_named']:.1f}%)")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"version": GTA_VERSION, "match": args.match, "params": vars(params),
             "cross_chunk": args.cross_chunk, "per_group": stats, "report": rep}, indent=2),
            encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
