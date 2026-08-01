"""Run the Stage-2 identity solve on a real broadcast match (lineup roster, per half).

Adapter between :mod:`generator.identity_solve` and this repo's match artifacts. Everything the
solver needs already exists on disk:

* tracklets -- the GTA connector (splitter OFF, ``tau`` frozen at 0.040) over the registry's aligned
  positions, exactly as :func:`tools.gta_carrier.merge_match` builds them;
* OCR reads -- ``outputs/identity/<match>_named_tracks_both2_prtreid.parquet`` (anchor reads,
  98.6% verified), entering as *soft votes* with the frozen digit-confusion prior;
* roster -- ``outputs/identity/<match>_lineup_assign.parquet``: name, shirt, position and the
  ``on_h1``/``on_h2`` availability flags that gate substitutions;
* appearance -- the per-detection PRTreID caches under ``<aligned parent>/gta/``.

Two things differ from the GSR driver on purpose. The identity gallery is pooled over the **whole
half** (every chunk's directly-read tracklets of that player), because cross-chunk appearance is the
only evidence that can name a tracklet in a chunk where nobody's number was legible. And the roster
carries the goalkeepers, whose shirt numbers OCR never reads, so the role gate is the only thing that
can fill those slots -- the defect ``results/MANUTD_IDENTITY_PROFILE.md`` records as
"0 of 1,664 named tracks is a goalkeeper".

The config is the one frozen on the SoccerNet-GSR DEV split; nothing is refitted here, so this is a
transfer check, not a tuned arm.

CLI::

    python -m tools.identity_match --matches manutd_liverpool,manutd_tottenham,manutd_brighton
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from core import registry
from generator.gta_link import GtaParams, connect, mean_embeddings
from generator.identity_solve import (
    SOLVER_VERSION,
    Identity,
    SolverConfig,
    Tracklet,
    exclusion_groups,
    posterior,
    solve_assignment,
)
from generator.track_relink import summarize_fragments
from tools.gta_match import NAMED_TRACKS, cache_dir, load_chunk_cache

logger = logging.getLogger("identity_match")

#: Roster table written by the lineup-assignment stage (name, shirt, position, availability).
LINEUP = Path("outputs/identity/{match}_lineup_assign.parquet")
#: Frozen Stage-1 partition (the 80.1%-merge-precision operating point).
PARTITION = GtaParams(tau=0.040)
#: Frozen Stage-2 calibration (fitted on the GSR DEV split only).
CONFIG_PATH = Path("results/identity_solver_config.json")
#: The shipped per-crop aggregation rule: the highest-density DEV floor that passed the OCR gate
#: (``results/OCR_DENSIFICATION.md`` §5, read precision 0.8578 on TEST-38).
PERCROP_FLOOR = "0.85"
# ponytail: a full match is ~800 tracklets per chunk against ~20 identities, so the raw
# every-crop-against-every-crop similarity is ~10^11 dot products. Both sides are subsampled
# evenly in time; raise the caps if the appearance term ever becomes the binding factor.
#: Max query crops per tracklet used for the gallery similarity.
QUERY_CAP = 32
#: Max crops kept per anchoring tracklet inside an identity gallery.
GALLERY_CAP = 32
#: Candidate identities kept per tracklet, and the HiGHS budget per chunk. A match chunk carries
# ~700 merged tracklets against a 22-slot roster; the unpruned program does not close.
MAX_CANDIDATES = 6
MILP_TIME_LIMIT_S = 60.0


def _cap(emb: np.ndarray, n: int) -> np.ndarray:
    """Evenly-spread subsample of at most ``n`` rows (pure)."""
    if len(emb) <= n:
        return emb
    return emb[np.unique(np.linspace(0, len(emb) - 1, n).astype(int))]


def roster(match_id: str, half: str) -> list[Identity]:
    """Lineup slots available in ``half`` (``h1``/``h2``) as solver identities."""
    lu = pd.read_parquet(str(LINEUP).format(match=match_id))
    on = lu[lu[f"on_{half}"].astype(bool)]
    return [Identity(key=str(r.name_), team=int(r.team), number=int(r.shirt),
                     role="goalkeeper" if str(r.position) == "G" else "player")
            for r in on.rename(columns={"name": "name_"}).itertuples()]


def _chunk_tracklets(
    sub: pd.DataFrame, det: dict[int, tuple[np.ndarray, np.ndarray]],
    reads: dict[int, list[tuple[int, float]]],
) -> tuple[list[Tracklet], dict[int, int], list[list[int]]]:
    """Merge one chunk's fragments and summarise them -> (tracklets, remap, alive-by-frame)."""
    embs, counts = mean_embeddings(det)
    remap = connect(summarize_fragments(sub), embs, counts, PARTITION)
    sub = sub.copy()
    sub["mid"] = sub["track_id"].map(lambda t: int(remap.get(int(t), int(t))))
    per_mid: dict[int, list] = defaultdict(list)
    for tid, (_f, e) in det.items():
        per_mid[int(remap.get(int(tid), int(tid)))].append(e)
    order = sorted(sub["mid"].unique().tolist())
    index = {m: i for i, m in enumerate(order)}
    tracklets: list[Tracklet] = []
    for mid in order:
        grp = sub[sub["mid"] == mid]
        tc = Counter(int(v) for v in grp["team"] if np.isfinite(v))
        team, frac = (None, 0.0)
        if tc:
            team, cnt = tc.most_common(1)[0]
            frac = cnt / sum(tc.values())
        rc = Counter(str(v) for v in grp["role"])
        n = sum(rc.values())
        merged_reads = [r for tid in grp["track_id"].astype(int).unique()
                        for r in reads.get(int(tid), [])]
        tracklets.append(Tracklet(
            track_id=int(mid), n_rows=int(len(grp)), team=team, team_frac=float(frac),
            role_frac={k: v / n for k, v in rc.items()}, reads=tuple(merged_reads),
            emb=np.concatenate(per_mid[mid]) if per_mid.get(mid) else np.zeros((0, 256), np.float32),
            span=(int(grp["frame"].min()), int(grp["frame"].max()))))
    alive = [[index[int(m)] for m in g["mid"].unique()] for _f, g in sub.groupby("frame")]
    return tracklets, remap, alive


def anchor_reads(match_id: str) -> dict[str, dict[int, list[tuple[int, float]]]]:
    """On-record evidence: the gated close-up anchor chain, one read per named tracklet."""
    named = pd.read_parquet(str(NAMED_TRACKS).format(match=match_id))
    reads: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for c, t, j in zip(named["chunk"], named["track_id"], named["jersey_number"]):
        if pd.notna(j):
            reads[str(c)][int(t)].append((int(j), 1.0))
    return reads


def percrop_reads(match_id: str, floor: str = PERCROP_FLOOR, *, variant: str = "",
                  min_votes: int | None = None
                  ) -> dict[str, dict[int, list[tuple[int, float]]]]:
    """Densified evidence: per-crop OCR (``tools.ocr_match``) under a frozen aggregation rule.

    Args:
        match_id: Registry match id.
        floor: Which DEV precision floor's frozen rule to apply (``results/ocr_density_rule.json``).
        variant: Which per-crop cache to read (``tools.ocr_match.percrop_dir``).
        min_votes: Override the frozen rule's vote bar. Exploratory only -- the shipped rule is
            ``floor="0.85"``, ``min_votes=5``, and any other value is post-hoc.

    Returns:
        ``{chunk: {track_id: [(number, confidence)]}}`` -- the shape :func:`anchor_reads` returns.
    """
    from tools.ocr_density import RULE_PATH, reads_for_sequence  # noqa: PLC0415
    from tools.ocr_match import percrop_dir  # noqa: PLC0415

    src = json.loads(RULE_PATH.read_text(encoding="utf-8"))["rules"][floor]
    rule = {k: v for k, v in src.items()
            if k in {"min_crop_conf", "min_votes", "emit_all", "min_legibility"}}
    if min_votes is not None:
        rule["min_votes"] = int(min_votes)
    reads: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(percrop_dir(match_id, variant).glob("*.parquet")):
        frame = pd.read_parquet(path)
        chunk = str(frame["chunk"].iloc[0])
        for tid, votes in reads_for_sequence(frame, rule).items():
            reads[chunk][int(tid)].extend(votes)
    return reads


def solve_match(match_id: str, cfg: SolverConfig, *, percrop: bool = False,
                floor: str = PERCROP_FLOOR, variant: str = "", min_votes: int | None = None
                ) -> tuple[dict[tuple[str, int], str], dict[tuple[str, int], float], dict]:
    """Solve every chunk of both halves -> ``(names, posteriors, stats)``.

    The gallery for an identity pools every directly-read tracklet of that player in the same half,
    with the query's own tracklet excluded, so a name read in one chunk can reach another.

    Args:
        match_id: Registry match id.
        cfg: Frozen solver configuration.
        percrop: Consume :func:`percrop_reads` instead of the close-up :func:`anchor_reads`.
        floor: Frozen per-crop rule to use when ``percrop`` is set.
        variant: Per-crop cache variant (crop geometry) to consume.
        min_votes: Exploratory override of the frozen rule's vote bar.

    Returns:
        ``({(chunk, track_id): player_name}, {(chunk, track_id): posterior}, stats)``. The posterior
        is the MILP's probability for the identity it assigned -- the dial every
        coverage-at-precision frontier needs.

    Note:
        The MILP solve unit is the **chunk** (~10 min of play, ~15k frames), not the half: frame
        numbering restarts per chunk, so the mutual-exclusion groups are only defined within one.
        Galleries *are* pooled half-wide.
    """
    match = registry.get(match_id)
    df = pd.read_parquet(match.aligned)
    people = df[df["role"].isin(["player", "goalkeeper"])].copy()
    reads_by_chunk = (percrop_reads(match_id, floor, variant=variant, min_votes=min_votes)
                      if percrop else anchor_reads(match_id))

    out: dict[tuple[str, int], str] = {}
    conf: dict[tuple[str, int], float] = {}
    stats = Counter()
    for half in ("h1", "h2"):
        ids = roster(match_id, half)
        chunks = sorted(c for c in people["chunk"].unique() if str(c).startswith(half))
        prepared: dict[str, tuple] = {}
        for chunk in chunks:
            cache = cache_dir(match_id) / f"detemb_{chunk}.npz"
            if not cache.exists():
                logger.warning("%s %s: no per-detection cache, skipping", match_id, chunk)
                continue
            sub = people[people["chunk"] == chunk]
            prepared[chunk] = _chunk_tracklets(sub, load_chunk_cache(match_id, chunk),
                                               reads_by_chunk[str(chunk)])
        # Half-wide galleries keyed by identity index: every tracklet whose own read is that shirt,
        # tagged with its source so a query is excluded from its own gallery (and nothing else is).
        anchors: dict[int, list[tuple[tuple[str, int], np.ndarray]]] = defaultdict(list)
        for chunk, (tracklets, _remap, _alive) in prepared.items():
            for trk in tracklets:
                if not trk.reads or trk.emb.size == 0:
                    continue
                num = Counter(n for n, _c in trk.reads).most_common(1)[0][0]
                for i, ident in enumerate(ids):
                    if ident.number == num and ident.team == trk.team:
                        anchors[i].append(((str(chunk), trk.track_id), _cap(trk.emb, GALLERY_CAP)))
        gallery = {i: (np.concatenate([e for _s, e in v]),
                       np.concatenate([[j] * len(e) for j, (_s, e) in enumerate(v)]),
                       [s for s, _e in v])
                   for i, v in anchors.items() if v}
        for chunk, (tracklets, remap, alive) in prepared.items():
            if not tracklets:
                continue
            sims = np.full((len(tracklets), len(ids)), np.nan)
            for k, trk in enumerate(tracklets):
                if trk.emb.size == 0:
                    continue
                q = _cap(trk.emb, QUERY_CAP)
                own = (str(chunk), trk.track_id)
                for i, (emb, src, keys) in gallery.items():
                    keep = src != next((j for j, s in enumerate(keys) if s == own), -1)
                    if not keep.any():
                        continue
                    sims[k, i] = float(np.mean(np.sort((q @ emb[keep].T).ravel())[-cfg.topk:]))
            probs = np.stack([posterior(t, ids, sims[k], cfg) for k, t in enumerate(tracklets)])
            probs[:, -1] = cfg.r_abstain
            pick = solve_assignment(
                probs, np.array([t.n_rows for t in tracklets], float),
                exclusion_groups(alive) if cfg.use_mutex else [], [i.team for i in ids],
                max_concurrent=cfg.max_concurrent, max_candidates=MAX_CANDIDATES,
                time_limit=MILP_TIME_LIMIT_S)
            by_mid = {t.track_id: (None if p < 0 else ids[p]) for t, p in zip(tracklets, pick)}
            # The MILP's own probability for the identity it chose. Persisted because every
            # coverage-at-precision frontier needs a dial, and this one was being thrown away.
            post = {t.track_id: (float(probs[k, pick[k]]) if pick[k] >= 0 else 0.0)
                    for k, t in enumerate(tracklets)}
            for tid in people[people["chunk"] == chunk]["track_id"].astype(int).unique():
                mid = int(remap.get(int(tid), int(tid)))
                ident = by_mid.get(mid)
                if ident is not None:
                    out[(str(chunk), int(tid))] = str(ident.key)
                    conf[(str(chunk), int(tid))] = post.get(mid, 0.0)
            stats["tracklets"] += len(tracklets)
            stats["named"] += sum(1 for v in by_mid.values() if v is not None)
            stats["gk_tracklets"] += sum(1 for t in tracklets
                                         if t.role_frac.get("goalkeeper", 0.0) > 0.5)
            stats["gk_named"] += sum(
                1 for t, p in zip(tracklets, pick)
                if t.role_frac.get("goalkeeper", 0.0) > 0.5 and p >= 0)
    return out, conf, dict(stats)


def gk_report(match_id: str, names: dict[tuple[str, int], str]) -> dict:
    """How many goalkeeper slots and goalkeeper-role tracklets the name set actually covers."""
    lu = pd.read_parquet(str(LINEUP).format(match=match_id))
    keepers = {str(r["name"]) for _i, r in lu.iterrows() if str(r["position"]) == "G"}
    assigned = set(names.values())
    df = pd.read_parquet(registry.get(match_id).aligned)
    gk_rows = df[df["role"] == "goalkeeper"]
    gk_tracks = {(str(c), int(t)) for c, t in zip(gk_rows["chunk"], gk_rows["track_id"])}
    named_gk = [names[k] for k in gk_tracks if k in names]
    return {
        "keepers_in_lineup": sorted(keepers),
        "keepers_named": sorted(keepers & assigned),
        "n_keepers_named": len(keepers & assigned),
        "n_gk_role_tracks_named_with_a_keeper": sum(1 for p in named_gk if p in keepers),
        "n_gk_role_tracks": len(gk_tracks),
        "n_gk_role_tracks_named": sum(1 for k in gk_tracks if k in names),
        "n_named_tracks": len(names),
    }


def baseline_names(match_id: str) -> dict[tuple[str, int], str]:
    """The Stage-1 arm's name set: connector merges + unanimous OCR-name propagation."""
    from tools.gta_carrier import merge_match, propagate_names  # noqa: PLC0415

    merged, _ = merge_match(match_id, PARTITION)
    names, _ = propagate_names(match_id, merged)
    return names


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matches", default="manutd_liverpool,manutd_tottenham,manutd_brighton")
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--out", type=Path, default=Path("results/identity_match_solver.json"))
    ap.add_argument("--names-dir", type=Path, default=Path("outputs/identity/solver"))
    ap.add_argument("--percrop", action="store_true",
                    help="consume per-crop OCR evidence (tools.ocr_match) instead of close-ups")
    ap.add_argument("--floor", default=PERCROP_FLOOR, help="frozen per-crop rule, with --percrop")
    ap.add_argument("--percrop-variant", default="",
                    help="per-crop cache variant (crop geometry), e.g. _w125")
    ap.add_argument("--min-votes", type=int, default=None,
                    help="EXPLORATORY override of the frozen rule's vote bar")
    args = ap.parse_args()
    cfg = SolverConfig.load(args.config)
    payload = {"version": SOLVER_VERSION, "percrop": args.percrop, "floor": args.floor,
               "percrop_variant": args.percrop_variant, "min_votes": args.min_votes,
               "config": {k: v for k, v in vars(cfg).items() if k != "confusion"}, "matches": {}}
    args.names_dir.mkdir(parents=True, exist_ok=True)
    for match_id in args.matches.split(","):
        names, conf, stats = solve_match(match_id, cfg, percrop=args.percrop, floor=args.floor,
                                         variant=args.percrop_variant, min_votes=args.min_votes)
        # The Stage-1 baseline needs the close-up anchor chain; skip it where that artifact cannot
        # exist (e.g. FOOTPASS: anonymised players, no usable name captions).
        has_anchors = Path(str(NAMED_TRACKS).format(match=match_id)).exists()
        base = baseline_names(match_id) if has_anchors else {}
        payload["matches"][match_id] = {
            "solver": {**stats, **gk_report(match_id, names)},
            "baseline": gk_report(match_id, base) if has_anchors else None,
        }
        pd.DataFrame([{"chunk": c, "track_id": t, "player_name": p,
                       "confidence": conf.get((c, t), 0.0)}
                      for (c, t), p in sorted(names.items())]).to_parquet(
            args.names_dir / f"{match_id}_solver_names.parquet", index=False)
        s = payload["matches"][match_id]
        if not has_anchors:
            logger.info("%s: solver names %d tracks (%d GK-role of %d); no anchor baseline",
                        match_id, s["solver"]["n_named_tracks"],
                        s["solver"]["n_gk_role_tracks_named"], s["solver"]["n_gk_role_tracks"])
            continue
        logger.info("%s: solver names %d tracks (%d GK-role of %d, %d keeper slots filled); "
                    "baseline names %d tracks (%d GK-role, %d keeper slots)", match_id,
                    s["solver"]["n_named_tracks"], s["solver"]["n_gk_role_tracks_named"],
                    s["solver"]["n_gk_role_tracks"], s["solver"]["n_keepers_named"],
                    s["baseline"]["n_named_tracks"], s["baseline"]["n_gk_role_tracks_named"],
                    s["baseline"]["n_keepers_named"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
