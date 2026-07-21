"""B2 Stage-2c runner: wire the 226 gated close-up anchors into named tactical tracks.

Reads the step-3 survivor crops (``results/closeup_anchor_probe/spotcheck_step3/_survivors/``),
attaches each to a ByteTrack fragment by ReID margin (:mod:`generator.anchor_wire`), applies the
propagation guard, resolves numbers to names via the cached Sofascore oracle, writes the named-tracks
parquet, and validates a visible-minutes proxy against the oracle.

Run (GPU, one job -- OSNet embeddings only, no re-detection)::

    python -m tools.wire_anchors
    python -m tools.wire_anchors --match manutd_liverpool

Outputs (``<match>`` = ``--match``, ``brighton_manutd`` keeps its original bare filenames):
    outputs/identity/<match>_named_tracks<tag>.parquet
    results/identity/NAMED_TRACKS.md                    (brighton_manutd)
    results/identity/NAMED_TRACKS_<match><tag>.md        (any other match)
"""
from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from core import registry
from generator import anchor_wire as aw
from generator import live_play
from generator.team_anchor import estimate_player_box
from generator.track_relink import OsnetEmbedder
from tools.action_spot_probe import SOFASCORE_MATCH_ID

MATCH_ID = "brighton_manutd"
SURVIVORS = Path("results/closeup_anchor_probe/spotcheck_step3/_survivors")
VIDEO_ROOT = Path("matches")
OUT_PARQUET = Path("outputs/identity/brighton_manutd_named_tracks.parquet")
REPORT = Path("results/identity/NAMED_TRACKS.md")
NEAR_WINDOW_S = 2.0  # matches the probe's cut window
MAX_CAND_CROPS = 25  # safety cap on candidate crops embedded per wide frame


def _oracle_path(match_id: str) -> Path:
    """Sofascore player-stats parquet for a registered match (ids shared with action_spot_probe)."""
    return Path(f"outputs/oracle/sofascore/player_stats_{SOFASCORE_MATCH_ID[match_id]}.parquet")


def _out_paths(match_id: str) -> tuple[Path, Path]:
    """``(out_parquet, report)`` base paths for a match; brighton_manutd keeps its bare filenames."""
    if match_id == "brighton_manutd":
        return OUT_PARQUET, REPORT
    return (Path(f"outputs/identity/{match_id}_named_tracks.parquet"),
            Path(f"results/identity/NAMED_TRACKS_{match_id}.md"))


def _team_id_resolver(match: registry.Match, oracle: pd.DataFrame):
    """Build ``oracle teamName -> positions-table team id`` from the registry's ``match.teams``.

    ``match.teams[i]`` (registry order) is matched to the oracle's actual ``teamName`` strings by
    first-word substring overlap (e.g. registry ``"Man Utd"`` <-> oracle ``"Manchester United"``).
    """
    oracle_names = sorted(oracle["teamName"].dropna().unique())
    id_by_name: dict[str, int] = {}
    for tid, reg_name in enumerate(match.teams):
        tok = reg_name.split()[0].lower()
        for on in oracle_names:
            if tok in on.lower() or on.split()[0].lower() in reg_name.lower():
                id_by_name[on] = tid
    missing = set(oracle_names) - set(id_by_name)
    if missing:
        print(f"WARN oracle team name(s) unresolved against registry teams {match.teams}: {missing}")

    def _team_id(team_name: str) -> int | None:
        return id_by_name.get(team_name)

    return _team_id


def _video_for(chunk_key: str, match_id: str = MATCH_ID) -> Path:
    half, num = chunk_key.split("_chunk_")
    return VIDEO_ROOT / match_id / half / f"chunk_{num}.mp4"


def _ascii(s: str) -> str:
    """cp1252-safe rendering for console prints (accents -> nearest ASCII)."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode() or "?"


def _embedder_label(model_name: str, weights: str | None, embedder: str = "osnet") -> str:
    """Report label naming the ACTUAL ReID embedder used (backbone + weights source).

    ``OsnetEmbedder(weights=None)`` loads ImageNet-classification weights; any non-None ``weights`` (a
    model-zoo key or checkpoint path) loads re-ID-objective weights. The label must reflect what was
    passed so an AIN/MSMT run is never mislabelled as the ImageNet baseline.
    """
    if embedder == "prtreid":
        return "football-domain PRTreID (part-based BPBreID/HRNet-32, SoccerNet-trained)"
    if weights is None:
        return f"ImageNet-classification OSNet (`{model_name}`)"
    return f"re-ID-objective OSNet (`{model_name}`, weights=`{weights}`)"


def build_embedder(embedder: str, model_name: str, weights: str | None):
    """Build the appearance embedder for the attachment gate (``osnet`` default, or ``prtreid``).

    Both classes expose the same ``embed(crops) -> (N, D)`` L2-normalised interface, so the
    attachment code path is identical across arms.

    Args:
        embedder: ``"osnet"`` (default, back-compatible) or ``"prtreid"``.
        model_name: OSNet backbone (ignored for the PRTreID arm).
        weights: OSNet re-ID weights key/path (ignored for the PRTreID arm).

    Returns:
        An embedder instance.
    """
    if embedder == "prtreid":
        from tools.prtreid_probe import PrtreidEmbedder  # noqa: PLC0415

        return PrtreidEmbedder()
    return OsnetEmbedder(model_name, weights=weights)


def load_anchors(survivors: Path = SURVIVORS) -> list[aw.Anchor]:
    """Parse the step-3 (or step-4 best-arm) survivor crops into :class:`anchor_wire.Anchor` rows."""
    anchors: list[aw.Anchor] = []
    for p in sorted(survivors.glob("*.jpg")):
        parsed = aw.parse_survivor_name(p.name)
        if parsed is None:
            continue
        chunk, frame, number, conf = parsed
        anchors.append(aw.Anchor(chunk, frame, number, conf, str(p)))
    return anchors


def wide_frames_by_chunk(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """``chunk -> sorted array of live_wide frame indices`` via the shipped shot classifier."""
    out: dict[str, np.ndarray] = {}
    for chunk, dfc in df.groupby("chunk", sort=True):
        cls = live_play.classify_dense(dfc)
        out[chunk] = np.array(
            sorted(int(f) for f in cls.index[cls["shot_type"] == live_play.SHOT_LIVE_WIDE]), int)
    return out


def build_candidate_embeddings(
    df: pd.DataFrame, need: dict[str, set[int]], embedder: OsnetEmbedder,
    match_id: str = MATCH_ID,
) -> dict[tuple[str, int], dict[int, tuple[np.ndarray, int]]]:
    """Embed every player/GK track crop on each needed wide frame (one video read per frame).

    Args:
        df: The match positions table.
        need: ``chunk -> {wide_frame indices needed}``.
        embedder: The OSNet appearance embedder.
        match_id: Registry match id (selects the broadcast chunk video root).

    Returns:
        ``(chunk, wide_frame) -> {track_id: (embedding, team)}``.
    """
    cache: dict[tuple[str, int], dict[int, tuple[np.ndarray, int]]] = {}
    players = df[df["role"].isin(live_play.PLAYER_ROLES)]
    for chunk, frames in need.items():
        if not frames:
            continue
        video = _video_for(chunk, match_id)
        if not video.exists():
            print(f"WARN missing video {video}")
            continue
        cap = cv2.VideoCapture(str(video))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        dchunk = players[players["chunk"] == chunk]
        for wf in sorted(frames):
            rows = dchunk[dchunk["frame"] == wf]
            if rows.empty:
                cache[(chunk, wf)] = {}
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, wf)
            ok, bgr = cap.read()
            if not ok:
                cache[(chunk, wf)] = {}
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            crops, meta = [], []
            for r in rows.itertuples(index=False):
                x1, y1, x2, y2 = estimate_player_box(r.image_x, r.image_y, fh, fw)
                crop = rgb[y1:y2, x1:x2]
                if crop.size:
                    crops.append(crop)
                    meta.append((int(r.track_id), int(r.team)))
                if len(crops) >= MAX_CAND_CROPS:
                    break
            feats = embedder.embed(crops) if crops else np.zeros((0, 1), np.float32)
            cache[(chunk, wf)] = {
                tid: (feats[i], team) for i, (tid, team) in enumerate(meta)}
        cap.release()
    return cache


def embed_anchor_crops(anchors: list[aw.Anchor], embedder: OsnetEmbedder) -> dict[str, np.ndarray]:
    """Embed each survivor crop; return ``crop_path -> L2-normalised embedding``."""
    crops, paths = [], []
    for a in anchors:
        bgr = cv2.imread(a.crop_path)
        if bgr is None:
            continue
        crops.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        paths.append(a.crop_path)
    feats = embedder.embed(crops) if crops else np.zeros((0, 1), np.float32)
    return {p: feats[i] for i, p in enumerate(paths)}


def attach_anchors(
    anchors: list[aw.Anchor], anchor_embs: dict[str, np.ndarray],
    cand_cache: dict[tuple[str, int], dict[int, tuple[np.ndarray, int]]],
    wide_by_chunk: dict[str, np.ndarray], teams_by_number: dict[int, set[int]],
    win_by_chunk: dict[str, int], *, min_sim: float = aw.REID_MIN_SIM,
    min_margin: float = aw.REID_MIN_MARGIN,
) -> tuple[list[aw.Attachment], Counter]:
    """Attach each anchor to a fragment by the ReID margin gate; tally unattached reasons."""
    attachments: list[aw.Attachment] = []
    reasons: Counter = Counter()
    for a in anchors:
        emb = anchor_embs.get(a.crop_path)
        if emb is None:
            reasons["no_anchor_crop"] += 1
            continue
        wf = aw.nearest_wide_frame(a.frame, wide_by_chunk.get(a.chunk, np.array([], int)),
                                   win_by_chunk[a.chunk])
        if wf is None:
            reasons["no_wide_frame"] += 1
            continue
        teams = teams_by_number.get(a.number, {0, 1})
        cands = cand_cache.get((a.chunk, wf), {})
        sims = {tid: float(np.dot(emb, cemb)) for tid, (cemb, team) in cands.items()
                if team in teams and cemb.shape == emb.shape}
        tid, basis = aw.choose_track(sims, min_margin=min_margin, min_sim=min_sim)
        if tid is None:
            reasons[basis] += 1
            continue
        team = cands[tid][1]
        attachments.append(aw.Attachment(a.chunk, tid, a.number, team, a.conf, basis))
    return attachments, reasons


def visible_seconds_by_player(
    resolved: list[aw.ResolvedFragment], df: pd.DataFrame, match: registry.Match,
) -> dict[tuple[int, int], float]:
    """Visible-time proxy (seconds) per ``(team, number)``: union of its fragments' frame spans.

    A fragment's on-screen span is ``(max_frame - min_frame) / fps``; spans of the same player within
    a chunk are unioned (:func:`anchor_wire.merge_intervals`) and summed across chunks.
    """
    spans: dict[tuple[int, int], dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list))
    idx = df.set_index(["chunk", "track_id"]).sort_index()
    for r in resolved:
        try:
            rows = idx.loc[(r.chunk, r.track_id)]
        except KeyError:
            continue
        frames = rows["frame"]
        lo, hi = int(np.min(frames)), int(np.max(frames))
        spans[(r.team, r.number)][r.chunk].append((lo, hi))
    out: dict[tuple[int, int], float] = {}
    for key, by_chunk in spans.items():
        total_frames = 0.0
        for chunk, intervals in by_chunk.items():
            total_frames += aw.merge_intervals(intervals) / match.chunk_fps(chunk)
        out[key] = round(total_frames, 1)
    return out


def _spearman(a: list[float], b: list[float]) -> float | None:
    """Spearman rank correlation of two equal-length lists (None if < 3 points)."""
    if len(a) < 3:
        return None
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return None
    return round(float(np.corrcoef(ra, rb)[0, 1]), 3)


def main(model_name: str = "osnet_x0_25", weights: str | None = None,
         survivors: Path = SURVIVORS, tag: str | None = None,
         precision_pending: bool = False, match_id: str = MATCH_ID,
         embedder_name: str = "osnet", min_sim: float = aw.REID_MIN_SIM,
         min_margin: float = aw.REID_MIN_MARGIN) -> None:
    """Attach -> guard -> name -> validate; write the named-tracks parquet and report.

    Args:
        model_name: OSNet backbone for the ReID embedder (``osnet_x1_0``/``osnet_ain_x1_0`` for
            re-ID-objective weights).
        weights: torchreid re-ID weights key/path, or ``None`` for the ImageNet baseline embedder.
            When set, outputs default to ``*_<weights>`` variant paths so the baseline artifacts
            (ImageNet run) are never clobbered.
        survivors: directory of gated anchor survivor crops to wire (step-3 by default; a step-4
            best-arm survivor dir for the lever re-measurement).
        tag: explicit output-path suffix, overriding the ``weights``-derived one (so a best-arm
            re-wire on the AIN embedder writes to its own artifacts).
        precision_pending: when set, the report marks every read precision-UNVERIFIED (the Koshkina
            PARSeq arm, whose confidence gate is toothless and whose spot-check is not yet in).
        match_id: registry match id to wire (default ``brighton_manutd``, output paths and Sofascore
            oracle id all resolve from this).
        embedder_name: ``"osnet"`` (default, reproduces every shipped artifact) or ``"prtreid"``.
        min_sim: Minimum best cosine for the attachment gate. Cosine scale is embedder-specific --
            PRTreID's same-kit distribution sits far higher than OSNet's, so its operating point is
            set from the GT-audited sweep in ``tools/prtreid_probe.py``, not shared with OSNet.
        min_margin: Minimum best-minus-second-best cosine gap for the attachment gate.
    """
    tag = tag if tag is not None else (f"_{weights}" if weights else "")
    if embedder_name != "osnet" and not tag:
        tag = f"_{embedder_name}"
    base_out_parquet, base_report = _out_paths(match_id)
    out_parquet = base_out_parquet.with_name(base_out_parquet.stem + tag + base_out_parquet.suffix)
    report = base_report.with_name(base_report.stem + tag + base_report.suffix)
    match = registry.get(match_id)
    df = match.load_aligned()
    oracle = pd.read_parquet(_oracle_path(match_id))
    team_id = _team_id_resolver(match, oracle)
    name_by, teams_by = aw.build_roster_maps(oracle, team_id, number_col="shirtNumber")
    # jerseyNumber cross-map (only to quantify the shirtNumber-vs-jerseyNumber discrepancy).
    jersey_by, _ = aw.build_roster_maps(oracle, team_id, number_col="jerseyNumber")

    anchors = load_anchors(survivors)
    print(f"loaded {len(anchors)} survivor anchors")
    hist = Counter(a.number for a in anchors)
    print(f"anchor number histogram: {dict(sorted(hist.items()))}")

    wide_by_chunk = wide_frames_by_chunk(df)
    win_by_chunk = {c: int(round(NEAR_WINDOW_S * match.chunk_fps(c))) for c in df["chunk"].unique()}

    # Which wide frames do we need candidate embeddings for?
    need: dict[str, set[int]] = defaultdict(set)
    for a in anchors:
        wf = aw.nearest_wide_frame(a.frame, wide_by_chunk.get(a.chunk, np.array([], int)),
                                   win_by_chunk[a.chunk])
        if wf is not None:
            need[a.chunk].add(wf)
    print(f"need candidate embeddings on {sum(len(v) for v in need.values())} wide frames")

    embedder = build_embedder(embedder_name, model_name, weights)
    print(f"embedder={embedder_name} ({model_name}, weights={weights}) on {embedder.device}; "
          f"gate min_sim={min_sim} min_margin={min_margin}")
    anchor_embs = embed_anchor_crops(anchors, embedder)
    cand_cache = build_candidate_embeddings(df, need, embedder, match_id)

    attachments, reasons = attach_anchors(anchors, anchor_embs, cand_cache, wide_by_chunk,
                                          teams_by, win_by_chunk, min_sim=min_sim,
                                          min_margin=min_margin)
    print(f"attached {len(attachments)} / {len(anchors)} anchors; unattached: {dict(reasons)}")

    # No match in this registry carries a relink remap (relink is benchmark-side only, 35% merge
    # precision) -- every fragment stands alone, so the >=2-anchor merge clause is not exercised on
    # real data here (it is covered by the synthetic test). Pass remap=None.
    resolved, flags = aw.resolve_identities(attachments, remap=None)
    print(f"resolved {len(resolved)} named fragments; {len(flags)} disagreement flags")

    rows = []
    unmatched_numbers: Counter = Counter()
    team_name = {0: match.teams[0], 1: match.teams[1]}
    for r in resolved:
        name = name_by.get((r.team, r.number))
        if name is None:
            unmatched_numbers[(r.team, r.number)] += 1
            continue
        rows.append({
            "chunk": r.chunk, "track_id": r.track_id, "player_name": name,
            "jersey_number": r.number, "team": team_name[r.team],
            "n_anchors": r.n_anchors, "confidence_basis": r.basis,
        })
    named = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    named.to_parquet(out_parquet, index=False)
    print(f"wrote {len(named)} named-track rows -> {out_parquet}")

    _write_report(match, df, oracle, anchors, hist, reasons, attachments, resolved, flags,
                  named, name_by, jersey_by, unmatched_numbers, team_name, report, team_id,
                  precision_pending=precision_pending, model_name=model_name, weights=weights,
                  embedder_name=embedder_name, min_sim=min_sim, min_margin=min_margin)


def _write_report(match, df, oracle, anchors, hist, reasons, attachments, resolved, flags, named,
                  name_by, jersey_by, unmatched_numbers, team_name, report=REPORT,
                  team_id=None, *,
                  precision_pending: bool = False, model_name: str = "osnet_x0_25",
                  weights: str | None = None, embedder_name: str = "osnet",
                  min_sim: float = aw.REID_MIN_SIM,
                  min_margin: float = aw.REID_MIN_MARGIN) -> None:
    """Write results/identity/NAMED_TRACKS<_match>.md: funnel, guard decisions, validation, caveats."""
    if team_id is None:
        team_id = _team_id_resolver(match, oracle)
    n_attached = len(attachments)
    # Validation table: per named player vs oracle.
    vis = visible_seconds_by_player(resolved, df, match)
    orc = oracle.copy()
    orc["_num"] = pd.to_numeric(orc["shirtNumber"], errors="coerce")
    orc["_team"] = orc["teamName"].map(team_id)
    orc_by = {(int(t), int(n)): row for row, t, n in
              zip(orc.to_dict("records"), orc["_team"], orc["_num"])
              if pd.notna(t) and pd.notna(n)}

    val_rows = []
    for (team, num), sec in sorted(vis.items(), key=lambda kv: -kv[1]):
        name = name_by.get((team, num), f"#{num}")
        frags = [r for r in resolved if r.team == team and r.number == num]
        n_anc = sum(r.n_anchors for r in frags)
        o = orc_by.get((team, num), {})
        val_rows.append({
            "player": name, "team": team_name[team], "number": num,
            "our_anchors": n_anc, "our_named_frags": len(frags),
            "our_visible_min": round(sec / 60.0, 1),
            "oracle_min": o.get("minutesPlayed"), "oracle_touches": o.get("touches"),
        })
    vdf = pd.DataFrame(val_rows)
    with_oracle = vdf[vdf["oracle_min"].notna()] if not vdf.empty else vdf
    rho = (_spearman(with_oracle["our_visible_min"].tolist(),
                     with_oracle["oracle_min"].tolist()) if len(with_oracle) >= 3 else None)

    # shirtNumber vs jerseyNumber discrepancy (the number->name contradiction).
    disc = []
    for row in oracle.itertuples(index=False):
        sn, jn = str(getattr(row, "shirtNumber")), str(getattr(row, "jerseyNumber"))
        if sn != jn and sn not in ("None", "nan") and jn not in ("None", "nan"):
            disc.append((aw._norm_name(getattr(row, "name")), jn, sn))

    prec_phrase = ("precision PENDING -- see banner" if precision_pending
                   else "98.6% verified read precision")
    lines: list[str] = []
    lines.append(f"# Named tracks -- {match.id} (B2 Stage-2c: anchors wired to tracks)\n")
    if precision_pending:
        lines.append("> **PRECISION PENDING (Koshkina arm).** These reads come from the Koshkina "
                     "PARSeq recognizer (`--reader koshkina` both2 arm), NOT the shipped easyocr "
                     "reader. PARSeq emits ~1.0 confidence on nearly every crop, so the confidence "
                     "gate is toothless here and the kit + OCR-agreement gates are the sole "
                     "precision guard. Montage first-pass looks >=95%, but human spot-check "
                     "verdicts on `montage_new_both2.png` are NOT YET in. Treat every number below "
                     "as precision-UNVERIFIED.\n")
    lines.append(f"Generated by `tools/wire_anchors.py` from the {len(anchors)} gated "
                 f"close-up anchors ({prec_phrase}). ReID attachment uses "
                 f"{_embedder_label(model_name, weights, embedder_name)}; margin gate "
                 f"`min_margin={min_margin}`, `min_sim={min_sim}` (frozen by reasoning -- no "
                 "annotated anchor->track map exists for this match).\n")

    lines.append("## Number->name source: shirtNumber, NOT jerseyNumber (contradicts the task "
                 "premise)\n")
    lines.append("The recognizer reads the physical back-of-shirt number. In the oracle parquet that "
                 "is **`shirtNumber`**, not `jerseyNumber` -- the two disagree for "
                 f"{len(disc)} players, e.g.:\n")
    lines.append("| player | jerseyNumber | shirtNumber (back) |")
    lines.append("|---|---|---|")
    for name, jn, sn in disc[:8]:
        lines.append(f"| {name} | {jn} | {sn} |")
    lines.append("")
    lines.append("Decisive case: our verified `20` back-reads are Diogo Dalot (shirtNumber 20); no "
                 "Man Utd player has jerseyNumber 20 (Dalot's jerseyNumber is 2), so under "
                 "`jerseyNumber` every verified Man Utd `20` anchor would be **unmatchable**. This "
                 "module uses `shirtNumber` and cross-checks `jerseyNumber` only to quantify the "
                 "gap.\n")

    lines.append("## Attachment funnel\n")
    lines.append("| stage | count |")
    lines.append("|---|---|")
    lines.append(f"| gated close-up anchors (step-3 survivors) | {len(anchors)} |")
    lines.append(f"| attached to a fragment (ReID margin gate) | {n_attached} |")
    for reason, n in reasons.most_common():
        lines.append(f"| unattached: {reason} | {n} |")
    lines.append(f"| named fragments after guard | {len(resolved)} |")
    lines.append(f"| named-track rows written | {len(named)} |")
    lines.append("")
    lines.append(f"Anchor number histogram (shirtNumber): `{dict(sorted(hist.items()))}` -- "
                 "hero-shot concentration (Bruno #8 dominates).\n")

    lines.append("## Propagation-guard decisions\n")
    lines.append(f"- Fragment-level naming: {len([r for r in resolved if r.basis == 'fragment'])} "
                 "fragments named by their own anchors.\n")
    lines.append(f"- Fragment disagreement flags (two anchors, different numbers, one fragment -> "
                 f"named neither): {len(flags)}.")
    for f in flags[:10]:
        lines.append(f"  - {f['chunk']} track {f['track_id']}: numbers {f['numbers']}")
    lines.append("- Relink-merge clause (>=2 agreeing anchors to cross a merge): **not exercised** "
                 "-- brighton_manutd carries no relink remap (relink is benchmark-side only, 35% "
                 "merge precision). Covered by the synthetic test `test_anchor_wire.py`.\n")

    lines.append("## Validation: our visible-minutes proxy vs oracle (directional sanity)\n")
    lines.append("Visible-minutes != played-minutes. Broadcast shows ~37% of the pitch-time of any "
                 "one player, and our proxy is the union of frame-spans of *named* fragments only "
                 "(sparse, ReID-limited). Expected direction: **ours << oracle**. The test is "
                 "ORDERING consistency, not calibration.\n")
    if vdf.empty:
        lines.append("_No players were named -- nothing to validate. See caveats._\n")
    else:
        lines.append("| player | team | # | our anchors | named frags | our visible min | "
                     "oracle min | oracle touches |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in val_rows:
            lines.append(f"| {r['player']} | {r['team']} | {r['number']} | {r['our_anchors']} | "
                         f"{r['our_named_frags']} | {r['our_visible_min']} | {r['oracle_min']} | "
                         f"{r['oracle_touches']} |")
        lines.append("")
        lines.append(f"Spearman(our_visible_min, oracle_min) over {len(with_oracle)} named players "
                     f"with oracle minutes: **{rho}** (None if <3 players or degenerate).\n")
        lines.append("Touch-count proxy: **skipped**. It needs per-frame ball-possession association "
                     "on top of sparse named fragments; the post-`link_ball` usable-ball track and "
                     "the ReID-limited naming make it noise, not signal, at this yield. Stated per "
                     "the task's 'else skip and say so'.\n")

    lines.append("## Honest caveats\n")
    lines.append("- **Hero-shot concentration:** ~5-6 distinct back-numbers surface legible close-up "
                 "reads this match; #8 (Bruno) is 82% of anchors. This names the players who get "
                 "repeated close-up hero shots, not a uniform XI.\n")
    lines.append("- **ReID is the bottleneck, not anchor precision.** OSNet is kit-dominated "
                 "(Stage-2a: median cosine 0.81 on same-team pairs; 35% merge precision), so the "
                 "margin gate rejects most same-kit disambiguations -- that is why attachment yield "
                 f"({n_attached}) is far below the anchor count ({len(anchors)}). Anchor *read* "
                 + ("precision is PENDING human spot-check (Koshkina arm)" if precision_pending
                    else "precision is ~99%")
                 + "; carrying reads onto the right *track* is the hard, unsolved half.\n")
    lines.append("- **Season-scale extrapolation:** at ~5-6 hero-shot players/match, a 38-match "
                 "season yields close-up anchors concentrated on the same marquee names (Bruno, "
                 "Rashford, ...). Close-up anchors **supplement** roster/relink priors for those "
                 "players; uniform per-player naming still needs the cluster/VLM close-up reader "
                 "(Sem 2).\n")

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote report -> {report}")
    print("VALIDATION TABLE:")
    if not vdf.empty:
        show = vdf.copy()
        show["player"] = show["player"].map(_ascii)
        print(show.to_string(index=False))
        print(f"Spearman(our_visible_min, oracle_min) = {rho}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-name", default="osnet_x0_25", help="OSNet backbone")
    ap.add_argument("--weights", default=None,
                    help="torchreid re-ID weights key/path (default: ImageNet baseline)")
    ap.add_argument("--survivors", default=str(SURVIVORS),
                    help="dir of gated anchor survivor crops to wire (default: step-3 226)")
    ap.add_argument("--tag", default=None,
                    help="explicit output suffix (overrides the weights-derived one)")
    ap.add_argument("--precision-pending", action="store_true",
                    help="mark all reads precision-UNVERIFIED (Koshkina PARSeq arm)")
    ap.add_argument("--match", default=MATCH_ID, choices=sorted(SOFASCORE_MATCH_ID),
                    help=f"registry match id to wire (default: {MATCH_ID})")
    ap.add_argument("--embedder", default="osnet", choices=["osnet", "prtreid"],
                    help="appearance embedder for the attachment gate (default: osnet)")
    ap.add_argument("--min-sim", type=float, default=aw.REID_MIN_SIM,
                    help="minimum best cosine to attach (embedder-specific scale)")
    ap.add_argument("--min-margin", type=float, default=aw.REID_MIN_MARGIN,
                    help="minimum best-minus-second-best cosine gap to attach")
    a = ap.parse_args()
    main(model_name=a.model_name, weights=a.weights, survivors=Path(a.survivors), tag=a.tag,
         precision_pending=a.precision_pending, match_id=a.match, embedder_name=a.embedder,
         min_sim=a.min_sim, min_margin=a.min_margin)
