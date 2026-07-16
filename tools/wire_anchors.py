"""B2 Stage-2c runner: wire the 226 gated close-up anchors into named tactical tracks.

Reads the step-3 survivor crops (``results/closeup_anchor_probe/spotcheck_step3/_survivors/``),
attaches each to a ByteTrack fragment by ReID margin (:mod:`generator.anchor_wire`), applies the
propagation guard, resolves numbers to names via the cached Sofascore oracle, writes the named-tracks
parquet, and validates a visible-minutes proxy against the oracle.

Run (GPU, one job -- OSNet embeddings only, no re-detection)::

    python -m tools.wire_anchors

Outputs:
    outputs/identity/brighton_manutd_named_tracks.parquet
    results/identity/NAMED_TRACKS.md
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

MATCH_ID = "brighton_manutd"
ORACLE = Path("outputs/oracle/sofascore/player_stats_12436888.parquet")
SURVIVORS = Path("results/closeup_anchor_probe/spotcheck_step3/_survivors")
VIDEO_ROOT = Path("matches")
OUT_PARQUET = Path("outputs/identity/brighton_manutd_named_tracks.parquet")
REPORT = Path("results/identity/NAMED_TRACKS.md")
NEAR_WINDOW_S = 2.0  # matches the probe's cut window
MAX_CAND_CROPS = 25  # safety cap on candidate crops embedded per wide frame


def _team_id(team_name: str) -> int | None:
    """Oracle ``teamName`` -> positions-table team id (0 = Man Utd red anchor, 1 = Brighton)."""
    n = team_name.lower()
    if "man" in n or "united" in n:
        return 0
    if "brighton" in n:
        return 1
    return None


def _video_for(chunk_key: str) -> Path:
    half, num = chunk_key.split("_chunk_")
    return VIDEO_ROOT / MATCH_ID / half / f"chunk_{num}.mp4"


def _ascii(s: str) -> str:
    """cp1252-safe rendering for console prints (accents -> nearest ASCII)."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode() or "?"


def load_anchors() -> list[aw.Anchor]:
    """Parse the 226 step-3 survivor crops into :class:`anchor_wire.Anchor` records."""
    anchors: list[aw.Anchor] = []
    for p in sorted(SURVIVORS.glob("*.jpg")):
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
    df: pd.DataFrame, need: dict[str, set[int]], embedder: OsnetEmbedder
) -> dict[tuple[str, int], dict[int, tuple[np.ndarray, int]]]:
    """Embed every player/GK track crop on each needed wide frame (one video read per frame).

    Args:
        df: The match positions table.
        need: ``chunk -> {wide_frame indices needed}``.
        embedder: The OSNet appearance embedder.

    Returns:
        ``(chunk, wide_frame) -> {track_id: (embedding, team)}``.
    """
    cache: dict[tuple[str, int], dict[int, tuple[np.ndarray, int]]] = {}
    players = df[df["role"].isin(live_play.PLAYER_ROLES)]
    for chunk, frames in need.items():
        if not frames:
            continue
        video = _video_for(chunk)
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
    win_by_chunk: dict[str, int],
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
        tid, basis = aw.choose_track(sims)
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


def main() -> None:
    """Attach -> guard -> name -> validate; write the named-tracks parquet and report."""
    match = registry.get(MATCH_ID)
    df = match.load_aligned()
    oracle = pd.read_parquet(ORACLE)
    name_by, teams_by = aw.build_roster_maps(oracle, _team_id, number_col="shirtNumber")
    # jerseyNumber cross-map (only to quantify the shirtNumber-vs-jerseyNumber discrepancy).
    jersey_by, _ = aw.build_roster_maps(oracle, _team_id, number_col="jerseyNumber")

    anchors = load_anchors()
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

    embedder = OsnetEmbedder()
    print(f"OSNet on {embedder.device}")
    anchor_embs = embed_anchor_crops(anchors, embedder)
    cand_cache = build_candidate_embeddings(df, need, embedder)

    attachments, reasons = attach_anchors(anchors, anchor_embs, cand_cache, wide_by_chunk,
                                          teams_by, win_by_chunk)
    print(f"attached {len(attachments)} / {len(anchors)} anchors; unattached: {dict(reasons)}")

    # brighton_manutd carries NO relink remap (relink is benchmark-side only, 35% precision) -- every
    # fragment stands alone, so the >=2-anchor merge clause is not exercised on real data here (it is
    # covered by the synthetic test). Pass remap=None.
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
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    named.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {len(named)} named-track rows -> {OUT_PARQUET}")

    _write_report(match, df, oracle, anchors, hist, reasons, attachments, resolved, flags,
                  named, name_by, jersey_by, unmatched_numbers, team_name)


def _write_report(match, df, oracle, anchors, hist, reasons, attachments, resolved, flags, named,
                  name_by, jersey_by, unmatched_numbers, team_name) -> None:
    """Write results/identity/NAMED_TRACKS.md: funnel, guard decisions, validation, caveats."""
    n_attached = len(attachments)
    # Validation table: per named player vs oracle.
    vis = visible_seconds_by_player(resolved, df, match)
    orc = oracle.copy()
    orc["_num"] = pd.to_numeric(orc["shirtNumber"], errors="coerce")
    orc["_team"] = orc["teamName"].map(_team_id)
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

    lines: list[str] = []
    lines.append("# Named tracks -- brighton_manutd (B2 Stage-2c: anchors wired to tracks)\n")
    lines.append(f"Generated by `tools/wire_anchors.py` from the {len(anchors)} step-3 gated "
                 "close-up anchors (98.6% verified read precision). ReID attachment uses ImageNet "
                 "OSNet (`generator.track_relink.OsnetEmbedder`); margin gate "
                 f"`REID_MIN_MARGIN={aw.REID_MIN_MARGIN}`, `REID_MIN_SIM={aw.REID_MIN_SIM}` (frozen "
                 "by reasoning -- no annotated anchor->track map exists for this match).\n")

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
                 f"({n_attached}) is far below the anchor count ({len(anchors)}). Anchor *reads* are "
                 "~99% precise; carrying them onto the right *track* is the hard, unsolved half.\n")
    lines.append("- **Season-scale extrapolation:** at ~5-6 hero-shot players/match, a 38-match "
                 "season yields close-up anchors concentrated on the same marquee names (Bruno, "
                 "Rashford, ...). Close-up anchors **supplement** roster/relink priors for those "
                 "players; uniform per-player naming still needs the cluster/VLM close-up reader "
                 "(Sem 2).\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote report -> {REPORT}")
    print("VALIDATION TABLE:")
    if not vdf.empty:
        show = vdf.copy()
        show["player"] = show["player"].map(_ascii)
        print(show.to_string(index=False))
        print(f"Spearman(our_visible_min, oracle_min) = {rho}")


if __name__ == "__main__":
    main()
