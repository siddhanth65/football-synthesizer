"""Feasibility probe: closed-set identification of the ball CARRIER at pass moments.

Today's player attribution names tracks globally (open set, ~750-1045 track ids per chunk for 22
players) and hopes a pass lands inside a named fragment -> 1-4% attribution
(``results/PLAYER_ANALYSIS_v2.md``). This probe tests the opposite framing: we only need the name of
the carrier at ~500-1000 pass instants, which turns an open-set tracking problem into a CLOSED-SET
match against the 22-player roster.

MEASURED (``results/CARRIER_ATTRIBUTION_PROBE.md``): the closed set lifts attribution coverage from
0.9-2.0% to 11.6-15.3%, but the *framing* premise -- "the camera is centred on the ball at pass
moments, so the carrier is well framed" -- is FALSE here: carrier crops are the same ~90 px height as
any random tracked player. The lift comes from the small candidate set, not from image quality.

Staged funnel (each stage's own ceiling is reported; a stage may kill the idea on its own):

1. **Gallery** -- per rostered player, PRTreID embeddings sampled from the frames of tracks that the
   existing identity chain already named (``outputs/identity/<match>_named_tracks_both2_prtreid``).
   Gallery crops come from the SAME wide-shot domain as the queries (not the close-up anchor crops),
   so query/gallery are domain-matched.
2. **Carrier extraction** -- the kick-moment carrier for every operating-point PASS event, reusing
   :mod:`tools.event_ledger` (``_contact_frame`` + ``nearest_carrier``) unchanged.
3. **Closed-set id** -- embed the carrier crop, score it against the gallery restricted to the
   carrier's tracked team (11, not 22), assign on ``best >= min_sim`` AND ``best - runner_up >=
   min_margin``, else abstain. Leave-one-track-out (LOTO): the carrier's OWN track is dropped from
   the gallery, so an assignment is never a restatement of the existing named-fragment overlap.
4. **Scoring** -- carrier-resolvable / gallery-coverage / assignment / abstention rates and the
   headline ATTRIBUTION COVERAGE, against the 1-4% baseline.
5. **Indirect validation** (no manual labels yet) -- Spearman + top-5 vs the Sofascore per-player
   ``totalPass`` oracle, same-track identity consistency, and roster/substitute sanity.

Threshold discipline: the operating point is selected on :data:`DEV_MATCH` only, written to
:data:`FROZEN_PATH`, and re-used verbatim on the hold-out matches.

Run (GPU, one job)::

    python -m tools.carrier_attribution_probe --stage funnel      # CPU, stage-2 ceiling only
    python -m tools.carrier_attribution_probe --stage embed --match manutd_liverpool
    python -m tools.carrier_attribution_probe --stage select      # freeze threshold on DEV_MATCH
    python -m tools.carrier_attribution_probe --stage score       # all matches at the frozen point
    python -m tools.carrier_attribution_probe --stage labelpack   # export Sid's labelling set
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import registry
from core.registry import Match
from tools import event_ledger as el
from tools.action_spot_probe import SOFASCORE_MATCH_ID
from tools.bas_validate import load_actions, op_events

#: Matches in the probe. The first is the threshold-selection (dev) match; the rest are hold-outs.
PROBE_MATCHES = ("manutd_liverpool", "manutd_tottenham", "manutd_brighton")
DEV_MATCH = PROBE_MATCHES[0]
OUT_DIR = Path("results/carrier_attr")
FROZEN_PATH = OUT_DIR / "frozen_threshold.json"
REPORT_PATH = Path("results/CARRIER_ATTRIBUTION_PROBE.md")
LABELPACK_DIR = OUT_DIR / "labelpack"

#: Max gallery crops sampled per named track (spread evenly over the track's frames).
GALLERY_PER_TRACK = 6
#: Min pixel height of a crop for it to be embedded at all (tiny far-side boxes are noise).
MIN_CROP_H = 24
#: Candidate operating points swept on the dev match only.
SIM_GRID = (0.80, 0.84, 0.88, 0.90, 0.92, 0.94, 0.96)
MARGIN_GRID = (0.0, 0.01, 0.02, 0.04, 0.08)
#: Pre-declared selection rule (fixed BEFORE any score was seen): among dev-match operating points
#: whose same-track identity consistency reaches this bar, take the one with the highest assignment
#: rate. Consistency is label-free, so this rule needs no ground truth.
CONSISTENCY_BAR = 0.90
#: A consistency point needs at least this many multi-pass tracks to count.
MIN_CONSISTENCY_TRACKS = 10
#: Labelling pack: how many pass moments, and how many context frames each side of the kick.
LABELPACK_N = 100
LABELPACK_CONTEXT = 2
LABELPACK_STRIDE = 10


def named_tracks_path(match_id: str) -> Path:
    """Path of the PRTreID named-tracks parquet for a match."""
    return Path(f"outputs/identity/{match_id}_named_tracks_both2_prtreid.parquet")


def oracle_path(match_id: str) -> Path:
    """Path of the Sofascore per-player stats parquet for a match."""
    return Path(f"outputs/oracle/sofascore/player_stats_{SOFASCORE_MATCH_ID[match_id]}.parquet")


def _video_for(match_id: str, chunk_key: str) -> Path:
    half, num = chunk_key.split("_chunk_")
    return registry.VIDEO_ROOT / match_id / half / f"chunk_{num}.mp4"


def track_team(df: pd.DataFrame) -> dict[tuple[str, int], int]:
    """Majority team per ``(chunk, track_id)``.

    The aligned table's ``team`` is a per-detection kit assignment and flips frame to frame (10-25%
    of a named track's rows can carry the wrong team). The per-track majority is the stable label and
    is used for BOTH the gallery tag and the query's team restriction.

    Args:
        df: aligned positions table.

    Returns:
        ``{(chunk, track_id): team}``.
    """
    g = (df[df["role"].isin(["player", "goalkeeper"])]
         .groupby(["chunk", "track_id"])["team"]
         .agg(lambda s: int(s.mode().iloc[0])))
    return {(str(c), int(t)): int(v) for (c, t), v in g.items()}


# === Stage 2: carrier extraction (CPU) ===========================================================
def carrier_events(match: Match) -> pd.DataFrame:
    """Kick-moment carrier for every operating-point PASS event of a match.

    Reuses :mod:`tools.event_ledger` end to end (``_contact_frame`` picks the window sample where a
    player is closest to the ball; ``nearest_carrier`` picks that player). Nothing is re-derived.

    Args:
        match: registry match.

    Returns:
        One row per filtered PASS event with ``resolvable`` and, when resolvable, the carrier's
        chunk/frame/track/team plus its image-space foot point (for cropping).
    """
    actions = load_actions(match.id)
    df = match.load_aligned()
    tteam = track_team(df)
    rows: list[dict] = []
    for chunk_key in sorted(actions):
        dfc = df[df["chunk"] == chunk_key]
        if dfc.empty:
            continue
        pf, by = el._players_by_frame(dfc)
        bf, bxy = el._ball_track(match, chunk_key, dfc)
        img = {(int(f), int(t)): (float(x), float(y)) for f, t, x, y in
               zip(dfc["frame"], dfc["track_id"], dfc["image_x"], dfc["image_y"])}
        for frame_index, conf in op_events(actions[chunk_key], "PASS"):
            g = el.bas_to_grid_frame(frame_index)
            contact = el._contact_frame(pf, by, bf, bxy, g)
            row: dict = {"chunk": chunk_key, "frame_index": int(frame_index),
                         "bas_conf": float(conf), "half": chunk_key[:2],
                         "ball_found": contact is not None, "resolvable": False,
                         "frame": -1, "track_id": -1, "team": -1, "dist_m": np.nan,
                         "image_x": np.nan, "image_y": np.nan}
            if contact is not None and contact["dist"] <= el.TEAM_MAX_M:
                pos, teams, tracks = by[contact["frame"]]
                team, track_id, dist = el.nearest_carrier(pos, teams, tracks, contact["ball"])
                xy = img.get((int(contact["frame"]), int(track_id)))
                if xy is not None and not (np.isnan(xy[0]) or np.isnan(xy[1])):
                    row.update(resolvable=True, frame=int(contact["frame"]), track_id=int(track_id),
                               team=tteam.get((chunk_key, int(track_id)), int(team)),
                               dist_m=round(float(dist), 2), image_x=xy[0], image_y=xy[1])
            rows.append(row)
    return pd.DataFrame(rows)


# === Stage 1: gallery spec (CPU) =================================================================
def gallery_spec(match: Match, per_track: int = GALLERY_PER_TRACK) -> pd.DataFrame:
    """Frames to embed for the per-match player gallery.

    Args:
        match: registry match.
        per_track: max crops sampled per named track, spread evenly across its frame span.

    Returns:
        Rows of ``chunk, frame, track_id, team, player, image_x, image_y``.
    """
    path = named_tracks_path(match.id)
    if not path.exists():
        return pd.DataFrame(columns=["chunk", "frame", "track_id", "team", "player",
                                     "image_x", "image_y"])
    named = pd.read_parquet(path)
    # The gallery's team label comes from the ROSTER (the identity chain already resolved the named
    # track's club), not from the noisy per-frame kit assignment: they disagree on ~7-9% of crops.
    roster_team = {n: i for i, n in enumerate(match.teams)}
    df = match.load_aligned()
    tteam = track_team(df)
    df = df[df["role"].isin(["player", "goalkeeper"])]
    idx = df.set_index(["chunk", "track_id"]).sort_index()
    rows: list[dict] = []
    for r in named.itertuples(index=False):
        try:
            sub = idx.loc[(r.chunk, int(r.track_id))]
        except KeyError:
            continue
        sub = sub.dropna(subset=["image_x", "image_y"])
        if isinstance(sub, pd.Series) or sub.empty:
            continue
        sub = sub.sort_values("frame")
        team = roster_team.get(r.team, tteam.get((r.chunk, int(r.track_id)),
                                                 int(sub["team"].iloc[0])))
        pick = np.linspace(0, len(sub) - 1, min(per_track, len(sub))).astype(int)
        for i in np.unique(pick):
            s = sub.iloc[int(i)]
            rows.append({"chunk": r.chunk, "frame": int(s["frame"]),
                         "track_id": int(r.track_id), "team": team,
                         "player": r.player_name, "image_x": float(s["image_x"]),
                         "image_y": float(s["image_y"])})
    return pd.DataFrame(rows)


# === Embedding (GPU) =============================================================================
def embed_requests(match_id: str, req: pd.DataFrame, embedder) -> np.ndarray:
    """Embed one crop per request row, decoding each chunk video in a single forward pass.

    Args:
        match_id: registry match id (locates the chunk videos).
        req: rows with ``chunk, frame, image_x, image_y`` (order is preserved in the output).
        embedder: any object exposing ``embed(list[rgb]) -> (N, D)`` L2-normalised features.

    Returns:
        ``(len(req), D)`` float32 embeddings; all-zero rows mark crops that could not be read.
    """
    import cv2  # noqa: PLC0415

    from generator.team_anchor import estimate_player_box  # noqa: PLC0415

    out: np.ndarray | None = None
    for chunk_key, g in req.groupby("chunk", sort=True):
        video = _video_for(match_id, str(chunk_key))
        if not video.exists():
            print(f"WARN missing video {video}")
            continue
        cap = cv2.VideoCapture(str(video))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        crops: list[np.ndarray] = []
        slots: list[int] = []
        for pos_i, r in zip(g.index, g.sort_values("frame").itertuples(index=True)):
            del pos_i
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
            ok, bgr = cap.read()
            if not ok:
                continue
            x1, y1, x2, y2 = estimate_player_box(r.image_x, r.image_y, fh, fw)
            if y2 - y1 < MIN_CROP_H or x2 - x1 < 8:
                continue
            crop = bgr[y1:y2, x1:x2]
            if not crop.size:
                continue
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            slots.append(int(r.Index))
        if not crops:
            cap.release()
            continue
        feats = embedder.embed(crops)
        if out is None:
            out = np.zeros((len(req), feats.shape[1]), np.float32)
        pos = {ix: k for k, ix in enumerate(req.index)}
        for k, ix in enumerate(slots):
            out[pos[ix]] = feats[k]
        cap.release()
        print(f"  {chunk_key}: embedded {len(crops)}/{len(g)} crops")
    return np.zeros((len(req), 1), np.float32) if out is None else out


def build_embedder():
    """PRTreID embedder (same class the identity chain uses)."""
    from tools.prtreid_probe import PrtreidEmbedder  # noqa: PLC0415

    return PrtreidEmbedder()


def run_embed(match_id: str) -> None:
    """Compute and cache carrier-query + gallery embeddings for one match (single GPU job)."""
    match = registry.get(match_id)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ev = carrier_events(match)
    gal = gallery_spec(match)
    ev.to_parquet(OUT_DIR / f"{match_id}_events.parquet", index=False)
    gal.to_parquet(OUT_DIR / f"{match_id}_gallery.parquet", index=False)
    q = ev[ev["resolvable"]].reset_index(drop=True)
    print(f"{match_id}: {len(ev)} PASS events, {len(q)} resolvable carriers, "
          f"{len(gal)} gallery crops over {gal['player'].nunique() if len(gal) else 0} players")
    embedder = build_embedder()
    print("embedding gallery ...")
    ge = embed_requests(match_id, gal, embedder)
    np.save(OUT_DIR / f"{match_id}_gallery_emb.npy", ge)
    print("embedding carrier queries ...")
    qe = embed_requests(match_id, q, embedder)
    np.save(OUT_DIR / f"{match_id}_query_emb.npy", qe)
    print(f"wrote embeddings for {match_id}")


# === Stage 3: closed-set identification ==========================================================
def identify(ev: pd.DataFrame, qe: np.ndarray, gal: pd.DataFrame, ge: np.ndarray, *,
             loto: bool = True) -> pd.DataFrame:
    """Score every carrier query against the per-team gallery; return best/runner-up per query.

    Args:
        ev: carrier-event rows (``resolvable`` subset, aligned with ``qe``).
        qe: ``(nq, D)`` query embeddings.
        gal: gallery rows aligned with ``ge``.
        ge: ``(ng, D)`` gallery embeddings.
        loto: leave-one-track-out -- drop gallery samples from the query's OWN track, so an
            assignment cannot be a restatement of the existing named-fragment overlap.

    Returns:
        ``ev`` plus ``best_player``, ``best_sim``, ``second_sim`` (best of the runner-up player),
        ``margin``, ``n_gallery`` (candidate players available) and ``own_track_named``.
    """
    ok_g = np.linalg.norm(ge, axis=1) > 0.5
    gal = gal.reset_index(drop=True)
    players = gal["player"].to_numpy()
    g_team = gal["team"].to_numpy()
    g_track = gal["track_id"].to_numpy()
    g_chunk = gal["chunk"].to_numpy()
    named_pairs = set(zip(gal["chunk"], gal["track_id"]))
    rows: list[dict] = []
    for i, r in enumerate(ev.itertuples(index=False)):
        q = qe[i]
        rec = {"best_player": None, "best_sim": np.nan, "second_sim": np.nan, "margin": np.nan,
               "n_gallery": 0,
               "own_track_named": (r.chunk, int(r.track_id)) in named_pairs}
        mask = ok_g & (g_team == r.team)
        if loto:
            mask = mask & ~((g_chunk == r.chunk) & (g_track == int(r.track_id)))
        if np.linalg.norm(q) > 0.5 and mask.any():
            sims = ge[mask] @ q
            pl = players[mask]
            best_by: dict[str, float] = {}
            for p, s in zip(pl, sims):
                if s > best_by.get(p, -2.0):
                    best_by[p] = float(s)
            order = sorted(best_by.items(), key=lambda kv: -kv[1])
            rec["n_gallery"] = len(order)
            rec["best_player"] = order[0][0]
            rec["best_sim"] = order[0][1]
            rec["second_sim"] = order[1][1] if len(order) > 1 else -1.0
            rec["margin"] = rec["best_sim"] - rec["second_sim"]
        rows.append(rec)
    return pd.concat([ev.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def assign(scored: pd.DataFrame, min_sim: float, min_margin: float) -> pd.Series:
    """Assigned player name per query at an operating point (``None`` = abstain)."""
    ok = (scored["best_sim"] >= min_sim) & (scored["margin"] >= min_margin)
    return scored["best_player"].where(ok, other=None)


def track_consistency(scored: pd.DataFrame, assigned: pd.Series) -> tuple[float | None, int]:
    """Same-track identity agreement: share of multi-assignment tracks with a single identity.

    Args:
        scored: scored query rows.
        assigned: assigned names aligned with ``scored``.

    Returns:
        ``(consistency, n_tracks)``; consistency is ``None`` when too few tracks carry 2+
        assignments to measure it.
    """
    d = scored.assign(_a=assigned.to_numpy())
    d = d[d["_a"].notna()]
    grp = d.groupby(["chunk", "track_id"])["_a"].nunique()
    n_multi = d.groupby(["chunk", "track_id"])["_a"].size()
    multi = grp[n_multi >= 2]
    if len(multi) == 0:
        return None, 0
    return float((multi == 1).mean()), int(len(multi))


# === Stage 4/5: scoring + indirect validation ====================================================
def _spearman(a: pd.Series, b: pd.Series) -> float | None:
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return round(float(a.corr(b, method="spearman")), 3)


def oracle_table(match_id: str) -> pd.DataFrame:
    """Sofascore per-player ``name, totalPass, minutesPlayed, substitute, teamName``."""
    df = pd.read_parquet(oracle_path(match_id))
    keep = ["name", "totalPass", "minutesPlayed", "substitute", "teamName"]
    out = df[[c for c in keep if c in df.columns]].copy()
    out["totalPass"] = pd.to_numeric(out.get("totalPass"), errors="coerce")
    return out


def gallery_completeness(match_id: str) -> dict:
    """How much of the real passing volume the closed set can even represent.

    The "closed set" is only closed if every player who touches the ball is in the gallery. Players
    the identity chain never named are absent, so any pass they make CANNOT be assigned correctly --
    a confident assignment there is guaranteed wrong. Weighting by the oracle's ``totalPass`` turns
    the raw roster count into the share of real passes the gallery could in principle cover.

    Args:
        match_id: registry match id with a cached gallery.

    Returns:
        ``n_players``, ``n_oracle_players``, ``pass_share`` (oracle passes by gallery players over
        all oracle passes) and the per-team split.
    """
    gal = pd.read_parquet(OUT_DIR / f"{match_id}_gallery.parquet")
    orc = oracle_table(match_id).dropna(subset=["totalPass"])
    have = set(gal["player"])
    inn = orc[orc["name"].isin(have)]
    by_team = {}
    for team, g in orc.groupby("teamName"):
        tot = float(g["totalPass"].sum())
        got = float(g[g["name"].isin(have)]["totalPass"].sum())
        by_team[str(team)] = {"players": int(g["name"].isin(have).sum()),
                              "of": int(len(g)),
                              "pass_share": round(got / tot, 3) if tot else None}
    tot = float(orc["totalPass"].sum())
    return {"n_players": len(have & set(orc["name"])), "n_oracle_players": int(len(orc)),
            "pass_share": round(float(inn["totalPass"].sum()) / tot, 3) if tot else None,
            "by_team": by_team}


def validate(scored: pd.DataFrame, assigned: pd.Series, match_id: str) -> dict:
    """Indirect validation of an assignment set (no manual labels).

    Args:
        scored: scored query rows.
        assigned: assigned names aligned with ``scored``.
        match_id: registry match id (for the oracle join).

    Returns:
        ``spearman``, ``n_join``, ``top5_overlap``, ``consistency``, ``n_consistency_tracks``,
        ``off_roster`` (assignments to a name absent from the oracle), ``sub_early`` (share of a
        substitute's assignments landing in the first half of the broadcast, per player).
    """
    d = scored.assign(_a=assigned.to_numpy())
    named = d[d["_a"].notna()]
    counts = named.groupby("_a").size().rename("attr_pass").reset_index()
    orc = oracle_table(match_id)
    merged = counts.merge(orc, left_on="_a", right_on="name", how="left")
    both = merged.dropna(subset=["totalPass"])
    rho = _spearman(both["attr_pass"], both["totalPass"]) if len(both) >= 3 else None
    top5_ours = set(counts.sort_values("attr_pass", ascending=False)["_a"].head(5))
    orc_pl = orc.dropna(subset=["totalPass"])
    # oracle top-5 restricted to the players our gallery can even produce
    pool = set(scored["best_player"].dropna())
    top5_orc = set(orc_pl[orc_pl["name"].isin(pool)]
                   .sort_values("totalPass", ascending=False)["name"].head(5))
    cons, n_tracks = track_consistency(scored, assigned)
    h1_share = named.assign(_h1=named["half"] == "h1").groupby("_a")["_h1"].mean()
    sub_rows = []
    for rec in merged.to_dict("records"):
        if pd.isna(rec.get("substitute")):
            continue
        player = rec["_a"]
        sub_rows.append({"player": player, "substitute": bool(rec["substitute"]),
                         "minutes": rec.get("minutesPlayed"), "n": int(rec["attr_pass"]),
                         "h1_share": round(float(h1_share.get(player, np.nan)), 2)})
    return {
        "spearman": rho, "n_join": int(len(both)),
        "top5_overlap": len(top5_ours & top5_orc), "top5_ours": sorted(top5_ours),
        "top5_oracle": sorted(top5_orc),
        "consistency": None if cons is None else round(cons, 3),
        "n_consistency_tracks": n_tracks,
        "off_roster": int(merged["name"].isna().sum()),
        "sub_rows": sub_rows,
        "counts": counts.sort_values("attr_pass", ascending=False),
    }


def load_scored(match_id: str, *, loto: bool = True) -> pd.DataFrame:
    """Load cached embeddings for a match and score every resolvable carrier query."""
    ev = pd.read_parquet(OUT_DIR / f"{match_id}_events.parquet")
    gal = pd.read_parquet(OUT_DIR / f"{match_id}_gallery.parquet")
    qe = np.load(OUT_DIR / f"{match_id}_query_emb.npy")
    ge = np.load(OUT_DIR / f"{match_id}_gallery_emb.npy")
    q = ev[ev["resolvable"]].reset_index(drop=True)
    return identify(q, qe, gal, ge, loto=loto)


def sweep(scored: pd.DataFrame) -> pd.DataFrame:
    """Assignment rate + label-free consistency over the candidate operating-point grid."""
    rows = []
    for s in SIM_GRID:
        for m in MARGIN_GRID:
            a = assign(scored, s, m)
            cons, n_tracks = track_consistency(scored, a)
            rows.append({"min_sim": s, "min_margin": m,
                         "assign_rate": round(float(a.notna().mean()), 3),
                         "n_assigned": int(a.notna().sum()),
                         "consistency": None if cons is None else round(cons, 3),
                         "n_tracks": n_tracks})
    return pd.DataFrame(rows)


def gallery_loto(match_id: str) -> pd.DataFrame:
    """Leave-one-track-out self-identification of the gallery against itself.

    Every gallery crop is re-identified against the same-team gallery with its OWN track removed.
    The identity-chain label is the pseudo-truth (itself ~95%-precision, so this is an upper-bound
    proxy, not ground truth) -- but the task is the exact 8-16-way closed-set problem the carrier
    queries face, so its precision curve is the only pre-label evidence available for choosing an
    operating point.

    Args:
        match_id: registry match id with cached gallery embeddings.

    Returns:
        Rows of ``true, pred, sim, margin, team``.
    """
    gal = pd.read_parquet(OUT_DIR / f"{match_id}_gallery.parquet")
    ge = np.load(OUT_DIR / f"{match_id}_gallery_emb.npy")
    ok = np.linalg.norm(ge, axis=1) > 0.5
    gal, ge = gal[ok].reset_index(drop=True), ge[ok]
    rows: list[dict] = []
    for t in sorted(gal["team"].unique()):
        m = (gal["team"] == t).to_numpy()
        sub, g = gal[m].reset_index(drop=True), ge[m]
        sims = g @ g.T
        for i in range(len(sub)):
            own = ((sub["track_id"] == sub["track_id"][i]) & (sub["chunk"] == sub["chunk"][i]))
            s = sims[i].copy()
            s[own.to_numpy()] = -2.0
            best: dict[str, float] = {}
            for p, v in zip(sub["player"], s):
                if v > best.get(p, -3.0):
                    best[p] = float(v)
            order = sorted(best.items(), key=lambda kv: -kv[1])
            if not order:
                continue
            second = order[1][1] if len(order) > 1 else -1.0
            rows.append({"true": sub["player"][i], "pred": order[0][0], "sim": order[0][1],
                         "margin": order[0][1] - second, "team": int(t)})
    return pd.DataFrame(rows)


def gallery_sweep(loto: pd.DataFrame) -> pd.DataFrame:
    """Pseudo-truth precision + retention of the gallery LOTO task over the operating-point grid."""
    rows = []
    for s in SIM_GRID:
        for m in MARGIN_GRID:
            k = loto[(loto["sim"] >= s) & (loto["margin"] >= m)]
            rows.append({"min_sim": s, "min_margin": m, "n": len(k),
                         "retention": round(len(k) / len(loto), 3) if len(loto) else None,
                         "precision": round(float((k["true"] == k["pred"]).mean()), 3)
                         if len(k) else None})
    return pd.DataFrame(rows)


def select_threshold(sw: pd.DataFrame, gsw: pd.DataFrame | None = None) -> dict:
    """Apply the pre-declared selection rule to a dev-match sweep, with a stated fallback.

    Primary rule (fixed before scores were seen): among points with ``consistency >=``
    :data:`CONSISTENCY_BAR` measured on at least :data:`MIN_CONSISTENCY_TRACKS` tracks, take the
    highest assignment rate; ties break to the higher ``min_sim``.

    That rule turned out to be unmeasurable on real data (carrier tracks are ~1 pass long, so almost
    no track carries two assignments). The stated fallback uses the gallery-LOTO precision curve at
    the project's pre-existing :data:`tools.prtreid_probe.PRECISION_BAR` (0.80): the maximum-retention
    grid point whose dev-match pseudo-truth precision clears the bar.

    Args:
        sw: dev-match query sweep (:func:`sweep`).
        gsw: dev-match gallery-LOTO sweep (:func:`gallery_sweep`); required for the fallback.

    Returns:
        The frozen operating point plus which rule produced it.
    """
    ok = sw[(sw["consistency"].notna()) & (sw["consistency"] >= CONSISTENCY_BAR)
            & (sw["n_tracks"] >= MIN_CONSISTENCY_TRACKS)]
    if not ok.empty:
        best = ok.sort_values(["assign_rate", "min_sim"], ascending=[False, False]).iloc[0]
        return {"min_sim": float(best["min_sim"]), "min_margin": float(best["min_margin"]),
                "dev_match": DEV_MATCH, "dev_assign_rate": float(best["assign_rate"]),
                "dev_consistency": float(best["consistency"]), "rule":
                f"PRIMARY: max assign_rate s.t. same-track consistency >= {CONSISTENCY_BAR} "
                f"on >= {MIN_CONSISTENCY_TRACKS} tracks (dev match only)"}
    if gsw is None:
        return {"min_sim": None, "min_margin": None, "rule": "primary rule unmeasurable, no fallback"}
    from tools.prtreid_probe import PRECISION_BAR  # noqa: PLC0415

    cand = gsw[(gsw["precision"].notna()) & (gsw["precision"] >= PRECISION_BAR) & (gsw["n"] >= 50)]
    if cand.empty:
        return {"min_sim": None, "min_margin": None,
                "rule": f"FALLBACK failed: no grid point reaches precision {PRECISION_BAR}"}
    best = cand.sort_values(["retention", "min_sim"], ascending=[False, False]).iloc[0]
    return {"min_sim": float(best["min_sim"]), "min_margin": float(best["min_margin"]),
            "dev_match": DEV_MATCH, "dev_gallery_precision": float(best["precision"]),
            "dev_gallery_retention": float(best["retention"]),
            "rule": f"FALLBACK (primary rule unmeasurable -- carrier tracks are ~1 pass long): max "
                    f"retention s.t. dev-match gallery-LOTO pseudo-truth precision >= "
                    f"{PRECISION_BAR} on >= 50 crops"}


# === Stage 6: labelling pack =====================================================================
def export_labelpack(matches_: tuple[str, ...] = PROBE_MATCHES, n: int = LABELPACK_N) -> Path:
    """Export ~``n`` pass moments (crop + context frames, prediction HIDDEN) plus a blank CSV.

    Stratified by match, half and carrier crop size (small/medium/large box height) so the set is not
    all easy close-ups.

    Args:
        matches_: match ids to sample from.
        n: target number of pass moments.

    Returns:
        The labelling-pack directory.
    """
    import cv2  # noqa: PLC0415

    from generator.team_anchor import estimate_player_box  # noqa: PLC0415

    LABELPACK_DIR.mkdir(parents=True, exist_ok=True)
    picks: list[pd.DataFrame] = []
    rng = np.random.default_rng(7)
    per_match = max(1, n // len(matches_))
    for mid in matches_:
        p = OUT_DIR / f"{mid}_events.parquet"
        if not p.exists():
            continue
        ev = pd.read_parquet(p)
        q = ev[ev["resolvable"]].copy()
        if q.empty:
            continue
        q["match"] = mid
        q["size_band"] = pd.qcut(q["image_y"], 3, labels=["far", "mid", "near"],
                                 duplicates="drop")
        take: list[pd.DataFrame] = []
        for _, g in q.groupby(["half", "size_band"], observed=True):
            k = max(1, per_match // 6)
            take.append(g.iloc[rng.choice(len(g), size=min(k, len(g)), replace=False)])
        picks.append(pd.concat(take))
    sel = pd.concat(picks).reset_index(drop=True) if picks else pd.DataFrame()
    rows = []
    for i, r in enumerate(sel.itertuples(index=False)):
        video = _video_for(r.match, r.chunk)
        if not video.exists():
            continue
        cap = cv2.VideoCapture(str(video))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        stem = f"{i:03d}_{r.match}_{r.chunk}_f{r.frame}"
        ok_any = False
        for k in range(-LABELPACK_CONTEXT, LABELPACK_CONTEXT + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(r.frame) + k * LABELPACK_STRIDE, 0))
            ok, bgr = cap.read()
            if not ok:
                continue
            x1, y1, x2, y2 = estimate_player_box(r.image_x, r.image_y, fh, fw)
            marked = bgr.copy()
            cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.imwrite(str(LABELPACK_DIR / f"{stem}_ctx{k:+d}.jpg"), marked,
                        [cv2.IMWRITE_JPEG_QUALITY, 82])
            if k == 0:
                pad = 12
                crop = bgr[max(y1 - pad, 0):min(y2 + pad, fh),
                           max(x1 - pad, 0):min(x2 + pad, fw)]
                if crop.size:
                    cv2.imwrite(str(LABELPACK_DIR / f"{stem}_crop.jpg"), crop)
                ok_any = True
        cap.release()
        if ok_any:
            rows.append({"id": stem, "match": r.match, "chunk": r.chunk, "frame": int(r.frame),
                         "team_tracked": int(r.team), "carrier_dist_m": r.dist_m,
                         "player_name": "", "confidence_1to3": "", "notes": ""})
    pd.DataFrame(rows).to_csv(LABELPACK_DIR / "labels.csv", index=False, encoding="utf-8")
    (LABELPACK_DIR / "README.txt").write_text(
        "Carrier labelling pack. For each id: <id>_crop.jpg is the carrier crop at the kick "
        "moment; <id>_ctx-2..+2.jpg are full frames around it with the carrier boxed in yellow.\n"
        "Fill labels.csv: player_name (exact Sofascore spelling, or UNKNOWN / WRONG_PLAYER_BOXED), "
        "confidence_1to3, notes.\nModel predictions are deliberately NOT included.\n",
        encoding="utf-8")
    print(f"labelpack: {len(rows)} moments -> {LABELPACK_DIR}")
    return LABELPACK_DIR


# === Reporting ===================================================================================
def funnel_row(match_id: str) -> dict:
    """Stage-1/2 ceiling numbers for one match (CPU only, no embeddings needed)."""
    match = registry.get(match_id)
    ev = carrier_events(match)
    gal = gallery_spec(match)
    n = len(ev)
    res = int(ev["resolvable"].sum())
    named = pd.read_parquet(named_tracks_path(match_id))
    return {
        "match": match_id, "n_pass": n,
        "ball_found": int(ev["ball_found"].sum()),
        "resolvable": res, "resolvable_rate": round(res / n, 3) if n else None,
        "median_dist_m": round(float(ev.loc[ev["resolvable"], "dist_m"].median()), 2) if res else None,
        "named_tracks": int(len(named)), "gallery_crops": int(len(gal)),
        "gallery_players": int(gal["player"].nunique()) if len(gal) else 0,
    }


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=["funnel", "embed", "select", "score", "labelpack"])
    ap.add_argument("--match", default=None, help="match id (embed stage)")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.stage == "funnel":
        rows = [funnel_row(m) for m in PROBE_MATCHES]
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        df.to_csv(OUT_DIR / "funnel_stage12.csv", index=False, encoding="utf-8")
    elif a.stage == "embed":
        run_embed(a.match or DEV_MATCH)
    elif a.stage == "select":
        scored = load_scored(DEV_MATCH)
        sw = sweep(scored)
        print(sw.to_string(index=False))
        sw.to_csv(OUT_DIR / "dev_sweep.csv", index=False, encoding="utf-8")
        gsw = gallery_sweep(gallery_loto(DEV_MATCH))
        print(gsw.to_string(index=False))
        gsw.to_csv(OUT_DIR / "dev_gallery_sweep.csv", index=False, encoding="utf-8")
        frozen = select_threshold(sw, gsw)
        FROZEN_PATH.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
        print(f"frozen: {frozen}")
    elif a.stage == "score":
        frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
        out = []
        for mid in PROBE_MATCHES:
            if not (OUT_DIR / f"{mid}_query_emb.npy").exists():
                continue
            scored = load_scored(mid)
            asg = assign(scored, frozen["min_sim"], frozen["min_margin"])
            val = validate(scored, asg, mid)
            ev = pd.read_parquet(OUT_DIR / f"{mid}_events.parquet")
            comp = gallery_completeness(mid)
            gl = gallery_loto(mid)
            gk = gl[(gl["sim"] >= frozen["min_sim"]) & (gl["margin"] >= frozen["min_margin"])]
            rec = {"match": mid, "n_pass": len(ev), "resolvable": int(ev["resolvable"].sum()),
                   "assigned": int(asg.notna().sum()),
                   "coverage": round(float(asg.notna().sum()) / len(ev), 4),
                   "abstain_rate": round(float(asg.isna().mean()), 3),
                   "gallery_players": comp["n_players"], "gallery_pass_share": comp["pass_share"],
                   "gallery_loto_precision": round(float((gk["true"] == gk["pred"]).mean()), 3)
                   if len(gk) else None,
                   "spearman": val["spearman"], "n_join": val["n_join"],
                   "top5_overlap": val["top5_overlap"], "consistency": val["consistency"],
                   "n_consistency_tracks": val["n_consistency_tracks"],
                   "off_roster": val["off_roster"]}
            out.append(rec)
            print(rec)
            val["counts"].to_csv(OUT_DIR / f"{mid}_player_counts.csv", index=False,
                                 encoding="utf-8")
            pd.DataFrame(val["sub_rows"]).to_csv(OUT_DIR / f"{mid}_sub_sanity.csv", index=False,
                                                 encoding="utf-8")
        pd.DataFrame(out).to_csv(OUT_DIR / "scores.csv", index=False, encoding="utf-8")
    elif a.stage == "labelpack":
        export_labelpack()


if __name__ == "__main__":
    main()


def _selftest() -> None:
    """Smallest runnable check of the pure identification seams."""
    gal = pd.DataFrame({"chunk": ["c", "c", "c"], "track_id": [1, 2, 3], "team": [0, 0, 1],
                        "player": ["A", "B", "C"], "image_x": [0.0] * 3, "image_y": [0.0] * 3})
    ge = np.array([[1.0, 0.0], [0.9, 0.436], [0.0, 1.0]], np.float32)
    ev = pd.DataFrame({"chunk": ["c"], "track_id": [1], "team": [0], "half": ["h1"],
                       "frame": [10], "dist_m": [1.0]})
    qe = np.array([[1.0, 0.0]], np.float32)
    s = identify(ev, qe, gal, ge, loto=True)
    assert s["best_player"].iloc[0] == "B", s["best_player"].iloc[0]  # own track 1 dropped
    assert s["n_gallery"].iloc[0] == 1  # team 1 excluded
    s2 = identify(ev, qe, gal, ge, loto=False)
    assert s2["best_player"].iloc[0] == "A"
    assert abs(float(s2["margin"].iloc[0]) - 0.1) < 1e-3
    print("selftest ok")
