"""Phase-A: per-named-player profiles for Manchester United from the PRTreID identity artifacts.

Turns the PRECISION identity fragments (``outputs/identity/<id>_named_tracks_both2_prtreid.parquet``)
plus the validated event ledger (:mod:`tools.event_ledger`) into a per-player, per-match profile:

1. **Visible geometry** -- mean attack-oriented ``(x, y)`` and its spread on trusted frames (the
   105x68 pitch, Man Utd oriented to attack ``+x`` via the per-chunk keeper resolve reused from
   :mod:`fingerprint.pass_network`); visible-tracked-named minutes (broadcast follows the ball, so
   this is a small fraction of 90 -- reported honestly).
2. **Line composition** -- Man Utd outfield entities (named players + unnamed tracked fragments) are
   1-D clustered by oriented ``x`` into three bands; the rearmost centroid is the defensive line.
   Named players are placed in a band by nearest centroid; the report names each line's anchors.
3. **Involvement** -- carrier-attributed PASS volume, a touch proxy (PASS+DRIVE carried), and the
   fraction of Man Utd possession-links the player appears in (either endpoint), from the ledger
   built on the *PRTreID* identities (not the koshkina recall arm).
4. **Impact framing** -- at 4-12% attribution coverage there is NO honest plus-minus. Impact here =
   validated involvement rate + territorial influence (mean advance, final-third presence), with the
   coverage stated per player.

**Validation** (per match, Man Utd only): Spearman of attributed passes vs Sofascore ``totalPass``
and of the touch proxy vs Sofascore ``touches``, plus top-5 set overlap.

Auto-extends: any registered match that gains a ``*_both2_prtreid.parquet`` is picked up with no code
change; oracle validation runs where a Sofascore ``player_stats`` parquet is registered
(:data:`tools.event_ledger.PLAYER_TRUTH`).

Run (CPU)::

    python -m fingerprint.player_profiles
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core import registry
from core.pitch import PITCH_LEN, PITCH_WID
from core.registry import Match
from fingerprint.pass_network import build_edges
from fingerprint.structural_metrics import resolve_attack_directions
from tools import event_ledger

MANU = "Man Utd"
PRTREID_TMPL = "outputs/identity/{}_named_tracks_both2_prtreid.parquet"
FRAME_S = event_ledger.STEP / 25.0        # aligned parquet is 25 fps sampled every STEP frames
MIN_FRAG_FR = 10                          # an unnamed track needs this many frames to seed a band
MIN_TRUST_FR = 25                         # a named player needs this many frames for a trusted position
POS_RANK = {"D": 0, "M": 1, "F": 2}       # oracle outfield position -> advance rank (keepers dropped)
FINAL_THIRD_X = 2.0 * PITCH_LEN / 3.0     # oriented x beyond this = attacking third (own goal at 0)
BAND_NAMES = ("defensive", "midfield", "attacking")
REPORT_PATH = Path("results/PLAYER_ANALYSIS_v1.md")


# === discovery ===================================================================================
def prtreid_matches() -> list[tuple[Match, pd.DataFrame]]:
    """Registered, processed matches that have a PRTreID named-tracks parquet, with it loaded.

    Returns:
        ``(match, named_df)`` per match, ``named_df`` = PRTreID fragments
        (``chunk, track_id, player_name, jersey_number, team, n_anchors``).
    """
    out: list[tuple[Match, pd.DataFrame]] = []
    for match in registry.matches(processed_only=True):
        path = Path(PRTREID_TMPL.format(match.id))
        if path.exists() and MANU in match.teams:
            named = pd.read_parquet(path).rename(columns={"player_name": "player_name"})
            out.append((match, named))
    return out


def _manu_int(match: Match) -> int:
    """Anchored team id (0/1) that maps to Man Utd for this match."""
    return match.teams.index(MANU)


# === oriented positions ==========================================================================
def oriented_positions(match: Match, named_df: pd.DataFrame) -> pd.DataFrame:
    """Every trusted outfield/keeper row, oriented so each team attacks ``+x``, with PRTreID names.

    Args:
        match: registry match.
        named_df: PRTreID fragments for this match.

    Returns:
        ``player, track_id, team_int, role, chunk, frame, x, y`` (``player`` NA when unnamed;
        chunks/teams without keeper evidence are dropped so orientation is never guessed).
    """
    df = match.load_aligned()
    df = df[df["role"].isin(["player", "goalkeeper"])].dropna(subset=["pitch_x", "pitch_y"])
    parts: list[pd.DataFrame] = []
    for ck, g in df.groupby("chunk"):
        adir = resolve_attack_directions(g)
        name_by = event_ledger._named_by_track(named_df, str(ck))
        sub = g[g["team"].map(lambda t: adir.get(int(t)) is not None)].copy()
        if sub.empty:
            continue
        sign = sub["team"].map(lambda t: adir[int(t)]).to_numpy()
        px, py = sub["pitch_x"].to_numpy(), sub["pitch_y"].to_numpy()
        sub["x"] = np.where(sign > 0, px, PITCH_LEN - px)
        sub["y"] = np.where(sign > 0, py, PITCH_WID - py)
        sub["player"] = sub["track_id"].map(lambda t: name_by.get(int(t)))
        sub["team_int"] = sub["team"].astype(int)
        parts.append(sub[["player", "track_id", "team_int", "role", "chunk", "frame", "x", "y"]])
    cols = ["player", "track_id", "team_int", "role", "chunk", "frame", "x", "y"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def player_geometry(pos: pd.DataFrame, roster: dict[str, str]) -> pd.DataFrame:
    """Per named player: mean oriented ``(x, y)``, positional spread, frames, visible minutes.

    Args:
        pos: :func:`oriented_positions` frame.
        roster: ``{player_name: team_name}`` from the PRTreID roster (authoritative team label).

    Returns:
        ``player, team_name, x, y, spread_x, spread_y, spread_r, final_third_frac, n_frames,
        minutes, trusted`` sorted by team then advance. ``trusted`` = at least
        :data:`MIN_TRUST_FR` frames (below that the mean position is a few-frame artifact).
    """
    named = pos[pos["player"].notna()]
    rows: list[dict] = []
    for player, g in named.groupby("player"):
        n_frames = g.drop_duplicates(["chunk", "frame"]).shape[0]
        rows.append({
            "player": player, "team_name": roster.get(player, "?"),
            "x": round(float(g["x"].mean()), 2), "y": round(float(g["y"].mean()), 2),
            "spread_x": round(float(g["x"].std(ddof=0)), 2),
            "spread_y": round(float(g["y"].std(ddof=0)), 2),
            "spread_r": round(float(np.hypot(g["x"] - g["x"].mean(),
                                             g["y"] - g["y"].mean()).mean()), 2),
            "final_third_frac": round(float((g["x"] > FINAL_THIRD_X).mean()), 3),
            "n_frames": int(n_frames), "minutes": round(n_frames * FRAME_S / 60.0, 1),
            "trusted": bool(n_frames >= MIN_TRUST_FR),
        })
    cols = ["player", "team_name", "x", "y", "spread_x", "spread_y", "spread_r",
            "final_third_frac", "n_frames", "minutes", "trusted"]
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values(["team_name", "x"]).reset_index(drop=True)


# === line composition (1-D banding of Man Utd) ===================================================
def kmeans_1d(x: np.ndarray, k: int = 3, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 1-D k-means (centres seeded at evenly spaced quantiles), centres ascending.

    Args:
        x: values to cluster.
        k: cluster count.
        iters: max Lloyd iterations.

    Returns:
        ``(labels, centres)`` with ``centres`` sorted ascending and ``labels`` indexing into them.
    """
    x = np.asarray(x, float)
    if x.size == 0:
        return np.empty(0, int), np.full(k, np.nan)
    k = min(k, len(np.unique(x)))
    c = np.quantile(x, np.linspace(0.0, 1.0, k * 2 + 1)[1::2])
    for _ in range(iters):
        lab = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)
        new = np.array([x[lab == j].mean() if np.any(lab == j) else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    order = np.argsort(c)
    remap = {old: new for new, old in enumerate(order)}
    lab = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)
    return np.array([remap[j] for j in lab]), c[order]


def line_composition(pos: pd.DataFrame, manu_int: int, geom: pd.DataFrame) -> dict:
    """Cluster Man Utd outfield entities into three bands; name the anchors of each.

    Entities = each named Man Utd outfield player (pooled) + each unnamed Man Utd outfield fragment
    ``(chunk, track_id)`` with enough frames. The rearmost centroid is the defensive line.

    Args:
        pos: :func:`oriented_positions` frame.
        manu_int: Man Utd anchored team id.
        geom: :func:`player_geometry` output (used for the named players' pooled x).

    Returns:
        ``{"centres_m": [def, mid, att], "n_entities": int, "bands": {band_name: [player, ...]}}``.
    """
    outfield = pos[(pos["team_int"] == manu_int) & (pos["role"] == "player")]
    frag = outfield[outfield["player"].isna()].groupby(["chunk", "track_id"])["x"]
    frag_x = [float(v.mean()) for _, v in frag if v.size >= MIN_FRAG_FR]
    manu_named = geom[(geom["team_name"] == MANU) & geom["trusted"]]
    named_x = manu_named["x"].to_numpy(float)
    ent_x = np.array(frag_x + named_x.tolist(), float)
    if ent_x.size < 3:
        return {"centres_m": [], "n_entities": int(ent_x.size), "bands": {}}
    _, centres = kmeans_1d(ent_x, 3)
    bands: dict[str, list[str]] = {b: [] for b in BAND_NAMES}
    for _, r in manu_named.iterrows():
        j = int(np.argmin(np.abs(centres - r["x"])))
        bands[BAND_NAMES[j]].append(f"{r['player']} ({r['x']:.0f}m)")
    return {"centres_m": [round(float(c), 1) for c in centres],
            "n_entities": int(ent_x.size), "bands": bands}


# === involvement (ledger on PRTreID identities) ==================================================
def build_ledger_prtreid(match: Match, named_df: pd.DataFrame) -> pd.DataFrame:
    """Event ledger for one match using the PRTreID identities (mirror of :func:`event_ledger.build_ledger`).

    Args:
        match: registry match.
        named_df: PRTreID fragments (the ``player_name`` source for carrier naming).

    Returns:
        The attributed PASS/DRIVE ledger (same schema as :func:`tools.event_ledger.build_ledger`).
    """
    actions = event_ledger.load_actions(match.id)
    df = match.load_aligned()
    rows: list[dict] = []
    for chunk_key in sorted(actions):
        dfc = df[df["chunk"] == chunk_key]
        if dfc.empty:
            continue
        rows += event_ledger.attribute_chunk(
            match, chunk_key, dfc, actions[chunk_key],
            event_ledger._named_by_track(named_df, chunk_key))
    cols = ["half", "chunk", "frame_index", "t_s", "t_chunk_s", "class", "team", "team_name",
            "player", "bas_conf", "ball_found", "carrier_dist_m", "n_players", "player_dist_m"]
    return pd.DataFrame(rows, columns=cols)


def involvement(ledger: pd.DataFrame, manu_int: int, roster: dict[str, str]) -> pd.DataFrame:
    """Per Man Utd named player: attributed passes, touch proxy, possession-link appearance share.

    Args:
        ledger: :func:`build_ledger_prtreid` frame.
        manu_int: Man Utd anchored team id.
        roster: ``{player_name: team_name}`` (to keep Man Utd players only).

    Returns:
        ``player, attr_pass, touch_proxy, poss_link_appear, poss_link_frac`` (Man Utd players).
    """
    manu_players = {p for p, t in roster.items() if t == MANU}
    passes = ledger[(ledger["class"] == "PASS") & (ledger["team"] == manu_int)]
    carried = ledger[ledger["class"].isin(["PASS", "DRIVE"]) & ledger["player"].notna()]
    edges = build_edges(ledger)
    manu_edges = edges[edges["team"] == manu_int] if len(edges) else edges
    n_links = int(len(manu_edges))
    rows: list[dict] = []
    for player in sorted(manu_players):
        appear = 0
        if n_links:
            appear = int(((manu_edges["src"] == player) | (manu_edges["dst"] == player)).sum())
        rows.append({
            "player": player,
            "attr_pass": int((passes["player"] == player).sum()),
            "touch_proxy": int((carried["player"] == player).sum()),
            "poss_link_appear": appear,
            "poss_link_frac": round(appear / n_links, 3) if n_links else 0.0,
        })
    df = pd.DataFrame(rows, columns=["player", "attr_pass", "touch_proxy",
                                     "poss_link_appear", "poss_link_frac"])
    df.attrs["n_manu_links"] = n_links
    return df.sort_values("attr_pass", ascending=False).reset_index(drop=True)


# === oracle validation ===========================================================================
def _oracle_manu(match_id: str) -> pd.DataFrame | None:
    """Sofascore Man Utd per-player passes/touches/minutes/position (whole match), or None."""
    if match_id not in event_ledger.PLAYER_TRUTH:
        return None
    path = Path(event_ledger.PLAYER_TRUTH[match_id][0])
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    mu = df[df["teamName"].str.contains("United", case=False, na=False)].copy()
    mu["truth_pass"] = pd.to_numeric(mu["totalPass"], errors="coerce")
    mu["truth_touch"] = pd.to_numeric(mu["touches"], errors="coerce")
    mu["truth_min"] = pd.to_numeric(mu["minutesPlayed"], errors="coerce")
    return mu[["name", "position", "truth_pass", "truth_touch", "truth_min"]]


def _spearman(a: pd.Series, b: pd.Series, *, n_min: int = 4) -> float | None:
    """Spearman rho of two aligned series, or None below ``n_min`` finite pairs / no variance."""
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < n_min or d.iloc[:, 0].nunique() < 2 or d.iloc[:, 1].nunique() < 2:
        return None
    r = float(d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman"))
    return None if np.isnan(r) else round(r, 3)


def validate(inv: pd.DataFrame, geom: pd.DataFrame, oracle: pd.DataFrame | None) -> dict:
    """Validate each layer against the Sofascore oracle, honestly separating strong from dead signal.

    Three checks, in decreasing support:

    - **Position ordering** (strong): trusted mean advance ``x`` vs oracle position rank (D<M<F).
    - **Visibility** (strong): tracked-named frames vs oracle minutes played.
    - **Event involvement** (dead at this coverage): attributed passes vs oracle ``totalPass`` -- the
      attributed vector is almost all zeros, so this Spearman is reported but flagged non-meaningful.

    Args:
        inv: :func:`involvement` frame.
        geom: :func:`player_geometry` frame (Man Utd rows).
        oracle: :func:`_oracle_manu` frame or None.

    Returns:
        ``{spearman_posx, n_posx, spearman_min, n_min, spearman_pass, n_pass_nonzero, top5_overlap,
        n_matched, merged}``.
    """
    if oracle is None:
        return {"spearman_posx": None, "n_posx": 0, "spearman_min": None, "n_min": 0,
                "spearman_pass": None, "n_pass_nonzero": int((inv["attr_pass"] > 0).sum()),
                "top5_overlap": None, "n_matched": 0, "merged": inv}
    gm = geom.merge(oracle, left_on="player", right_on="name", how="left")
    trusted = gm[gm["trusted"] & gm["position"].isin(POS_RANK)]
    posx = _spearman(trusted["x"], trusted["position"].map(POS_RANK))
    vis = gm.dropna(subset=["truth_min"])
    smin = _spearman(vis["n_frames"], vis["truth_min"])
    m = inv.merge(oracle, left_on="player", right_on="name", how="left")
    both = m.dropna(subset=["truth_pass"])
    spass = _spearman(both["attr_pass"], both["truth_pass"], n_min=3)
    mine = list(inv.sort_values("attr_pass", ascending=False)["player"].head(5))
    theirs = list(oracle.sort_values("truth_pass", ascending=False)["name"].head(5))
    return {"spearman_posx": posx, "n_posx": int(len(trusted)),
            "spearman_min": smin, "n_min": int(len(vis)),
            "spearman_pass": spass, "n_pass_nonzero": int((inv["attr_pass"] > 0).sum()),
            "top5_overlap": len(set(mine) & set(theirs)),
            "n_matched": int(len(both)), "merged": m}


# === per-match assembly ==========================================================================
def match_profile(match: Match, named_df: pd.DataFrame) -> dict:
    """Full player profile for one match (geometry, lines, involvement, validation)."""
    roster = {n: g["team"].mode().iloc[0] for n, g in named_df.groupby("player_name")}
    pos = oriented_positions(match, named_df)
    geom = player_geometry(pos, roster)
    manu_int = _manu_int(match)
    lines = line_composition(pos, manu_int, geom)
    ledger = build_ledger_prtreid(match, named_df)
    inv = involvement(ledger, manu_int, roster)
    oracle = _oracle_manu(match.id)
    manu_geom = geom[geom["team_name"] == MANU].reset_index(drop=True)
    val = validate(inv, manu_geom, oracle)
    n_manu_pass = int((ledger["class"].eq("PASS") & ledger["team"].eq(manu_int)).sum())
    n_named_pass = int(inv["attr_pass"].sum())
    return {
        "match": match.id, "manager": match.manager, "home_team": match.home_team,
        "teams": list(match.teams), "geom": manu_geom, "lines": lines, "inv": inv,
        "val": val, "n_manu_links": inv.attrs.get("n_manu_links", 0),
        "n_manu_pass": n_manu_pass, "n_named_pass": n_named_pass,
        "named_pass_cov": round(n_named_pass / n_manu_pass, 3) if n_manu_pass else None,
        "n_manu_named": int(len(manu_geom)),
    }


# === cross-match aggregation =====================================================================
def cross_match(profiles: list[dict]) -> pd.DataFrame:
    """Man Utd players named in >= 2 matches, aggregated (volume totals, mean geometry).

    Returns:
        ``player, n_matches, matches, attr_pass, touch_proxy, poss_link_frac, x, final_third_frac,
        minutes`` (per-match means for the geometry columns).
    """
    grec: dict[str, list] = {}
    for p in profiles:
        gmap = {r["player"]: r for _, r in p["geom"].iterrows()}
        imap = {r["player"]: r for _, r in p["inv"].iterrows()}
        for player in set(gmap) | set(imap):
            grec.setdefault(player, []).append((p["match"], gmap.get(player), imap.get(player)))
    rows: list[dict] = []
    for player, recs in grec.items():
        if len(recs) < 2:
            continue
        # Geometry means use TRUSTED per-match records only (thin few-frame positions excluded).
        trust = [g for _, g, _ in recs if g is not None and g["trusted"]]
        xs = [g["x"] for g in trust]
        f3 = [g["final_third_frac"] for g in trust]
        frames = [g["n_frames"] for _, g, _ in recs if g is not None]
        plf = [i["poss_link_frac"] for _, _, i in recs if i is not None]
        rows.append({
            "player": player, "n_matches": len(recs), "n_trusted": len(trust),
            "matches": ",".join(m.split("_")[0] if "manutd" in m else m.replace("_manutd", "")
                                 for m, _, _ in recs),
            "attr_pass": int(sum(i["attr_pass"] for _, _, i in recs if i is not None)),
            "touch_proxy": int(sum(i["touch_proxy"] for _, _, i in recs if i is not None)),
            "poss_link_frac": round(float(np.mean(plf)), 3) if plf else 0.0,
            "frames": int(sum(frames)),
            "x": round(float(np.mean(xs)), 1) if xs else float("nan"),
            "final_third_frac": round(float(np.mean(f3)), 3) if f3 else float("nan"),
        })
    cols = ["player", "n_matches", "n_trusted", "matches", "attr_pass", "touch_proxy",
            "poss_link_frac", "frames", "x", "final_third_frac"]
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values(["n_matches", "frames"], ascending=False).reset_index(drop=True)


# === reporting ===================================================================================
def _fmt_geom(g: pd.DataFrame) -> list[str]:
    lines = ["| player | x (m) | y (m) | spread r (m) | final-3rd % | frames | trusted |",
             "|--------|------:|------:|-------------:|------------:|-------:|:-------:|"]
    for _, r in g.iterrows():
        mark = "yes" if r["trusted"] else "thin"
        lines.append(f"| {r['player']} | {r['x']:.1f} | {r['y']:.1f} | {r['spread_r']:.1f} | "
                     f"{r['final_third_frac'] * 100:.0f}% | {r['n_frames']} | {mark} |")
    return lines


def _fmt_inv(inv: pd.DataFrame, val: dict) -> list[str]:
    m = val["merged"]
    lines = ["| player | attr pass | oracle pass | touch proxy | oracle touch | poss-link % |",
             "|--------|----------:|------------:|------------:|-------------:|------------:|"]
    for _, r in m.iterrows():
        tp = r.get("truth_pass")
        tt = r.get("truth_touch")
        tps = "-" if tp is None or (isinstance(tp, float) and np.isnan(tp)) else int(tp)
        tts = "-" if tt is None or (isinstance(tt, float) and np.isnan(tt)) else int(tt)
        lines.append(f"| {r['player']} | {int(r['attr_pass'])} | {tps} | {int(r['touch_proxy'])} | "
                     f"{tts} | {r['poss_link_frac'] * 100:.1f}% |")
    return lines


def _fmt_match(p: dict) -> list[str]:
    v = p["val"]
    lines = [f"### {p['match']}  ({p['manager']}, home={p['home_team']})", "",
             f"Man Utd named players: {p['n_manu_named']} | attributed Man Utd passes: "
             f"{p['n_manu_pass']} | named-attributed: {p['n_named_pass']} "
             f"(coverage {(p['named_pass_cov'] or 0) * 100:.1f}%) | Man Utd poss-links: "
             f"{p['n_manu_links']}.", "",
             "Validation vs Sofascore (Man Utd):",
             f"- position ordering: Spearman(mean advance x, position D<M<F) = "
             f"{v['spearman_posx']} over {v['n_posx']} trusted players.",
             f"- visibility: Spearman(tracked-named frames, minutes played) = "
             f"{v['spearman_min']} over {v['n_min']} players.",
             f"- event involvement (DEAD -- {v['n_pass_nonzero']} players with any attributed pass): "
             f"Spearman(attr pass, totalPass) = {v['spearman_pass']}, top-5 overlap "
             f"{v['top5_overlap']}/5 -- computed over a near-all-zero vector, NOT meaningful.", ""]
    lc = p["lines"]
    if lc["centres_m"]:
        lines.append(f"**Lines** (centre m from own goal, {lc['n_entities']} entities clustered): "
                     f"def {lc['centres_m'][0]} / mid {lc['centres_m'][1]} / att "
                     f"{lc['centres_m'][2]}.")
        for band in BAND_NAMES:
            anchors = ", ".join(lc["bands"][band]) or "(no named anchor)"
            lines.append(f"- {band}: {anchors}")
        lines.append("")
    lines += ["**Visible geometry** (Man Utd, oriented to attack +x):", ""]
    lines += _fmt_geom(p["geom"])
    lines += ["", "**Involvement + oracle validation:**", ""]
    lines += _fmt_inv(p["inv"], p["val"])
    lines.append("")
    return lines


def format_report(profiles: list[dict], cross: pd.DataFrame) -> str:
    """Render ``results/PLAYER_ANALYSIS_v1.md``."""
    covs = [p["named_pass_cov"] or 0 for p in profiles]
    posx = [p["val"]["spearman_posx"] for p in profiles if p["val"]["spearman_posx"] is not None]
    smin = [p["val"]["spearman_min"] for p in profiles if p["val"]["spearman_min"] is not None]
    lines = [
        "# Player analysis v1 (Phase A) -- Manchester United named players", "",
        "Per-named-player profiles from the PRTreID PRECISION identity artifacts "
        "(`outputs/identity/<id>_named_tracks_both2_prtreid.parquet`) + the validated event ledger. "
        "Man Utd is oriented to attack +x on the 105x68 pitch. Auto-extends to any match that gains "
        "a PRTreID parquet.", "",
        "## Coverage statement (read first -- this bounds every claim below)", "",
        f"- **Identity matches: {len(profiles)}** (brighton_manutd, manutd_liverpool, "
        "manutd_tottenham -- all correct team-mapping, none of the two known flips).",
        "- **What has real support: the positional layer, not events.** Where players play (mean "
        "advance, line band) and how visible they are rest on hundreds of tracked-named frames per "
        "player. Visibility is validated in all three matches -- Spearman(frames, minutes played) = "
        f"{min(smin) if smin else float('nan'):.2f}-{max(smin) if smin else float('nan'):.2f} "
        "(positive every leg: the more a player is tracked-and-named, the more he actually played). "
        "Position ordering Spearman(mean advance x, D<M<F) = "
        f"{min(posx) if posx else float('nan'):.2f}-{max(posx) if posx else float('nan'):.2f}: "
        "positive in 2/3 matches but NEGATIVE in brighton_manutd, where the few trusted frames of an "
        "attacking full-back (Mazraoui, nominally D) landed him at 72 m and inverted the naive "
        "defender-deep ordering. So mean-x recovers role directionally but is fooled by advanced "
        "full-backs and sparse-frame players -- read it with the frame count.",
        f"- **What is DEAD: per-player event involvement.** Only {min(covs) * 100:.1f}-"
        f"{max(covs) * 100:.1f}% of Man Utd's attributed passes carry a named player (the known "
        "4-12% floor, a tracking limit not an identity error): 2-5 named passes per match. The "
        "attributed-pass / touch / possession-link counts are a **floor an order of magnitude below "
        "the truth**, and their Spearman vs the oracle is computed over a near-all-zero vector -- it "
        "is reported but is NOT a meaningful ranking. This confirms the `PASS_NETWORKS_v1` negative "
        "result at the player level.",
        "- **No plus-minus, no per-90 rate cards, no involvement-based impact ranking.** At this "
        "coverage on/off-ball impact cannot be estimated. The impact section below leads with what "
        "the positional layer CAN say (role, territory, line anchoring) and states plainly what it "
        "cannot.", "",
        "## Per-match named-player tables", "",
    ]
    for p in profiles:
        lines += _fmt_match(p)
    lines += ["## Cross-match aggregation (Man Utd players named in 2+ matches)", "",
              "`frames` and volume columns are totals; `x` / `final-3rd %` are means over the "
              "player's TRUSTED (>=25-frame) per-match positions only (`trust` = how many of the "
              "matches gave a trusted position). Event columns stay near zero -- see the coverage "
              "note; the positional columns are the validated content.", "",
              "| player | matches | n | trust | frames | x (m) | final-3rd % | attr pass | "
              "touch proxy | poss-link % |",
              "|--------|---------|--:|------:|-------:|------:|------------:|----------:|"
              "------------:|------------:|"]
    for _, r in cross.iterrows():
        xs = "-" if np.isnan(r["x"]) else f"{r['x']:.0f}"
        f3 = "-" if np.isnan(r["final_third_frac"]) else f"{r['final_third_frac'] * 100:.0f}%"
        lines.append(f"| {r['player']} | {r['matches']} | {r['n_matches']} | {r['n_trusted']} | "
                     f"{r['frames']} | {xs} | {f3} | {r['attr_pass']} | {r['touch_proxy']} | "
                     f"{r['poss_link_frac'] * 100:.1f}% |")
    lines += ["", "## Who is the impact player? (the honest answer)", ""]
    lines += _impact_section(profiles, cross)
    lines += ["", "## What this CANNOT claim", "",
              "- **Not a plus-minus or a rating.** No goals/assists-added, no on/off splits -- the "
              "attribution coverage (4-12%) forbids it. A player's low attributed count can be low "
              "involvement OR low broadcast visibility; the two are not separable here.",
              "- **Line bands are geometric, not tactical roles.** The clustering splits Man Utd "
              "outfield entities by mean advance on trusted frames; a full-back bombing on reads as "
              "higher, a dropping striker as lower. It is a shape descriptor, not a formation call.",
              "- **Possession-link % over-counts.** An edge links the next *attributed* carrier and "
              "skips unlabeled touches; and a named carrier can be the opponent's nearest player. "
              "Read it as relative involvement, not a pass-completion figure.",
              "- **Frames != played minutes.** The `frames` column counts only frames where the "
              "player's fragment is named and on-screen (broadcast follows the ball), so it tracks "
              "minutes only in *rank* (validated: frames-vs-minutes Spearman above), not in scale.",
              "- **n=3 matches, distinct opponents.** No opponent is measured twice, so nothing here "
              "separates a player-stable trait from a single-match matchup.", ""]
    return "\n".join(lines)


def _impact_section(profiles: list[dict], cross: pd.DataFrame) -> list[str]:
    """The honest answer to 'who is the biggest impact player' given a dead involvement layer."""
    lines = [
        "**Straight answer: this corpus cannot name an impact player, and it would be dishonest to.**"
        " Impact needs involvement volume or on/off value, and the event layer is dead here -- 2-5 "
        "named passes per match (coverage note above). Ranking anyone 'biggest impact' off three "
        "attributed passes would be noise dressed as a finding. What the data DOES support, and what "
        "checks out against the oracle, is *where each player operates and how central their zone is "
        "to the shape* -- so that is what is reported.", ""]
    if not len(cross):
        lines.append("No Man Utd player is named in 2+ matches -- no cross-match read.")
        return lines
    terr = cross.dropna(subset=["x"]).copy()
    deepest = terr.sort_values("x").head(3)
    highest = terr.sort_values("x", ascending=False).head(3)
    seen = cross.sort_values("frames", ascending=False).head(3)
    lines += [
        "What the validated positional layer supports (players in 2+ matches, trusted positions):", "",
        "- **Deepest builders (rearmost mean advance):** "
        + ", ".join(f"{r['player']} ({r['x']:.0f} m)" for _, r in deepest.iterrows())
        + " -- these anchor the defensive/first line of the build-up.",
        "- **Highest / most territorial:** "
        + ", ".join(f"{r['player']} ({r['x']:.0f} m, {r['final_third_frac'] * 100:.0f}% final-3rd)"
                    for _, r in highest.iterrows())
        + " -- the players carrying Man Utd furthest up the pitch on trusted frames.",
        "- **Most on-ball-visible (a biased proxy for centrality, NOT impact):** "
        + ", ".join(f"{r['player']} ({r['frames']} frames)" for _, r in seen.iterrows())
        + " -- read as 'the pipeline sees them on the ball most', which conflates true involvement "
        "with broadcast/tracking visibility; do not read it as most valuable.", "",
        "So the closest defensible statement to Sid's question is territorial, not a rating: Man "
        "Utd's build-up is anchored deep by "
        f"{deepest.iloc[0]['player']} and carried highest by {highest.iloc[0]['player']}. Naming a "
        "single 'biggest impact player' needs the plus-minus this coverage forbids -- flagged, not "
        "faked."]
    return lines


# === orchestration ===============================================================================
def build_all() -> tuple[list[dict], pd.DataFrame]:
    """Build every PRTreID match's profile and the cross-match aggregation."""
    profiles: list[dict] = []
    for match, named in prtreid_matches():
        print(f"player-profiles {match.id} ...")
        p = match_profile(match, named)
        profiles.append(p)
        v = p["val"]
        print(f"  ManU named {p['n_manu_named']} | named-pass cov "
              f"{(p['named_pass_cov'] or 0) * 100:.1f}% | pos-order rho {v['spearman_posx']} "
              f"(n={v['n_posx']}) | vis rho {v['spearman_min']} (n={v['n_min']})")
    return profiles, cross_match(profiles)


def main() -> None:
    """Build profiles, write ``results/PLAYER_ANALYSIS_v1.md``."""
    profiles, cross = build_all()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(format_report(profiles, cross), encoding="utf-8")
    print(f"cross-match players (2+): {len(cross)} | wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
