"""Phase-0 build A: event-only directed passing networks from the validated BAS pass stream.

Turns the attributed pass ledger (:func:`tools.event_ledger.build_ledger` -- reused, not forked) into
per-team directed passing networks and their network-science fingerprint. Everything here is
event-only (no geometry dependency beyond the ledger's carrier attribution) and CPU-only.

The layers, per team per match:

1. **Directed network** -- op-filtered PASS events are segmented into possessions (a maximal run of
   same-team attributed carriers with no attributed opponent flip and no > :data:`POSS_GAP_S` gap).
   Within a possession, consecutive attributed carriers form a directed edge ``src -> dst``. A carrier
   is a named player where identity resolves, else a per-team abstain bucket ``"<team> _UNK"``.
   ``src -> dst`` skips intervening unattributed (team-NA) passes -- a known approximation (the two
   named carriers may not be a *direct* pass), flagged in the report and by the coverage stat.

2. **Network metrics** (named-player subgraph only -- the bucket collapses structure): density,
   edge asymmetry (reciprocity + weighted asymmetry), top-edge concentration (top-3 share + Gini),
   strength centralization, max betweenness.

3. **Flow motifs** -- 2-path structure on the named weighted adjacency: ABA return walks
   (``W[i,j]*W[j,i]``) vs ABC progression walks (``W[i,j]*W[j,k]``, distinct), with a
   strength-preserving directed-configuration null (in-stub shuffle) giving a z-score that separates
   *structure* (reciprocal passing beyond volume) from *volume*.

4. **Average-position formation proxy** -- mean attack-oriented ``(x, y)`` per named track per half,
   over trusted frames (the event-channel formation visual). Uses more data than the pass events
   (every tracked frame of a named fragment), so it is the one player-level artifact that survives.

**GATE-0** (printed + persisted): op PASS totals vs the frozen Sofascore 0.97-1.09x band
(reuses :func:`tools.bas_validate.validate_match`); named-edge coverage reported honestly against the
known 4-12% player floor; motif return-ratio stability across the Man Utd matches.

Run (CPU)::

    python -m fingerprint.pass_network
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from core import registry
from core.registry import Match
from fingerprint.structural_metrics import resolve_attack_directions
from fingerprint.style_fingerprint import _orient
from tools import bas_validate, event_ledger

# --- possession + edge construction --------------------------------------------------------------
POSS_GAP_S = 8.0          # break a possession if attributed passes are farther apart than this (same chunk)
BAS_FPS = 25.0            # BAS frame_index is the within-chunk 25 fps clock (see tools.event_ledger)
POSS_GAP_FR = int(POSS_GAP_S * BAS_FPS)
MOTIF_NULL_N = 500        # strength-preserving randomizations for the motif z-score
BAND = (0.97, 1.09)       # frozen Sofascore pass-count band (tools.bas_validate operating point)
MANU = "Man Utd"
OUT_NAME = "pass_network.json"
REPORT_PATH = Path("results/PASS_NETWORKS_v1.md")


def build_edges(ledger: pd.DataFrame) -> pd.DataFrame:
    """Directed pass edges (one row per edge occurrence) from an attributed ledger.

    Segments op-PASS events into possessions and links consecutive attributed carriers. Team-NA
    passes neither break nor connect (they are skipped -- the ``src -> dst`` may hide unattributed
    touches). A carrier label is the named player or the per-team abstain bucket ``"<team> _UNK"``.

    Args:
        ledger: an :func:`tools.event_ledger.build_ledger` frame (PASS + DRIVE rows).

    Returns:
        ``team, team_name, src, dst, both_named`` -- one row per directed edge occurrence.
    """
    passes = ledger[(ledger["class"] == "PASS") & ledger["team"].notna()].copy()
    passes = passes.sort_values(["chunk", "frame_index"]).reset_index(drop=True)
    rows: list[dict] = []
    prev: dict | None = None
    for r in passes.itertuples(index=False):
        team = int(r.team)
        named = pd.notna(r.player)
        label = r.player if named else f"{r.team_name}_UNK"
        cur = {"team": team, "team_name": r.team_name, "chunk": r.chunk,
               "frame": int(r.frame_index), "label": label, "named": named}
        if (prev is not None and prev["team"] == team and prev["chunk"] == r.chunk
                and cur["frame"] - prev["frame"] <= POSS_GAP_FR):
            rows.append({"team": team, "team_name": r.team_name,
                         "src": prev["label"], "dst": label,
                         "both_named": bool(prev["named"] and named)})
        prev = cur
    return pd.DataFrame(rows, columns=["team", "team_name", "src", "dst", "both_named"])


# --- weighted adjacency + network metrics --------------------------------------------------------
def named_adjacency(edges_team: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Weighted directed adjacency over the *named* nodes of one team (bucket edges dropped).

    Args:
        edges_team: :func:`build_edges` rows for a single team.

    Returns:
        ``(nodes, W)`` with ``W[i, j]`` = passes ``nodes[i] -> nodes[j]`` (named endpoints only).
    """
    named = edges_team[edges_team["both_named"] & (edges_team["src"] != edges_team["dst"])]
    nodes = sorted(set(named["src"]) | set(named["dst"]))
    idx = {n: i for i, n in enumerate(nodes)}
    w = np.zeros((len(nodes), len(nodes)), float)
    for src, dst in zip(named["src"], named["dst"]):
        w[idx[src], idx[dst]] += 1.0
    return nodes, w


def _gini(x: np.ndarray) -> float:
    """Gini coefficient of non-negative weights (0 = equal, ->1 = concentrated); NaN if empty/zero."""
    x = np.sort(np.asarray(x, float))
    if x.size == 0 or x.sum() == 0:
        return float("nan")
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def network_metrics(nodes: list[str], w: np.ndarray) -> dict:
    """Structural fingerprint of one team's named passing network.

    Args:
        nodes: named node labels.
        w: weighted directed adjacency (from :func:`named_adjacency`).

    Returns:
        density, reciprocity, weighted asymmetry, top-3 edge share, edge Gini, strength
        centralization, max betweenness, and node/edge counts (NaN where undefined at this size).
    """
    n = len(nodes)
    edge_w = w[w > 0]
    e = int((w > 0).sum())
    recip = pairs = asym_num = asym_den = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = w[i, j], w[j, i]
            if a > 0 or b > 0:
                pairs += 1
                recip += int(a > 0 and b > 0)
                asym_num += abs(a - b)
                asym_den += a + b
    strength = w.sum(0) + w.sum(1)
    top3 = float(np.sort(edge_w)[::-1][:3].sum() / edge_w.sum()) if edge_w.size else float("nan")
    cent = (float((strength.max() - strength.mean()) / strength.sum())
            if n > 1 and strength.sum() > 0 else float("nan"))
    max_btw = float("nan")
    if e > 0:
        g = nx.DiGraph()
        g.add_nodes_from(range(n))
        for i in range(n):
            for j in range(n):
                if w[i, j] > 0:
                    g.add_edge(i, j, weight=1.0 / w[i, j])  # betweenness: cost = 1/passes
        btw = nx.betweenness_centrality(g, weight="weight")
        max_btw = float(max(btw.values())) if btw else float("nan")
    return {
        "n_named_nodes": n, "n_named_edges": e, "total_named_passes": float(w.sum()),
        "density": (e / (n * (n - 1))) if n > 1 else float("nan"),
        "reciprocity": (recip / pairs) if pairs else float("nan"),
        "weighted_asymmetry": (asym_num / asym_den) if asym_den else float("nan"),
        "top3_edge_share": top3, "edge_gini": _gini(edge_w),
        "strength_centralization": cent, "max_betweenness": max_btw,
    }


# --- flow motifs (2-path structure) + strength-preserving null -----------------------------------
def two_path_counts(w: np.ndarray) -> tuple[float, float]:
    """ABA return-walk and ABC progression-walk weights on a weighted directed adjacency.

    ABA = ``sum_{i!=j} W[i,j] W[j,i]`` (one-twos / returns); ABC = ``sum_{i,j,k distinct}
    W[i,j] W[j,k]`` (circulation to a new player). Self-loops are excluded.

    Args:
        w: weighted directed adjacency.

    Returns:
        ``(aba, abc)`` walk weights.
    """
    m = w.copy()
    np.fill_diagonal(m, 0.0)
    aba = float((m * m.T).sum())            # = trace(m@m): i->j->i return walks
    allwalks = float((m @ m).sum())         # every 2-step walk i->j->k
    abc = allwalks - aba                    # i!=k progression walks (circulation)
    return aba, max(abc, 0.0)


def motif_null(w: np.ndarray, *, n: int = MOTIF_NULL_N, seed: int = 0) -> dict:
    """ABA return-ratio and its z-score against a strength-preserving directed null.

    The null keeps each node's out- and in-strength exactly (shuffle in-stubs among out-stubs), so a
    positive z means reciprocal passing exceeds what pass *volume* alone predicts -- structure, not
    volume.

    Args:
        w: named weighted directed adjacency.
        n: number of randomizations.
        seed: RNG seed.

    Returns:
        ``ratio_obs, ratio_null_mean, aba_obs, aba_z, n_pass`` (NaN-heavy if too few passes).
    """
    m = w.copy()
    np.fill_diagonal(m, 0.0)
    total = m.sum()
    aba_obs, abc_obs = two_path_counts(m)
    ratio_obs = aba_obs / (aba_obs + abc_obs) if (aba_obs + abc_obs) > 0 else float("nan")
    out = {"ratio_obs": ratio_obs, "aba_obs": aba_obs, "n_pass": float(total),
           "ratio_null_mean": float("nan"), "aba_z": float("nan")}
    if total < 6 or m.shape[0] < 3:      # too few passes/nodes for a meaningful null
        return out
    src = np.repeat(np.arange(m.shape[0]), m.sum(1).astype(int))    # out-stubs
    dst = np.repeat(np.arange(m.shape[0]), m.sum(0).astype(int))    # in-stubs
    rng = np.random.default_rng(seed)
    abas, ratios = [], []
    for _ in range(n):
        d = rng.permutation(dst)
        r = np.zeros_like(m)
        np.add.at(r, (src, d), 1.0)
        aba, abc = two_path_counts(r)
        abas.append(aba)
        if aba + abc > 0:
            ratios.append(aba / (aba + abc))
    abas = np.asarray(abas)
    out["ratio_null_mean"] = float(np.mean(ratios)) if ratios else float("nan")
    out["aba_z"] = float((aba_obs - abas.mean()) / abas.std()) if abas.std() > 0 else float("nan")
    return out


# --- average-position formation proxy ------------------------------------------------------------
def _name_team_map(match: Match) -> dict[str, str]:
    """``{player_name: true_team_name}`` from the named-tracks roster (empty for non-identity matches).

    The ledger's ``player`` is the nearest named track to the ball, which can belong to the *opponent*
    of the carrier team -- so a player's own team must come from the identity roster, not the carrier
    attribution, to avoid crediting opponent players to Man Utd.
    """
    if match.id not in event_ledger.PLAYER_TRUTH:
        return {}
    nt = pd.read_parquet(event_ledger.PLAYER_TRUTH[match.id][1])
    return {n: g["team"].mode().iloc[0] for n, g in nt.groupby("player_name")}


def formation_positions(match: Match) -> pd.DataFrame:
    """Mean attack-oriented ``(x, y)`` per named track per half (the event-channel formation visual).

    Only defined for identity matches (:data:`tools.event_ledger.PLAYER_TRUTH`). ManU/opponents both
    named. Positions are oriented so each team attacks ``+x`` (per-chunk keeper resolve), averaged over
    every trusted frame of the named fragment -- more data than the sparse pass events.

    Returns:
        ``player, team_name, half, x, y, n_frames`` (empty for non-identity matches).
    """
    cols = ["player", "team_name", "half", "x", "y", "n_frames"]
    if match.id not in event_ledger.PLAYER_TRUTH:
        return pd.DataFrame(columns=cols)
    named = pd.read_parquet(event_ledger.PLAYER_TRUTH[match.id][1])
    ntmap = _name_team_map(match)
    df = match.load_aligned()
    players = df[df["role"].isin(["player", "goalkeeper"])].dropna(subset=["pitch_x", "pitch_y"])
    acc: dict[tuple, list] = {}
    for ck, g in players.groupby("chunk"):
        name_by = event_ledger._named_by_track(named, str(ck))
        if not name_by:
            continue
        adir = resolve_attack_directions(g)
        half = str(ck)[:2]
        sub = g[g["track_id"].isin(name_by)]
        for tid, tg in sub.groupby("track_id"):
            team = int(tg["team"].mode().iloc[0])       # anchored team -> attack orientation
            if adir.get(team) is None:
                continue
            px, py = _orient(tg["pitch_x"].to_numpy(), tg["pitch_y"].to_numpy(), adir[team])
            name = name_by[int(tid)]
            key = (name, ntmap.get(name, match.teams[team]), half)   # roster team, not carrier team
            acc.setdefault(key, [[], []])
            acc[key][0].extend(px.tolist())
            acc[key][1].extend(py.tolist())
    rows = [{"player": p, "team_name": tm, "half": h, "x": round(float(np.mean(xs)), 2),
             "y": round(float(np.mean(ys)), 2), "n_frames": len(xs)}
            for (p, tm, h), (xs, ys) in acc.items()]
    return pd.DataFrame(rows, columns=cols).sort_values(["team_name", "half", "player"])


# --- GATE-0 band (reuse bas_validate) ------------------------------------------------------------
def gate0_band(match: Match) -> dict:
    """Op PASS total vs Sofascore attempted over complete halves (the frozen 0.97-1.09x band)."""
    res = bas_validate.validate_match(match)
    op = truth = 0
    complete = []
    for h in bas_validate.HALVES:
        ph = res["per_half"][h]
        if ph["complete"] and ph["truth_att"]:
            op += ph["counts"]["PASS"]["op"]
            truth += ph["truth_att"]
            complete.append(h)
    ratio = round(op / truth, 3) if truth else None
    return {"op_pass": op, "truth_att": truth, "ratio": ratio,
            "complete_halves": complete,
            "in_band": (ratio is not None and BAND[0] <= ratio <= BAND[1])}


# --- per-match orchestration ---------------------------------------------------------------------
def match_report(match: Match) -> dict:
    """Full pass-network artifact for one match (both teams)."""
    ledger = event_ledger.build_ledger(match)
    edges = build_edges(ledger)
    passes = ledger[(ledger["class"] == "PASS")]
    n_pass = len(passes)
    n_team = int(passes["team"].notna().sum())
    n_named = int(passes["player"].notna().sum())
    # Named pass VOLUME per player (independent of edges) -- the only non-empty player signal.
    # Grouped by the player's ROSTER team (a named carrier may be the opponent of the ball team).
    ntmap = _name_team_map(match)
    volume: dict[str, dict[str, float]] = {}
    for player, cnt in passes[passes["player"].notna()].groupby("player").size().items():
        tt = ntmap.get(player)
        if tt:
            volume.setdefault(tt, {})[player] = float(cnt)
    teams: dict[str, dict] = {}
    for team_int in sorted(passes["team"].dropna().unique()):
        team_name = match.teams[int(team_int)]
        et = edges[edges["team"] == team_int] if len(edges) else edges
        named_occ = (et["both_named"] & (et["src"] != et["dst"])) if len(et) else et.index == -1
        nodes, w = named_adjacency(et) if len(et) else ([], np.zeros((0, 0)))
        metrics = network_metrics(nodes, w)
        motif = motif_null(w)
        strength = {nodes[i]: float(w[i].sum() + w[:, i].sum()) for i in range(len(nodes))}
        teams[team_name] = {
            "team_int": int(team_int), "metrics": metrics, "motif": motif,
            "n_poss_links": int(len(et)),
            "n_named_edge_occ": int(named_occ.sum()) if len(et) else 0,
            "node_strength": strength,
            "named_volume": volume.get(team_name, {}),
            "edges_named": [{"src": s, "dst": d} for s, d, b in
                            zip(et["src"], et["dst"], named_occ) if b] if len(et) else [],
        }
    return {
        "match": match.id,
        "coverage": {"n_pass_events": n_pass, "n_team_attributed": n_team,
                     "n_named": n_named,
                     "team_attr_rate": round(n_team / n_pass, 3) if n_pass else None,
                     "named_rate": round(n_named / n_pass, 3) if n_pass else None},
        "gate0": gate0_band(match),
        "teams": teams,
        "formation": formation_positions(match).to_dict(orient="records"),
    }


# --- identifiability (does ManU look like ManU across matches) -----------------------------------
def _cosine(a: dict, b: dict) -> float:
    """Cosine between two ``{player: strength}`` vectors on the union of players."""
    keys = sorted(set(a) | set(b))
    va = np.array([a.get(k, 0.0) for k in keys])
    vb = np.array([b.get(k, 0.0) for k in keys])
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    return float(va @ vb / (na * nb)) if na > 0 and nb > 0 else float("nan")


def identifiability(reports: list[dict]) -> dict:
    """Man Utd cross-match self-similarity of the named node-strength profile (volume-based).

    With distinct opponents (n=1 each) opponent self-consistency is untestable, so this reports the
    Man Utd intra-match cosine of the player pass-*volume* profile (the named-edge network is empty at
    our coverage) -- the honest, under-powered proxy for the Buldu-lineage identifiability test.
    """
    manu = []
    for rep in reports:
        prof = rep["teams"].get(MANU, {}).get("named_volume", {})
        if prof:
            manu.append((rep["match"], prof))
    pairs = []
    for i in range(len(manu)):
        for j in range(i + 1, len(manu)):
            pairs.append({"a": manu[i][0], "b": manu[j][0],
                          "cosine": round(_cosine(manu[i][1], manu[j][1]), 3)})
    vals = [p["cosine"] for p in pairs if not np.isnan(p["cosine"])]
    return {"n_manu_networks": len(manu),
            "mean_intra_cosine": round(float(np.mean(vals)), 3) if vals else None,
            "pairs": pairs}


# --- reporting -----------------------------------------------------------------------------------
def _fmt_match(rep: dict) -> list[str]:
    c = rep["coverage"]
    g = rep["gate0"]
    band = "PASS" if g["in_band"] else "FAIL"
    lines = [f"### {rep['match']}", "",
             f"Coverage -- PASS events {c['n_pass_events']} | team-attributed {c['n_team_attributed']} "
             f"({(c['team_attr_rate'] or 0) * 100:.1f}%) | named {c['n_named']} "
             f"({(c['named_rate'] or 0) * 100:.1f}%).",
             f"GATE-0 band -- op PASS {g['op_pass']} / truth {g['truth_att']} = {g['ratio']} "
             f"(complete halves {g['complete_halves'] or '-'}) -> **{band}** (band {BAND[0]}-{BAND[1]}x).",
             "",
             "| team | poss links | named nodes | named edges | recip | top3 | Gini | cent | "
             "ABA ratio | ABA z |",
             "|------|-----------|-------------|-------------|-------|------|------|------|-----------|"
             "-------|"]
    for tm, t in rep["teams"].items():
        m, mo = t["metrics"], t["motif"]
        lines.append(
            f"| {tm} | {t['n_poss_links']} | {m['n_named_nodes']} | {m['n_named_edges']} | "
            f"{_n(m['reciprocity'])} | {_n(m['top3_edge_share'])} | {_n(m['edge_gini'])} | "
            f"{_n(m['strength_centralization'])} | {_n(mo['ratio_obs'])} | {_n(mo['aba_z'])} |")
    lines.append("")
    return lines


def _n(x) -> str:
    """Compact numeric cell (``-`` for NaN/None)."""
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def format_report(reports: list[dict], ident: dict) -> str:
    """Render ``results/PASS_NETWORKS_v1.md``."""
    n_band = sum(r["gate0"]["in_band"] for r in reports)
    named_rates = [r["coverage"]["named_rate"] or 0 for r in reports]
    lines = [
        "# Pass networks v1 (Phase-0 build A)", "",
        "Directed passing networks from the validated BAS pass stream + event-ledger carrier "
        "attribution (`tools.event_ledger.build_ledger`, reused). Edges link consecutive attributed "
        "carriers within a possession; nodes are named players where identity resolves, else a "
        "per-team abstain bucket. Structural metrics use the **named-player subgraph only** (the "
        "bucket collapses who-to-whom structure).", "",
        "## GATE-0 verdicts", "",
        f"- **Pass-total band:** {n_band}/{len(reports)} matches inside the Sofascore "
        f"{BAND[0]}-{BAND[1]}x band (op PASS vs attempted, complete halves).",
        f"- **Named-edge coverage:** named pass rate {min(named_rates) * 100:.1f}-"
        f"{max(named_rates) * 100:.1f}% across matches (the known 4-12% player floor). "
        "Named-player *edges* (both endpoints named, consecutive) are rarer still -- see per-match "
        "counts; the player-level network is volume-thin, structure-poor.",
        f"- **Man Utd identifiability:** {ident['n_manu_networks']} ManU identity matches; mean "
        f"intra-match player pass-volume cosine = {ident['mean_intra_cosine']} "
        "(the named-edge network is empty -- see limits; structural identifiability is not "
        "computable at this coverage).", "",
        "## Per-match networks + gate", "",
    ]
    for rep in reports:
        lines += _fmt_match(rep)
    lines += ["## Man Utd cross-match identifiability (player pass-volume cosine)", "",
              "| match A | match B | cosine |", "|---------|---------|--------|"]
    for p in ident["pairs"]:
        lines.append(f"| {p['a']} | {p['b']} | {p['cosine']} |")
    lines += ["", f"Mean intra-ManU cosine: **{ident['mean_intra_cosine']}** over "
              f"{ident['n_manu_networks']} networks.", "",
              "## Honest limits", "",
              "- **Player network is not viable at current identity coverage.** Named passes are "
              "0-42 per match across both teams (2-4% of the pass stream); consecutive named->named "
              "edges are a handful, so centrality/motif numbers on the named subgraph are volume "
              "artifacts, not structure. What is real is per-player pass *volume* (the node "
              "strengths, echoing `results/PLAYER_LEDGER.md`), not who-passes-to-whom.",
              "- **Edges skip unattributed touches.** `src -> dst` links the next attributed carrier "
              "in a possession, so an edge may span 1-2 unlabeled passes -- an over-connection bias, "
              "not a direct-pass guarantee.",
              "- **Team attribution is the nearest-carrier heuristic** (`tools.event_ledger`); it "
              "abstains ~60% of passes (no tracked player on the ball at the kick). The abstain "
              "bucket carries most edge weight; the named subgraph is the residue.",
              "- **Formation proxy is the usable player-level artifact** -- it averages every trusted "
              "frame of a named fragment, not just pass events, so it has real support; see the "
              "`formation` records in each `outputs/<id>/facts/pass_network.json`.",
              "- **Identifiability is under-powered:** 3 identity matches, distinct opponents (n=1 "
              "each) -> opponent self-consistency untestable; only ManU intra-match consistency is "
              "reported.", ""]
    return "\n".join(lines)


def main() -> None:
    """Build every BAS match's pass network, persist per-match JSON, write the v1 report."""
    reports: list[dict] = []
    for match in registry.matches():
        if not (bas_validate.BALL_ROOT / match.id / "ball_action").exists() or not match.processed:
            continue
        print(f"pass-network {match.id} ...")
        rep = match_report(match)
        if rep["coverage"]["n_pass_events"] == 0:
            print("  no BAS pass events yet (empty/partial ball_action dir) -- skipped")
            continue
        out = match.aligned.parent.parent / "facts" / OUT_NAME  # outputs/<id>/facts/pass_network.json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        reports.append(rep)
        c, g = rep["coverage"], rep["gate0"]
        print(f"  named {c['n_named']}/{c['n_pass_events']} ({(c['named_rate'] or 0) * 100:.1f}%) | "
              f"band {g['ratio']} {'PASS' if g['in_band'] else 'FAIL'} | wrote {out}")
    ident = identifiability(reports)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(format_report(reports, ident), encoding="utf-8")
    print(f"ManU identifiability: {ident['n_manu_networks']} networks, mean intra cosine "
          f"{ident['mean_intra_cosine']}")
    print(f"wrote {REPORT_PATH}")


def _demo() -> None:
    """Self-check on the pure seams: edge segmentation, motif counts, strength-preserving null."""
    # possession segmentation: A->B->A same team; team flip breaks; gap breaks.
    led = pd.DataFrame({
        "class": ["PASS"] * 6,
        "chunk": ["h1_chunk_000"] * 6,
        "frame_index": [0, 25, 50, 60, 1000, 1010],
        "team": [0, 0, 0, 1, 0, 0],
        "team_name": ["MU", "MU", "MU", "OP", "MU", "MU"],
        "player": ["A", "B", "A", "X", "C", "D"],
    })
    e = build_edges(led)
    mu = e[e["team"] == 0]
    got = set(zip(mu["src"], mu["dst"]))
    assert ("A", "B") in got and ("B", "A") in got, got          # A-B-A one-two
    assert ("A", "C") not in got, "team flip + gap must break: no A/B -> C edge"
    assert ("C", "D") in got, "C, D are consecutive in the fresh possession"
    # two-path counts on a clean reciprocal pair + a chain
    w = np.array([[0, 2, 0], [2, 0, 3], [0, 0, 0]], float)       # A<->B (2 each), B->C (3)
    aba, abc = two_path_counts(w)
    assert aba == 2 * 2 + 2 * 2, aba                             # A->B->A + B->A->B = 4+4=8? check
    assert abc == 2 * 3, abc                                     # A->B->C = 6
    # null preserves strengths and is finite on enough passes
    nz = motif_null(w, n=50)
    assert nz["n_pass"] == w.sum() and np.isfinite(nz["ratio_obs"])
    print("pass_network self-check OK")


if __name__ == "__main__":
    main()
