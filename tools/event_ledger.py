"""B-3 stage 2: attribute the validated BAS event stream to teams and named players.

Stage 1 (:mod:`tools.bas_validate`) froze an operating-point filter that brings BAS pass counts to
~1.0x Sofascore. This stage takes those exact filtered events (via
:func:`tools.bas_validate.op_events` -- reused, not re-derived) and answers *who*:

1. **Team attribution** -- at each event frame, the ball-carrier's team. The carrier is the tracked
   player nearest the ball (Viterbi ``final/ball`` track) at the aligned frame nearest the event.
   Abstain when no ball position is available within +-1 s or the nearest player is a loose-ball
   distance away. Validated against Sofascore per-team attempted passes (the stage-2 gate).
2. **Player attribution** (identity matches only) -- the named track nearest the ball at the event
   frame, within a tight carrier radius. Coverage is sparse (named-fragment coverage is sparse);
   validated by rank-correlation + ratio against Sofascore per-player ``totalPass`` for the named
   subset only.

Frame bookkeeping (get this right or everything downstream is garbage): BAS ``frame_index`` is the
25 fps within-chunk index; the aligned / ball parquets sample the SAME within-chunk clock at stride
:data:`STEP` (=5). So a BAS event maps to grid frame ``round(frame_index / 5) * 5`` in the *same*
chunk key -- verified by :mod:`tests.test_event_ledger`.

Run (CPU)::

    python -m tools.event_ledger
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core import registry
from core.registry import Match
from tools.bas_validate import load_actions, op_events

# --- Frame + attribution parameters --------------------------------------------------------------
STEP = 5                 # aligned/ball parquet frame stride (25 fps sampled every 5 frames)
BALL_GAP_FR = 25         # accept a ball/player frame within +-1 s (25 fps) of the event
TEAM_MAX_M = 3.0         # on-the-ball radius: abstain if nearest player is farther (carrier untracked)
PLAYER_MAX_M = 3.0       # named-track carrier radius for player attribution
# The BAS peak fires at the kick, when the ball is already leaving the passer's feet -- so scan a
# short window biased just BEFORE the peak and take the frame where a player is closest to the ball
# (the moment of contact). -0.6 s .. +0.2 s at 25 fps. Naive "at-peak" catches the ball mid-flight
# (median carrier 3.3 m); window-min lands the passer on the ball (median 1.3 m).
CONTACT_OFFSETS = tuple(range(-15, 6, STEP))
CLASSES = ("PASS", "DRIVE")
HALVES = ("h1", "h2")

# match_id -> Sofascore team-stats parquet (per-half attempted passes). Mirrors bas_validate.TRUTH.
TEAM_TRUTH = {
    "brighton_manutd": "outputs/oracle/sofascore/team_stats_12436888.parquet",
    "manutd_liverpool": "outputs/oracle/sofascore/team_stats_12436920.parquet",
    "manutd_fulham": "outputs/oracle/sofascore/team_stats_12436870.parquet",
    "manutd_tottenham": "outputs/oracle/sofascore/team_stats_12436995.parquet",
}
# match_id -> (player-stats parquet, named-tracks parquet). Identity matches only.
PLAYER_TRUTH = {
    "brighton_manutd": ("outputs/oracle/sofascore/player_stats_12436888.parquet",
                        "outputs/identity/brighton_manutd_named_tracks_koshkina.parquet"),
    "manutd_liverpool": ("outputs/oracle/sofascore/player_stats_12436920.parquet",
                         "outputs/identity/manutd_liverpool_named_tracks_koshkina.parquet"),
    "manutd_tottenham": ("outputs/oracle/sofascore/player_stats_12436995.parquet",
                         "outputs/identity/manutd_tottenham_named_tracks_koshkina.parquet"),
}
REPORT_PATH = Path("results/PLAYER_LEDGER.md")


# === Pure seams (unit-tested) ====================================================================
def bas_to_grid_frame(frame_index: int, step: int = STEP) -> int:
    """Map a 25 fps BAS ``frame_index`` to the nearest stride-``step`` grid frame (same chunk).

    Args:
        frame_index: within-chunk 25 fps frame index from a BAS action.
        step: aligned/ball parquet frame stride.

    Returns:
        The grid frame (a multiple of ``step``) nearest ``frame_index``.
    """
    return int(round(frame_index / step)) * step


def _nearest_row(frames: np.ndarray, target: int, max_gap: int) -> int | None:
    """Index into ``frames`` (ascending) of the value nearest ``target`` within ``max_gap``, else None."""
    if frames.size == 0:
        return None
    i = int(np.searchsorted(frames, target))
    best, best_d = None, max_gap + 1
    for j in (i - 1, i):
        if 0 <= j < frames.size:
            d = abs(int(frames[j]) - target)
            if d < best_d:
                best, best_d = j, d
    return best


def nearest_carrier(
    pos: np.ndarray, teams: np.ndarray, tracks: np.ndarray, ball_xy: tuple[float, float]
) -> tuple[int, int, float] | None:
    """Nearest player to the ball among a frame's players (the ball-carrier rule).

    Args:
        pos: ``(n, 2)`` player pitch coords (metres) at one frame.
        teams: ``(n,)`` team ids aligned with ``pos``.
        tracks: ``(n,)`` track ids aligned with ``pos``.
        ball_xy: ball ``(x, y)`` pitch coords (metres).

    Returns:
        ``(team, track_id, distance_m)`` of the nearest player, or ``None`` when no players.
    """
    if pos.shape[0] == 0:
        return None
    d = np.hypot(pos[:, 0] - ball_xy[0], pos[:, 1] - ball_xy[1])
    k = int(np.argmin(d))
    return int(teams[k]), int(tracks[k]), float(d[k])


# === Per-chunk lookups ===========================================================================
def _players_by_frame(dfc: pd.DataFrame) -> tuple[np.ndarray, dict[int, tuple[np.ndarray, ...]]]:
    """Group one chunk's outfield rows by frame.

    Args:
        dfc: aligned rows for a single chunk.

    Returns:
        ``(sorted_frames, {frame: (pos(n,2), teams(n,), tracks(n,))})`` over player/keeper rows.
    """
    out = dfc[dfc["role"].isin(["player", "goalkeeper"])]
    out = out[out["pitch_x"].notna() & out["pitch_y"].notna()]
    by: dict[int, tuple[np.ndarray, ...]] = {}
    for f, g in out.groupby("frame"):
        by[int(f)] = (g[["pitch_x", "pitch_y"]].to_numpy(float),
                      g["team"].to_numpy(int), g["track_id"].to_numpy(int))
    return np.array(sorted(by), dtype=int), by


def _ball_track(match: Match, chunk_key: str, dfc: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Union ball positions for a chunk: Viterbi ``final/ball`` preferred, aligned detections fill gaps.

    The Viterbi track is cleaner but sparser (~78% of events have a position within +-1 s); the raw
    aligned ``role == 'ball'`` detections cover ~95%. Prefer Viterbi where present, fall back to the
    aligned detection -> ~97% coverage.

    Args:
        match: registry match.
        chunk_key: e.g. ``h1_chunk_000``.
        dfc: aligned rows for this chunk (source of the fallback ball detections).

    Returns:
        ``(frames, xy(n,2))`` ascending by frame, NaN-position rows dropped.
    """
    pos: dict[int, tuple[float, float]] = {}
    ball = dfc[dfc["role"] == "ball"]
    for f, x, y in zip(ball["frame"], ball["pitch_x"], ball["pitch_y"]):
        if not (np.isnan(x) or np.isnan(y)):
            pos[int(f)] = (float(x), float(y))
    for ck, path in match.ball_chunks():
        if ck == chunk_key and path.exists():
            b = pd.read_parquet(path)
            for f, x, y in zip(b["frame"], b["x"], b["y"]):
                if not (np.isnan(x) or np.isnan(y)):
                    pos[int(f)] = (float(x), float(y))  # Viterbi overwrites the aligned detection
            break
    if not pos:
        return np.empty(0, dtype=int), np.empty((0, 2))
    frames = np.array(sorted(pos), dtype=int)
    return frames, np.array([pos[f] for f in frames], dtype=float)


def _named_by_track(named: pd.DataFrame, chunk_key: str) -> dict[int, str]:
    """``{track_id: player_name}`` for one chunk (highest ``n_anchors`` wins on collision)."""
    sub = named[named["chunk"] == chunk_key].sort_values("n_anchors")
    return dict(zip(sub["track_id"].astype(int), sub["player_name"]))


# === Attribution =================================================================================
def attribute_chunk(match: Match, chunk_key: str, dfc: pd.DataFrame, actions: dict,
                    named_by: dict[int, str]) -> list[dict]:
    """Attribute every filtered PASS/DRIVE event in one chunk to a team and (maybe) a named player.

    Args:
        match: registry match.
        chunk_key: e.g. ``h1_chunk_000``.
        dfc: aligned rows for this chunk.
        actions: ``{"fps", "actions"}`` BAS blob for this chunk.
        named_by: ``{track_id: name}`` for this chunk (empty for team-only matches).

    Returns:
        One dict per event with team/player attribution and evidence flags.
    """
    pf, by = _players_by_frame(dfc)
    bf, bxy = _ball_track(match, chunk_key, dfc)
    half = chunk_key[:2]
    dur_s = (float(dfc["frame"].max()) / 25.0) if len(dfc) else 0.0
    rows: list[dict] = []
    for cls in CLASSES:
        for frame_index, conf in op_events(actions, cls):
            g = bas_to_grid_frame(frame_index)
            contact = _contact_frame(pf, by, bf, bxy, g)
            row = {
                "half": half, "chunk": chunk_key, "frame_index": int(frame_index),
                "t_s": frame_index / 25.0, "t_chunk_s": round(dur_s, 1), "class": cls,
                "bas_conf": float(conf), "team": pd.NA, "team_name": pd.NA, "player": pd.NA,
                "ball_found": contact is not None, "carrier_dist_m": pd.NA,
                "n_players": 0 if contact is None else contact["n_players"], "player_dist_m": pd.NA,
            }
            if contact is not None and contact["dist"] <= TEAM_MAX_M:
                team = contact["team"]
                row["team"] = int(team)
                row["team_name"] = match.teams[int(team)]
                row["carrier_dist_m"] = round(contact["dist"], 2)
                if named_by:  # nearest NAMED track to the ball at the contact frame, tight radius
                    pos, _, tracks = by[contact["frame"]]
                    mask = np.array([int(t) in named_by for t in tracks])
                    if mask.any():
                        nd = np.hypot(pos[mask, 0] - contact["ball"][0],
                                      pos[mask, 1] - contact["ball"][1])
                        kk = int(np.argmin(nd))
                        if nd[kk] <= PLAYER_MAX_M:
                            row["player"] = named_by[int(tracks[mask][kk])]
                            row["player_dist_m"] = round(float(nd[kk]), 2)
            rows.append(row)
    return rows


def _contact_frame(pf: np.ndarray, by: dict, bf: np.ndarray, bxy: np.ndarray,
                   g: int) -> dict | None:
    """Scan :data:`CONTACT_OFFSETS` around grid frame ``g`` for the moment a player is on the ball.

    Args:
        pf: sorted player frames for the chunk.
        by: ``{frame: (pos, teams, tracks)}`` player lookup.
        bf: sorted ball frames.
        bxy: ball ``(n, 2)`` positions aligned with ``bf``.
        g: event grid frame.

    Returns:
        The window sample minimising nearest-player-to-ball distance as
        ``{"frame", "ball", "team", "dist", "n_players"}``, or ``None`` if no ball+player sample.
    """
    best: dict | None = None
    for off in CONTACT_OFFSETS:
        gf = g + off
        pj = _nearest_row(pf, gf, STEP)
        bi = _nearest_row(bf, gf, STEP)
        if pj is None or bi is None:
            continue
        frame = int(pf[pj])
        pos, teams, tracks = by[frame]
        ball = (float(bxy[bi, 0]), float(bxy[bi, 1]))
        carrier = nearest_carrier(pos, teams, tracks, ball)
        if carrier is None:
            continue
        team, _track, dist = carrier
        if best is None or dist < best["dist"]:
            best = {"frame": frame, "ball": ball, "team": team, "dist": dist,
                    "n_players": int(pos.shape[0])}
    return best


def build_ledger(match: Match) -> pd.DataFrame:
    """Full attributed event ledger for one match (all chunks, PASS + DRIVE)."""
    actions = load_actions(match.id)
    df = match.load_aligned()
    named = pd.DataFrame(columns=["chunk", "track_id", "player_name", "n_anchors"])
    if match.id in PLAYER_TRUTH:
        named = pd.read_parquet(PLAYER_TRUTH[match.id][1])
    rows: list[dict] = []
    for chunk_key in sorted(actions):
        dfc = df[df["chunk"] == chunk_key]
        if dfc.empty:
            continue
        rows += attribute_chunk(match, chunk_key, dfc, actions[chunk_key],
                                _named_by_track(named, chunk_key))
    cols = ["half", "chunk", "frame_index", "t_s", "t_chunk_s", "class", "team", "team_name",
            "player", "bas_conf", "ball_found", "carrier_dist_m", "n_players", "player_dist_m"]
    return pd.DataFrame(rows, columns=cols)


# === Validation ==================================================================================
def _team_truth(match: Match) -> dict[int, dict[str, int]] | None:
    """``{team_int: {"h1": att, "h2": att, "match": att, "side": "home"/"away"}}`` or ``None``."""
    path = Path(TEAM_TRUTH.get(match.id, ""))
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    home_int = 0 if match.teams[0] == match.home_team else 1
    out: dict[int, dict[str, int]] = {}
    for team_int in (0, 1):
        col = "homeValue" if team_int == home_int else "awayValue"
        side = "home" if team_int == home_int else "away"
        rec: dict[str, int] = {"side": side}
        tot = 0
        for half, period in (("h1", "1ST"), ("h2", "2ND")):
            r = df[(df["key"] == "passes") & (df["period"] == period)]
            rec[half] = int(r[col].iloc[0]) if not r.empty else 0
            tot += rec[half]
        rec["match"] = tot
        out[team_int] = rec
    return out


def team_gate(ledger: pd.DataFrame, match: Match) -> dict:
    """Per-team PASS split (attributed / abstained) vs Sofascore attempted, per half + match."""
    passes = ledger[ledger["class"] == "PASS"]
    truth = _team_truth(match)
    per_team: dict[int, dict] = {}
    for team_int in (0, 1):
        rec: dict = {"name": match.teams[team_int]}
        for scope, sub in (("h1", passes[passes["half"] == "h1"]),
                           ("h2", passes[passes["half"] == "h2"]), ("match", passes)):
            n = int((sub["team"] == team_int).sum())
            t = truth[team_int][scope] if truth else None
            rec[scope] = {"attr": n, "truth": t,
                          "ratio": round(n / t, 3) if t else None}
            if truth:
                rec[scope]["side"] = truth[team_int]["side"]
        per_team[team_int] = rec
    abstain = int(passes["team"].isna().sum())
    return {"per_team": per_team, "n_pass": len(passes), "abstain": abstain,
            "abstain_rate": round(abstain / len(passes), 3) if len(passes) else None}


def _player_truth(match_id: str) -> pd.DataFrame | None:
    """Sofascore per-player ``name -> (totalPass, touches)`` (whole match, wide-format), or ``None``."""
    path = Path(PLAYER_TRUTH[match_id][0])
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "totalPass" not in df.columns:
        return None
    out = df[["name", "totalPass", "touches"]].copy()
    out = out.rename(columns={"totalPass": "truth_pass", "touches": "truth_touch"})
    out["truth_pass"] = pd.to_numeric(out["truth_pass"], errors="coerce")
    return out.dropna(subset=["truth_pass"])


def player_table(ledger: pd.DataFrame, match: Match) -> dict:
    """Per-named-player attributed PASS counts vs Sofascore ``totalPass`` (named subset only)."""
    passes = ledger[(ledger["class"] == "PASS") & ledger["team"].notna()]
    named = passes[passes["player"].notna()]
    counts = named.groupby("player").size().rename("attr_pass").reset_index()
    truth = _player_truth(match.id)
    merged = counts
    corr = None
    if truth is not None:
        merged = counts.merge(truth, left_on="player", right_on="name", how="left")
        both = merged.dropna(subset=["truth_pass"])
        if len(both) >= 3:
            corr = float(both["attr_pass"].corr(both["truth_pass"], method="spearman"))
    return {
        "rows": merged.sort_values("attr_pass", ascending=False),
        "n_team_pass": len(passes), "n_named_pass": len(named),
        "coverage": round(len(named) / len(passes), 3) if len(passes) else None,
        "spearman": None if corr is None or np.isnan(corr) else round(corr, 3),
    }


# === Reporting ===================================================================================
def _fmt_gate(gate: dict, match: Match) -> list[str]:
    lines = [f"### {match.id} -- team gate", "",
             "| team | side | h1 attr/truth | h2 attr/truth | match attr/truth | ratio |",
             "|------|------|---------------|---------------|------------------|-------|"]
    for team_int in (0, 1):
        r = gate["per_team"][team_int]
        m = r["match"]
        side = m.get("side", "-")
        lines.append(
            f"| {r['name']} | {side} | {r['h1']['attr']}/{r['h1']['truth']} | "
            f"{r['h2']['attr']}/{r['h2']['truth']} | {m['attr']}/{m['truth']} | "
            f"{m['ratio'] if m['ratio'] is not None else '-'} |")
    lines += ["", f"Filtered PASS events: {gate['n_pass']} | team-attributed: "
              f"{gate['n_pass'] - gate['abstain']} | abstained (no on-ball carrier): "
              f"{gate['abstain']} ({(gate['abstain_rate'] or 0) * 100:.1f}%)", ""]
    return lines


def _fmt_players(pt: dict, match: Match) -> list[str]:
    n_players = int(pt["rows"]["player"].nunique())
    lines = [f"### {match.id} -- named-player passes vs oracle", "",
             f"Team-attributed passes: {pt['n_team_pass']} | with a named player: "
             f"{pt['n_named_pass']} (coverage {(pt['coverage'] or 0) * 100:.1f}%) | "
             f"Spearman(attr, totalPass) = {pt['spearman']} over {n_players} named players "
             f"(N too small to be conclusive; counts are a floor)", "",
             "| player | attr pass | oracle totalPass |",
             "|--------|-----------|------------------|"]
    for _, r in pt["rows"].iterrows():
        truth = r.get("truth_pass")
        tv = "-" if truth is None or (isinstance(truth, float) and np.isnan(truth)) else int(truth)
        lines.append(f"| {r['player']} | {int(r['attr_pass'])} | {tv} |")
    lines.append("")
    return lines


def format_report(results: list[dict]) -> str:
    """Render ``results/PLAYER_LEDGER.md`` from per-match gate + player-table dicts."""
    lines = [
        "# Player-action ledger (B-3 stage 2)", "",
        "Team attribution: ball-carrier (nearest tracked player to the Viterbi ball) team at the "
        f"aligned frame nearest each filtered PASS event; abstain if no ball within +-1 s or the "
        f"nearest player is > {TEAM_MAX_M:.0f} m away. Player attribution (identity matches only): "
        f"nearest NAMED track to the ball within {PLAYER_MAX_M:.0f} m. Truth = Sofascore attempted "
        "passes (team) / totalPass (player).", "",
        "## Team gate (all matches)", "",
    ]
    for res in results:
        lines += _fmt_gate(res["gate"], res["match"])
    lines += ["## Player attribution (identity matches)", ""]
    for res in results:
        if res.get("players") is not None:
            lines += _fmt_players(res["players"], res["match"])
    lines += [
        "## Honest limits", "",
        "- Coverage ceiling is tracking, not the method: ~35-42% of filtered passes get an on-ball "
        "carrier within 3 m; the rest abstain because no tracked player is on the ball at the kick "
        "(broadcast detects players in a minority of frames -- the industry regime, cf. the plan "
        "doc). Where a carrier IS found the fit is tight (median ~1.3 m).",
        "- The per-team split holds for manutd_liverpool (attr 181:174 vs truth 507:464), "
        "manutd_fulham (166:144 vs 482:384), and manutd_tottenham (attr Man Utd 120 : Tottenham "
        "218 vs truth 395:636 -- the 0-3 possession loser stays behind by a clear margin), but "
        "INVERTS for brighton_manutd (attr over-weights Brighton 223 vs Man Utd 191, while truth "
        "has Man Utd ahead 511:477). Three holds vs one invert: the nearest-carrier split "
        "preserves the possession winner when the true gap is large (tottenham 395:636), but is "
        "marginal-to-unreliable when the two teams' attempted passes are near-level -- brighton's "
        "truth (511:477, ~7%) is exactly where noise flips the direction. A real ceiling, not "
        "smoothed over.",
        "- Team attribution rests on the nearest-player-to-ball heuristic, not a possession model; a "
        "loose ball between two players attributes to whoever is closest. The gate ratio is the "
        "check that this holds in aggregate.",
        "- Player coverage is bounded by named-fragment coverage (sparse): only passes whose carrier "
        "is a named track within the radius get a player, so counts are a floor, not a total.",
        "- `manutd_fulham` has no identity artifacts -> team-level only.",
        "- The event stream itself is BAS PASS/DRIVE; E2E-Spot goals/shots/cards live in "
        "`results/action_spotting_probe/` and are not merged here (needs a shared 2 fps<->25 fps "
        "half-clock; add when the ledger needs multi-class rows).", "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Build + write every match's ledger and the combined ``results/PLAYER_LEDGER.md``."""
    results: list[dict] = []
    for match in registry.matches():
        if match.id not in TEAM_TRUTH:
            continue
        print(f"attributing {match.id} ...")
        ledger = build_ledger(match)
        out = match.aligned.parent.parent / "ledger.parquet"  # outputs/<match>/ledger.parquet
        ledger.to_parquet(out)
        gate = team_gate(ledger, match)
        players = player_table(ledger, match) if match.id in PLAYER_TRUTH else None
        results.append({"match": match, "gate": gate, "players": players})
        for team_int in (0, 1):
            r = gate["per_team"][team_int]["match"]
            print(f"  {match.teams[team_int]:>10}: {r['attr']} attr / {r['truth']} truth "
                  f"= {r['ratio']}")
        print(f"  abstain {gate['abstain']} ({(gate['abstain_rate'] or 0) * 100:.1f}%) | "
              f"wrote {out}")
        if players is not None:
            print(f"  player coverage {(players['coverage'] or 0) * 100:.1f}% | "
                  f"spearman {players['spearman']}")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(format_report(results), encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
