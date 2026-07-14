"""Per-match grounded FACT STORE: every computed metric, with provenance, in one JSON.

The P0 audit fix for "the pundit report consumes 4 numbers": this module runs the WHOLE metric
inventory (tendencies, phases, transitions/counter-press, passes/PPDA, ball-xT, set pieces, synchrony,
style distance, space control, formation) over a registered match and writes
``outputs/facts/<match>.json`` — CV metrics + the FIFA PMSR ground truth side by side, stamped with
``metrics_version``. The pundit/narration layer reads THIS file; nothing narrates un-stored numbers,
which is what makes the reports auditable (docs/PROJECT_AUDIT_2026-07.md sections 3.2/6).

Ball-dependent metrics aggregate per chunk (frame numbers restart per chunk, so possession chains are
built within a chunk and summed across them); position-only metrics run over the whole aligned table.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.pitch import METRICS_VERSION
from core.registry import Match, get, matches
from fingerprint.phase_metrics import ball_phase_by_frame
from fingerprint.possession_metrics import PASS_MAX_GAP_S, extract_passes, ppda
from fingerprint.roles import assign_roles, infer_formation, track_means
from fingerprint.set_pieces import detect_set_pieces, set_piece_summary
from fingerprint.structural_metrics import (
    resolve_attack_directions,
    resolve_attack_directions_from_ball,
)
from fingerprint.style_metrics import style_distance, team_synchrony
from fingerprint.theory_metrics import (
    REGAIN_WINDOWS_S,
    complete_directions,
    counterpress_curve,
    lane_occupation,
    line_breaks,
    local_overload,
    pressing_intensity,
    verticality,
)
from fingerprint.transitions import transition_metrics
from fingerprint.xt import ball_xt_timeline
from generator.ball import assign_possession
from generator.impute import line_estimates
from synthesizer.predict import predict_tendencies

FACTS_DIR = Path("outputs/facts")
CALIB_MAX_M = 1.0   # per-frame calibration gate, same as the C6 tools
COUNTERPRESS_WINDOW_S = 5.0   # StatsBomb open-play counter-press window (transitions recovery window)
IN_PH = ("build_up", "progression", "final_third")
OUT_PH = ("high_press", "mid_block", "low_block")


def _clean(o):
    """JSON-safe: numpy scalars -> python, NaN/inf -> None, recursively."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    return o


def _ball_facts(m: Match, aligned: pd.DataFrame) -> dict:
    """Aggregate every ball-dependent metric family over the match's linked-ball chunks."""
    phase_counts = {t: {p: 0 for p in IN_PH + OUT_PH} for t in (0, 1)}
    trans = {t: {"won": 0, "lost": 0, "pressed": 0, "rec": [], "cxt": [], "high": 0} for t in (0, 1)}
    passes = {t: {"n": 0, "len": 0.0} for t in (0, 1)}
    press = {t: {"passes_allowed": 0, "pressures": 0} for t in (0, 1)}
    xt = {t: {"xt_created": 0.0, "n_moves": 0} for t in (0, 1)}
    set_pieces: dict[str, int] = {}
    # P1 theory-metric accumulators (weighted so per-chunk reads combine into one match number)
    th_press = {t: [0.0, 0] for t in (0, 1)}          # [Σ intensity·frames, Σ frames]
    th_lb = {t: [0, 0] for t in (0, 1)}               # [breaks, in-poss frames]
    th_ov = {t: [0.0, 0.0, 0] for t in (0, 1)}        # [Σ mean_ov·f, Σ share·f, f]
    th_vert = {t: [0.0, 0] for t in (0, 1)}           # [Σ verticality·spells, spells]
    th_lane: dict[int, dict[str, list[float]]] = {0: {}, 1: {}}   # field -> [Σ share·w, w]
    th_cp = {t: {"losses": 0, **{w: 0 for w in REGAIN_WINDOWS_S}} for t in (0, 1)}
    n_chunks = 0
    for ck, path in m.ball_chunks():
        ball = pd.read_parquet(path)
        fps = m.chunk_fps(ck)
        pos = aligned[(aligned["chunk"] == ck) & (aligned["calib_error_m"] <= CALIB_MAX_M)].dropna(
            subset=["pitch_x"])
        if pos.empty or ball.empty:
            continue
        dirs = resolve_attack_directions_from_ball(pos, ball)
        if len(dirs) < 2:
            dirs = resolve_attack_directions(pos)
        dirs = complete_directions(dirs)  # opposite-goal constraint for the single-goalmouth view
        if len(dirs) < 2:
            continue
        n_chunks += 1
        poss = assign_possession(ball, pos, smooth=True)
        for r in ball_phase_by_frame(ball, poss, pos).itertuples(index=False):
            if int(r.team) in phase_counts and r.phase in phase_counts[int(r.team)]:
                phase_counts[int(r.team)][r.phase] += 1
        window_frames = max(1, round(COUNTERPRESS_WINDOW_S * fps))  # 5 s in this chunk's native frames
        for r in transition_metrics(poss, pos, ball, dirs,
                                    window_frames=window_frames).itertuples(index=False):
            t = int(r.team)
            if t not in trans:
                continue
            trans[t]["won"] += int(r.won)
            trans[t]["lost"] += int(r.lost)
            if np.isfinite(r.counterpress_rate):
                trans[t]["pressed"] += int(round(r.counterpress_rate * r.lost))
            if np.isfinite(r.mean_recovery_frames):
                # store recovery in SECONDS (native frames / this chunk's true fps) so the cross-chunk
                # weighted mean is fps-consistent across a mixed-fps corpus (25 vs 59.94 Hz)
                trans[t]["rec"].append((float(r.mean_recovery_frames) / fps, int(r.lost)))
            if np.isfinite(r.counter_xt):
                trans[t]["cxt"].append((float(r.counter_xt), int(r.won)))
            trans[t]["high"] += int(r.high_regains)
        pdf = extract_passes(poss, pos, max_gap_s=PASS_MAX_GAP_S, fps=fps)
        for t, g in pdf.groupby("team"):
            if int(t) in passes:
                passes[int(t)]["n"] += len(g)
                passes[int(t)]["len"] += float(g["length_m"].sum())
        for r in ppda(poss, pos, directions=dirs).itertuples(index=False):
            if int(r.team) in press:
                press[int(r.team)]["passes_allowed"] += int(r.passes_allowed)
                press[int(r.team)]["pressures"] += int(r.pressures)
        for r in ball_xt_timeline(ball, poss, dirs).itertuples(index=False):
            if int(r.team) in xt:
                xt[int(r.team)]["xt_created"] += float(r.xt_created)
                xt[int(r.team)]["n_moves"] += int(r.n_moves)
        for k, v in set_piece_summary(detect_set_pieces(ball, fps=fps)).items():
            set_pieces[k] = set_pieces.get(k, 0) + int(v)
        # --- P1 theory metrics (ball-anchored) ---
        for r in pressing_intensity(ball, poss, pos, fps=fps).itertuples(index=False):
            if int(r.team) in th_press:
                th_press[int(r.team)][0] += r.pressing_intensity * r.press_frames
                th_press[int(r.team)][1] += r.press_frames
        for r in line_breaks(ball, poss, pos, dirs).itertuples(index=False):
            if int(r.team) in th_lb:
                th_lb[int(r.team)][0] += int(r.line_breaks)
                th_lb[int(r.team)][1] += int(r.in_poss_frames)
        for r in local_overload(ball, poss, pos).itertuples(index=False):
            if int(r.team) in th_ov:
                th_ov[int(r.team)][0] += r.mean_overload * r.frames
                th_ov[int(r.team)][1] += r.overload_share * r.frames
                th_ov[int(r.team)][2] += int(r.frames)
        for r in verticality(ball, poss, dirs).itertuples(index=False):
            if int(r.team) in th_vert and np.isfinite(r.verticality):
                th_vert[int(r.team)][0] += r.verticality * r.spells
                th_vert[int(r.team)][1] += int(r.spells)
        for r in counterpress_curve(ball, poss, pos, fps=fps).itertuples(index=False):
            t = int(r.team)
            if t not in th_cp:
                continue
            th_cp[t]["losses"] += int(r.losses)
            for w in REGAIN_WINDOWS_S:
                th_cp[t][w] += int(round(getattr(r, f"regain_{int(w)}s") * r.losses))
        lo = lane_occupation(pos, dirs)
        for r in lo.itertuples(index=False):
            t = int(r.team)
            if t not in th_lane:
                continue
            w = int((pos["team"] == t).sum())  # weight by this chunk's player-frames
            for f in lo.columns:
                if f == "team":
                    continue
                acc = th_lane[t].setdefault(f, [0.0, 0])
                acc[0] += float(getattr(r, f)) * w
                acc[1] += w

    def wmean(pairs):
        num = sum(v * w for v, w in pairs)
        den = sum(w for _, w in pairs)
        return num / den if den else float("nan")

    out: dict = {"n_ball_chunks": n_chunks, "set_pieces": set_pieces}
    for t in (0, 1):
        pc, tv = phase_counts[t], trans[t]
        in_tot = sum(pc[p] for p in IN_PH) or 1
        out_tot = sum(pc[p] for p in OUT_PH) or 1
        out.setdefault("phases_pct", {})[t] = (
            {p: 100 * pc[p] / in_tot for p in IN_PH} | {p: 100 * pc[p] / out_tot for p in OUT_PH})
        out.setdefault("transitions", {})[t] = {
            "turnovers_won": tv["won"], "turnovers_lost": tv["lost"],
            "counterpress_rate": tv["pressed"] / tv["lost"] if tv["lost"] else float("nan"),
            "mean_recovery_s": wmean(tv["rec"]) if tv["rec"] else float("nan"),  # rec already in seconds
            "counter_xt": wmean(tv["cxt"]) if tv["cxt"] else float("nan"),
            "high_regains": tv["high"]}
        out.setdefault("passing", {})[t] = {
            "n_passes": passes[t]["n"],
            "mean_pass_m": passes[t]["len"] / passes[t]["n"] if passes[t]["n"] else float("nan"),
            "ppda": (press[t]["passes_allowed"] / press[t]["pressures"]
                     if press[t]["pressures"] else float("nan")),
            "pressures": press[t]["pressures"]}
        out.setdefault("ball_xt", {})[t] = xt[t]
        # --- P1 theory metrics (combined across chunks) ---
        pr, lb, ov, ve = th_press[t], th_lb[t], th_ov[t], th_vert[t]
        cp = th_cp[t]
        lane = {f: (a[0] / a[1] if a[1] else float("nan")) for f, a in th_lane[t].items()}
        out.setdefault("theory", {})[t] = {
            "pressing_intensity": pr[0] / pr[1] if pr[1] else float("nan"),
            "press_frames": pr[1],
            "line_breaks": lb[0], "line_break_in_poss_frames": lb[1],
            "mean_overload": ov[0] / ov[2] if ov[2] else float("nan"),
            "overload_share": ov[1] / ov[2] if ov[2] else float("nan"),
            "verticality": ve[0] / ve[1] if ve[1] else float("nan"),
            "halfspace_share": lane.get("halfspace_share", float("nan")),
            "wing_share": lane.get("wing_share", float("nan")),
            "centre_share": lane.get("centre_share", float("nan")),
            "final_third_lane_share": lane.get("final_third_share", float("nan")),
            "counterpress_losses": cp["losses"],
            "counterpress_regain_curve": {
                f"{int(w)}s": (cp[w] / cp["losses"] if cp["losses"] else float("nan"))
                for w in REGAIN_WINDOWS_S},
            "lane_matrix": {f: lane[f] for f in lane
                            if f not in ("halfspace_share", "wing_share", "centre_share",
                                         "final_third_share")},
        }
    return out


def _formations(aligned: pd.DataFrame, roles: pd.DataFrame | None = None) -> dict:
    """Modal formation (+ mean fit cost) per team across chunks (reuses an assign_roles table if given)."""
    if roles is not None and not roles.empty:
        votes: dict[int, list] = {0: [], 1: []}
        for r in roles.drop_duplicates(["chunk", "team"]).itertuples(index=False):
            if int(r.team) in votes:
                votes[int(r.team)].append((r.formation, float(r.fit_cost_m)))
        return _modal_formations(votes)
    votes = {0: [], 1: []}
    groups = aligned.groupby("chunk") if "chunk" in aligned.columns else [(None, aligned)]
    for _, g in groups:
        dirs = resolve_attack_directions(g)
        for t in (0, 1):
            if t not in dirs:
                continue
            name, cost, _ = infer_formation(track_means(g, team=t, attack_dir=dirs[t]))
            if np.isfinite(cost):
                votes[t].append((name, cost))
    return _modal_formations(votes)


def _modal_formations(votes: dict[int, list]) -> dict:
    """Reduce per-chunk (formation, cost) votes to the modal formation per team."""
    out = {}
    for t, v in votes.items():
        if not v:
            continue
        names = [n for n, _ in v]
        modal = max(set(names), key=names.count)
        out[t] = {"formation": modal, "mean_fit_cost_m": float(np.mean([c for n, c in v if n == modal])),
                  "chunks_agreeing": names.count(modal), "chunks_total": len(v)}
    return out


def build_facts(match_id: str) -> dict:
    """Compute the full grounded fact bundle for one registered match."""
    m = get(match_id)
    if not m.processed:
        raise FileNotFoundError(f"{match_id}: aligned parquet missing ({m.aligned})")
    aligned = m.load_aligned()
    name = {t: m.teams[t] for t in (0, 1)}
    # True native fps for the position-only metrics that run over the whole aligned table (velocity is
    # frame_delta/fps; 'frame' is the native frame index). Constant within a match (single broadcast),
    # so one representative rate is exact; falls back to the registry default when videos are absent.
    ck_all = sorted(aligned["chunk"].unique()) if "chunk" in aligned.columns else []
    match_fps = float(np.median([m.chunk_fps(ck) for ck in ck_all])) if ck_all else m.chunk_fps("")

    cv: dict = {"tendencies": {}, "style": {}}
    for t in (0, 1):
        td = predict_tendencies(aligned, t, tier="A")
        cv["tendencies"][name[t]] = td or None
    sync = team_synchrony(aligned, fps=match_fps)
    for r in (sync.itertuples(index=False) if not sync.empty else []):
        if int(r.team) in name:
            cv["style"].setdefault("velocity_synchrony", {})[name[int(r.team)]] = float(
                r.velocity_synchrony)
    cv["style"]["style_distance_m"] = style_distance(aligned)
    try:
        from fingerprint.pitch_control import space_control_metrics  # noqa: PLC0415
        for r in space_control_metrics(aligned, fps=match_fps).itertuples(index=False):
            if int(r.team) in name:
                cv.setdefault("space", {})[name[int(r.team)]] = {
                    "space_control": float(r.space_control),
                    "att_third_control": float(r.att_third_control)}
    except Exception:  # noqa: BLE001 — space control is enrichment, never block the store
        pass
    roles_df = assign_roles(aligned)
    cv["formation"] = {name[t]: v for t, v in _formations(aligned, roles_df).items()}
    # P2 de-biased defensive line (validated vs FIFA per-phase to ~5 m; the biased raw line is retained
    # for reference). See docs/PROJECT_AUDIT_2026-07.md P2 and tools/line_c6.py.
    le = line_estimates(aligned, roles_df)
    for t in (0, 1):
        s = le[le["team"] == t]
        if not s.empty:
            cv.setdefault("line_height", {})[name[t]] = {
                "def_line_debiased_m": round(float(s["line_debiased"].mean()), 1),
                "def_line_raw_m": round(float(s["line_raw"].mean()), 1),
                "n_back_visible_mean": round(float(s["n_back"].mean()), 2)}

    bf = _ball_facts(m, aligned)
    for fam in ("phases_pct", "transitions", "passing", "ball_xt", "theory"):
        if fam in bf:
            cv[fam] = {name[t]: v for t, v in bf[fam].items()}
    cv["set_pieces"] = bf["set_pieces"]
    cv["n_ball_chunks"] = bf["n_ball_chunks"]

    return _clean({
        "match": m.id, "teams": list(m.teams),
        "metrics_version": METRICS_VERSION,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {"cv": str(m.aligned), "fifa": str(m.pmsr) if m.pmsr else None},
        "cv": cv,
        "fifa": m.load_pmsr(),
    })


def write_facts(match_id: str) -> Path:
    """Build + persist the fact store for one match; returns the JSON path."""
    facts = build_facts(match_id)
    FACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = FACTS_DIR / f"{match_id}.json"
    out.write_text(json.dumps(facts, indent=1), encoding="utf-8")
    return out


def load_facts(match_id: str) -> dict | None:
    """Read a previously written fact store (None if absent)."""
    p = FACTS_DIR / f"{match_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", default="all", help="registered match id, or 'all' (processed only)")
    args = ap.parse_args()
    ids = ([m.id for m in matches(processed_only=True)] if args.match == "all" else [args.match])
    for mid in ids:
        print(f"[facts] {mid} ...", flush=True)
        print(f"  wrote {write_facts(mid)}")


if __name__ == "__main__":
    main()
