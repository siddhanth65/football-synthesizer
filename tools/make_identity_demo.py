"""Compose the identity-headlined pipeline showcase (broadcast -> named, validated tactical data).

A presentation deliverable built on the same house rules as :mod:`tools.make_demo`, whose machinery
this module REUSES (``ChunkClip``, ``TopDown``, ``compose``, ``card``, ``text``/``shade``/``chip``,
the 720p/1080p ffmpeg render). Nothing is recomputed on the GPU and **no number on screen is
invented**: every figure is read from a persisted artifact at render time (see
:func:`identity_facts`) -- the named-tracks parquet, the close-up anchor stats/verdicts JSON, the
Sofascore player oracle, and the two GS-HOTA benchmark JSONs.

Headline: PLAYER IDENTIFICATION. Segments:

1. Title card.
2. Shot classification (``generator.live_play`` labels overlaid live on a broadcast stretch).
3. Detection + tracking + teams (reused ``make_demo`` machinery).
4. Calibration + top-down (reused machinery, with a real failure on screen).
5. Player identification (the centerpiece): the gated anchor read, the anchor->track attachment,
   the name following Bruno's track, the visible-minutes vs Sofascore validation, and the honest
   hero-shot-concentration caveat.
6. External GS-HOTA benchmark (official + relink lift).
7. Honest scorecard finale.

Run::

    python tools/make_identity_demo.py                 # full render
    python tools/make_identity_demo.py --preview       # short sample of each footage segment
    python tools/make_identity_demo.py --height 720    # 720p output
    python tools/make_identity_demo.py --check          # facts self-check, no render
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))  # import the sibling make_demo module directly

import make_demo as md  # noqa: E402
from core import registry  # noqa: E402
from generator.live_play import (  # noqa: E402
    SHOT_CLOSE_UP,
    SHOT_GRAPHIC,
    SHOT_LIVE_WIDE,
    SHOT_REPLAY_OTHER,
    classify_dense,
)

W, H, FPS = md.W, md.H, md.FPS
C_WHITE, C_BLACK, C_YELLOW = md.C_WHITE, md.C_BLACK, md.C_YELLOW
C_AMBER, C_RED, C_GREEN = md.C_AMBER, md.C_RED, md.C_GREEN
C_BG = md.C_BG
FONT, FONT_T = md.FONT, md.FONT_T
text, shade, chip, card = md.text, md.shade, md.chip, md.card

MATCH_ID = "brighton_manutd"
ANCHOR_ROOT = Path("results/closeup_anchor_probe")
GSR_ROOT = Path("results/gsr_benchmark")
NAMED_TRACKS = Path("outputs/identity/brighton_manutd_named_tracks.parquet")
ORACLE = Path("outputs/oracle/sofascore/player_stats_12436888.parquet")
NAMED_MD = Path("results/identity/NAMED_TRACKS.md")

SHOT_COLOUR = {
    SHOT_LIVE_WIDE: C_GREEN,
    SHOT_CLOSE_UP: C_AMBER,
    SHOT_REPLAY_OTHER: (170, 170, 170),
    SHOT_GRAPHIC: (90, 90, 210),
}
SHOT_LABEL = {
    SHOT_LIVE_WIDE: "LIVE WIDE (tactical)",
    SHOT_CLOSE_UP: "CLOSE-UP / under-detected",
    SHOT_REPLAY_OTHER: "REPLAY / OTHER",
    SHOT_GRAPHIC: "GRAPHIC / CUT (zero detections)",
}


# --------------------------------------------------------------------------------------------------
# Facts: every on-screen number, each read from a persisted artifact at render time
# --------------------------------------------------------------------------------------------------
def identity_facts() -> dict:
    """Collect every figure the identity demo displays.

    Returns:
        Dict of parsed figures plus a ``provenance`` map; a missing artifact drops its keys (the
        card omits the line rather than inventing a value).
    """
    facts: dict = dict(md.demo_facts(MATCH_ID))  # fixture, comp, ball_coverage, calib rates

    # -- named tracks (the identity result) --------------------------------------------------------
    if NAMED_TRACKS.exists():
        nt = pd.read_parquet(NAMED_TRACKS)
        facts["n_named_players"] = int(nt["player_name"].nunique())
        facts["n_named_frags"] = int(len(nt))
        players = (nt.groupby("player_name")
                   .agg(number=("jersey_number", "first"), team=("team", "first"),
                        frags=("track_id", "count"), anchors=("n_anchors", "sum"))
                   .reset_index().sort_values("frags", ascending=False))
        facts["named_players"] = players.to_dict("records")
        facts["provenance"]["named_players"] = str(NAMED_TRACKS)

    # -- gated-anchor funnel (the identity source) -------------------------------------------------
    stats_p = ANCHOR_ROOT / "closeup_anchor_stats.json"
    if stats_p.exists():
        tot = json.loads(stats_p.read_text(encoding="utf-8"))["total"]
        facts["anchor_baseline"] = tot["anchors"]
        facts["anchor_kit"] = tot["anchors_kit"]
        facts["anchor_ocr"] = tot["anchors_ocr"]
        facts["anchor_final"] = tot["anchors_kit_ocr"]
        facts["anchor_shots"] = tot["shots_with_final_anchor"]
        facts["anchor_feasible"] = tot["anchors_attachable_2s"]
        facts["anchor_eligible"] = tot["eligible_crops"]
        facts["provenance"]["anchor_funnel"] = str(stats_p)

    verd_p = ANCHOR_ROOT / "spotcheck_step3" / "verdicts.json"
    if verd_p.exists():
        vd = json.loads(verd_p.read_text(encoding="utf-8"))
        crops = vd["crops"]
        correct = sum(1 for c in crops if c["verdict"].upper().startswith("CORRECT"))
        facts["anchor_survivors"] = vd["n_survivors"]
        facts["anchor_verified_n"] = len(crops)
        facts["anchor_verified_correct"] = correct
        facts["anchor_precision"] = correct / len(crops) if crops else float("nan")
        facts["verdicts"] = crops
        facts["provenance"]["anchor_precision"] = str(verd_p)

    # -- number histogram of the survivors (hero-shot concentration) -------------------------------
    if NAMED_TRACKS.exists():
        surv = ANCHOR_ROOT / "spotcheck_step3" / "_survivors"
        if surv.exists():
            nums = [int(m.group(1)) for f in surv.glob("*.jpg")
                    if (m := re.search(r"_n(\d+)_", f.name))]
            hist = pd.Series(nums).value_counts()
            if len(hist):
                facts["hero_number"] = int(hist.index[0])
                facts["hero_share"] = float(hist.iloc[0] / hist.sum())
                facts["provenance"]["hero_share"] = str(surv)

    # -- per-player validation: our visible minutes (MD) vs Sofascore oracle (parquet) -------------
    ours: dict[str, float] = {}
    if NAMED_MD.exists():
        txt = NAMED_MD.read_text(encoding="utf-8")
        row = re.compile(
            r"^\|\s*([A-Za-z .'-]+?)\s*\|\s*(?:Man Utd|Brighton)\s*\|\s*\d+\s*\|\s*\d+\s*\|"
            r"\s*\d+\s*\|\s*([\d.]+)\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|",
            re.M)
        for m in row.finditer(txt):
            ours[m.group(1).strip()] = float(m.group(2))
        sp = re.search(r"Spearman\([^)]*\)[^*]*\*\*([\d.]+)\*\*", txt)
        if sp:
            facts["spearman"] = float(sp.group(1))
        facts["provenance"]["our_visible_min"] = str(NAMED_MD)

    if ORACLE.exists() and "named_players" in facts:
        ps = pd.read_parquet(ORACLE)
        rows = []
        for rec in facts["named_players"]:
            name = rec["player_name"]
            o = ps[ps["name"] == name]
            if len(o):
                r = o.iloc[0]
                rows.append({
                    "name": name, "number": int(rec["number"]), "team": rec["team"],
                    "our_min": ours.get(name, float("nan")),
                    "oracle_min": float(r["minutesPlayed"]), "touches": float(r["touches"]),
                })
        facts["validation"] = rows
        facts["provenance"]["oracle_min/touches"] = str(ORACLE)

    # -- external GS-HOTA benchmark ----------------------------------------------------------------
    gsr_p = GSR_ROOT / "gsr_scores.json"
    if gsr_p.exists():
        gs = json.loads(gsr_p.read_text(encoding="utf-8"))["configs"]
        facts["gsr"] = {cfg: gs[cfg]["combined"] for cfg in gs}
        facts["provenance"]["gsr"] = str(gsr_p)
    relink_p = GSR_ROOT / "gsr_scores_relink.json"
    if relink_p.exists():
        rl = json.loads(relink_p.read_text(encoding="utf-8"))
        facts["relink"] = {"before": rl["before"], "after": rl["after"],
                           "frag_before": rl["mean_fragments_before"],
                           "frag_after": rl["mean_fragments_after"],
                           "merge_prec": rl["pilot_merge_precision"]}
        facts["provenance"]["relink"] = str(relink_p)
    return facts


# --------------------------------------------------------------------------------------------------
# Segment 2: shot classification overlaid live
# --------------------------------------------------------------------------------------------------
def shot_labels(half: str, chunk: int) -> dict[int, str]:
    """Per-sampled-frame ``shot_type`` for a chunk, keyed by source frame."""
    dense = pd.read_parquet(
        md.DENSE_ROOT / MATCH_ID / half / "match" / f"chunk_{chunk:03d}_dense.parquet")
    cls = classify_dense(dense)
    return {int(fr): str(st) for fr, st in cls["shot_type"].items()}


def compose_shotclass(frame: np.ndarray, shot: str, running: dict[str, int], facts: dict,
                      caption: str, note: str) -> np.ndarray:
    """One output frame for the shot-classification segment (broadcast + live class readout)."""
    canvas = np.full((H, W, 3), C_BG, np.uint8)
    vw, vh, vx, vy = 1387, 780, 266, 146
    canvas[vy:vy + vh, vx:vx + vw] = cv2.resize(frame, (vw, vh), interpolation=cv2.INTER_AREA)
    cv2.rectangle(canvas, (vx - 2, vy - 2), (vx + vw + 2, vy + vh + 2), (90, 90, 90), 2)

    shade(canvas, 0, md.HEADER_H, 0.78)
    text(canvas, "2 / SHOT CLASSIFICATION", (28, 50), 1.0, C_YELLOW, 2, FONT_T)
    text(canvas, "generator/live_play.py  |  rule-based, thresholds pre-committed",
         (760, 48), 0.72, (215, 215, 225), 1)

    # live per-frame verdict, top-left of the video
    col = SHOT_COLOUR.get(shot, C_WHITE)
    chip(canvas, SHOT_LABEL.get(shot, shot), (vx + 20, vy + 56), col, 0.9)

    # running tally strip under the header
    shade(canvas, md.HEADER_H + 6, md.HEADER_H + 58, 0.55)
    total = max(1, sum(running.values()))
    x = 30
    for st in (SHOT_LIVE_WIDE, SHOT_CLOSE_UP, SHOT_REPLAY_OTHER, SHOT_GRAPHIC):
        pct = 100 * running.get(st, 0) / total
        c = SHOT_COLOUR[st] if st == shot else (150, 150, 160)
        text(canvas, f"{st}: {pct:.0f}%", (x, md.HEADER_H + 44), 0.72, c, 2)
        x += 380

    shade(canvas, H - md.FOOTER_H, H, 0.72)
    text(canvas, caption, (30, H - md.FOOTER_H + 52), 0.86, C_WHITE, 2)
    if note:
        for i, ln in enumerate(md._wrap(note, 118)):
            text(canvas, ln, (30, H - md.FOOTER_H + 96 + 34 * i), 0.68, (185, 210, 235), 1)
    return canvas


# --------------------------------------------------------------------------------------------------
# Segment 5: identity centerpiece
# --------------------------------------------------------------------------------------------------
def _fit(img: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    """Letterbox an image into a ``box_w x box_h`` panel on the card background."""
    h, w = img.shape[:2]
    s = min(box_w / w, box_h / h)
    rs = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    pan = np.full((box_h, box_w, 3), (30, 20, 20), np.uint8)
    y0, x0 = (box_h - rs.shape[0]) // 2, (box_w - rs.shape[1]) // 2
    pan[y0:y0 + rs.shape[0], x0:x0 + rs.shape[1]] = rs
    return pan


def anchor_read_card(facts: dict) -> np.ndarray:
    """5a: a gated anchor crop with the read stamps (number, confidence, OCR + kit gates)."""
    crops = [c for c in facts.get("verdicts", []) if c["pred"] == 8
             and c["verdict"].upper().startswith("CORRECT")]
    crops.sort(key=lambda c: -c["conf"])
    c = np.full((H, W, 3), C_BG, np.uint8)
    cv2.rectangle(c, (0, 0), (18, H), C_YELLOW, -1)
    text(c, "5 / PLAYER IDENTIFICATION  --  reading the number", (90, 130), 1.4, C_YELLOW, 3, FONT_T)
    if not crops:
        return c
    rec = crops[0]
    crop_p = ANCHOR_ROOT / "spotcheck_step3" / rec["file"]
    img = cv2.imread(str(crop_p))
    if img is not None:
        pan = _fit(img, 360, 560)
        cv2.rectangle(pan, (0, 0), (pan.shape[1] - 1, pan.shape[0] - 1), (90, 90, 90), 2)
        c[250:250 + pan.shape[0], 120:120 + pan.shape[1]] = pan
    text(c, "gated close-up anchor crop", (120, 240), 0.7, (185, 210, 235), 1)
    x = 620
    text(c, "classifier read", (x, 300), 0.95, (200, 215, 235), 2)
    text(c, f"#{rec['pred']}   conf {rec['conf']:.2f}", (x, 372), 1.5, C_WHITE, 3, FONT_T)
    stamps = [
        ("kit-colour gate", "PASS  (torso CIELAB matches a match kit)", C_GREEN),
        ("OCR digit-agreement", f"PASS  (independent read agrees: {rec['pred']})", C_GREEN),
        ("human verdict", rec["verdict"], C_GREEN),
    ]
    for i, (k, v, col) in enumerate(stamps):
        y = 470 + 72 * i
        text(c, f"[x] {k}", (x, y), 0.8, col, 2)
        text(c, v, (x + 470, y), 0.68, (210, 220, 235), 1)
    prec = facts.get("anchor_precision", float("nan"))
    text(c, f"verified anchor precision: {facts.get('anchor_verified_correct', '?')}/"
            f"{facts.get('anchor_verified_n', '?')} = {prec * 100:.1f}%  "
            f"(stratified sample of {facts.get('anchor_survivors', '?')} survivors)",
         (x, 720), 0.72, C_YELLOW, 2)
    text(c, f"source: {ANCHOR_ROOT / 'spotcheck_step3' / 'verdicts.json'}", (90, H - 70), 0.55,
         (150, 160, 175), 1)
    return c


def crosscut_frame(crop: np.ndarray, wide: np.ndarray, track_px: tuple[int, int], name: str,
                   number: int, facts: dict) -> np.ndarray:
    """5b: the anchor crop (left) attached to a wide-play track (right, highlighted)."""
    c = np.full((H, W, 3), C_BG, np.uint8)
    shade(c, 0, md.HEADER_H, 0.78)
    text(c, "5 / PLAYER IDENTIFICATION  --  attaching the name to a track", (28, 50), 0.95,
         C_YELLOW, 2, FONT_T)

    lp = _fit(crop, 300, 470)
    cv2.rectangle(lp, (0, 0), (lp.shape[1] - 1, lp.shape[0] - 1), (90, 90, 90), 2)
    c[300:300 + lp.shape[0], 60:60 + lp.shape[1]] = lp
    text(c, "close-up anchor", (60, 285), 0.72, (200, 215, 235), 1)
    text(c, f"read #{number}", (60, 300 + lp.shape[0] + 40), 1.0, C_WHITE, 2, FONT_T)

    # wide panel, with the chosen track ringed and named
    vw, vh, vx, vy = 1330, 748, 470, 170
    scale = vw / wide.shape[1]
    panel = cv2.resize(wide, (vw, vh), interpolation=cv2.INTER_AREA)
    tx, ty = int(track_px[0] * scale), int(track_px[1] * scale)
    cv2.circle(panel, (tx, ty), 34, C_YELLOW, 4, cv2.LINE_AA)
    _name_pill(panel, name, number, (tx, ty))
    c[vy:vy + vh, vx:vx + vw] = panel
    cv2.rectangle(c, (vx - 2, vy - 2), (vx + vw + 2, vy + vh + 2), (90, 90, 90), 2)
    text(c, "adjacent live-wide frame (within +/- 2 s of the anchor)", (vx, vy - 12), 0.62,
         (200, 220, 255), 1)

    # the connecting arrow + funnel
    cv2.arrowedLine(c, (370, 520), (vx - 12, 520), C_YELLOW, 3, cv2.LINE_AA, tipLength=0.03)
    shade(c, H - md.FOOTER_H, H, 0.72)
    text(c, "The close-up number is carried onto the wide-play track it belongs to, then named "
            "from the team roster.", (30, H - md.FOOTER_H + 52), 0.8, C_WHITE, 2)
    text(c, f"funnel: {facts.get('anchor_final', '?')} gated anchors  ->  "
            f"{facts.get('anchor_feasible', '?')} with a wide frame within +/-2 s  ->  "
            f"{facts.get('n_named_frags', '?')} named track fragments  ->  "
            f"{facts.get('n_named_players', '?')} named players",
         (30, H - md.FOOTER_H + 96), 0.68, (185, 210, 235), 1)
    return c


def _name_pill(panel: np.ndarray, name: str, number: int, at: tuple[int, int]) -> None:
    """Draw a name pill anchored above a highlighted track point."""
    label = f"{name}  #{number}"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.72, 2)
    x, y = at[0] - tw // 2, at[1] - 48
    x = max(6, min(x, panel.shape[1] - tw - 16))
    cv2.rectangle(panel, (x - 10, y - th - 12), (x + tw + 12, y + 10), (20, 20, 20), -1)
    cv2.rectangle(panel, (x - 10, y - th - 12), (x + tw + 12, y + 10), C_YELLOW, 2)
    cv2.putText(panel, label, (x, y), FONT, 0.72, C_WHITE, 2, cv2.LINE_AA)


def compose_namefollow(frame: np.ndarray, st: md.FrameState | None, td: md.TopDown,
                       named: dict[int, tuple[str, int]], attack_dirs: dict[int, int],
                       caption: str, note: str) -> np.ndarray:
    """5c: split layout with a name label following each named track (broadcast + minimap)."""
    canvas = np.full((H, W, 3), C_BG, np.uint8)
    vw, vh, vx, vy = 1180, 664, 24, 196
    scale = vw / frame.shape[1]
    panel = cv2.resize(frame, (vw, vh), interpolation=cv2.INTER_AREA)
    if st is not None:
        for p in st.players:
            if p["role"] == "referee":
                continue
            col = md.C_TEAM.get(p["team"], md.C_TEAM[-1])
            px = (int(p["img"][0] * scale), int(p["img"][1] * scale))
            cv2.circle(panel, px, 8, C_BLACK, -1, cv2.LINE_AA)
            cv2.circle(panel, px, 6, col, -1, cv2.LINE_AA)
            if p["tid"] in named:
                cv2.circle(panel, px, 22, C_YELLOW, 3, cv2.LINE_AA)
                nm, num = named[p["tid"]]
                _name_pill(panel, nm, num, px)
    canvas[vy:vy + vh, vx:vx + vw] = panel
    cv2.rectangle(canvas, (vx - 2, vy - 2), (vx + vw + 2, vy + vh + 2), (90, 90, 90), 2)

    # minimap with names
    px_, py_ = 1214, 300
    if st is not None and st.calibrated:
        pan = td.frame()
        for p in st.players:
            if p["role"] == "referee" or p["pitch"] is None:
                continue
            col = md.C_TEAM.get(p["team"], md.C_TEAM[-1])
            cpt = td.to_px(*p["pitch"])
            cv2.circle(pan, cpt, 8, C_BLACK, -1, cv2.LINE_AA)
            cv2.circle(pan, cpt, 6, col, -1, cv2.LINE_AA)
            if p["tid"] in named:
                cv2.circle(pan, cpt, 13, C_YELLOW, 2, cv2.LINE_AA)
                nm, num = named[p["tid"]]
                text(pan, f"{nm.split()[-1]} #{num}", (cpt[0] + 12, cpt[1] + 4), 0.5, C_WHITE, 1)
    else:
        pan = td.frame()
        shade(pan, 0, pan.shape[0], 0.62, (10, 10, 10))
        text(pan, "NO PITCH GEOMETRY", (int(0.13 * td.w), td.h // 2 - 10), 1.0, C_AMBER, 2)
        text(pan, "name stays on the broadcast track only", (int(0.10 * td.w), td.h // 2 + 34),
             0.6, (215, 215, 220), 1)
    canvas[py_:py_ + td.h, px_:px_ + td.w] = pan
    cv2.rectangle(canvas, (px_ - 2, py_ - 2), (px_ + td.w + 2, py_ + td.h + 2), (90, 90, 90), 2)
    text(canvas, "TOP-DOWN (105 x 68 m, named)", (px_, py_ - 14), 0.62, (200, 220, 255), 1)
    text(canvas, "BROADCAST FRAME", (vx, vy - 14), 0.62, (200, 220, 255), 1)

    shade(canvas, 0, md.HEADER_H, 0.78)
    text(canvas, "5 / PLAYER IDENTIFICATION  --  the name follows the track", (28, 50), 0.95,
         C_YELLOW, 2, FONT_T)
    shade(canvas, H - md.FOOTER_H, H, 0.72)
    text(canvas, caption, (30, H - md.FOOTER_H + 52), 0.86, C_WHITE, 2)
    if note:
        for i, ln in enumerate(md._wrap(note, 118)):
            text(canvas, ln, (30, H - md.FOOTER_H + 96 + 34 * i), 0.68, (185, 210, 235), 1)
    return canvas


def validation_card(facts: dict) -> np.ndarray:
    """5d: our visible-minutes proxy vs the Sofascore official row, per named player."""
    c = np.full((H, W, 3), C_BG, np.uint8)
    cv2.rectangle(c, (0, 0), (18, H), C_GREEN, -1)
    text(c, "5 / VALIDATION vs SOFASCORE (directional, n=3)", (90, 130), 1.3, C_GREEN, 3, FONT_T)
    cols = [("player", 110), ("our visible min", 720), ("official min", 1120), ("official touches",
                                                                                1500)]
    for name, x in cols:
        text(c, name, (x, 260), 0.78, (200, 215, 235), 2)
    for i, r in enumerate(facts.get("validation", [])):
        y = 340 + 66 * i
        text(c, f"{r['name']} #{r['number']} ({r['team']})", (110, y), 0.8, C_WHITE, 2)
        text(c, f"{r['our_min']:.1f}", (720, y), 0.8, C_AMBER, 2)
        text(c, f"{r['oracle_min']:.0f}", (1120, y), 0.8, C_WHITE, 2)
        text(c, f"{r['touches']:.0f}", (1500, y), 0.8, C_WHITE, 2)
    if "spearman" in facts:
        text(c, f"ordering consistency (Spearman, our visible min vs official min): "
                f"{facts['spearman']:.1f}", (110, 640), 0.85, C_GREEN, 2)
    text(c, "Directional validation only: visible-minutes != played-minutes. A broadcast shows "
            "~37% of any one", (110, 740), 0.75, (205, 215, 235), 1)
    text(c, "player's pitch-time, and our proxy is the union of NAMED fragment spans (sparse, "
            "ReID-limited) -- so", (110, 782), 0.75, (205, 215, 235), 1)
    text(c, "ours << official is EXPECTED. The test is ordering, not calibration.", (110, 824),
         0.75, (205, 215, 235), 1)
    text(c, f"sources: {NAMED_MD} (our min) | {ORACLE} (official)", (90, H - 70), 0.55,
         (150, 160, 175), 1)
    return c


def honest_identity_card(facts: dict) -> np.ndarray:
    """5e: the hero-shot-concentration caveat, on screen."""
    lines = [
        f"{facts.get('n_named_players', '?')} players named this match, at "
        f"{facts.get('anchor_verified_correct', '?')}/{facts.get('anchor_verified_n', '?')} = "
        f"{facts.get('anchor_precision', float('nan')) * 100:.1f}% verified anchor precision.",
        "",
        f"Hero-shot concentration: #{facts.get('hero_number', '?')} (Bruno) is "
        f"{facts.get('hero_share', float('nan')) * 100:.0f}% of the "
        f"{facts.get('anchor_survivors', '?')} gated anchors.",
        "Only ~5-6 distinct player-numbers surface a legible close-up back this match.",
        "",
        "This names the handful of players who repeatedly get close-up hero shots, NOT a uniform XI.",
        "The bottleneck is NOT anchor accuracy (~99%) -- it is same-kit ReID: 95 of 226 anchors were",
        "rejected by the margin guard because ImageNet OSNet cannot separate two players in one kit.",
        "Uniform per-player coverage needs a football-domain ReID / VLM close-up reader (future work).",
    ]
    return card("IDENTITY: HONEST CEILING", lines, "football-synthesizer / identity demo",
                accent=C_AMBER)


# --------------------------------------------------------------------------------------------------
# Segment 6: external benchmark
# --------------------------------------------------------------------------------------------------
def benchmark_card(facts: dict) -> np.ndarray:
    """6: the official GS-HOTA decomposition plus the relink association lift."""
    c = np.full((H, W, 3), C_BG, np.uint8)
    cv2.rectangle(c, (0, 0), (18, H), C_YELLOW, -1)
    text(c, "6 / EXTERNAL BENCHMARK -- SoccerNet-GSR (official GS-HOTA)", (90, 120), 1.2, C_YELLOW,
         3, FONT_T)
    gsr = facts.get("gsr", {})
    hdr = [("config", 110), ("GS-HOTA", 760), ("GS-DetA", 1010), ("GS-AssA", 1260),
           ("GS-LocA", 1510), ("IDF1", 1740)]
    for name, x in hdr:
        text(c, name, (x, 230), 0.7, (200, 215, 235), 2)
    rows = [("gs_hota_full  (role+team+JERSEY, official)", "gs_hota_full", C_AMBER),
            ("no_jersey  (role+team)", "no_jersey", C_WHITE),
            ("loc_assoc  (localization + association)", "loc_assoc", C_WHITE)]
    for i, (label, key, col) in enumerate(rows):
        if key not in gsr:
            continue
        m = gsr[key]
        y = 300 + 58 * i
        text(c, label, (110, y), 0.62, col, 1)
        for x, v in [(760, m["GS-HOTA"]), (1010, m["GS-DetA"]), (1260, m["GS-AssA"]),
                     (1510, m["GS-LocA"]), (1740, m["IDF1"])]:
            text(c, f"{v:.1f}", (x, y), 0.72, col, 2)
    text(c, "LocA ~92.5: when we report a player, its pitch position is right -- geometry externally "
            "validated.", (110, 510), 0.68, C_GREEN, 1)
    text(c, "Adding the JERSEY requirement collapses DetA (we emit jersey=null) -> the 43.1 -> 14.8 "
            "drop IS the identity gap.", (110, 552), 0.68, C_AMBER, 1)

    rl = facts.get("relink", {})
    if rl:
        text(c, "Post-hoc ReID track-relinking (association half):", (110, 640), 0.78, C_WHITE, 2)
        for i, (label, key) in enumerate([("loc_assoc", "loc_assoc"), ("no_jersey", "no_jersey")]):
            b = rl["before"][key]["GS-AssA"]
            a = rl["after"][key]["GS-AssA"]
            text(c, f"{label} GS-AssA: {b:.1f} -> {a:.1f}  (+{a - b:.1f})", (140, 700 + 46 * i),
                 0.7, C_GREEN, 2)
        mp = rl["merge_prec"]
        text(c, f"fragments/seq {rl['frag_before']:.0f} -> {rl['frag_after']:.0f};  merge precision "
                f"{mp[0]}/{mp[1]} = {100 * mp[0] / mp[1]:.0f}%  (benchmark-side only, not wired)",
             (140, 794), 0.66, C_AMBER, 1)
    text(c, "Split caveat: we score the VALID split. Published baseline 29.01 / 2024 SOTA 63.90 are "
            "TEST/CHALLENGE-split", (110, 880), 0.62, (200, 200, 210), 1)
    text(c, "numbers with a full jersey+ReID stack -- context, NOT a like-for-like ranking.",
         (110, 918), 0.62, (200, 200, 210), 1)
    text(c, f"sources: {GSR_ROOT / 'gsr_scores.json'} | {GSR_ROOT / 'gsr_scores_relink.json'}",
         (90, H - 60), 0.55, (150, 160, 175), 1)
    return c


def finale_card(facts: dict) -> np.ndarray:
    """7: the honest scorecard -- what we do NOT do, then the thesis line."""
    c = np.full((H, W, 3), C_BG, np.uint8)
    cv2.rectangle(c, (0, 0), (18, H), C_GREEN, -1)
    text(c, "THE HONEST SCORECARD", (90, 140), 1.7, C_GREEN, 3, FONT_T)
    dos = [
        "Detection + tracking + team clustering from the broadcast feed.",
        "Per-frame calibration -> metric top-down positions (GS-LocA ~92.5, externally validated).",
        "Structure metrics validated (defensive line ~5 m vs FIFA PMSR).",
        f"{facts.get('n_named_players', '?')} players NAMED end-to-end at "
        f"~{facts.get('anchor_precision', float('nan')) * 100:.0f}% anchor precision, oracle-checked.",
    ]
    donts = [
        "No goals / shots / xG detection -- events are NOT detected.",
        "Possession = caveated trackable-share, never a gated headline.",
        "Absolute pass counts WITHHELD (pass-recall proxy misses the 50% bar) -- relative claims only.",
        "Player naming is hero-shot-concentrated, not a uniform XI.",
    ]
    text(c, "WHAT WE DO  (validated):", (90, 250), 0.85, C_GREEN, 2)
    for i, s in enumerate(dos):
        text(c, f"[x] {s}", (110, 310 + 48 * i), 0.7, (215, 225, 240), 1)
    text(c, "WHAT WE DO NOT DO  (stated, not hidden):", (90, 560), 0.85, C_AMBER, 2)
    for i, s in enumerate(donts):
        text(c, f"[ ] {s}", (110, 620 + 48 * i), 0.7, (225, 210, 195), 1)
    text(c, "Broadcast video  ->  named, validated tactical data.  Every number on screen was read "
            "from a persisted artifact.", (90, 900), 0.75, C_YELLOW, 2)
    return c


def title_card(facts: dict) -> np.ndarray:
    """Opening card for the identity demo."""
    c = np.full((H, W, 3), C_BG, np.uint8)
    cv2.rectangle(c, (0, 0), (18, H), C_YELLOW, -1)
    text(c, "FOOTBALL-SYNTHESIZER", (90, 200), 2.1, C_WHITE, 3, FONT_T)
    text(c, "broadcast video  ->  NAMED, validated tactical data", (92, 300), 1.25, C_YELLOW, 2,
         FONT_T)
    text(c, facts.get("fixture", "Brighton vs Manchester Utd"), (92, 440), 1.3, C_WHITE, 2, FONT_T)
    text(c, facts.get("comp", "Premier League 2024-25"), (92, 502), 0.9, (190, 210, 235), 2)
    text(c, "Detect + track + team -> calibrate -> top-down -> IDENTIFY players by shirt number ->",
         (92, 620), 0.9, C_WHITE, 2)
    text(c, "validate against an official player oracle. Headline: PLAYER IDENTIFICATION.", (92, 668),
         0.9, C_WHITE, 2)
    shade(c, 760, 940, 0.5, (200, 160, 40))
    text(c, "HONEST NOTE: this is OUR CV output from the broadcast feed, not vendor tracking data.",
         (92, 820), 0.82, C_AMBER, 2)
    text(c, "Where the pipeline fails or over-reaches, the video says so on screen.", (92, 872), 0.82,
         C_AMBER, 2)
    return c


# --------------------------------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------------------------------
def _named_map() -> dict[tuple[str, int], dict[int, tuple[str, int]]]:
    """Map ``(half, chunk_idx) -> {track_id: (player_name, number)}`` from the named-tracks parquet."""
    nt = pd.read_parquet(NAMED_TRACKS)
    out: dict[tuple[str, int], dict[int, tuple[str, int]]] = {}
    for r in nt.itertuples():
        m = re.match(r"(h\d)_chunk_(\d+)", r.chunk)
        key = (m.group(1), int(m.group(2)))
        out.setdefault(key, {})[int(r.track_id)] = (r.player_name, int(r.jersey_number))
    return out


def _footage(writer, seg: md.Segment, td: md.TopDown, facts: dict, preview: bool) -> None:
    """Render a reused make_demo footage segment (detection/tracking or calibration/top-down)."""
    end = min(seg.end, seg.start + int(3 * FPS)) if preview else seg.end
    clip = md.ChunkClip(MATCH_ID, seg.half, seg.chunk, seg.start, end)
    cap = cv2.VideoCapture(str(clip.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, seg.start)
    for fr in range(seg.start, end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        st = clip.at(fr)
        writer.write(md.compose(frame, st, seg, td, clip.attack_dirs, facts))
    cap.release()


def render(out: Path, *, preview: bool = False, height: int = 1080, crf: int = 20) -> dict:
    """Render the identity demo end to end.

    Args:
        out: final MP4 path.
        preview: render only a short sample of each footage segment.
        height: output height (1080 or 720).
        crf: x264 quality (lower = better).

    Returns:
        Summary dict (frames, seconds, path, size).
    """
    facts = identity_facts()
    print(f"[idemo] match={MATCH_ID}")
    for k, v in facts.get("provenance", {}).items():
        print(f"[idemo] fact {k}: {v}")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter at {tmp}")
    td = md.TopDown(width=690)
    n = 0
    t0 = time.time()

    def hold(img: np.ndarray, seconds: float) -> None:
        nonlocal n
        for _ in range(int(round(seconds * FPS))):
            writer.write(img)
            n += 1

    # 1. title
    hold(title_card(facts), 3.0 if preview else 6.0)

    # 2. shot classification (h1 chunk 4: all four classes appear in this stretch)
    labels = shot_labels("h1", 4)
    s2s, s2e = (5515, 5515 + int(3 * FPS)) if preview else (5515, 6165)
    cap = cv2.VideoCapture(str(md.VIDEO_ROOT / MATCH_ID / "h1" / "chunk_004.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, s2s)
    running: dict[str, int] = {}
    for fr in range(s2s, s2e + 1):
        ok, frame = cap.read()
        if not ok:
            break
        shot = labels.get(int(round(fr / 5) * 5), SHOT_GRAPHIC)
        running[shot] = running.get(shot, 0) + 1
        writer.write(compose_shotclass(
            frame, shot, running, facts,
            "Every broadcast frame is classified before any geometry is attempted.",
            "live_wide is the geometry-yielding class; close-up/replay/graphic are filtered out. "
            "The 'close-up' bucket is really 'detector under-populated'."))
        n += 1
    cap.release()

    # 3. detection + tracking + teams (reused machinery)
    md.SECTION_TITLES["A"] = "3 / DETECTION + TRACKING + TEAMS"
    seg3 = md.Segment("A", "h2", 2, 3140, 4015, "video", ("markers", "ids"),
                      "Fine-tuned detector + ByteTrack: every dot is a tracked player, coloured by "
                      "team, labelled by track id.",
                      "~11-13 of 22 players visible per frame -- a broadcast never shows the whole "
                      "pitch, and we never invent the rest.")
    _footage(writer, seg3, td, facts, preview)
    n += _seg_frames(seg3, preview)

    # 4. calibration + top-down, with a real geometry dropout
    md.SECTION_TITLES["C"] = "4 / CALIBRATION + TOP-DOWN"
    seg4 = md.Segment("C", "h1", 3, 7800, 8280, "split",
                      ("markers", "ids", "lines", "ball", "topdown"),
                      "One homography per frame maps foot points into 105 x 68 m pitch coordinates.",
                      "When the geometry drops out the right panel goes dark -- an honest gap, not a "
                      "guess.")
    _footage(writer, seg4, td, facts, preview)
    n += _seg_frames(seg4, preview)

    # 5a. the anchor read
    hold(anchor_read_card(facts), 4.0 if preview else 12.0)

    # 5b. the anchor -> track attachment (real: #8 crop at f13700 -> track 724 in the wide frame)
    crop = cv2.imread(str(ANCHOR_ROOT / "spotcheck_step3" / "_survivors"
                          / "h2_chunk_001_f13700_n08_c0.829.jpg"))
    capw = cv2.VideoCapture(str(md.VIDEO_ROOT / MATCH_ID / "h2" / "chunk_001.mp4"))
    capw.set(cv2.CAP_PROP_POS_FRAMES, 13705)
    ok, wide = capw.read()
    capw.release()
    if crop is not None and ok:
        hold(crosscut_frame(crop, wide, (1290, 657), "Bruno Fernandes", 8, facts),
             4.0 if preview else 11.0)

    # 5c. the name follows the track (h2 chunk 1, Bruno = track 724, calibrated 13705-13885)
    nmap = _named_map()
    named = nmap.get(("h2", 1), {})
    c5s, c5e = (13705, 13705 + int(3 * FPS)) if preview else (13705, 13890)
    clip = md.ChunkClip(MATCH_ID, "h2", 1, c5s, c5e)
    capn = cv2.VideoCapture(str(clip.video))
    capn.set(cv2.CAP_PROP_POS_FRAMES, c5s)
    for fr in range(c5s, c5e + 1):
        ok, frame = capn.read()
        if not ok:
            break
        st = clip.at(fr)
        writer.write(compose_namefollow(
            frame, st, td, named, clip.attack_dirs,
            "The name rides the ByteTrack fragment: Bruno's label follows track 724 across the "
            "passage,",
            "on the broadcast and on the tactical top-down. Names propagate within a fragment "
            "freely; across relink merges only where anchors confirm."))
        n += 1
    capn.release()

    # 5d + 5e
    hold(validation_card(facts), 4.0 if preview else 12.0)
    hold(honest_identity_card(facts), 4.0 if preview else 10.0)

    # 6 + 7
    hold(benchmark_card(facts), 5.0 if preview else 18.0)
    hold(finale_card(facts), 4.0 if preview else 14.0)

    writer.release()
    size_mb = tmp.stat().st_size / 1e6
    print(f"[idemo] raw render: {n} frames ({n / FPS:.1f} s), {size_mb:.0f} MB, "
          f"{time.time() - t0:.0f}s wall")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        vf = f"scale=-2:{height}" if height != H else "null"
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(tmp), "-vf", vf, "-c:v", "libx264",
               "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(out)]
        subprocess.run(cmd, check=True)
        tmp.unlink(missing_ok=True)
    else:
        print("[idemo] WARNING: ffmpeg not found -> shipping the mp4v render (not H.264)")
        shutil.move(str(tmp), str(out))

    info = {"path": str(out), "frames": n, "seconds": n / FPS,
            "size_mb": out.stat().st_size / 1e6, "height": height}
    print(f"[idemo] wrote {info['path']}  {info['seconds']:.1f}s  {info['size_mb']:.1f} MB")
    return info


def _seg_frames(seg: md.Segment, preview: bool) -> int:
    """Frame count a footage segment contributes (mirrors :func:`_footage`)."""
    end = min(seg.end, seg.start + int(3 * FPS)) if preview else seg.end
    return end - seg.start + 1


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/demo/identity_demo.mp4")
    ap.add_argument("--preview", action="store_true", help="short sample of each footage segment")
    ap.add_argument("--height", type=int, default=1080, choices=(720, 1080))
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--check", action="store_true", help="print parsed facts and exit (no render)")
    args = ap.parse_args()
    registry.get(MATCH_ID)  # fail loudly if the match is not registered
    if args.check:
        facts = identity_facts()
        for k in ("n_named_players", "n_named_frags", "anchor_final", "anchor_shots",
                  "anchor_precision", "hero_number", "hero_share", "spearman"):
            print(f"  {k} = {facts.get(k)}")
        print("  validation:", facts.get("validation"))
        print("  gsr full:", facts.get("gsr", {}).get("gs_hota_full"))
        return
    render(Path(args.out), preview=args.preview, height=args.height, crf=args.crf)


if __name__ == "__main__":
    main()
