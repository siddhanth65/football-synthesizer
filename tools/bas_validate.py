"""B-3 stage 1: turn raw ball-action-spotting output into validated pass/drive counts.

The ball-action-spotting (BAS) model fires a PASS/DRIVE peak every time it *sees* a pass -- so a
pass shown live and then in replay is double-counted, and low-confidence noise peaks inflate the
total. Raw brighton_manutd PASS = 1367 vs Sofascore's 988 attempted (1.38x over).

Two candidate corrections are measured here, both CPU-only (no GPU, no video re-decode -- the shot
classifier reads the already-computed aligned parquet):

1. **Replay/close-up filter (pre-committed hypothesis).** Keep a peak only if it lands inside a
   *live-wide* broadcast segment (:mod:`generator.live_play` ``shot_type == live_wide``, run-length
   segments with gap-merge + edge pad). VERDICT: this fails. The shipped classifier separates
   *wide-geometry* from *tight-geometry*, not *live* from *replay* -- a replay shown in a wide
   framing is still ``live_wide``, and a real pass in a medium/tight live shot is dropped. The
   filter over-cuts to ~0.82x (removes real passes) without cleanly removing replays.

2. **Confidence threshold + de-duplication (what actually works).** Drop peaks below a confidence
   floor and merge same-class peaks within ~1 s (the model's own temporal tolerance). A floor
   frozen on brighton h1 alone (0.40) brings the full match to 0.98x and generalises to the held-out
   h2 (0.97x). Confidence thresholding explains ~93% of the over-count; the over-count is spurious
   low-confidence peaks, NOT replay re-showing.

Both corrections are reported per match / per half in ``results/bas_validation.md``. Coverage is
honest: a half's ratio is quoted only when every aligned chunk has a BAS file; partial halves report
counts with a ``PARTIAL`` flag so a re-run picks up new chunks and the ratio appears once complete.

Run (CPU)::

    python -m tools.bas_validate
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import registry
from core.registry import Match
from generator import live_play

# --- FROZEN operating point (confidence + dedup arm) ---------------------------------------------
# OP_THRESHOLD was picked on brighton_manutd h1 ONLY (target = h1 attempted 543): the 1 s-dedup
# count is 557 @ 0.35, 538 @ 0.40, 533 @ 0.42 -> 0.40 minimises |count - truth| (dev 5). Frozen here
# and applied unchanged to every match/half, so h2 and the liverpool halves are genuine hold-outs.
OP_THRESHOLD = 0.40
DEDUP_S = 1.0            # merge same-class peaks within this many seconds (model NMS tolerance)

# --- Replay/close-up filter params (pre-committed hypothesis arm) --------------------------------
LIVE_GAP_MERGE_S = 2.0   # bridge a <=2 s non-wide interruption between two live-wide runs
LIVE_PAD_S = 1.0         # extend each live-wide segment by 1 s each side (coarse sampling slack)

# A half's ratio is quoted once BAS covers >= this fraction of the half's aligned frame span (by
# time, not chunk count -- brighton h2's missing chunk_005 is a ~1.4 min stoppage tail = 97% covered,
# a fair comparison; liverpool h2 at ~39% covered stays PARTIAL with no ratio).
COVERAGE_MIN = 0.90

# match_id -> Sofascore team-stats parquet (per-half `passes`); no field for this in the registry.
TRUTH = {
    "brighton_manutd": Path("outputs/oracle/sofascore/team_stats_12436888.parquet"),
    "manutd_liverpool": Path("outputs/oracle/sofascore/team_stats_12436920.parquet"),
    "manutd_fulham": Path("outputs/oracle/sofascore/team_stats_12436870.parquet"),
    "palace_manutd": Path("outputs/oracle/sofascore/team_stats_12436962.parquet"),
    "manutd_tottenham": Path("outputs/oracle/sofascore/team_stats_12436995.parquet"),
    "southampton_manutd": Path("outputs/oracle/sofascore/team_stats_12436949.parquet"),
}
BALL_ROOT = Path("outputs")               # outputs/<match>/ball_action/<half>/chunk_*_actions.json
LIVEPLAY_CACHE = Path("outputs")          # outputs/<match>/live_play/<chunk>.json
REPORT_PATH = Path("results/bas_validation.md")
HALVES = ("h1", "h2")


# === Pure seams (unit-tested) ====================================================================
def dedup_peaks(frames: list[int], confs: list[float], gap_frames: int) -> list[int]:
    """Suppress peaks within ``gap_frames`` of a kept, higher-confidence peak (greedy NMS).

    Args:
        frames: peak frame indices (one class only).
        confs: matching peak confidences (same order/length as ``frames``).
        gap_frames: suppression radius in frames; ``<= 0`` keeps everything.

    Returns:
        Kept frame indices, ascending.
    """
    if gap_frames <= 0:
        return sorted(frames)
    kept: list[int] = []
    for i in np.argsort(-np.asarray(confs, dtype=float)):
        f = frames[int(i)]
        if all(abs(f - k) > gap_frames for k in kept):
            kept.append(f)
    return sorted(kept)


def build_live_segments(
    grid: np.ndarray, is_live: np.ndarray, gap_merge_frames: int, pad_frames: int
) -> list[tuple[int, int]]:
    """Run-length ``is_live`` into ``(lo, hi)`` segments, merging small gaps and padding edges.

    Args:
        grid: ascending frame indices of the sampled classification grid.
        is_live: boolean array (same length as ``grid``); ``True`` where the frame is live-wide.
        gap_merge_frames: merge two live runs whose frame gap is ``<=`` this (bridges a brief cut).
        pad_frames: widen each merged segment by this many frames on each side.

    Returns:
        Padded, gap-merged inclusive ``(lo, hi)`` frame segments, ascending and non-overlapping.
    """
    runs: list[list[int]] = []
    i, n = 0, len(grid)
    while i < n:
        if is_live[i]:
            j = i
            while j + 1 < n and is_live[j + 1]:
                j += 1
            runs.append([int(grid[i]), int(grid[j])])
            i = j + 1
        else:
            i += 1
    if not runs:
        return []
    merged = [runs[0]]
    for lo, hi in runs[1:]:
        if lo - merged[-1][1] <= gap_merge_frames:
            merged[-1][1] = hi
        else:
            merged.append([lo, hi])
    return [(lo - pad_frames, hi + pad_frames) for lo, hi in merged]


def in_any_segment(frame: int, segments: list[tuple[int, int]]) -> bool:
    """True iff ``frame`` lies inside any inclusive ``(lo, hi)`` segment (linear scan)."""
    return any(lo <= frame <= hi for lo, hi in segments)


# === IO + classification =========================================================================
def _chunk_key(half: str, path: Path) -> str:
    """``h1``, ``.../chunk_003_actions.json`` -> ``h1_chunk_003``."""
    return f"{half}_{path.stem.replace('_actions', '')}"


def load_actions(match_id: str) -> dict[str, dict]:
    """Load every ``chunk_*_actions.json`` for a match (missing chunks simply do not appear).

    Args:
        match_id: registry match id.

    Returns:
        ``{chunk_key: {"fps": float, "actions": [{frame_index, class, confidence}, ...]}}``.
    """
    out: dict[str, dict] = {}
    for half in HALVES:
        d = BALL_ROOT / match_id / "ball_action" / half
        if not d.exists():
            continue
        for p in sorted(d.glob("chunk_*_actions.json")):
            blob = json.loads(p.read_text(encoding="utf-8"))
            out[_chunk_key(half, p)] = {"fps": float(blob["fps"]), "actions": blob["actions"]}
    return out


def chunk_classification(match: Match, chunk_key: str, dfc: pd.DataFrame) -> tuple[np.ndarray,
                                                                                   np.ndarray]:
    """Per-frame ``(grid, shot_type)`` for one chunk, cached under ``outputs/<match>/live_play/``.

    The classification is derived from the aligned parquet (CPU-only, no video decode). The sampled
    grid is reconstructed at the parquet's modal frame step; frames absent from the parquet (zero
    detections) become :data:`live_play.SHOT_GRAPHIC`.

    Args:
        match: registry match.
        chunk_key: e.g. ``h1_chunk_003``.
        dfc: aligned rows for this chunk.

    Returns:
        ``(grid frame indices, shot_type strings)`` aligned element-wise.
    """
    cache = LIVEPLAY_CACHE / match.id / "live_play" / f"{chunk_key}.json"
    if cache.exists():
        blob = json.loads(cache.read_text(encoding="utf-8"))
        return np.asarray(blob["grid"], dtype=int), np.asarray(blob["shot_type"], dtype=object)

    fr = np.array(sorted(dfc["frame"].unique()), dtype=int)
    if fr.size < 2:
        grid = fr
        shot = np.array([live_play.SHOT_GRAPHIC] * fr.size, dtype=object)
    else:
        step = max(int(np.median(np.diff(fr))), 1)
        grid_idx = pd.Index(range(int(fr.min()), int(fr.max()) + 1, step))
        cls = live_play.classify_dense(dfc, grid_frames=grid_idx)
        grid = cls.index.to_numpy(dtype=int)
        shot = cls["shot_type"].to_numpy(dtype=object)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"grid": grid.tolist(), "shot_type": list(shot)}),
                     encoding="utf-8")
    return grid, shot


# === Counting ====================================================================================
def _peaks(actions: list[dict], cls: str, min_conf: float) -> tuple[list[int], list[float]]:
    """Frame indices + confidences of one action class at/above ``min_conf``."""
    fr, cf = [], []
    for a in actions:
        if a["class"] == cls and a["confidence"] >= min_conf:
            fr.append(int(a["frame_index"]))
            cf.append(float(a["confidence"]))
    return fr, cf


def op_events(chunk: dict, cls: str) -> list[tuple[int, float]]:
    """Operating-point filtered events for one action class (confidence floor + dedup).

    This is the single source of truth for the ``op`` arm: the same threshold + same-class NMS the
    validation counts use, but returning the kept ``(frame_index, confidence)`` pairs instead of a
    count -- so stage 2 attributes exactly the events stage 1 validated. Reuse this; do not re-derive
    the filter.

    Args:
        chunk: ``{"fps", "actions"}`` for one chunk (as loaded by :func:`load_actions`).
        cls: ``"PASS"`` or ``"DRIVE"``.

    Returns:
        Kept ``(frame_index, confidence)`` pairs, ascending by frame.
    """
    dedup_g = int(round(DEDUP_S * chunk["fps"]))
    fr, cf = _peaks(chunk["actions"], cls, OP_THRESHOLD)
    kept = set(dedup_peaks(fr, cf, dedup_g))
    best: dict[int, float] = {}
    for f, c in zip(fr, cf):
        if f in kept:
            best[f] = max(best.get(f, 0.0), c)
    return sorted(best.items())


def op_event_frames(chunk: dict, cls: str) -> list[int]:
    """Kept frame indices only (see :func:`op_events`)."""
    return [f for f, _ in op_events(chunk, cls)]


def count_chunk(chunk: dict, grid: np.ndarray, shot: np.ndarray, cls: str) -> dict[str, int]:
    """Raw / live-filtered / operating-point counts for one action class in one chunk.

    Args:
        chunk: ``{"fps", "actions"}`` for the chunk.
        grid: classification grid frames.
        shot: per-grid-frame shot types.
        cls: ``"PASS"`` or ``"DRIVE"``.

    Returns:
        ``{"raw", "live", "op"}`` peak counts.
    """
    fps = chunk["fps"]
    dedup_g = int(round(DEDUP_S * fps))

    fr_raw, cf_raw = _peaks(chunk["actions"], cls, 0.0)
    raw = len(fr_raw)

    is_live = shot == live_play.SHOT_LIVE_WIDE
    segs = build_live_segments(grid, is_live, int(round(LIVE_GAP_MERGE_S * fps)),
                               int(round(LIVE_PAD_S * fps)))
    fr_l = [f for f, c in zip(fr_raw, cf_raw) if in_any_segment(f, segs)]
    cf_l = [c for f, c in zip(fr_raw, cf_raw) if in_any_segment(f, segs)]
    live = len(dedup_peaks(fr_l, cf_l, dedup_g))

    op = len(op_event_frames(chunk, cls))
    return {"raw": raw, "live": live, "op": op}


def _truth_passes(match_id: str) -> dict[str, tuple[int, int]] | None:
    """Per-half attempted passes ``{"h1": (att, acc), "h2": ...}`` from Sofascore, or ``None``."""
    path = TRUTH.get(match_id)
    if path is None or not path.exists():
        return None
    df = pd.read_parquet(path)
    per = {}
    for half, period in (("h1", "1ST"), ("h2", "2ND")):
        att = df[(df["key"] == "passes") & (df["period"] == period)]
        acc = df[(df["key"] == "accuratePasses") & (df["period"] == period)]
        if att.empty:
            return None
        att_n = int(att["homeValue"].iloc[0] + att["awayValue"].iloc[0])
        acc_n = int(acc["homeValue"].iloc[0] + acc["awayValue"].iloc[0]) if not acc.empty else 0
        per[half] = (att_n, acc_n)
    return per


def validate_match(match: Match) -> dict:
    """Compute the full per-half raw/live/op PASS and DRIVE counts + coverage for one match.

    Args:
        match: registry match.

    Returns:
        A structured dict consumed by :func:`format_report`.
    """
    actions = load_actions(match.id)
    df = match.load_aligned()
    aligned_chunks = {h: sorted(c for c in df["chunk"].unique() if c.startswith(h)) for h in HALVES}
    bas_chunks = {h: sorted(c for c in actions if c.startswith(h)) for h in HALVES}
    truth = _truth_passes(match.id)

    def _span(ck: str) -> int:
        fr = df.loc[df["chunk"] == ck, "frame"]
        return int(fr.max() - fr.min()) if len(fr) else 0

    span = {c: _span(c) for c in df["chunk"].unique()}

    per_half: dict[str, dict] = {}
    for half in HALVES:
        agg = {cls: {"raw": 0, "live": 0, "op": 0} for cls in ("PASS", "DRIVE")}
        for ck in bas_chunks[half]:
            dfc = df[df["chunk"] == ck]
            if dfc.empty:
                continue
            grid, shot = chunk_classification(match, ck, dfc)
            for cls in ("PASS", "DRIVE"):
                for k, v in count_chunk(actions[ck], grid, shot, cls).items():
                    agg[cls][k] += v
        aligned_span = sum(span[c] for c in aligned_chunks[half])
        bas_span = sum(span.get(c, 0) for c in bas_chunks[half])
        coverage = bas_span / aligned_span if aligned_span else 0.0
        complete = coverage >= COVERAGE_MIN
        per_half[half] = {
            "counts": agg,
            "n_bas": len(bas_chunks[half]),
            "n_aligned": len(aligned_chunks[half]),
            "coverage": coverage,
            "complete": complete,
            "truth_att": truth[half][0] if truth else None,
            "truth_acc": truth[half][1] if truth else None,
        }
    return {"match": match.id, "per_half": per_half}


# === Reporting ===================================================================================
def _ratio(n: int, truth: int | None) -> str:
    """``n/truth`` as ``x.xxx`` or ``-`` when truth is missing/zero."""
    return f"{n / truth:.3f}" if truth else "-"


def format_report(results: list[dict]) -> str:
    """Render the pre-committed validation tables (Markdown) from :func:`validate_match` dicts."""
    lines = [
        "# BAS pass/drive validation (B-3 stage 1)",
        "",
        f"Operating point (frozen on brighton_manutd h1): PASS confidence >= {OP_THRESHOLD}, "
        f"same-class peaks merged within {DEDUP_S:.0f} s.",
        f"Replay filter (pre-committed hypothesis): keep peaks in live_wide segments "
        f"(gap-merge {LIVE_GAP_MERGE_S:.0f} s, pad {LIVE_PAD_S:.0f} s).",
        "",
        "`raw` = every BAS peak. `live` = replay-filter arm (+dedup). `op` = confidence+dedup arm. "
        f"`ratio_*` vs Sofascore attempted; a half's ratio is shown only when BAS covers "
        f">= {COVERAGE_MIN * 100:.0f}% of the half's frame span (else PARTIAL).",
        "",
    ]
    for res in results:
        mid = res["match"]
        ph = res["per_half"]
        lines.append(f"## {mid}")
        lines.append("")
        # Coverage line.
        cov = ", ".join(f"{h}: {ph[h]['n_bas']}/{ph[h]['n_aligned']} chunks "
                        f"({ph[h]['coverage'] * 100:.0f}% of frames"
                        + ("" if ph[h]["complete"] else ", PARTIAL") + ")" for h in HALVES)
        lines.append(f"Coverage -- {cov}")
        lines.append("")
        # PASS table.
        lines.append("### PASS")
        lines.append("")
        lines.append("| half | raw | live | op | truth att | ratio_raw | ratio_live | ratio_op |")
        lines.append("|------|-----|------|----|-----------|-----------|------------|----------|")
        tot = {"raw": 0, "live": 0, "op": 0}
        tot_truth = 0
        all_complete = True
        for h in HALVES:
            c = ph[h]["counts"]["PASS"]
            t = ph[h]["truth_att"]
            complete = ph[h]["complete"]
            show_t = t if complete else None
            flag = "" if complete else " (PARTIAL)"
            lines.append(
                f"| {h}{flag} | {c['raw']} | {c['live']} | {c['op']} | "
                f"{t if t is not None else '-'} | {_ratio(c['raw'], show_t)} | "
                f"{_ratio(c['live'], show_t)} | {_ratio(c['op'], show_t)} |"
            )
            for k in tot:
                tot[k] += c[k]
            if complete and t:
                tot_truth += t
            all_complete &= complete
        show_tt = tot_truth if (all_complete and tot_truth) else None
        lines.append(
            f"| **match** | {tot['raw']} | {tot['live']} | {tot['op']} | "
            f"{tot_truth if tot_truth else '-'} | {_ratio(tot['raw'], show_tt)} | "
            f"{_ratio(tot['live'], show_tt)} | {_ratio(tot['op'], show_tt)} |"
        )
        lines.append("")
        # DRIVE table (no truth).
        lines.append("### DRIVE (no ground truth -- raw vs filtered only)")
        lines.append("")
        lines.append("| half | raw | live | op |")
        lines.append("|------|-----|------|----|")
        dtot = {"raw": 0, "live": 0, "op": 0}
        for h in HALVES:
            c = ph[h]["counts"]["DRIVE"]
            flag = "" if ph[h]["complete"] else " (PARTIAL)"
            lines.append(f"| {h}{flag} | {c['raw']} | {c['live']} | {c['op']} |")
            for k in dtot:
                dtot[k] += c[k]
        lines.append(f"| **match** | {dtot['raw']} | {dtot['live']} | {dtot['op']} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Validate every match with BAS output and write ``results/bas_validation.md``."""
    results = []
    for match in registry.matches():
        if not (BALL_ROOT / match.id / "ball_action").exists():
            continue
        print(f"validating {match.id} ...")
        res = validate_match(match)
        results.append(res)
        for h in HALVES:
            ph = res["per_half"][h]
            c = ph["counts"]["PASS"]
            print(f"  {h}: {ph['n_bas']}/{ph['n_aligned']} chunks ({ph['coverage'] * 100:.0f}%) | "
                  f"PASS raw {c['raw']} live {c['live']} op {c['op']} | truth {ph['truth_att']}"
                  f"{'' if ph['complete'] else ' PARTIAL'}")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(format_report(results), encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
