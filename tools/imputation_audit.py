"""Audit B4 truth-data candidates before any imputer is built (validated-or-nothing).

Metrica Sports open data is the PRIMARY full-pitch truth candidate; SkillCorner open
data is a broadcast domain-transfer check (NOT truth). This script measures, for each:
players-per-frame distribution, gap/NaN rates, frame rate, coordinate system and license
notes, and (SkillCorner) the detected-vs-extrapolated fraction. Run it before building the
censoring simulator -- if Metrica is not genuinely 22-player full-pitch, that gates B4.

Usage:
    python tools/imputation_audit.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
METRICA = REPO / "data" / "imputation" / "metrica"
SKILLCORNER = REPO / "data" / "imputation" / "skillcorner"


def load_metrica_team(csv_path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load one Metrica raw-tracking CSV into arrays.

    Metrica CSVs have three header rows (team label, jersey number, column name) then
    per-frame rows of ``Period, Frame, Time [s], (x, y) * players, Ball_x, Ball_y``.

    Args:
        csv_path: Path to a ``*_RawTrackingData_{Home,Away}_Team.csv`` file.

    Returns:
        Tuple of ``(xy, player_names, time_s)`` where ``xy`` has shape
        ``(n_frames, n_players, 2)`` in normalised [0, 1] pitch coordinates (NaN when the
        player is off the sheet), ``player_names`` is the ordered player list (Ball
        excluded), and ``time_s`` is the per-frame timestamp in seconds.
    """
    with csv_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    name_row = rows[2]
    # Columns 0..2 are Period, Frame, Time; player x-columns start at index 3, step 2.
    players: list[str] = []
    x_idx: list[int] = []
    for idx in range(3, len(name_row), 2):
        label = name_row[idx].strip()
        if label and label.lower() != "ball":
            players.append(label)
            x_idx.append(idx)
    data = rows[3:]
    n_frames = len(data)
    xy = np.full((n_frames, len(players), 2), np.nan, dtype=float)
    time_s = np.full(n_frames, np.nan, dtype=float)
    for r, row in enumerate(data):
        time_s[r] = _to_float(row[2])
        for p, col in enumerate(x_idx):
            xy[r, p, 0] = _to_float(row[col])
            xy[r, p, 1] = _to_float(row[col + 1])
    return xy, players, time_s


def _to_float(token: str) -> float:
    """Parse a CSV token to float, mapping blanks/``NaN`` to ``np.nan``."""
    token = token.strip()
    if not token or token.lower() == "nan":
        return float("nan")
    return float(token)


def _pct(arr: np.ndarray, q: float) -> float:
    """Percentile helper tolerant of empty arrays."""
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def audit_metrica() -> dict[str, object]:
    """Audit the Metrica full-pitch truth candidate and print measured numbers."""
    print("=" * 72)
    print("METRICA AUDIT  (primary full-pitch truth candidate)")
    print("=" * 72)
    home_xy, home_names, t_home = load_metrica_team(
        METRICA / "Sample_Game_1_RawTrackingData_Home_Team.csv"
    )
    away_xy, away_names, _ = load_metrica_team(
        METRICA / "Sample_Game_1_RawTrackingData_Away_Team.csv"
    )
    n = min(home_xy.shape[0], away_xy.shape[0])
    home_xy, away_xy, t_home = home_xy[:n], away_xy[:n], t_home[:n]

    # frame rate from median timestamp delta
    dt = np.diff(t_home)
    fps = 1.0 / float(np.median(dt[np.isfinite(dt)]))
    print(f"frames={n}  roster: home={len(home_names)} away={len(away_names)} slots")
    print(f"fps (median dt)={fps:.3f}   duration={t_home[-1]/60:.1f} min")

    # players-per-frame (x not NaN)
    home_present = np.isfinite(home_xy[:, :, 0]).sum(axis=1)
    away_present = np.isfinite(away_xy[:, :, 0]).sum(axis=1)
    total_present = home_present + away_present
    for tag, arr in (("home", home_present), ("away", away_present), ("both", total_present)):
        print(
            f"players/frame [{tag:4s}] mean={arr.mean():5.2f} min={arr.min():2d} "
            f"p50={int(np.median(arr)):2d} max={arr.max():2d}"
        )
    frac_full_22 = float((total_present == 22).mean())
    frac_ge_21 = float((total_present >= 21).mean())
    print(f"frac frames with exactly 22 present = {frac_full_22:.4f}")
    print(f"frac frames with >=21 present       = {frac_ge_21:.4f}")

    # coordinate system extents
    finite = np.isfinite(home_xy)
    xs = home_xy[:, :, 0][finite[:, :, 0]]
    ys = home_xy[:, :, 1][finite[:, :, 1]]
    print(
        f"coord extents  x[{xs.min():.3f},{xs.max():.3f}]  y[{ys.min():.3f},{ys.max():.3f}]"
        "  (normalised 0-1, top-left origin, pitch 105x68 m per README)"
    )

    # per-player NaN/gap rate: fraction of frames NaN while that player was ever on pitch
    print("per-player coverage (home; on-pitch span only):")
    _print_player_gaps(home_xy, home_names, fps)
    print("license: Metrica README -- 'be responsible; acknowledge the source if public'")
    verdict = frac_ge_21 > 0.98
    print(f"VERDICT: {'PASS' if verdict else 'FAIL'} full-pitch 22-player truth")
    return {
        "fps": fps,
        "frames": n,
        "mean_players_per_frame": float(total_present.mean()),
        "frac_full_22": frac_full_22,
        "frac_ge_21": frac_ge_21,
        "pass": verdict,
    }


def _print_player_gaps(xy: np.ndarray, names: list[str], fps: float) -> None:
    """Print NaN-within-span gap stats per player (substitutes handled via first/last seen)."""
    for p, name in enumerate(names):
        present = np.isfinite(xy[:, p, 0])
        if not present.any():
            print(f"  {name:9s} never on pitch")
            continue
        first, last = np.argmax(present), len(present) - np.argmax(present[::-1])
        span = present[first:last]
        gaps = int((~span).sum())
        print(
            f"  {name:9s} span={span.size:6d} fr  gaps_in_span={gaps:4d} "
            f"({100.0 * gaps / span.size:5.2f}%)"
        )


def audit_skillcorner() -> dict[str, object]:
    """Audit the SkillCorner broadcast transfer set: detected-vs-extrapolated + coverage."""
    print("=" * 72)
    print("SKILLCORNER AUDIT  (broadcast transfer-check, NOT truth)")
    print("=" * 72)
    path = next(SKILLCORNER.glob("*_tracking_extrapolated.jsonl"))
    per_frame_players: list[int] = []
    per_frame_detected: list[int] = []
    n_points = 0
    n_detected = 0
    ball_detected = 0
    ball_total = 0
    times: list[float] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            pdat = rec.get("player_data") or []
            if not pdat:
                continue
            det = sum(1 for p in pdat if p.get("is_detected"))
            per_frame_players.append(len(pdat))
            per_frame_detected.append(det)
            n_points += len(pdat)
            n_detected += det
            bd = rec.get("ball_data") or {}
            if bd.get("x") is not None:
                ball_total += 1
                ball_detected += 1 if bd.get("is_detected") else 0
            ts = rec.get("timestamp")
            if ts:
                times.append(_hms_to_s(ts))
    ppf = np.array(per_frame_players)
    dpf = np.array(per_frame_detected)
    dt = np.diff(np.array(times))
    dt = dt[(dt > 0) & (dt < 1)]
    fps = 1.0 / float(np.median(dt)) if dt.size else float("nan")
    print(f"populated frames={ppf.size}  fps (median dt)={fps:.3f}")
    print(
        f"players/frame (incl. extrapolated) mean={ppf.mean():5.2f} "
        f"min={ppf.min()} p50={int(np.median(ppf))} max={ppf.max()}"
    )
    print(
        f"DETECTED players/frame            mean={dpf.mean():5.2f} "
        f"p10={_pct(dpf, 10):.1f} p50={int(np.median(dpf))} p90={_pct(dpf, 90):.1f}"
    )
    frac_extrap = 1.0 - n_detected / n_points
    print(f"player points: total={n_points}  detected={n_detected}")
    print(f"FRACTION EXTRAPOLATED (players) = {frac_extrap:.4f}")
    if ball_total:
        print(
            f"ball: frames_with_xy={ball_total}  detected={ball_detected} "
            f"({100.0 * ball_detected / ball_total:.1f}%)  extrap={100.0 * (1 - ball_detected / ball_total):.1f}%"
        )
    print("coord system: metres, pitch-centred origin (x +/-52.5, y +/-34)")
    print("license: SkillCorner opendata -- CC BY-NC, acknowledge source, non-commercial")
    print("NOTE: extrapolated points are model guesses, not ground truth -> transfer check only")
    return {
        "fps": fps,
        "frames": int(ppf.size),
        "mean_players_per_frame": float(ppf.mean()),
        "mean_detected_per_frame": float(dpf.mean()),
        "frac_extrapolated": frac_extrap,
    }


def _hms_to_s(ts: str) -> float:
    """Convert a ``HH:MM:SS.ss`` SkillCorner timestamp to seconds."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def main() -> None:
    """Run both audits."""
    audit_metrica()
    print()
    audit_skillcorner()


if __name__ == "__main__":
    main()
