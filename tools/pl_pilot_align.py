"""Assemble the Brighton-Man Utd pilot's per-chunk dense parquets into an anchored match table.

Mirrors the WC ``final/match_aligned.parquet`` layout so every existing consumer (registry,
``report/facts.py``, the fingerprint engine) works unchanged. For each half it loads
``outputs/brighton_manutd/<half>/match/chunk_<NNN>_dense.parquet``, tags the chunk with the
half-prefixed key (``h1_chunk_000`` -- matching the WC ``aligned['chunk']`` convention), then
kit-anchors team identity across BOTH halves together via
:func:`generator.team_anchor.align_teams_by_color` (``dark_is_team0`` -> the darker kit becomes team
0). Man Utd wore red (home) and Brighton blue/white stripes; a colour diagnostic reports which team
id each kit maps to, and an attack-direction check per chunk verifies the halftime split (the same
team should attack opposite directions in h1 vs h2).

Run (CPU only -- decodes video for jersey colour + a colour-ID sample):
    python -m tools.pl_pilot_align
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from fingerprint.structural_metrics import resolve_attack_directions
from generator.team_anchor import PLAYER_ROLES, estimate_player_box, align_teams_by_color

MATCH_ID = "brighton_manutd"
OUT_ROOT = Path("outputs") / MATCH_ID
VIDEO_ROOT = Path("matches") / MATCH_ID
HALVES = ("h1", "h2")
_CHUNK_RE = re.compile(r"chunk_(\d+)_dense\.parquet$")


def load_half(half: str) -> pd.DataFrame:
    """Load a half's per-chunk dense parquets, tagging each with its half-prefixed chunk key."""
    dfs = []
    match_dir = OUT_ROOT / half / "match"
    for f in sorted(match_dir.glob("chunk_*_dense.parquet")):
        num = _CHUNK_RE.search(f.name).group(1)
        d = pd.read_parquet(f)
        d["chunk"] = f"{half}_chunk_{num}"
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def team_colour_report(aligned: pd.DataFrame, video_map: dict[str, str], *,
                       per_chunk: int = 40) -> dict[int, dict]:
    """Mean torso BGR/hue per anchored team id, to name which kit (red=Utd / stripes=Brighton) is 0/1."""
    import cv2  # noqa: PLC0415

    acc: dict[int, list] = {0: [], 1: []}
    for chunk, g in aligned.groupby("chunk"):
        vid = video_map.get(chunk)
        if not vid or not Path(vid).exists():
            continue
        pl = g[g["role"].isin(PLAYER_ROLES) & g["team"].isin([0, 1])].dropna(
            subset=["image_x", "image_y"])
        if pl.empty:
            continue
        cap = cv2.VideoCapture(vid)
        fh, fw = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frames = np.sort(pl["frame"].unique())
        pick = frames[np.linspace(0, len(frames) - 1, min(per_chunk, len(frames))).astype(int)]
        by_frame = {fr: gg for fr, gg in pl[pl["frame"].isin(pick)].groupby("frame")}
        for fr in pick:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
            ok, bgr = cap.read()
            if not ok:
                continue
            for r in by_frame.get(fr, pd.DataFrame()).itertuples(index=False):
                x1, y1, x2, y2 = estimate_player_box(r.image_x, r.image_y, fh, fw)
                crop = bgr[y1:y2, x1:x2]
                if crop.size:
                    acc[int(r.team)].append(crop.reshape(-1, 3).mean(axis=0))
        cap.release()
    out: dict[int, dict] = {}
    for t in (0, 1):
        if not acc[t]:
            continue
        mean_bgr = np.mean(acc[t], axis=0)
        hsv = cv2.cvtColor(np.uint8([[mean_bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
        b, g_, r = mean_bgr
        kit = "red (Man Utd)" if r > b + 12 else ("blue/stripes (Brighton)" if b > r + 12 else "ambiguous")
        out[t] = {"n": len(acc[t]), "bgr": [round(float(c), 1) for c in mean_bgr],
                  "hue": int(hsv[0]), "kit": kit}
    return out


def halftime_direction_check(aligned: pd.DataFrame) -> str:
    """Per-chunk attack direction for team 0; opposite signs h1 vs h2 confirm the halftime split."""
    lines = []
    for chunk, g in sorted(aligned.groupby("chunk")):
        gg = g[(g["calib_error_m"] <= 1.0) & g["pitch_x"].notna()]
        if gg.empty:
            lines.append(f"  {chunk}: no calibrated frames")
            continue
        dirs = resolve_attack_directions(gg)
        d0 = dirs.get(0, float("nan"))
        lines.append(f"  {chunk}: team0 attack_dir={d0:+.0f}  (players/frame "
                     f"{gg[gg['team'].isin([0, 1])].groupby('frame').size().mean():.1f})")
    return "\n".join(lines)


def main() -> None:
    """Anchor teams across both halves, write match_aligned + per-half tables, print diagnostics."""
    halves = {h: load_half(h) for h in HALVES}
    halves = {h: d for h, d in halves.items() if not d.empty}
    if not halves:
        print("no dense parquets yet under outputs/brighton_manutd/<half>/match -- run extraction first")
        return
    combined = pd.concat(halves.values(), ignore_index=True)
    video_map = {}
    for h, d in halves.items():
        for ck in d["chunk"].unique():
            num = ck.split("_")[-1]
            video_map[ck] = str(VIDEO_ROOT / h / f"chunk_{num}.mp4")

    print(f"[align] anchoring {combined['chunk'].nunique()} chunks "
          f"({sum(len(d) for d in halves.values())} rows) by jersey colour ...", flush=True)
    aligned = align_teams_by_color(combined, video_map, dark_is_team0=True)

    (OUT_ROOT / "final").mkdir(parents=True, exist_ok=True)
    aligned.to_parquet(OUT_ROOT / "final" / "match_aligned.parquet", index=False)
    for h in HALVES:
        sub = aligned[aligned["chunk"].str.startswith(h)]
        if not sub.empty:
            sub.to_parquet(OUT_ROOT / "final" / f"{h}_aligned.parquet", index=False)
    print(f"[align] wrote {OUT_ROOT / 'final' / 'match_aligned.parquet'} "
          f"({len(aligned)} rows)", flush=True)

    print("\n[align] TEAM COLOUR IDENTITY (anchored id -> kit):")
    rep = team_colour_report(aligned, video_map)
    for t in (0, 1):
        if t in rep:
            print(f"  team {t}: {rep[t]['kit']:26s} bgr={rep[t]['bgr']} hue={rep[t]['hue']} "
                  f"(n={rep[t]['n']})")
    utd = next((t for t in (0, 1) if t in rep and "Utd" in rep[t]["kit"]), None)
    print(f"  => Man Utd maps to team id: {utd}")

    print("\n[align] HALFTIME / ATTACK-DIRECTION CHECK (team0 dir should flip h1 vs h2):")
    print(halftime_direction_check(aligned))


if __name__ == "__main__":
    main()
