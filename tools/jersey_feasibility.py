"""Feasibility probe: are jersey numbers legible on our own PL broadcast footage?

PROBE ONLY -- writes nothing into the pipeline. Decides Layer 2 (player identity) of
``docs/CAPABILITY_LEDGER.md``. The question: at 1920x1080 wide-camera distance, how tall (in
pixels) is a player torso / jersey number, and is that number human-legible?

Method:
  1. Content-type split from the dense aligned parquet (players-per-frame is the proxy the ledger
     already uses): ``>=8`` detected outfield people = WIDE tactical play, ``4-7`` = MEDIUM,
     ``<=3`` = CLOSE-UP / replay. This mirrors the pose-carry probe's close-up diagnosis.
  2. The parquet stores only foot points (image_x/image_y), no box height, so we re-run a detector
     on the sampled frames to recover person boxes. NOTE: the football-trained YOLO ``extract.py``
     uses caps box height at ~160 px -- it is blind to large foreground / close-up players (verified:
     on close-up frames it maxes ~150 px while COCO yolov8s finds boxes up to ~1050 px). So this
     probe uses **COCO yolov8s person detection**, which spans both tactical scale (agrees with the
     football detector on wide play) and close-up scale (the whole point of path 2).
  3. Torso = the upper 40% of the person box (where a back number sits). Record torso-crop height
     in px; build per-content-type contact-sheet montages (upscaled, px height burnt in) so a human
     -- or this agent -- can eyeball legibility.

Two viability paths, reported separately:
  * DIRECT: OCR the number straight off a wide-play crop. Needs a meaningful fraction of wide-play
    appearances to be legible.
  * CLOSE-UP-ANCHORED: a number read once during a close-up / replay (useless for geometry, but the
    torso is huge and the number legible) propagates along the whole track id. Close-ups are gold
    for identity even though we discard them for geometry.

Run (needs the ``[cv]`` stack; GPU used, one job)::

    python -m tools.jersey_feasibility --per-type 160 --seed 0
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from core import registry

OUT_DIR = Path("results/jersey_probe")
VIDEO_ROOT = Path("matches")
_CHUNK_RE = re.compile(r"(h\d)_chunk_(\d+)$")

# Content-type thresholds on detected outfield people per frame (ledger's live-wide-play proxy).
WIDE_MIN = 8
MEDIUM_MIN = 4
CONTENT_TYPES = ("wide", "medium", "close")

# Torso occupies the upper 40% of a standing-player box; a back number spans roughly the middle
# third of that torso, so number height ~= 0.35 * torso height (used only to translate the torso
# distribution into the ledger's ">= ~12 px legible number" bar; the montages are the real check).
TORSO_FRAC = 0.40
NUMBER_OF_TORSO = 0.35
NUMBER_LEGIBLE_PX = 12.0  # pre-committed human-legibility floor for a jersey number
# -> a >= 12 px number needs torso height >= 12 / 0.35 ~= 34 px; we report the 32 px bar alongside.


def _video_for(chunk_key: str) -> Path:
    """``h1_chunk_000`` -> ``matches/<id>/h1/chunk_000.mp4`` (id filled by the caller)."""
    mo = _CHUNK_RE.fullmatch(chunk_key)
    if mo is None:
        raise ValueError(f"unparseable chunk key {chunk_key!r}")
    return Path(mo.group(1)) / f"chunk_{mo.group(2)}.mp4"


def content_type(n_people: int) -> str:
    """Bucket a frame by its detected-people count (the broadcast content proxy)."""
    if n_people >= WIDE_MIN:
        return "wide"
    if n_people >= MEDIUM_MIN:
        return "medium"
    return "close"


def classify_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Per (chunk, frame): outfield-people count and content bucket, from the aligned parquet."""
    people = df[df["role"].isin(["player", "goalkeeper"])]
    counts = (
        people.groupby(["chunk", "frame"]).size().rename("n_people").reset_index()
    )
    counts["content"] = counts["n_people"].map(content_type)
    return counts


def sample_frames(counts: pd.DataFrame, per_type: int, seed: int) -> pd.DataFrame:
    """Draw ``per_type`` frames per content bucket, spread across chunks, deterministically."""
    rng = np.random.default_rng(seed)
    picks = []
    for ct in CONTENT_TYPES:
        pool = counts[counts["content"] == ct]
        n = min(per_type, len(pool))
        idx = rng.choice(pool.index.to_numpy(), size=n, replace=False)
        picks.append(pool.loc[idx])
    return pd.concat(picks).sort_values(["chunk", "frame"]).reset_index(drop=True)


class _Reservoir:
    """Seeded reservoir of ``cap`` torso crops (+ their px heights) for one content type."""

    def __init__(self, cap: int, seed: int):
        self.cap = cap
        self.rng = np.random.default_rng(seed)
        self.n_seen = 0
        self.items: list[tuple[np.ndarray, float]] = []

    def offer(self, crop: np.ndarray, height_px: float) -> None:
        self.n_seen += 1
        if len(self.items) < self.cap:
            self.items.append((crop, height_px))
        else:
            j = int(self.rng.integers(0, self.n_seen))
            if j < self.cap:
                self.items[j] = (crop, height_px)


def _montage(items: list[tuple[np.ndarray, float]], cols: int = 8, cell_h: int = 150) -> np.ndarray:
    """Grid of upscaled torso crops with px height burnt in; ordered tallest-first."""
    items = sorted(items, key=lambda t: -t[1])
    cell_w = int(cell_h * 0.6)
    rows = (len(items) + cols - 1) // cols
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 30, np.uint8)
    for k, (crop, h_px) in enumerate(items):
        c = crop
        if c.size == 0:
            continue
        scale = cell_h / max(c.shape[0], 1)
        w = max(int(c.shape[1] * scale), 1)
        interp = cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA
        r = cv2.resize(c, (min(w, cell_w), cell_h), interpolation=interp)
        ry, rx = k // cols * cell_h, k % cols * cell_w
        sheet[ry:ry + r.shape[0], rx:rx + r.shape[1]] = r
        cv2.putText(sheet, f"{h_px:.0f}px", (rx + 2, ry + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return sheet


def _pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if len(a) else float("nan")


def run(match_id: str, per_type: int, seed: int, montage_n: int) -> dict:
    """Execute the probe: sample, detect, measure torso heights, write montages + stats."""
    from ultralytics import YOLO  # noqa: PLC0415

    match = registry.get(match_id)
    df = match.load_aligned()
    counts = classify_frames(df)
    total = len(counts)
    overall_mix = {ct: float((counts["content"] == ct).mean()) for ct in CONTENT_TYPES}
    sampled = sample_frames(counts, per_type, seed)

    import torch  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO("yolov8s.pt")
    yolo.to(device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    heights: dict[str, list[float]] = {ct: [] for ct in CONTENT_TYPES}
    box_heights: dict[str, list[float]] = {ct: [] for ct in CONTENT_TYPES}
    # Per-frame tallest torso: the decisive path-2 metric -- a close-up has ONE big legible subject
    # amid many small background players, so the per-crop median hides it but the per-frame max shows
    # it. "Does this frame contain a legibly-large person?"
    frame_max: dict[str, list[float]] = {ct: [] for ct in CONTENT_TYPES}
    reservoirs = {ct: _Reservoir(montage_n, seed + i) for i, ct in enumerate(CONTENT_TYPES)}
    frames_done = {ct: 0 for ct in CONTENT_TYPES}

    for chunk_key, grp in sampled.groupby("chunk"):
        video = VIDEO_ROOT / match.id / _video_for(chunk_key)
        if not video.exists():
            print(f"WARN missing video {video}")
            continue
        cap = cv2.VideoCapture(str(video))
        for _, row in grp.iterrows():
            ct = row["content"]
            fi = int(row["frame"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame_bgr = cap.read()
            if not ret:
                continue
            frames_done[ct] += 1
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = yolo(rgb, verbose=False, conf=0.20, classes=[0])[0]  # COCO person only
            xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
            best_this_frame = 0.0
            for box in xyxy:
                x1, y1, x2, y2 = (int(v) for v in box)
                box_h = float(y2 - y1)
                if box_h <= 0:
                    continue
                torso_h = box_h * TORSO_FRAC
                ty2 = y1 + int(round(box_h * TORSO_FRAC))
                crop = frame_bgr[max(y1, 0):max(ty2, 0), max(x1, 0):max(x2, 0)]
                heights[ct].append(torso_h)
                box_heights[ct].append(box_h)
                best_this_frame = max(best_this_frame, torso_h)
                if crop.size > 0:
                    reservoirs[ct].offer(crop.copy(), torso_h)
            frame_max[ct].append(best_this_frame)
        cap.release()

    stats: dict = {
        "match": match.id,
        "resolution": "1920x1080",
        "content_proxy": {"wide": ">=8 people", "medium": "4-7", "close": "<=3"},
        "overall_content_mix_of_match": overall_mix,
        "total_frame_instances": total,
        "torso_frac_of_box": TORSO_FRAC,
        "number_height_px_estimate": f"~{NUMBER_OF_TORSO:.2f} x torso height",
        "legible_number_floor_px": NUMBER_LEGIBLE_PX,
        "per_type": {},
    }
    for ct in CONTENT_TYPES:
        h = np.array(heights[ct])
        bh = np.array(box_heights[ct])
        fm = np.array(frame_max[ct])
        num_est = h * NUMBER_OF_TORSO
        stats["per_type"][ct] = {
            "frames_sampled": frames_done[ct],
            "n_player_crops": int(len(h)),
            "torso_h_px_p10": round(_pct(h, 10), 1),
            "torso_h_px_p50": round(_pct(h, 50), 1),
            "torso_h_px_p90": round(_pct(h, 90), 1),
            "box_h_px_p50": round(_pct(bh, 50), 1),
            "frac_torso_ge_32px": round(float((h >= 32).mean()), 3) if len(h) else None,
            "frac_torso_ge_64px": round(float((h >= 64).mean()), 3) if len(h) else None,
            "frac_number_ge_12px_est": round(float((num_est >= NUMBER_LEGIBLE_PX).mean()), 3)
            if len(h) else None,
            "est_number_h_px_p50": round(_pct(num_est, 50), 1),
            "frame_max_torso_p50": round(_pct(fm, 50), 1),
            "frame_max_torso_p90": round(_pct(fm, 90), 1),
            "frac_frames_with_torso_ge_64px": round(float((fm >= 64).mean()), 3) if len(fm) else None,
            "frac_frames_with_torso_ge_100px": round(float((fm >= 100).mean()), 3)
            if len(fm) else None,
        }
        sheet = _montage(reservoirs[ct].items)
        path = OUT_DIR / f"montage_{ct}.png"
        cv2.imwrite(str(path), sheet)
        stats["per_type"][ct]["montage"] = str(path)

    (OUT_DIR / "jersey_probe_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    return stats


def _print_summary(stats: dict) -> None:
    print(f"\nJERSEY LEGIBILITY PROBE -- {stats['match']} @ {stats['resolution']}")
    mix = stats["overall_content_mix_of_match"]
    print("match content mix (frame-instances): "
          f"wide {mix['wide']:.1%}  medium {mix['medium']:.1%}  close {mix['close']:.1%}")
    print("-" * 78)
    print("PER-CROP torso height (all detected people, incl. small background players):")
    hdr = ("content", "frames", "crops", "torso p10/p50/p90", ">=32px", ">=64px", "num>=12px")
    print("{:<8}{:>7}{:>7}{:>22}{:>8}{:>8}{:>11}".format(*hdr))
    for ct in CONTENT_TYPES:
        s = stats["per_type"][ct]
        tri = f"{s['torso_h_px_p10']}/{s['torso_h_px_p50']}/{s['torso_h_px_p90']}"
        print("{:<8}{:>7}{:>7}{:>22}{:>8}{:>8}{:>11}".format(
            ct, s["frames_sampled"], s["n_player_crops"], tri,
            f"{s['frac_torso_ge_32px']:.0%}", f"{s['frac_torso_ge_64px']:.0%}",
            f"{s['frac_number_ge_12px_est']:.0%}"))
    print("-" * 78)
    print("PER-FRAME TALLEST torso (path-2 metric: a legibly-large subject present in the frame?):")
    print("{:<8}{:>18}{:>18}{:>16}".format(
        "content", "frame-max p50/p90", "frames >=64px", "frames >=100px"))
    for ct in CONTENT_TYPES:
        s = stats["per_type"][ct]
        fm = f"{s['frame_max_torso_p50']}/{s['frame_max_torso_p90']}"
        print("{:<8}{:>18}{:>18}{:>16}".format(
            ct, fm, f"{s['frac_frames_with_torso_ge_64px']:.0%}",
            f"{s['frac_frames_with_torso_ge_100px']:.0%}"))
    print("-" * 78)
    print(f"montages + stats -> {OUT_DIR}")


def main() -> None:
    """CLI entry point (needs the ``[cv]`` stack; uses the GPU)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", default="brighton_manutd")
    ap.add_argument("--per-type", type=int, default=160, help="frames sampled per content bucket")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--montage-n", type=int, default=48, help="crops per content-type montage")
    args = ap.parse_args()
    stats = run(args.match, args.per_type, args.seed, args.montage_n)
    _print_summary(stats)


if __name__ == "__main__":
    main()
