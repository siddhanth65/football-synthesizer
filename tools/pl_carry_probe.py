"""Prove or disprove the PL post-link coverage lever: temporal homography carry-over for the ball.

Phase-A PL probe context (see ``results/pl_probe/diagnosis/DIAGNOSIS.md`` and ``STATUS.md``): post-
``link_ball`` ball coverage is ~21% pooled despite 87% held-out detection recall. The binding
constraint is per-frame homography availability, not detection. This tool measures whether reusing
the nearest camera-continuous homography (:mod:`generator.ball_carry`) for frames that lack their own
lifts the ONLY number that counts -- post-``link_ball`` usable-track coverage -- end to end, OFF vs ON.

Two stages, VRAM-isolated (4 GB laptop):

* **detect (GPU, cached):** re-run the v6 ball detector on *all* dense frames of each segment (the
  baseline probe only detected on already-calibrated frames, so carry-over targets had no detection)
  and cache raw image-xy to ``outputs/pl_probe/<match>/<seg>_ball_imgxy.parquet``. One GPU pass.
* **project+link (CPU):** for OFF (own-frame homography only, reproduces the v6 baseline) and ON
  (carry-over) we project, ``link_ball`` at the true fps, and report post-link coverage plus a
  pre-link mechanism breakdown (own/carried/interp -- diagnostic only) and a physical-plausibility
  check on the carried positions.

Run (GPU stage is one-at-a-time; never with the full pytest suite):
    python -m tools.pl_carry_probe                    # all segments: detect (cached) + OFF/ON
    python -m tools.pl_carry_probe --skip-detect      # reuse cached detections (CPU only)
    python -m tools.pl_carry_probe --match manutd_fulham --skip-detect
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from generator.ball import BALL_MAX_SPEED_MS, link_ball
from generator.ball_carry import (
    DEFAULT_JACCARD_CUT,
    DEFAULT_WINDOW_FRAMES,
    _nearest_sources,
    camera_segment_ids,
    fit_frame_homographies,
    project_ball_carry,
)

PROBE_ROOT = Path("matches/pl_probe")
DENSE_ROOT = Path("outputs/pl_probe")
RESULTS_ROOT = Path("results/pl_probe")
V6_WEIGHTS = "outputs/ball_finetuned/tracknetv2_v6.pth"
MATCH_KEYS = ("brighton_manutd", "manutd_fulham", "manutd_liverpool")
PLAYER_ROLES = ("player", "goalkeeper")
BALL_THR = 0.5
SRC_LEN, SRC_WID = 105.0, 68.0


def _segments(match: str | None, seg: str | None) -> list[tuple[str, Path]]:
    """List ``(match_key, video_path)`` for the requested filter."""
    keys = [match] if match else list(MATCH_KEYS)
    out: list[tuple[str, Path]] = []
    for key in keys:
        for vid in sorted((PROBE_ROOT / key).glob("seg_*.mp4")):
            if seg is None or vid.stem == seg:
                out.append((key, vid))
    return out


def _true_fps(video: Path) -> float:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps if fps and fps > 1.0 else 25.0


def _detpath(key: str, stem: str) -> Path:
    return DENSE_ROOT / key / f"{stem}_ball_imgxy.parquet"


# === Stage 1: GPU ball detection over ALL dense frames (cached) ==================================
def detect_all_frames(segs: list[tuple[str, Path]], *, weights: str, thr: float,
                      skip_existing: bool = True) -> None:
    """Detect the v6 ball on every dense frame of each segment; cache raw image-xy to parquet."""
    todo = [(k, v) for k, v in segs if not (skip_existing and _detpath(k, v.stem).exists())]
    if not todo:
        print("[detect] all detection caches present; skipping GPU pass", flush=True)
        return
    from tools.ball_possession import _load_model, detect_ball_imagexy  # noqa: PLC0415

    model, dev = _load_model(weights)
    print(f"[detect] loaded v6 on {dev}; weights={weights} thr={thr}", flush=True)
    for key, vid in todo:
        dense = pd.read_parquet(DENSE_ROOT / key / f"{vid.stem}_dense.parquet")
        frames = sorted(int(f) for f in dense["frame"].unique())
        det = detect_ball_imagexy(str(vid), frames, model, dev, thr=thr)
        df = pd.DataFrame([{"frame": int(f), "bx": float(x), "by": float(y)}
                           for f, (x, y) in det.items()], columns=["frame", "bx", "by"])
        df.sort_values("frame").to_parquet(_detpath(key, vid.stem), index=False)
        print(f"[detect] {key}/{vid.stem}: fired {len(df)}/{len(frames)} frames "
              f"({len(df) / max(len(frames), 1):.0%}) -> {_detpath(key, vid.stem).name}", flush=True)
    try:
        import gc  # noqa: PLC0415

        import torch  # noqa: PLC0415
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# === Stage 2: CPU project + link, OFF vs ON ======================================================
def _plausibility(track: pd.DataFrame, carried_frames: set[int], fps: float) -> dict:
    """Physical-plausibility check on the ON linked track's carried, observed ball positions."""
    if track.empty:
        return {"n_carried_linked": 0, "off_pitch": 0, "max_speed": 0.0, "n_over_clamp": 0}
    obs = track[track["observed"]].sort_values("frame").reset_index(drop=True)
    carried_obs = obs[obs["frame"].isin(carried_frames)]
    x, y = carried_obs["x"].to_numpy(), carried_obs["y"].to_numpy()
    off = int(((x < 0) | (x > SRC_LEN) | (y < 0) | (y > SRC_WID)).sum())
    # frame-to-frame speed across ALL observed samples (carried positions must move physically vs nbrs)
    f = obs["frame"].to_numpy()
    ox, oy = obs["x"].to_numpy(), obs["y"].to_numpy()
    speeds = []
    for i in range(1, len(f)):
        dt = (f[i] - f[i - 1]) / fps
        if dt > 0:
            speeds.append(float(np.hypot(ox[i] - ox[i - 1], oy[i] - oy[i - 1]) / dt))
    speeds = np.array(speeds) if speeds else np.array([0.0])
    return {"n_carried_linked": int(len(carried_obs)), "off_pitch": off,
            "max_speed": float(speeds.max()), "n_over_clamp": int((speeds > BALL_MAX_SPEED_MS).sum())}


def carry_xval_error(dense: pd.DataFrame, ball_imgxy: dict[int, tuple[float, float]], *,
                     window: int, jaccard_cut: float) -> np.ndarray:
    """Leave-one-out carry-over error (metres) vs the trusted own-frame projection.

    For each frame that has BOTH its own homography and a ball detection, project the ball with its
    own homography (reference) and with carry-over from neighbours with that frame's homography
    removed from the source pool. The distance is a direct faithfulness measure independent of the
    on-pitch / speed heuristics (a small error means the camera barely moved across the window, so a
    carried homography is a valid stand-in).
    """
    import cv2  # noqa: PLC0415

    players = dense[dense["role"].isin(PLAYER_ROLES)]
    homs = fit_frame_homographies(players)
    seg = camera_segment_ids(dense, jaccard_cut=jaccard_cut)
    good = np.array(sorted(homs), dtype=int)

    def _proj(h, bx, by):
        p = cv2.perspectiveTransform(np.array([[[bx, by]]], np.float32), h)[0, 0]
        return float(p[0]), float(p[1])

    errs: list[float] = []
    for f, (bx, by) in ball_imgxy.items():
        if f not in homs:
            continue
        ox, oy = _proj(homs[f], bx, by)
        pool = good[good != f]
        left, right = _nearest_sources(f, pool, seg, window=window)
        if left is None and right is None:
            continue
        if left is not None and right is not None:
            xl, yl = _proj(homs[left], bx, by)
            xr, yr = _proj(homs[right], bx, by)
            w = (f - left) / (right - left)
            cx, cy = (1 - w) * xl + w * xr, (1 - w) * yl + w * yr
        else:
            cx, cy = _proj(homs[left if right is None else right], bx, by)
        errs.append(float(np.hypot(cx - ox, cy - oy)))
    return np.array(errs) if errs else np.array([])


def run_experiment(segs: list[tuple[str, Path]], *, window: int, jaccard_cut: float,
                   interpolate: bool) -> list[dict]:
    """Project + link OFF and ON for each segment; return per-segment result dicts."""
    results: list[dict] = []
    for key, vid in segs:
        dense = pd.read_parquet(DENSE_ROOT / key / f"{vid.stem}_dense.parquet")
        n_dense = int(dense["frame"].nunique())
        fps = _true_fps(vid)
        det_df = pd.read_parquet(_detpath(key, vid.stem))
        ball_imgxy = {int(r.frame): (float(r.bx), float(r.by)) for r in det_df.itertuples(index=False)}

        off_df, off_stats = project_ball_carry(ball_imgxy, dense, carry=False)
        on_df, on_stats = project_ball_carry(
            ball_imgxy, dense, carry=True, interpolate=interpolate,
            window=window, jaccard_cut=jaccard_cut)

        off_track = link_ball(off_df[["frame", "x", "y"]], fps=fps)
        on_track = link_ball(on_df[["frame", "x", "y"]], fps=fps)
        off_cov = len(off_track) / n_dense if n_dense else 0.0
        on_cov = len(on_track) / n_dense if n_dense else 0.0
        plaus = _plausibility(on_track, set(on_stats.carried_frames), fps)
        xval = carry_xval_error(dense, ball_imgxy, window=window, jaccard_cut=jaccard_cut)

        results.append({
            "xval": xval,
            "match": key, "seg": vid.stem, "n_dense": n_dense, "fps": fps,
            "n_det": len(ball_imgxy),
            "off_proj": len(off_df), "on_proj": len(on_df),
            "off_cov": off_cov, "on_cov": on_cov,
            "own": on_stats.own, "carried": on_stats.carried, "interp": on_stats.interpolated,
            "offpitch_dropped": on_stats.offpitch_dropped, "no_source": on_stats.no_source,
            "plaus": plaus,
        })
        print(f"[exp] {key}/{vid.stem}: OFF proj={len(off_df)} cov={off_cov:.1%} | "
              f"ON proj={len(on_df)} cov={on_cov:.1%} "
              f"(own={on_stats.own} carried={on_stats.carried} interp={on_stats.interpolated} "
              f"offpitch_drop={on_stats.offpitch_dropped}) "
              f"plaus[carried_linked={plaus['n_carried_linked']} offpitch={plaus['off_pitch']} "
              f"maxv={plaus['max_speed']:.0f} over40={plaus['n_over_clamp']}]", flush=True)
    return results


# === Table ========================================================================================
def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in rows])) if rows else 0.0


def build_table(results: list[dict], *, window: int, jaccard_cut: float, interpolate: bool) -> str:
    """Render the OFF-vs-ON coverage table (per segment / match / pooled) + mechanism diagnostics."""
    lines: list[str] = []
    lines.append("PL COVERAGE LEVER -- temporal homography carry-over for the ball (v6 detections)")
    lines.append(f"window=+/-{window} frames (~{window / 25:.1f}s @25fps)  cut=track-id Jaccard<"
                 f"{jaccard_cut}  interpolate={interpolate}")
    lines.append("The ONLY success metric is post-link coverage OFF vs ON. own/carried/interp are")
    lines.append("PRE-LINK diagnostics (how many frames each mechanism projected, before linking).")
    lines.append("")
    hdr = (f"{'segment':<26}{'n_dense':>8}{'OFF cov':>9}{'ON cov':>9}{'d(pp)':>7}"
           f"{'own':>6}{'carry':>6}{'interp':>7}{'offpit':>7}")
    lines.append(hdr)
    lines.append("-" * len(hdr))

    def seg_line(r: dict) -> str:
        d = (r["on_cov"] - r["off_cov"]) * 100
        return (f"{r['match'][:8]+'/'+r['seg']:<26}{r['n_dense']:>8}{r['off_cov']:>8.1%} "
                f"{r['on_cov']:>8.1%} {d:>+6.1f}{r['own']:>6}{r['carried']:>6}{r['interp']:>7}"
                f"{r['offpitch_dropped']:>7}")

    for key in MATCH_KEYS:
        rows = [r for r in results if r["match"] == key]
        if not rows:
            continue
        for r in sorted(rows, key=lambda x: x["seg"]):
            lines.append(seg_line(r))
        off_m, on_m = _mean(rows, "off_cov"), _mean(rows, "on_cov")
        lines.append(f"{'  '+key+' (match mean)':<26}{'':>8}{off_m:>8.1%} {on_m:>8.1%} "
                     f"{(on_m - off_m) * 100:>+6.1f}")
        lines.append("")
    off_p, on_p = _mean(results, "off_cov"), _mean(results, "on_cov")
    lines.append(f"{'POOLED (mean of 9 segs)':<26}{'':>8}{off_p:>8.1%} {on_p:>8.1%} "
                 f"{(on_p - off_p) * 100:>+6.1f}")
    lines.append("")
    lines.append("Baseline check: OFF should reproduce the v6 feasibility.md column "
                 "(brighton 29.4 / fulham 19.7 / liverpool 15.2 / pooled 21.4).")
    lines.append("")
    lines.append("PLAUSIBILITY of carried ball positions (post-link, observed carried samples):")
    tot_c = sum(r["plaus"]["n_carried_linked"] for r in results)
    tot_off = sum(r["plaus"]["off_pitch"] for r in results)
    maxv = max((r["plaus"]["max_speed"] for r in results), default=0.0)
    lines.append(f"  carried samples surviving link: {tot_c}   off-pitch: {tot_off}   "
                 f"max frame-to-frame speed: {maxv:.0f} m/s (clamp {BALL_MAX_SPEED_MS:.0f})")
    lines.append("")
    lines.append("CARRY FAITHFULNESS (leave-one-out: carried vs own-frame ball position, metres):")
    allx = np.concatenate([r["xval"] for r in results if len(r["xval"])]) if any(
        len(r["xval"]) for r in results) else np.array([])
    if len(allx):
        lines.append(f"  n={len(allx)}  median={np.median(allx):.2f}m  p90={np.percentile(allx, 90):.2f}m"
                     f"  within2m={np.mean(allx <= 2):.0%}  within5m={np.mean(allx <= 5):.0%}"
                     f"  max={allx.max():.1f}m")
        lines.append("  (a sub-metre median means the camera is near-static across the window, so a")
        lines.append("   carried homography is a valid stand-in; gross outliers hit the off-pitch gate.)")
    return "\n".join(lines)


def main() -> None:
    """Driver: GPU detect (cached) -> CPU OFF/ON project+link -> table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", choices=MATCH_KEYS, default=None)
    ap.add_argument("--segment", default=None, help="restrict to one segment stem, e.g. seg_2")
    ap.add_argument("--weights", default=V6_WEIGHTS)
    ap.add_argument("--thr", type=float, default=BALL_THR)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW_FRAMES)
    ap.add_argument("--jaccard-cut", type=float, default=DEFAULT_JACCARD_CUT)
    ap.add_argument("--no-interpolate", action="store_true", help="disable bracketing-H blend")
    ap.add_argument("--skip-detect", action="store_true", help="reuse cached ball detections")
    args = ap.parse_args()

    segs = _segments(args.match, args.segment)
    if not segs:
        print("no segments found under matches/pl_probe/ for the given filter")
        return
    if not args.skip_detect:
        detect_all_frames(segs, weights=args.weights, thr=args.thr)

    interpolate = not args.no_interpolate
    results = run_experiment(segs, window=args.window, jaccard_cut=args.jaccard_cut,
                             interpolate=interpolate)
    table = build_table(results, window=args.window, jaccard_cut=args.jaccard_cut,
                        interpolate=interpolate)
    print("\n" + table)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "carry_probe.md").write_text(
        "# PL coverage lever: temporal homography carry-over\n\n```\n" + table + "\n```\n",
        encoding="utf-8")
    print(f"\nwrote {RESULTS_ROOT / 'carry_probe.md'}")


if __name__ == "__main__":
    main()
