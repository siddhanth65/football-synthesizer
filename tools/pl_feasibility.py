"""Phase A feasibility probe: run the WC CV pipeline on Premier League segments, emit the gate table.

Standalone probe for ``docs/PL_PIVOT_PLAN.md`` Phase A. For each 2-minute segment cut from the PL
full-match replays (``matches/pl_probe/<match_key>/seg_<N>.mp4``) it runs the *exact* validated WC
stages -- no reimplementation:

* **Detection + tracking** -> :func:`generator.extract.extract_positions` with the football-role
  detector + ByteTrack (the batch_match path). Reports players-per-frame, track count, fragmentation.
* **Calibration** -> :class:`generator.calibrate.PnLCalibCalibrator` per-frame. Reports the fraction of
  frames PnLCalib solves (<= 2 m gate) and the >=6-player-correspondence yield -- the norway killer.
* **Ball (zero-shot v5)** -> the fine-tuned TrackNetV2 v5 detector at 512x288 loaded via WASB
  ``build_model`` (:func:`tools.ball_possession._load_model`). Reports the fire-rate at thr=0.5 (NOT
  recall -- we have no PL ball labels) and saves annotated crops for a human precision eyeball.
* **Post-link coverage** -> per-frame player-correspondence homography projection
  (:func:`tools.ball_possession.project_ball`) + :func:`generator.ball.link_ball` at the true fps.
  This is the ONLY coverage number that counts.

Two VRAM-isolated passes (4 GB laptop discipline): pass A holds the YOLO detector + PnLCalib HRNets;
pass B holds only the v5 ball net. Never run concurrently with another GPU job or the full pytest suite.

Run (GPU job -- one at a time):
    python -m tools.pl_feasibility                       # all 3 matches, all segments
    python -m tools.pl_feasibility --match manutd_fulham # one match
    python -m tools.pl_feasibility --segment seg_2 --match manutd_fulham   # one segment (timing)
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROBE_ROOT = Path("matches/pl_probe")
RESULTS_ROOT = Path("results/pl_probe")
DENSE_ROOT = Path("outputs/pl_probe")
DEFAULT_WEIGHTS = "outputs/ball_finetuned/tracknetv2_v5.pth"
MATCH_KEYS = ("brighton_manutd", "manutd_fulham", "manutd_liverpool")
PLAYER_ROLES = ("player", "goalkeeper")
CALIB_GATE_M = 2.0        # PnLCalib reprojection-error gate (mirrors MAX_REPROJ_ERROR_M)
MIN_CORR = 6              # project_ball's min player correspondences (the norway-killer threshold)
BALL_THR = 0.5           # v5 heatmap peak threshold
N_CROPS = 40             # annotated ball crops saved per match for the precision eyeball
CROP_HALF = 150          # half-size (px) of each saved annotated crop window

# WC baselines measured from the France dense parquets (same pipeline; means over 11-13 chunks each).
# post-link coverage from STATUS.md (the exact same probe path). iraq is the gate reference match.
WC_POSTLINK = {"senegal": 0.458, "iraq": 0.359, "norway": 0.380}
WC_PPF = {"senegal": 10.9, "iraq": 7.8, "norway": 8.8}
WC_CALIB = {"senegal": 0.85, "iraq": 0.88, "norway": 0.83}
WC_GE6 = {"senegal": 0.62, "iraq": 0.41, "norway": 0.49}


# === Pure metric seams (tested without the CV / GPU stack) =======================================
def players_per_frame(players: pd.DataFrame) -> float:
    """Mean number of tracked players (+GK) per frame that has any detection."""
    if players.empty:
        return 0.0
    return float(players.groupby("frame").size().mean())


def track_stats(players: pd.DataFrame, *, sample_every: int, fps: float) -> dict:
    """Track-stability summary from a players table.

    Returns track count, mean/median track length (in sampled frames and seconds), and a
    fragmentation index = ``n_tracks / mean_players_per_frame`` (ideal ~1: one track per on-screen
    slot; higher = tracks break or ID-switch, so more tracks than bodies).
    """
    if players.empty:
        return {"n_tracks": 0, "mean_track_frames": 0.0, "median_track_frames": 0.0,
                "mean_track_s": 0.0, "frag_index": 0.0}
    per_track = players.groupby("track_id")["frame"].nunique()
    ppf = players_per_frame(players)
    mean_frames = float(per_track.mean())
    return {
        "n_tracks": int(per_track.size),
        "mean_track_frames": mean_frames,
        "median_track_frames": float(per_track.median()),
        "mean_track_s": mean_frames * sample_every / fps if fps else 0.0,
        "frag_index": (per_track.size / ppf) if ppf else 0.0,
    }


def calibration_yield(dense: pd.DataFrame) -> dict:
    """Per-frame PnLCalib yield from a dense positions table.

    * ``pct_calibrated``: fraction of processed frames whose reprojection error passes the <=2 m gate
      (``calib_error_m <= CALIB_GATE_M``) -- the frames that get trustworthy pitch coordinates.
    * ``pct_ge6_corr``: fraction of processed frames with >= :data:`MIN_CORR` players carrying valid
      pitch coordinates -- exactly what :func:`tools.ball_possession.project_ball` needs, and the
      "min-6-correspondence" yield that gated norway.
    """
    n_frames = int(dense["frame"].nunique())
    if n_frames == 0:
        return {"n_frames": 0, "pct_calibrated": 0.0, "pct_ge6_corr": 0.0}
    err_per_frame = dense.groupby("frame")["calib_error_m"].first()
    calibrated = float((err_per_frame <= CALIB_GATE_M).mean())
    players = dense[dense["role"].isin(PLAYER_ROLES)].dropna(subset=["pitch_x", "pitch_y"])
    ge6 = 0.0
    if not players.empty:
        n_ge6 = int((players.groupby("frame")["track_id"].nunique() >= MIN_CORR).sum())
        ge6 = n_ge6 / n_frames
    return {"n_frames": n_frames, "pct_calibrated": calibrated, "pct_ge6_corr": ge6}


def post_link_coverage(n_track_rows: int, n_dense_frames: int) -> float:
    """Post-``link_ball`` usable-track coverage = linked samples / distinct dense frames."""
    return n_track_rows / n_dense_frames if n_dense_frames else 0.0


# === Segment probe state =========================================================================
@dataclass
class SegResult:
    """Per-segment probe outcome."""

    match: str
    seg: str
    n_frames: int = 0
    ppf: float = 0.0
    tracks: dict = field(default_factory=dict)
    calib: dict = field(default_factory=dict)
    fps: float = 25.0
    n_sampled_ball: int = 0
    n_fired: int = 0
    proj_stats: dict = field(default_factory=dict)
    n_linked: int = 0
    n_observed: int = 0
    postlink_cov: float = 0.0

    @property
    def fire_rate(self) -> float:
        """Fraction of sampled frames where the v5 detector fired above threshold (NOT recall)."""
        return self.n_fired / self.n_sampled_ball if self.n_sampled_ball else 0.0


def _segments(match: str | None, seg: str | None) -> list[tuple[str, Path]]:
    """List ``(match_key, segment_path)`` for the requested match/segment filter."""
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


# === Pass A: detection + tracking + calibration ==================================================
def run_extraction(segs: list[tuple[str, Path]], results: dict[tuple[str, str], SegResult], *,
                   sample_every: int, calib_period: int, skip_existing: bool = True) -> None:
    """Pass A: build PnLCalib once, extract dense positions per segment, fill detection/calib stats."""
    from generator.calibrate import PnLCalibCalibrator  # noqa: PLC0415
    from generator.extract import extract_positions  # noqa: PLC0415

    DENSE_ROOT.mkdir(parents=True, exist_ok=True)
    todo = [(k, v) for k, v in segs
            if not (skip_existing and (DENSE_ROOT / k / f"{v.stem}_dense.parquet").exists())]
    calibrator = PnLCalibCalibrator() if todo else None  # only pay the HRNet load if there is work
    for key, vid in segs:
        out = DENSE_ROOT / key / f"{vid.stem}_dense.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        if skip_existing and out.exists():
            print(f"[A] reuse {key}/{vid.stem} (dense exists)", flush=True)
            dense = pd.read_parquet(out)
        else:
            print(f"[A] extract {key}/{vid.stem} -> {out.name}", flush=True)
            dense = extract_positions(
                vid, out, sample_every=sample_every, calibrator=calibrator,
                detector_name="football", tracker_name="bytetrack", calib_period=calib_period,
            )
        players = dense[dense["role"].isin(PLAYER_ROLES)]
        fps = _true_fps(vid)
        r = results[(key, vid.stem)]
        r.fps = fps
        r.n_frames = int(dense["frame"].nunique())
        r.ppf = players_per_frame(players)
        r.tracks = track_stats(players, sample_every=sample_every, fps=fps)
        r.calib = calibration_yield(dense)
        print(f"    frames={r.n_frames} ppf={r.ppf:.1f} tracks={r.tracks['n_tracks']} "
              f"frag={r.tracks['frag_index']:.2f} calib={r.calib['pct_calibrated']:.0%} "
              f">=6corr={r.calib['pct_ge6_corr']:.0%}", flush=True)
    _free_gpu()


def _free_gpu() -> None:
    import gc  # noqa: PLC0415

    try:
        import torch  # noqa: PLC0415

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# === Pass B: ball detection + projection + link + crops ==========================================
def run_ball(segs: list[tuple[str, Path]], results: dict[tuple[str, str], SegResult], *,
             weights: str, thr: float) -> None:
    """Pass B: load v5 once, run detect -> project -> link per segment, save annotated crops."""
    from generator.ball import link_ball  # noqa: PLC0415
    from tools.ball_possession import _load_model, detect_ball_imagexy, project_ball  # noqa: PLC0415

    model, dev = _load_model(weights)
    print(f"[B] loaded v5 on {dev}; weights={weights} thr={thr}", flush=True)
    # per-match store of (segment_path, {frame: (x, y)}) for the crop sampler
    per_match_dets: dict[str, list[tuple[Path, dict]]] = {k: [] for k in MATCH_KEYS}
    for key, vid in segs:
        dense_path = DENSE_ROOT / key / f"{vid.stem}_dense.parquet"
        dense = pd.read_parquet(dense_path)
        players = dense[dense["role"].isin(PLAYER_ROLES)].dropna(
            subset=["image_x", "image_y", "pitch_x", "pitch_y"])
        n_dense = int(dense["frame"].nunique())
        frames = sorted(int(f) for f in players["frame"].unique())
        r = results[(key, vid.stem)]
        if not frames:
            print(f"[B] {key}/{vid.stem}: no calibrated player frames -> skip ball", flush=True)
            continue
        ball_img = detect_ball_imagexy(str(vid), frames, model, dev, thr=thr)
        ball_pitch, stats = project_ball(ball_img, players, return_stats=True)
        track = link_ball(ball_pitch, fps=r.fps)
        r.n_sampled_ball = len(frames)
        r.n_fired = len(ball_img)
        r.proj_stats = stats
        r.n_linked = int(len(track))
        r.n_observed = int(track["observed"].sum()) if len(track) else 0
        r.postlink_cov = post_link_coverage(r.n_linked, n_dense)
        out = DENSE_ROOT / key / f"{vid.stem}_ball.parquet"
        track.to_parquet(out, index=False)
        per_match_dets[key].append((vid, ball_img))
        print(f"[B] {key}/{vid.stem}: fire {r.n_fired}/{r.n_sampled_ball} ({r.fire_rate:.0%}) "
              f"proj {stats['projected']}/{stats['detected']} link {r.n_linked} "
              f"post-link {r.postlink_cov:.1%}", flush=True)
    _free_gpu()
    for key, dets in per_match_dets.items():
        if dets:
            save_ball_crops(key, dets, n=N_CROPS)


def save_ball_crops(match: str, seg_dets: list[tuple[Path, dict]], *, n: int) -> int:
    """Save ``n`` random annotated crops (predicted ball circled) for a human precision eyeball."""
    import cv2  # noqa: PLC0415

    pool: list[tuple[Path, int, float, float]] = []
    for vid, dets in seg_dets:
        for fr, (x, y) in dets.items():
            pool.append((vid, int(fr), float(x), float(y)))
    if not pool:
        return 0
    rng = random.Random(42)
    picks = rng.sample(pool, min(n, len(pool)))
    out_dir = RESULTS_ROOT / match
    out_dir.mkdir(parents=True, exist_ok=True)
    by_video: dict[Path, list[tuple[int, float, float]]] = {}
    for vid, fr, x, y in picks:
        by_video.setdefault(vid, []).append((fr, x, y))
    saved = 0
    for vid, items in by_video.items():
        cap = cv2.VideoCapture(str(vid))
        for fr, x, y in sorted(items):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, frame = cap.read()
            if not ok:
                continue
            ix, iy = int(round(x)), int(round(y))
            cv2.circle(frame, (ix, iy), 14, (0, 0, 255), 2)
            h, w = frame.shape[:2]
            x0, y0 = max(ix - CROP_HALF, 0), max(iy - CROP_HALF, 0)
            x1, y1 = min(ix + CROP_HALF, w), min(iy + CROP_HALF, h)
            crop = frame[y0:y1, x0:x1]
            if crop.size:
                cv2.imwrite(str(out_dir / f"{vid.stem}_f{fr:06d}.png"), crop)
                saved += 1
        cap.release()
    print(f"[B] saved {saved} annotated crops -> {out_dir}", flush=True)
    return saved


# === Gate table ==================================================================================
def _agg(results: list[SegResult], attr: str) -> float:
    """Mean of a scalar SegResult attribute over segments (0.0 if none)."""
    vals = [getattr(r, attr) for r in results]
    return float(np.mean(vals)) if vals else 0.0


def _agg_calib(results: list[SegResult], key: str) -> float:
    vals = [r.calib.get(key, 0.0) for r in results if r.calib]
    return float(np.mean(vals)) if vals else 0.0


def build_gate_table(results: dict[tuple[str, str], SegResult]) -> str:
    """Render the Phase A gate table (per match + pooled) with the WC baseline column."""
    by_match: dict[str, list[SegResult]] = {k: [] for k in MATCH_KEYS}
    for (key, _), r in results.items():
        by_match[key].append(r)
    matches_present = [k for k in MATCH_KEYS if by_match[k]]
    pooled = [r for k in matches_present for r in by_match[k]]

    def col(fn) -> str:
        cells = [f"{fn(by_match[k]):>16}" for k in matches_present] + [f"{fn(pooled):>16}"]
        return "".join(cells)

    hdr = "".join(f"{k:>16}" for k in matches_present) + f"{'POOLED':>16}"
    lines: list[str] = []
    lines.append("PHASE A FEASIBILITY GATE TABLE (Man Utd PL segments; 3 segments x 2 min each)")
    lines.append("")
    lines.append(f"{'metric':<34}{hdr}{'WC baseline':>22}")
    lines.append("-" * (34 + 16 * (len(matches_present) + 1) + 22))

    def row(label: str, fn, wc: str) -> None:
        lines.append(f"{label:<34}{col(fn)}{wc:>22}")

    row("players/frame (mean)", lambda rs: f"{_agg(rs, 'ppf'):.1f}",
        f"irq {WC_PPF['iraq']:.1f}/sen {WC_PPF['senegal']:.1f}")
    row("track count (mean/seg, 2min)",
        lambda rs: f"{np.mean([r.tracks.get('n_tracks', 0) for r in rs]):.0f}", "~180/2min (WC)")
    row("track frag index (n_trk/ppf)",
        lambda rs: f"{np.mean([r.tracks.get('frag_index', 0) for r in rs]):.2f}",
        "lower=better")
    row("mean track length (s)",
        lambda rs: f"{np.mean([r.tracks.get('mean_track_s', 0) for r in rs]):.1f}", "n/a")
    row("calibrated frames (<=2m gate)",
        lambda rs: f"{_agg_calib(rs, 'pct_calibrated'):.0%}", f"irq {WC_CALIB['iraq']:.0%}")
    row(">=6-corr yield (norway killer)",
        lambda rs: f"{_agg_calib(rs, 'pct_ge6_corr'):.0%}", f">= irq {WC_GE6['iraq']:.0%}")
    row("ball fire-rate @0.5 (NOT recall)",
        lambda rs: f"{_agg(rs, 'fire_rate'):.0%}", "no direct WC eq")
    row("POST-LINK coverage (the number)",
        lambda rs: f"{_agg(rs, 'postlink_cov'):.1%}",
        f"sen {WC_POSTLINK['senegal']:.0%}/irq {WC_POSTLINK['iraq']:.0%}/"
        f"nor {WC_POSTLINK['norway']:.0%}")
    lines.append("")
    lines.append("GATES NEEDING EXTERNAL DATA (honestly pending, not proxied):")
    lines.append("  - True ball recall on ~100 hand-labelled frames: PENDING -- needs hand labels.")
    lines.append("    (fire-rate above is a precision-eyeball proxy only; see results/pl_probe/<m>/).")
    lines.append("  - FBref possession within ~5pp: PENDING -- needs a full-match process + FBref pull.")
    lines.append("")
    lines.append("PER-SEGMENT DETAIL:")
    for k in matches_present:
        for r in sorted(by_match[k], key=lambda x: x.seg):
            ps = r.proj_stats or {}
            lines.append(
                f"  {r.match}/{r.seg}: frames={r.n_frames} ppf={r.ppf:.1f} "
                f"tracks={r.tracks.get('n_tracks', 0)} frag={r.tracks.get('frag_index', 0):.2f} "
                f"calib={r.calib.get('pct_calibrated', 0):.0%} "
                f">=6corr={r.calib.get('pct_ge6_corr', 0):.0%} "
                f"fire={r.fire_rate:.0%} "
                f"proj={ps.get('projected', 0)}/{ps.get('detected', 0)} "
                f"(too_few={ps.get('too_few_pts', 0)} homog_fail={ps.get('homography_failed', 0)}) "
                f"post-link={r.postlink_cov:.1%}")
    return "\n".join(lines)


def main() -> None:
    """Cut-agnostic driver: extraction pass, ball pass, then emit the gate table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", choices=MATCH_KEYS, default=None)
    ap.add_argument("--segment", default=None, help="restrict to one segment stem, e.g. seg_2")
    ap.add_argument("--sample-every", type=int, default=5)
    ap.add_argument("--calib-period", type=int, default=1,
                    help="1 = per-frame PnLCalib (honest yield); >1 = temporal reuse")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--thr", type=float, default=BALL_THR)
    ap.add_argument("--skip-extraction", action="store_true",
                    help="reuse existing dense parquets (ball pass only)")
    args = ap.parse_args()

    segs = _segments(args.match, args.segment)
    if not segs:
        print("no segments found under matches/pl_probe/ for the given filter")
        return
    results = {(k, v.stem): SegResult(match=k, seg=v.stem) for k, v in segs}

    if not args.skip_extraction:
        run_extraction(segs, results, sample_every=args.sample_every, calib_period=args.calib_period)
    else:
        for (k, stem), r in results.items():
            dense = pd.read_parquet(DENSE_ROOT / k / f"{stem}_dense.parquet")
            players = dense[dense["role"].isin(PLAYER_ROLES)]
            vid = PROBE_ROOT / k / f"{stem}.mp4"
            r.fps = _true_fps(vid)
            r.n_frames = int(dense["frame"].nunique())
            r.ppf = players_per_frame(players)
            r.tracks = track_stats(players, sample_every=args.sample_every, fps=r.fps)
            r.calib = calibration_yield(dense)

    run_ball(segs, results, weights=args.weights, thr=args.thr)

    table = build_gate_table(results)
    print("\n" + table)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "feasibility.md").write_text(
        "# Phase A feasibility probe\n\n```\n" + table + "\n```\n", encoding="utf-8")
    print(f"\nwrote {RESULTS_ROOT / 'feasibility.md'}")


if __name__ == "__main__":
    main()
