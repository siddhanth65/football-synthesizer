"""Zero-shot SoccerNet action-spotting probe (E2E-Spot, RegNet-Y 200MF + GSM + GRU).

Runs an *off-the-shelf* SoccerNet-v2 action-spotting model (17 classes: Goal, Corner, Shots
on/off target, cards, fouls, ...) over a registered match's broadcast chunks, with **no training**,
to answer the standing reviewer attack "your report can't see goals" (``docs/REVIEW_CRIB.md`` Q1).

This is a PROBE, not pipeline integration: it changes no pipeline module, reads match chunk videos
via :mod:`core.registry` (``matches/<id>/<half>/chunk_<NNN>.mp4``), and writes per-chunk score
checkpoints under ``results/action_spotting_probe/`` so a long inference resumes from disk state.

Model provenance: E2E-Spot (Hong et al., ECCV 2022), public weights ``soccer_rny002gsm_gru_rgb``
from ``github.com/jhong93/e2e-spot-models`` (18 MB checkpoint = 17 SoccerNet-v2 classes + background).
The repo code + weights live OUTSIDE this repo (default ``~/action-spot-env``); a one-line
timm-1.0 compatibility patch to ``model/shift.py`` (``ConvBnAct`` -> duck-typed ``ConvNormAct``) is
applied there. Verified to load strict on torch 2.11 / timm 1.0, ~760 MB VRAM per 100-frame clip.

Preprocessing matches training: frames at 2 FPS, scaled to 224 px high (native 16:9 width ~398),
ImageNet-normalized RGB, no crop (config ``crop_dim: null``). Inference uses overlapping windows of
``clip_len`` frames at 50% stride, averaging the per-frame softmax.

Usage:
    python -m tools.action_spot_probe --match brighton_manutd            # run / resume inference
    python -m tools.action_spot_probe --match brighton_manutd --analyze  # aggregate + report only
    python -m tools.action_spot_probe --selftest                         # peak-picker self-check
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from core.registry import VIDEO_ROOT, get

# Sofascore event ids, keyed by registry match id (same ids as tools.bas_validate.TRUTH).
SOFASCORE_MATCH_ID = {
    "brighton_manutd": 12436888,
    "manutd_liverpool": 12436920,
    "manutd_fulham": 12436870,
    "palace_manutd": 12436962,
    "manutd_tottenham": 12436995,
    "southampton_manutd": 12436949,
    "liverpool_manutd": 12436514,
    "manutd_brighton": 12436883,
    "fulham_manutd": 12436899,
    "manutd_palace": 12436925,
    "manutd_southampton": 12436516,
    "tottenham_manutd": 12436952,
}
SOFASCORE_MATCH_DICTS = Path(
    "outputs/oracle/sofascore/match_dicts_England_Premier_League_24_25.json"
)

# SoccerNet-v2 action classes, sorted (E2E-Spot class.txt order); model output index = i + 1,
# index 0 = background. This is load_classes()'s {name: i+1} convention, verified against the repo.
SOCCERNET_V2_CLASSES = [
    "Ball out of play", "Clearance", "Corner", "Direct free-kick", "Foul", "Goal",
    "Indirect free-kick", "Kick-off", "Offside", "Penalty", "Red card", "Shots off target",
    "Shots on target", "Substitution", "Throw-in", "Yellow card", "Yellow->red card",
]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_ENV = Path.home() / "action-spot-env"
OUT_DIR = Path("results/action_spotting_probe")
FRAME_FPS = 2.0


def class_index(name: str) -> int:
    """Model output index (1..17) for a SoccerNet-v2 class name; 0 is background."""
    return SOCCERNET_V2_CLASSES.index(name) + 1


# --------------------------------------------------------------------------------------------------
# Chunk enumeration + frame extraction
# --------------------------------------------------------------------------------------------------


def list_chunks(match_id: str) -> list[tuple[str, int, Path]]:
    """Ordered ``(half, idx, path)`` for a match's chunk videos under the registry VIDEO_ROOT."""
    out: list[tuple[str, int, Path]] = []
    for half in ("h1", "h2"):
        for p in sorted((VIDEO_ROOT / match_id / half).glob("chunk_*.mp4")):
            out.append((half, int(p.stem.split("_")[1]), p))
    return out


def extract_frames(video: Path, frame_dir: Path, fps: float = FRAME_FPS) -> int:
    """Extract frames from ``video`` at ``fps``, scaled to 224 px high, into ``frame_dir``.

    Returns the number of jpg frames written.
    """
    frame_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(video),
        "-vf", f"fps={fps},scale=-2:224", "-q:v", "2", str(frame_dir / "%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return len(list(frame_dir.glob("*.jpg")))


def load_frame_tensor(frame_dir: Path):
    """Stack all jpgs in ``frame_dir`` into a normalized ``(N, 3, H, W)`` float tensor (CPU)."""
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    paths = sorted(frame_dir.glob("*.jpg"))
    imgs = []
    for p in paths:
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        imgs.append(arr.transpose(2, 0, 1))
    return torch.from_numpy(np.stack(imgs)) if imgs else torch.empty(0)


# --------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------


def build_model(env: Path):
    """Load the E2E-Spot SoccerNet model from ``env`` (spot/ code + models/ weights)."""
    import torch  # noqa: PLC0415

    sys.path.insert(0, str(env / "spot"))
    from train_e2e import E2EModel  # noqa: PLC0415

    mdir = env / "models" / "soccer_rny002gsm_gru_rgb"
    cfg = json.loads((mdir / "config.json").read_text())
    model = E2EModel(
        cfg["num_classes"] + 1, cfg["feature_arch"], cfg["temporal_arch"],
        cfg["clip_len"], cfg["modality"], device="cuda",
    )
    sd = torch.load(mdir / "checkpoint_088.pt", map_location="cuda")
    model._model.load_state_dict(sd, strict=True)
    return model, cfg["clip_len"]


def infer_chunk(model, frames, clip_len: int, stride: int, batch: int) -> np.ndarray:
    """Per-frame softmax ``(N, 18)`` for a chunk, overlap-averaging ``clip_len`` windows."""
    import torch  # noqa: PLC0415

    n = frames.shape[0]
    acc = np.zeros((n, 18), dtype=np.float32)
    cnt = np.zeros(n, dtype=np.float32)
    starts = list(range(0, max(1, n - 1), stride))
    for i in range(0, len(starts), batch):
        clips, spans = [], []
        for s in starts[i : i + batch]:
            e = min(s + clip_len, n)
            clips.append(frames[s:e])
            spans.append((s, e))
        # pad clips in a batch to the same length (GSM undoes trailing pad internally)
        maxlen = max(c.shape[0] for c in clips)
        padded = torch.zeros((len(clips), maxlen, *clips[0].shape[1:]), dtype=torch.float32)
        for j, c in enumerate(clips):
            padded[j, : c.shape[0]] = c
        _, scores = model.predict(padded.cuda())  # (B, maxlen, 18)
        for j, (s, e) in enumerate(spans):
            acc[s:e] += scores[j, : e - s]
            cnt[s:e] += 1.0
    cnt[cnt == 0] = 1.0
    return acc / cnt[:, None]


# --------------------------------------------------------------------------------------------------
# Inference driver (resumable)
# --------------------------------------------------------------------------------------------------


def run_inference(match_id: str, env: Path, stride: int, batch: int) -> None:
    """Extract frames + infer each chunk, checkpointing scores to disk (skips done chunks)."""
    out = OUT_DIR / match_id
    out.mkdir(parents=True, exist_ok=True)
    chunks = list_chunks(match_id)
    print(f"[probe] {match_id}: {len(chunks)} chunks")
    model = clip_len = None
    for half, idx, video in chunks:
        npz = out / f"scores_{half}_chunk{idx:03d}.npz"
        if npz.exists():
            print(f"[skip] {npz.name} exists")
            continue
        if model is None:
            model, clip_len = build_model(env)
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            fdir = Path(tmp)
            nf = extract_frames(video, fdir)
            frames = load_frame_tensor(fdir)
        scores = infer_chunk(model, frames, clip_len, stride, batch)
        np.savez_compressed(
            npz, scores=scores.astype(np.float16), n_frames=nf, fps=FRAME_FPS,
            half=half, idx=idx,
        )
        print(f"[done] {npz.name}: {nf} frames in {time.time() - t0:.1f}s")
    print("[probe] inference complete")


# --------------------------------------------------------------------------------------------------
# Peak picking + analysis
# --------------------------------------------------------------------------------------------------


def find_peaks(signal: np.ndarray, thresh: float, min_sep: int) -> list[tuple[int, float]]:
    """Greedy NMS peaks: descending score, suppressing anything within ``min_sep`` samples.

    Args:
        signal: 1-D per-frame class probability.
        thresh: minimum peak score.
        min_sep: minimum separation between kept peaks, in samples.

    Returns:
        ``(index, score)`` kept peaks, sorted by index.
    """
    order = np.argsort(signal)[::-1]
    kept: list[tuple[int, float]] = []
    taken = np.zeros(signal.shape[0], dtype=bool)
    for i in order:
        if signal[i] < thresh:
            break
        if taken[i]:
            continue
        kept.append((int(i), float(signal[i])))
        lo, hi = max(0, i - min_sep), min(signal.shape[0], i + min_sep + 1)
        taken[lo:hi] = True
    return sorted(kept)


def load_half_timeline(match_id: str, half: str) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate a half's chunk scores in order. Returns ``(scores(F,18), time_s(F))``."""
    out = OUT_DIR / match_id
    npzs = sorted(out.glob(f"scores_{half}_chunk*.npz"))
    all_scores, all_times, offset = [], [], 0.0
    for p in npzs:
        d = np.load(p)
        sc = d["scores"].astype(np.float32)
        fps = float(d["fps"])
        t = offset + np.arange(sc.shape[0]) / fps
        all_scores.append(sc)
        all_times.append(t)
        offset = t[-1] + 1.0 / fps if sc.shape[0] else offset
    if not all_scores:
        return np.empty((0, 18)), np.empty(0)
    return np.concatenate(all_scores), np.concatenate(all_times)


def _fmt(t: float) -> str:
    """Seconds -> mm:ss."""
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def analyze(match_id: str, thresh: float, min_sep_s: float) -> dict:
    """Aggregate per-half timelines into per-class detected counts + goal-peak analysis."""
    halves = {h: load_half_timeline(match_id, h) for h in ("h1", "h2")}
    min_sep = int(round(min_sep_s * FRAME_FPS))
    per_class: dict[str, dict] = {}
    for name in SOCCERNET_V2_CLASSES:
        ci = class_index(name)
        total, byhalf = 0, {}
        for h, (sc, _t) in halves.items():
            if sc.shape[0] == 0:
                byhalf[h] = 0
                continue
            peaks = find_peaks(sc[:, ci], thresh, min_sep)
            byhalf[h] = len(peaks)
            total += len(peaks)
        per_class[name] = {"count": total, "by_half": byhalf}

    # Goal-specific: top-K peaks across the whole match with timestamps.
    goal_peaks = []
    for h, (sc, t) in halves.items():
        if sc.shape[0] == 0:
            continue
        for idx, score in find_peaks(sc[:, class_index("Goal")], 0.01, min_sep):
            goal_peaks.append({"half": h, "t": _fmt(t[idx]), "t_s": round(float(t[idx]), 1),
                               "score": round(score, 4)})
    goal_peaks.sort(key=lambda x: -x["score"])
    return {"per_class": per_class, "goal_peaks_top": goal_peaks[:12],
            "thresh": thresh, "min_sep_s": min_sep_s}


# Oracle (Sofascore, brighton_manutd match 12436888) whole-match counts for the non-goal mappable
# classes. Only validated for that match; other matches show "-" for these rows.
ORACLE_COUNTS = {
    "Corner": 8, "Shots on target": 9, "Shots off target": 11,
    "Yellow card": 3, "Red card": 0, "Foul": 22, "Offside": 6,
}


def goal_oracle(match_id: str) -> dict[str, int] | None:
    """Per-half goal counts for ``match_id`` from the cached Sofascore match-dicts JSON.

    Returns ``{"total": n, "h1": n, "h2": n}`` or ``None`` if the match/file isn't resolvable.
    """
    event_id = SOFASCORE_MATCH_ID.get(match_id)
    if event_id is None or not SOFASCORE_MATCH_DICTS.exists():
        return None
    matches = json.loads(SOFASCORE_MATCH_DICTS.read_text(encoding="utf-8"))
    m = next((x for x in matches if x["id"] == event_id), None)
    if m is None:
        return None
    home, away = m["homeScore"], m["awayScore"]
    h1 = home["period1"] + away["period1"]
    h2 = home["period2"] + away["period2"]
    return {"total": h1 + h2, "h1": h1, "h2": h2}


def print_report(match_id: str, res: dict) -> None:
    """ASCII per-class table (detected vs oracle) + goal-peak list."""
    print(f"\n=== action-spot probe: {match_id} "
          f"(thresh={res['thresh']}, min_sep={res['min_sep_s']}s) ===")
    goracle = goal_oracle(match_id)
    print(f"{'class':<20}{'detected':>9}{'oracle':>8}  by_half")
    for name in SOCCERNET_V2_CLASSES:
        pc = res["per_class"][name]
        orc = goracle["total"] if (name == "Goal" and goracle) else ORACLE_COUNTS.get(name, "-")
        bh = pc["by_half"]
        print(f"{name:<20}{pc['count']:>9}{str(orc):>8}  "
              f"h1={bh.get('h1', 0)} h2={bh.get('h2', 0)}")
    if goracle:
        print(f"\nGoal top peaks (oracle: {goracle['total']} goals, "
              f"{goracle['h1']} in H1, {goracle['h2']} in H2):")
    else:
        print("\nGoal top peaks (oracle: unknown - no Sofascore match id mapped):")
    for g in res["goal_peaks_top"]:
        print(f"  {g['half']} {g['t']}  score={g['score']}")


# --------------------------------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------------------------------


def _selftest() -> None:
    """Peak-picker sanity: two well-separated bumps -> two peaks; NMS collapses a close pair."""
    sig = np.zeros(200, dtype=np.float32)
    sig[50] = 0.9
    sig[52] = 0.8   # within min_sep of the first -> suppressed
    sig[150] = 0.7
    peaks = find_peaks(sig, thresh=0.5, min_sep=10)
    assert [p[0] for p in peaks] == [50, 150], peaks
    assert find_peaks(sig, thresh=0.95, min_sep=10) == [], "threshold gate failed"
    print("selftest OK")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", default="brighton_manutd")
    ap.add_argument("--env", type=Path, default=DEFAULT_ENV)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--thresh", type=float, default=0.30)
    ap.add_argument("--min-sep-s", type=float, default=30.0)
    ap.add_argument("--analyze", action="store_true",
                     help="ensure inference results exist (resumable), then aggregate + report")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    get(args.match)  # validate the match id against the registry
    # Inference is resumable per chunk (existing npz's are skipped), so both the default flow
    # and --analyze run it: one command per match either way, cached matches do no extra work.
    run_inference(args.match, args.env, args.stride, args.batch)
    res = analyze(args.match, args.thresh, args.min_sep_s)
    print_report(args.match, res)
    out_dir = OUT_DIR / args.match
    out_dir.mkdir(parents=True, exist_ok=True)
    summ = out_dir / "summary.json"
    summ.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n[probe] wrote {summ}")


if __name__ == "__main__":
    main()
