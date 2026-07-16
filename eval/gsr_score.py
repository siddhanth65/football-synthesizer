"""Score our CV pipeline with the official GS-HOTA metric on SoccerNet-GSR.

Game State Reconstruction (GSR) is the academic formalisation of our exact problem: a single moving
broadcast camera -> 2D pitch positions + role + team + jersey for every person. The task ships an
official metric, **GS-HOTA** (arXiv:2404.11335), implemented in the SoccerNet ``sn-trackeval`` fork
(``trackeval.datasets.SoccerNetGS`` + the stock ``HOTA``/``Identity`` metrics). We **never**
reimplement the metric: this module only (1) runs our existing pipeline
(:func:`generator.extract.extract_positions`: football-YOLO detect -> ByteTrack/BoT-SORT ->
PnLCalib calibrate -> project; CIELAB-KMeans team) on the GSR frame sequences, (2) converts the
positions table into the GSR prediction JSON the evaluator reads, and (3) drives
``trackeval.SoccerNetGS`` to produce GS-HOTA / GS-DetA / GS-AssA / IDF1.

GSR similarity is a Gaussian on the projected pitch bottom-middle point (sigma from a 5 m distance
tolerance), and an attribute mismatch (role AND team AND jersey) zeroes the similarity. Our pipeline
has **no jersey-number model yet**, so predictions carry ``jersey = null``: under the official
jersey-on config every GT player with a labelled number is therefore unmatchable -- exactly the
pre-Layer-2 baseline this benchmark is meant to expose. We report that honest headline plus
attribute ablations (jersey-off, team-off, role-off) so localization/association capability is
visible separately from the identity gap.

Coordinate convention: our pipeline emits the project's uncentred ``[0,105] x [0,68]`` metres frame;
GSR ground truth is PnLCalib's **centred** ``[-52.5,52.5] x [-34,34]`` frame, so we shift by
``(-PITCH_LEN/2, -PITCH_WID/2)``. Team ``0/1`` from unsupervised KMeans has no inherent left/right
meaning, so per sequence we resolve the two-way label permutation against GT (nearest match within
the tolerance), the standard resolution of an arbitrary cluster labelling -- documented, not tuned.

CLI::

    python -m eval.gsr_score --limit 3                 # pilot: 3 sequences end-to-end
    python -m eval.gsr_score                            # full valid split, resumable
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from core.pitch import PITCH_LEN, PITCH_WID

logger = logging.getLogger("gsr_score")

#: Shift from the project's uncentred pitch frame to the GSR centred frame (metres).
CENTRE_SHIFT_X = PITCH_LEN / 2.0
CENTRE_SHIFT_Y = PITCH_WID / 2.0

#: GSR distance tolerance (metres) used both by the metric (sigma) and our team-map resolution.
GSR_DIST_TOL_M = 5.0

#: GSR category ids (mirror the dataset's ``categories`` so predictions look native).
_ROLE_CATEGORY_ID = {"player": 1, "goalkeeper": 2, "referee": 3, "other": 7}

#: Default data / output locations (outputs stay lean, under the sanctioned dirs).
DEFAULT_DATA_DIR = Path("data/soccernet/gamestate-2024")
DEFAULT_OUT_DIR = Path("outputs/gsr")
DEFAULT_RESULTS_DIR = Path("results/gsr_benchmark")

#: Attribute configs reported by the benchmark (name -> trackeval USE_* flags).
EVAL_CONFIGS: dict[str, dict[str, bool]] = {
    "gs_hota_full": {"USE_ROLES": True, "USE_TEAMS": True, "USE_JERSEY_NUMBERS": True},
    "no_jersey": {"USE_ROLES": True, "USE_TEAMS": True, "USE_JERSEY_NUMBERS": False},
    "role_only": {"USE_ROLES": True, "USE_TEAMS": False, "USE_JERSEY_NUMBERS": False},
    "loc_assoc": {"USE_ROLES": False, "USE_TEAMS": False, "USE_JERSEY_NUMBERS": False},
}


# === Pure adapter seam (tested without the CV / metric stack) =====================================
def to_centred(pitch_x: float, pitch_y: float) -> tuple[float, float]:
    """Convert an uncentred ``[0,105]x[0,68]`` foot point to the GSR centred frame (metres)."""
    return pitch_x - CENTRE_SHIFT_X, pitch_y - CENTRE_SHIFT_Y


def _team_side(team: int, role: str, team_map: dict[int, str]) -> str | None:
    """Map our integer team to a GSR side (``left``/``right``), or ``None`` for non-team roles."""
    if role not in {"player", "goalkeeper"}:
        return None
    return team_map.get(int(team))


def row_to_prediction(row: dict, image_id: str, team_map: dict[int, str]) -> dict | None:
    """Convert one positions row to a GSR prediction dict, or ``None`` if it should be dropped.

    Rows without a trustworthy projected pitch position (NaN, gated out) or the ball are dropped;
    everything else becomes a person prediction with our known attributes (role, team) and a null
    jersey (no jersey model yet).

    Args:
        row: One positions row (keys ``role, team, track_id, pitch_x, pitch_y, image_x, image_y,
            conf``).
        image_id: The GSR ``image_id`` string of this frame.
        team_map: Mapping from our integer team ``{0,1}`` to a GSR side ``{"left","right"}``.

    Returns:
        A GSR prediction dict, or ``None`` when the row carries no usable pitch position.
    """
    role = str(row["role"])
    if role == "ball":
        return None
    px, py = row.get("pitch_x"), row.get("pitch_y")
    if px is None or py is None or not np.isfinite(px) or not np.isfinite(py):
        return None
    cx, cy = to_centred(float(px), float(py))
    track_id = int(row["track_id"])
    side = _team_side(row.get("team", -1), role, team_map)
    conf = float(row.get("conf", 1.0)) if np.isfinite(row.get("conf", 1.0)) else 1.0
    img_x = float(row.get("image_x", 0.0) or 0.0)
    img_y = float(row.get("image_y", 0.0) or 0.0)
    return {
        "id": f"{image_id}{track_id:04d}",
        "image_id": image_id,
        "track_id": track_id,
        "supercategory": "object",
        "category_id": _ROLE_CATEGORY_ID.get(role, 7),
        "attributes": {"role": role, "jersey": None, "team": side},
        "bbox_pitch": {
            "x_bottom_left": cx, "y_bottom_left": cy,
            "x_bottom_middle": cx, "y_bottom_middle": cy,
            "x_bottom_right": cx, "y_bottom_right": cy,
        },
        "bbox_image": {
            "x": img_x - 8, "y": img_y - 40, "x_center": img_x, "y_center": img_y - 20,
            "w": 16, "h": 40,
        },
        "confidence": conf,
    }


def build_submission(
    df: pd.DataFrame, image_id_by_frame: dict[int, str], team_map: dict[int, str]
) -> dict:
    """Build the GSR prediction JSON payload for one sequence from a positions table (pure).

    Args:
        df: Positions table (:data:`generator.extract.POSITIONS_COLUMNS`).
        image_id_by_frame: Maps our processed frame index to the GSR ``image_id`` string.
        team_map: Our integer team -> GSR side mapping for this sequence.

    Returns:
        ``{"predictions": [...]}`` with one entry per usable person detection.
    """
    preds: list[dict] = []
    for row in df.to_dict("records"):
        image_id = image_id_by_frame.get(int(row["frame"]))
        if image_id is None:
            continue
        pred = row_to_prediction(row, image_id, team_map)
        if pred is not None:
            preds.append(pred)
    return {"predictions": preds}


# === GT helpers ==================================================================================
def load_image_id_map(seq_dir: Path) -> dict[int, str]:
    """Map processed frame index (``000001.jpg`` -> 0) to the GSR ``image_id`` for a sequence."""
    gt = json.loads((seq_dir / "Labels-GameState.json").read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    for img in gt["images"]:
        idx = int(Path(img["file_name"]).stem) - 1  # 000001.jpg -> frame index 0
        out[idx] = img["image_id"]
    return out


def load_gt_people_by_frame(seq_dir: Path) -> dict[int, list[tuple[float, float, str, str]]]:
    """Load GT player/GK positions per frame index as ``(cx, cy, role, team_side)`` (centred m)."""
    gt = json.loads((seq_dir / "Labels-GameState.json").read_text(encoding="utf-8"))
    id_to_idx = {img["image_id"]: int(Path(img["file_name"]).stem) - 1 for img in gt["images"]}
    out: dict[int, list[tuple[float, float, str, str]]] = defaultdict(list)
    for ann in gt["annotations"]:
        if ann.get("supercategory") != "object":
            continue
        attrs = ann.get("attributes") or {}
        role = attrs.get("role")
        if role not in {"player", "goalkeeper"}:
            continue
        bp = ann.get("bbox_pitch")
        if not bp:
            continue
        idx = id_to_idx.get(ann["image_id"])
        if idx is None:
            continue
        out[idx].append((bp["x_bottom_middle"], bp["y_bottom_middle"], role, attrs.get("team")))
    return out


def resolve_team_map(
    df: pd.DataFrame, gt_people: dict[int, list[tuple[float, float, str, str]]]
) -> dict[int, str]:
    """Resolve our team ``{0,1}`` -> GSR side by agreement with GT among position-matched people.

    For every frame each of our player/GK detections is matched to the nearest GT player/GK within
    :data:`GSR_DIST_TOL_M`; the two-way label permutation that maximises team agreement over all
    matches is returned. This is the standard resolution of an arbitrary unsupervised cluster
    labelling -- disclosed, not fitted to the metric.

    Returns:
        ``{0: side0, 1: side1}`` with the two sides distinct; defaults to ``{0:'left',1:'right'}``
        when there is no matched evidence.
    """
    # counts[our_team][gt_side] = number of matched detections
    counts: dict[int, dict[str, int]] = {0: defaultdict(int), 1: defaultdict(int)}
    people = df[df["role"].isin(["player", "goalkeeper"])]
    for frame, grp in people.groupby("frame"):
        gts = gt_people.get(int(frame), [])
        if not gts:
            continue
        gt_xy = np.array([[g[0], g[1]] for g in gts])
        for r in grp.itertuples():
            if not (np.isfinite(r.pitch_x) and np.isfinite(r.pitch_y)):
                continue
            cx, cy = to_centred(float(r.pitch_x), float(r.pitch_y))
            d = np.hypot(gt_xy[:, 0] - cx, gt_xy[:, 1] - cy)
            j = int(np.argmin(d))
            if d[j] > GSR_DIST_TOL_M:
                continue
            gt_side = gts[j][3]
            our_team = int(r.team)
            if our_team in counts and gt_side in {"left", "right"}:
                counts[our_team][gt_side] += 1
    direct = counts[0]["left"] + counts[1]["right"]
    swapped = counts[0]["right"] + counts[1]["left"]
    if swapped > direct:
        return {0: "right", 1: "left"}
    return {0: "left", 1: "right"}


# === Pipeline extraction (GPU stage; resumable) ==================================================
def extract_sequence(
    seq_dir: Path, out_parquet: Path, calibrator, *, calib_period: int, detector: str,
    tracker: str, force: bool = False,
) -> pd.DataFrame:
    """Run our CV pipeline over a GSR frame sequence, writing/reusing a positions parquet.

    Frames are fed to :func:`generator.extract.extract_positions` through OpenCV's native
    image-sequence reader (``img1/%06d.jpg``); ``sample_every=1`` so every GT frame gets a
    prediction (needed for a fair GS-DetA). ``force`` overwrites an existing parquet (used to repair
    a stale/NaN-only cache).
    """
    if out_parquet.exists() and not force:
        return pd.read_parquet(out_parquet)
    from generator.extract import extract_positions  # noqa: PLC0415

    pattern = str(seq_dir / "img1" / "%06d.jpg")
    df = extract_positions(
        pattern, out_parquet, sample_every=1, calibrator=calibrator,
        detector_name=detector, tracker_name=tracker, calib_period=calib_period,
    )
    return df


# === Metric driver (official trackeval; never reimplemented) =====================================
def _run_trackeval(
    gt_dir: Path, trackers_dir: Path, tracker: str, seq_info: dict[str, int], cfg: dict[str, bool]
) -> dict:
    """Drive ``trackeval.SoccerNetGS`` over ``seq_info`` and return the raw ``output_res`` tree."""
    import trackeval  # noqa: PLC0415

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False,
        "TIME_PROGRESS": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False, "OUTPUT_EMPTY_CLASSES": False,
    })
    dataset_config = trackeval.datasets.SoccerNetGS.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(gt_dir), "TRACKERS_FOLDER": str(trackers_dir),
        "TRACKERS_TO_EVAL": [tracker], "TRACKER_SUB_FOLDER": "data",
        "SKIP_SPLIT_FOL": True, "SEQ_INFO": seq_info, "PRINT_CONFIG": False,
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/Labels-GameState.json",
        "EVAL_MODE": "distance", "EVAL_SPACE": "pitch", "EVAL_SIMILARITY_METRIC": "gaussian",
        "EVAL_DIST_TOL": GSR_DIST_TOL_M, **cfg,
    })
    metrics_config = {"METRICS": ["HOTA", "Identity"], "THRESHOLD": 0.5}
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.SoccerNetGS(dataset_config)
    metrics = [m(metrics_config) for m in (trackeval.metrics.HOTA, trackeval.metrics.Identity)]
    output_res, _ = evaluator.evaluate([dataset], metrics)
    return output_res["SoccerNetGS"][tracker]


def _scalar(node: dict, metric: str, field: str) -> float:
    """Mean of a (possibly per-alpha) metric field, as a percentage."""
    val = node[metric][field]
    return float(np.mean(val)) * 100.0


def summarise_res(res: dict, seq_list: list[str]) -> dict:
    """Extract combined + per-sequence GS-HOTA/DetA/AssA/IDF1 (percentages) from ``output_res``."""
    def pack(node: dict) -> dict:
        return {
            "GS-HOTA": _scalar(node, "HOTA", "HOTA"),
            "GS-DetA": _scalar(node, "HOTA", "DetA"),
            "GS-AssA": _scalar(node, "HOTA", "AssA"),
            "GS-LocA": _scalar(node, "HOTA", "LocA"),
            "IDF1": _scalar(node, "Identity", "IDF1"),
        }

    combined = pack(res["COMBINED_SEQ"]["person"])
    per_seq = {seq: pack(res[seq]["person"]) for seq in seq_list if seq in res}
    return {"combined": combined, "per_seq": per_seq}


def gs_hota(predictions: str | Path, ground_truth: str | Path, *, split: str = "valid",
            seq_info: dict[str, int] | None = None, **cfg: bool) -> dict:
    """Return GS-HOTA (and DetA/AssA/LocA/IDF1) for a folder of GSR predictions (official metric).

    The real entry point replacing the former stub. ``predictions`` is a trackers folder laid out as
    ``<predictions>/<tracker>/data/<seq>.json``; ``ground_truth`` holds ``<seq>/Labels-GameState``.

    Args:
        predictions: Trackers folder (contains a single ``predictions`` tracker subfolder).
        ground_truth: GSR ground-truth folder (sequence dirs with ``Labels-GameState.json``).
        split: Split label (metadata only; sequences are chosen by ``seq_info``).
        seq_info: ``{seq_name: n_frames}`` to evaluate; defaults to every sequence found.
        **cfg: ``USE_ROLES`` / ``USE_TEAMS`` / ``USE_JERSEY_NUMBERS`` overrides.

    Returns:
        ``{"combined": {...}, "per_seq": {...}}`` of metric percentages.
    """
    gt_dir, pred_dir = Path(ground_truth), Path(predictions)
    if seq_info is None:
        data = pred_dir / "predictions" / "data"
        seq_info = {p.stem: 0 for p in sorted(data.glob("*.json"))}
    res = _run_trackeval(gt_dir, pred_dir, "predictions", seq_info, cfg)
    return summarise_res(res, list(seq_info.keys()))


# === Orchestration ================================================================================
def _write_submission(df: pd.DataFrame, seq_dir: Path, dest: Path) -> dict[int, str]:
    """Resolve the team map, build and write one sequence's prediction JSON. Returns the team map."""
    image_ids = load_image_id_map(seq_dir)
    gt_people = load_gt_people_by_frame(seq_dir)
    team_map = resolve_team_map(df, gt_people)
    submission = build_submission(df, image_ids, team_map)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(submission), encoding="utf-8")
    return team_map


def run_benchmark(
    data_dir: Path, out_dir: Path, results_dir: Path, *, limit: int | None, calib_period: int,
    detector: str, tracker: str, skip_extract: bool = False,
) -> dict:
    """Run the full GSR benchmark: extract -> submissions -> multi-config GS-HOTA + write results.

    Resumable: per-sequence positions parquets and prediction JSONs are reused when present, so an
    interrupted run continues where it stopped.
    """
    seqs = sorted(p for p in data_dir.iterdir()
                  if p.is_dir() and (p / "Labels-GameState.json").exists())
    if limit:
        seqs = seqs[:limit]
    if not seqs:
        raise SystemExit(f"no GSR sequences under {data_dir}")
    logger.info("GSR benchmark: %d sequences, calib_period=%d, det=%s track=%s",
                len(seqs), calib_period, detector, tracker)

    pos_dir = out_dir / "positions"
    pred_data_dir = out_dir / "eval" / "predictions" / "data"
    pred_data_dir.mkdir(parents=True, exist_ok=True)
    team_maps: dict[str, dict[int, str]] = {}

    calibrator = None
    if not skip_extract:
        from generator.calibrate import PnLCalibCalibrator  # noqa: PLC0415

        calibrator = PnLCalibCalibrator()  # built once, reused across sequences

    seq_info: dict[str, int] = {}
    for i, seq_dir in enumerate(seqs):
        name = seq_dir.name
        seq_info[name] = len(load_image_id_map(seq_dir))
        pred_json = pred_data_dir / f"{name}.json"
        if pred_json.exists() and (pos_dir / f"{name}.parquet").exists():
            logger.info("[%d/%d] %s: reuse existing submission", i + 1, len(seqs), name)
            continue
        if skip_extract:  # score-only: never run the GPU stage, just skip unprocessed sequences
            logger.info("[%d/%d] %s: no submission, skipping (score-only)", i + 1, len(seqs), name)
            continue
        t0 = time.time()
        df = extract_sequence(seq_dir, pos_dir / f"{name}.parquet", calibrator,
                              calib_period=calib_period, detector=detector, tracker=tracker)
        team_map = _write_submission(df, seq_dir, pred_json)
        team_maps[name] = team_map
        valid = int(df.dropna(subset=["pitch_x"]).shape[0]) if len(df) else 0
        logger.info("[%d/%d] %s: %d rows (%d w/ pitch), team_map=%s (%.0fs)",
                    i + 1, len(seqs), name, len(df), valid, team_map, time.time() - t0)

    # Score every attribute configuration over the sequences that actually have a submission
    # (keeps a partial/resumed run scorable instead of erroring on a missing tracker file).
    scored = {name: n for name, n in seq_info.items() if (pred_data_dir / f"{name}.json").exists()}
    results: dict[str, dict] = {}
    for cfg_name, cfg in EVAL_CONFIGS.items():
        logger.info("scoring config '%s' (%s) over %d sequences", cfg_name, cfg, len(scored))
        results[cfg_name] = gs_hota(out_dir / "eval", data_dir, seq_info=scored, **cfg)

    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_sequences": len(scored), "n_target": len(seqs), "calib_period": calib_period,
        "detector": detector, "tracker": tracker, "dist_tol_m": GSR_DIST_TOL_M,
        "team_maps": team_maps, "configs": results,
    }
    (results_dir / "gsr_scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for cfg_name, r in results.items():
        c = r["combined"]
        logger.info("== %-12s GS-HOTA %.2f  DetA %.2f  AssA %.2f  LocA %.2f  IDF1 %.2f",
                    cfg_name, c["GS-HOTA"], c["GS-DetA"], c["GS-AssA"], c["GS-LocA"], c["IDF1"])
    return payload


# === Stage 2a: post-hoc track re-linking (appearance ReID) =======================================
#: Pilot sequences the re-link threshold is tuned on, then FROZEN before touching the other 55.
PILOT_SEQS = ("SNGS-021", "SNGS-022", "SNGS-023")


def _relink_and_write(
    seqs: list[Path], data_dir: Path, out_dir: Path, pos_dir: Path, pred_out: Path, *,
    params, audit_seqs: set[str], build_tools: bool,
) -> dict[str, dict]:
    """Relink every sequence's fragments and write relinked submissions -> per-seq stats.

    Embeddings are cached under ``out_dir/relink_cache`` (resumable); the OSNet embedder + football
    detector are built once, and only if some cache is cold (``build_tools``).
    """
    from generator.track_relink import OsnetEmbedder, relink_sequence  # noqa: PLC0415

    cache_dir = out_dir / "relink_cache"
    embedder = detector = None
    if build_tools:
        import torch  # noqa: PLC0415

        from generator.extract import _build_detector  # noqa: PLC0415

        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedder = OsnetEmbedder(params.model_name, device=device, weights=params.weights)
        detector = _build_detector(device, "football")
    pred_out.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}
    for i, seq_dir in enumerate(seqs):
        name = seq_dir.name
        parquet = pos_dir / f"{name}.parquet"
        if not parquet.exists():
            continue
        df = pd.read_parquet(parquet)
        relinked, st = relink_sequence(
            seq_dir, df, params=params, embedder=embedder, detector=detector,
            cache_path=cache_dir / f"{name}.npz", audit=name in audit_seqs)
        _write_submission(relinked, seq_dir, pred_out / f"{name}.json")
        stats[name] = st
        logger.info("[%d/%d] %s: frags %d -> %d (%d merges, %d embedded, %.1f crops/frag)%s",
                    i + 1, len(seqs), name, st["n_fragments_before"], st["n_fragments_after"],
                    st["n_merges"], st["n_embedded"], st["mean_crops"],
                    (f", merge-prec {st.get('merge_precision_correct')}/"
                     f"{st.get('merge_precision_total')}" if name in audit_seqs else ""))
    return stats


def _caches_cold(seqs: list[Path], out_dir: Path) -> bool:
    """True if any sequence lacks a cached embedding (so the GPU tools must be built)."""
    cache_dir = out_dir / "relink_cache"
    return any(not (cache_dir / f"{p.name}.npz").exists() for p in seqs)


def run_relink_benchmark(
    data_dir: Path, out_dir: Path, results_dir: Path, *, limit: int | None, threshold: float,
) -> dict:
    """Stage 2a: relink fragments, re-score the SAME sequences, write ``gsr_scores_relink.json``.

    Never overwrites the baseline submissions or ``gsr_scores.json``: relinked submissions go to
    ``out_dir/eval_relink`` and scores to ``results_dir/gsr_scores_relink.json``.
    """
    from generator.track_relink import RelinkParams  # noqa: PLC0415

    pos_dir = out_dir / "positions"
    seqs = sorted(p for p in data_dir.iterdir()
                  if p.is_dir() and (pos_dir / f"{p.name}.parquet").exists())
    if limit:
        seqs = seqs[:limit]
    params = RelinkParams(threshold=threshold)
    logger.info("relink benchmark: %d sequences, threshold=%.3f", len(seqs), threshold)
    pred_out = out_dir / "eval_relink"
    stats = _relink_and_write(
        seqs, data_dir, out_dir, pos_dir, pred_out / "predictions" / "data",
        params=params, audit_seqs=set(PILOT_SEQS), build_tools=_caches_cold(seqs, out_dir))

    scored = {p.name: 0 for p in seqs if (pred_out / "predictions" / "data" / f"{p.name}.json").exists()}
    after: dict[str, dict] = {}
    for cfg_name, cfg in EVAL_CONFIGS.items():
        logger.info("scoring relinked config '%s' over %d sequences", cfg_name, len(scored))
        after[cfg_name] = gs_hota(pred_out, data_dir, seq_info=scored, **cfg)

    baseline = json.loads((results_dir / "gsr_scores.json").read_text(encoding="utf-8"))["configs"]
    frags_before = float(np.mean([s["n_fragments_before"] for s in stats.values()]))
    frags_after = float(np.mean([s["n_fragments_after"] for s in stats.values()]))
    pilot_prec = _pilot_precision(stats)
    payload = {
        "n_sequences": len(scored), "threshold": threshold, "params": vars(params),
        "mean_fragments_before": frags_before, "mean_fragments_after": frags_after,
        "pilot_merge_precision": pilot_prec, "per_seq_stats": stats,
        "before": {k: v["combined"] for k, v in baseline.items()},
        "after": {k: v["combined"] for k, v in after.items()},
        "after_per_seq": {k: v["per_seq"] for k, v in after.items()},
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "gsr_scores_relink.json").write_text(json.dumps(payload, indent=2),
                                                        encoding="utf-8")
    for cfg_name in EVAL_CONFIGS:
        b, a = baseline[cfg_name]["combined"], after[cfg_name]["combined"]
        logger.info("== %-12s AssA %.2f -> %.2f (%+.2f)  HOTA %.2f -> %.2f  DetA %.2f -> %.2f",
                    cfg_name, b["GS-AssA"], a["GS-AssA"], a["GS-AssA"] - b["GS-AssA"],
                    b["GS-HOTA"], a["GS-HOTA"], b["GS-DetA"], a["GS-DetA"])
    logger.info("fragments/seq %.0f -> %.0f; pilot merge precision %d/%d = %.1f%%",
                frags_before, frags_after, pilot_prec[0], pilot_prec[1],
                100.0 * pilot_prec[0] / max(pilot_prec[1], 1))
    return payload


def _pilot_precision(stats: dict[str, dict]) -> tuple[int, int]:
    """Sum (correct, total) auditable merge pairs over the pilot sequences."""
    c = sum(stats[s].get("merge_precision_correct", 0) for s in PILOT_SEQS if s in stats)
    t = sum(stats[s].get("merge_precision_total", 0) for s in PILOT_SEQS if s in stats)
    return c, t


def tune_relink_threshold(
    data_dir: Path, out_dir: Path, thresholds: list[float],
) -> None:
    """Sweep re-link thresholds on the 3 pilot sequences: report loc_assoc AssA + merge precision.

    Embeddings are computed once (cached), so the sweep is CPU-only. Prints a table to pick and
    freeze the threshold before applying it to the full split.
    """
    from generator.track_relink import RelinkParams  # noqa: PLC0415

    pos_dir = out_dir / "positions"
    seqs = [data_dir / s for s in PILOT_SEQS]
    tune_out = out_dir / "eval_relink_tune"
    logger.info("tuning on pilot %s over thresholds %s", PILOT_SEQS, thresholds)
    print(f"{'thresh':>7} {'AssA':>7} {'HOTA_la':>8} {'frags':>12} {'merges':>7} {'merge_prec':>11}")
    for th in thresholds:
        params = RelinkParams(threshold=th)
        pred_out = tune_out / f"th_{th:.2f}"
        stats = _relink_and_write(
            seqs, data_dir, out_dir, pos_dir, pred_out / "predictions" / "data",
            params=params, audit_seqs=set(PILOT_SEQS), build_tools=_caches_cold(seqs, out_dir))
        scored = {s: 0 for s in PILOT_SEQS}
        r = gs_hota(pred_out, data_dir, seq_info=scored, **EVAL_CONFIGS["loc_assoc"])
        c, t = _pilot_precision(stats)
        fb = sum(stats[s]["n_fragments_before"] for s in PILOT_SEQS if s in stats)
        fa = sum(stats[s]["n_fragments_after"] for s in PILOT_SEQS if s in stats)
        prec = f"{c}/{t}={100.0 * c / max(t, 1):.0f}%"
        print(f"{th:>7.2f} {r['combined']['GS-AssA']:>7.2f} {r['combined']['GS-HOTA']:>8.2f} "
              f"{fb:>5} ->{fa:>5} {fb - fa:>7} {prec:>11}")


def main() -> None:
    """CLI entry point: extract our pipeline over GSR sequences and score GS-HOTA."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N sequences")
    ap.add_argument("--calib-period", type=int, default=1, help="PnLCalib every N frames (1=per-frame)")
    ap.add_argument("--detector", default="football")
    ap.add_argument("--tracker", default="bytetrack")
    ap.add_argument("--score-only", action="store_true", help="reuse submissions; skip the GPU stage")
    ap.add_argument("--relink", action="store_true",
                    help="Stage 2a: merge fragments by appearance, re-score into gsr_scores_relink.json")
    ap.add_argument("--relink-threshold", type=float, default=0.80,
                    help="frozen cosine merge threshold (tuned on the 3 pilot sequences)")
    ap.add_argument("--tune-relink", type=str, default=None,
                    help="comma-separated thresholds to sweep on the pilot (prints AssA + precision)")
    args = ap.parse_args()
    if args.tune_relink:
        tune_relink_threshold(args.data_dir, args.out_dir,
                              [float(t) for t in args.tune_relink.split(",")])
        return
    if args.relink:
        run_relink_benchmark(args.data_dir, args.out_dir, args.results_dir,
                             limit=args.limit, threshold=args.relink_threshold)
        return
    run_benchmark(
        args.data_dir, args.out_dir, args.results_dir, limit=args.limit,
        calib_period=args.calib_period, detector=args.detector, tracker=args.tracker,
        skip_extract=args.score_only,
    )


if __name__ == "__main__":
    main()
