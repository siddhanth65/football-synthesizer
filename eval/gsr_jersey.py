"""Attach Koshkina jersey reads to the GSR valid-split submissions and re-score GS-HOTA.

Bridge between :class:`generator.jersey_id.KoshkinaRecognizer` and :mod:`eval.gsr_score`. The shipped
benchmark emitted ``attributes.jersey = null`` for every detection, so the official ``gs_hota_full``
config (which gates similarity on role AND team AND jersey, but only for ``role == "player"``) could
never match a GT player carrying a labelled number -- the 14.8 floor. This module fills that field:

1. Per valid-split sequence, recover the player/GK crops from the cached positions parquet by
   re-detecting frames with the same football detector and matching persisted foot points (the exact
   crop-recovery :mod:`generator.track_relink` uses), then tracklet-vote each ``track_id``'s jersey
   number with the Koshkina chain (:func:`generator.jersey_id.aggregate_votes`) -- **NO roster mask**:
   GSR sequences ship no squad list, unlike our France match where both squads' numbers constrain the
   decode, so every number 1..99 is admissible here and the reader has no roster prior to lean on.
   Checkpointed ``{track_id: [number, conf]}`` per sequence (resumable by disk state; long GPU job).
2. Patch only ``attributes.jersey`` on the baseline submissions' player rows with the voted number as
   a string (matching GT's ``"7"`` form) -> ``eval_koshkina`` submissions; every other field (pitch
   position, role, team, track id, confidence) is copied verbatim, so the ONLY change scored is the
   jersey attribute.
3. Re-score all four attribute configs with the official evaluator. The jersey-attached arm is
   compared to the abstain-all arm (== the shipped ``gsr_scores.json`` ``gs_hota_full`` baseline);
   under GS-HOTA a wrong number on a track overlapping a GT player who has NO labelled number turns a
   would-be ``null == null`` match into a miss, so attaching can HURT -- both arms are reported.

CLI::

    python -m eval.gsr_jersey --limit 1        # pilot: read + score one sequence end-to-end
    python -m eval.gsr_jersey                   # full valid split (resumable)
    python -m eval.gsr_jersey --score-only      # patch + score from existing vote checkpoints
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from eval.gsr_score import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, DEFAULT_RESULTS_DIR, EVAL_CONFIGS, gs_hota
from generator.jersey_id import OCR_PERCROP_VERSION, aggregate_votes
from generator.track_relink import _frame_path, _sample_frames

logger = logging.getLogger("gsr_jersey")

#: Machine-local reproduced Koshkina pipeline paths (mirrors tools/closeup_anchor_probe.py).
#: ``$KOSHKINA_REPO`` / ``$KOSHKINA_SIDECAR_PY`` move the stack to another host (the A100 server
#: keeps the clone under ``~/src`` and runs the STR sidecar from a conda env); unset, every path is
#: the machine-local one every on-record number was measured with.
KOSHKINA_REPO = Path(os.environ.get("KOSHKINA_REPO") or Path.home() / "jersey-number-pipeline")
KOSHKINA_LEG_WEIGHTS = KOSHKINA_REPO / "models" / "legibility_resnet34_soccer_20240215.pth"
KOSHKINA_SIDECAR_PY = Path(os.environ.get("KOSHKINA_SIDECAR_PY")
                           or Path.home() / "jersey-str-env" / "Scripts" / "python.exe")
KOSHKINA_SIDECAR = Path("tools") / "koshkina_str_sidecar.py"
#: The INCUMBENT PARSeq reader: Koshkina's published SoccerNet fine-tune, resolved by glob because
#: its filename carries the training metrics. Every on-record number was measured with this file.
KOSHKINA_PARSEQ_GLOB = "models/parseq_epoch=24-*.ckpt"

#: Vote/decoding defaults (the reader's committed tracklet-vote threshold; NOT tuned to the metric).
MIN_CONF = 0.30
#: Crops sampled per track and foot-point match tolerance (same recovery as track_relink).
MAX_CROPS = 20
MATCH_TOL_PX = 6.0


def extract_track_crops(
    seq_dir: Path, df: pd.DataFrame, detector, out_dir: Path, *, max_crops: int = MAX_CROPS,
    match_tol_px: float = MATCH_TOL_PX, crop_scale: float = 1.0,
) -> dict[int, list[Path]]:
    """Recover up to ``max_crops`` person-box crops per player/GK track by re-detecting frames (IO/GPU).

    For each track up to ``max_crops`` evenly-spread sample frames are chosen; each sample's persisted
    foot point is matched to the nearest re-detected box (within ``match_tol_px``) and that box crop is
    written to disk. This is the crop-recovery of :func:`generator.track_relink.fragment_embeddings`,
    writing crop files (for the Koshkina reader) instead of embedding them.

    Args:
        seq_dir: GSR sequence directory (holds ``img1/``).
        df: The sequence's positions table.
        detector: A ``generator.extract`` detector (``detect(rgb) -> (xyxy, ...)``); MUST be the same
            football detector the parquet was extracted with, so boxes reproduce exactly.
        out_dir: Scratch dir for the crop JPGs (created; caller owns cleanup).
        max_crops: Max crops sampled per track.
        match_tol_px: Max foot-point distance (px) to accept a re-detected box as the crop source.
        crop_scale: Widening of the detector box about its foot point
            (:func:`tools.ocr_match.scale_box`). ``1.0`` reproduces the shipped geometry exactly.
            Unlike our broadcast pipeline -- where the box is *reconstructed* by
            ``generator.team_anchor.estimate_player_box`` and measured 0.814x too small
            (``results/OCR_DOMAIN_SHIFT.md`` section 7) -- the GSR box here is the detector's own,
            so this knob is a genuine widening, not a repair.

    Returns:
        ``{track_id: [crop_path, ...]}`` (BGR JPGs), player/GK tracks only.
    """
    import cv2  # noqa: PLC0415

    from tools.ocr_match import scale_box  # noqa: PLC0415

    people = df[df["role"].isin(["player", "goalkeeper"])
                & np.isfinite(df["image_x"]) & np.isfinite(df["image_y"])]
    need: dict[int, list[tuple[int, float, float]]] = defaultdict(list)  # frame -> [(tid, ix, iy)]
    for tid, grp in people.groupby("track_id"):
        grp = grp.sort_values("frame")
        frames = grp["frame"].astype(int).tolist()
        keep = set(_sample_frames(frames, max_crops))
        for r in grp.itertuples():
            if int(r.frame) in keep:
                need[int(r.frame)].append((int(tid), float(r.image_x), float(r.image_y)))

    out_dir.mkdir(parents=True, exist_ok=True)
    paths_by_tid: dict[int, list[Path]] = defaultdict(list)
    for frame_idx in sorted(need):
        bgr = cv2.imread(str(_frame_path(seq_dir, frame_idx)))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        xyxy = detector.detect(rgb)[0]
        if len(xyxy) == 0:
            continue
        foot = np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2.0, xyxy[:, 3]])
        for tid, ix, iy in need[frame_idx]:
            d = np.hypot(foot[:, 0] - ix, foot[:, 1] - iy)
            k = int(np.argmin(d))
            if d[k] > match_tol_px:
                continue
            box = tuple(int(v) for v in xyxy[k])
            x1, y1, x2, y2 = (box if crop_scale == 1.0
                              else scale_box(box, crop_scale, bgr.shape[0], bgr.shape[1]))
            crop = bgr[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)]
            if crop.size:
                p = out_dir / f"t{tid}_{frame_idx:06d}.jpg"
                cv2.imwrite(str(p), crop)
                paths_by_tid[tid].append(p)
    return paths_by_tid


def vote_tracks(
    recog, paths_by_tid: dict[int, list[Path]], *, min_conf: float = MIN_CONF
) -> dict[int, tuple[int, float]]:
    """Tracklet-vote a jersey number per track from its crops (one reader pass over the sequence).

    All the sequence's crops are read in a single :meth:`crop_probs` call (one legibility/pose batch
    and one PARSeq sidecar invocation), then the ``[N, 100]`` rows are grouped back per track and
    consolidated with :func:`generator.jersey_id.aggregate_votes` (confidence-weighted mean softmax,
    thresholded) -- **no roster mask**.

    Args:
        recog: A :class:`generator.jersey_id.KoshkinaRecognizer` (``crop_probs`` contract).
        paths_by_tid: ``{track_id: [crop_path, ...]}`` from :func:`extract_track_crops`.
        min_conf: Tracklet-level floor; below it (or when illegible wins) the track abstains (``-1``).

    Returns:
        ``{track_id: (number, confidence)}`` with ``number == -1`` meaning abstain.
    """
    flat: list[Path] = []
    index: list[int] = []
    for tid, ps in paths_by_tid.items():
        for p in ps:
            flat.append(p)
            index.append(tid)
    if not flat:
        return {}
    probs = recog.crop_probs(flat)  # [N, 100], aligned 1:1 with flat
    rows: dict[int, list[np.ndarray]] = defaultdict(list)
    for i, tid in enumerate(index):
        rows[tid].append(probs[i])
    votes: dict[int, tuple[int, float]] = {}
    for tid, rws in rows.items():
        num, conf = aggregate_votes(np.stack(rws), min_conf=min_conf)  # mask=None -> no roster prior
        votes[tid] = (int(num), float(conf))
    return votes


def resolve_parseq_ckpt(spec: str | Path | None = None) -> Path:
    """Resolve which PARSeq weights the reader loads (pure lookup + existence check).

    Args:
        spec: An explicit checkpoint path, or ``None`` for the incumbent
            (:data:`KOSHKINA_PARSEQ_GLOB` inside :data:`KOSHKINA_REPO`). Defaulting to the incumbent
            is deliberate: every number on record was measured with that file, so a caller must ask
            for a different reader by name before anything on record can move.

    Returns:
        The checkpoint path.

    Raises:
        SystemExit: If the requested checkpoint (or the incumbent glob) does not resolve, or if its
            path cannot be classified by strhub.

    Note:
        ``strhub.models.utils.load_from_checkpoint`` picks the model class by substring-testing the
        checkpoint **path** (``'parseq' in key``), so a file the trainer wrote as ``last.ckpt``
        raises ``InvalidModelError`` in the sidecar unless ``parseq`` appears somewhere in its path.
        Checked here, before a multi-hour GPU pass starts.
    """
    if spec is not None:
        p = Path(spec)
        if not p.exists():
            raise SystemExit(f"parseq checkpoint not found: {p}")
        if "parseq" not in str(p.resolve()):
            raise SystemExit(f"strhub resolves the model class from the path; rename/move so "
                             f"'parseq' appears in it: {p}")
        return p
    hits = sorted(KOSHKINA_REPO.glob(KOSHKINA_PARSEQ_GLOB))
    if not hits:
        raise SystemExit(f"no incumbent parseq checkpoint at {KOSHKINA_REPO / KOSHKINA_PARSEQ_GLOB}")
    return hits[0]


def koshkina_work_dir() -> Path:
    """Per-PROCESS scratch dir for the Koshkina chain (torso crops + the sidecar's JSON).

    :class:`generator.jersey_id.KoshkinaRecognizer` wipes this directory at construction and names
    its torso crops ``c{call}_{i}.jpg`` from a per-process counter, so two readers sharing one
    directory delete each other's crops AND collide on those names -- silently mixing one
    sequence's PARSeq reads into another's. The pid keeps concurrent workers (the server runs the
    percrop pass sharded across GPU workers) isolated; a single-process run is unchanged apart
    from the directory name.
    """
    return DEFAULT_OUT_DIR / "koshkina_jersey" / f"_work_{os.getpid()}"


def _build_recognizer(device: str | None = None, parseq_ckpt: str | Path | None = None,
                      edl_head: str | Path | None = None, leg_thresh: float = 0.5):  # noqa: ANN202
    """Construct the Koshkina reader (wipes + reloads legibility/pose; call once per sequence).

    Args:
        device: ``"cuda"``/``"cpu"``; auto-detected when ``None``.
        parseq_ckpt: Reader weights; ``None`` = the incumbent (see :func:`resolve_parseq_ckpt`).
        edl_head: Dirichlet evidential head checkpoint; ``None`` = the incumbent PARSeq-only read.
        leg_thresh: External ResNet34 legibility floor; ``0.0`` disables the gate. The shipped
            ``0.5`` is the default so every existing call site is unchanged.
    """
    from generator.jersey_id import KoshkinaRecognizer  # noqa: PLC0415

    parseq_ckpt = resolve_parseq_ckpt(parseq_ckpt)
    return KoshkinaRecognizer(
        legibility_weights=KOSHKINA_LEG_WEIGHTS,
        sidecar_python=KOSHKINA_SIDECAR_PY,
        sidecar_script=KOSHKINA_SIDECAR,
        parseq_ckpt=parseq_ckpt,
        parseq_repo=KOSHKINA_REPO,
        work_dir=koshkina_work_dir(),
        device=device,
        min_conf=MIN_CONF,
        leg_thresh=leg_thresh,
        edl_head=edl_head,
    )


def read_sequence_jerseys(
    seq_dir: Path, parquet: Path, ckpt_json: Path, detector, *, force: bool = False,
) -> dict[int, tuple[int, float]]:
    """Read + checkpoint one sequence's per-track jersey votes (resumable by ``ckpt_json``).

    Args:
        seq_dir: GSR sequence directory.
        parquet: The sequence's cached positions parquet.
        ckpt_json: Per-sequence vote checkpoint (``{track_id: [number, conf]}``); reused if present.
        detector: The football detector (built once, reused across sequences).
        force: Recompute even if ``ckpt_json`` exists.

    Returns:
        ``{track_id: (number, conf)}``.
    """
    if ckpt_json.exists() and not force:
        blob = json.loads(ckpt_json.read_text(encoding="utf-8"))
        return {int(k): (int(v[0]), float(v[1])) for k, v in blob["votes"].items()}
    df = pd.read_parquet(parquet)
    crop_dir = ckpt_json.parent / "_crops" / seq_dir.name
    import shutil  # noqa: PLC0415

    if crop_dir.exists():
        shutil.rmtree(crop_dir, ignore_errors=True)
    paths_by_tid = extract_track_crops(seq_dir, df, detector, crop_dir)
    recog = _build_recognizer()
    votes = vote_tracks(recog, paths_by_tid, min_conf=MIN_CONF)
    shutil.rmtree(crop_dir, ignore_errors=True)  # crops are transient; the checkpoint is the artifact
    n_read = sum(1 for n, _ in votes.values() if n >= 1)
    ckpt_json.parent.mkdir(parents=True, exist_ok=True)
    ckpt_json.write_text(json.dumps({
        "seq": seq_dir.name, "n_tracks": len(votes), "n_read": n_read,
        "n_crops": int(sum(len(p) for p in paths_by_tid.values())),
        "votes": {str(t): [n, round(c, 4)] for t, (n, c) in votes.items()},
    }), encoding="utf-8")
    return votes


#: Per-crop evidence cache (NEW artifact; the aggregated koshkina_jersey/*.json is left untouched).
PERCROP_SUBDIR = "koshkina_percrop"


def percrop_frame(
    paths_by_tid: dict[int, list[Path]], probs: np.ndarray, detail: dict, index: list[int],
    flat: list[Path],
) -> pd.DataFrame:
    """Assemble one sequence's per-crop OCR evidence into a tidy frame (pure).

    Args:
        paths_by_tid: Unused beyond documenting provenance; the row order comes from ``flat``.
        probs: ``[N, NUM_CLASSES]`` folded per-crop distributions from ``crop_reads``.
        detail: The ``crop_reads`` detail dict (``leg``, ``torso``, ``p0``, ``p1``).
        index: ``track_id`` per row of ``probs``.
        flat: Crop path per row (``t<tid>_<frame>.jpg``), used to recover the frame index.

    Returns:
        One row per crop: track/frame, legibility, torso flag, the crop's best number and its mass,
        the illegible mass, and the raw PARSeq positional softmaxes (list columns ``p0``/``p1``).
    """
    del paths_by_tid
    best = probs[:, 1:].argmax(axis=1) + 1
    conf = probs[np.arange(probs.shape[0]), best]
    out = pd.DataFrame({
        "track_id": np.asarray(index, dtype=np.int32),
        "frame": np.array([int(p.stem.split("_")[-1]) for p in flat], dtype=np.int32),
        "legibility": detail["leg"].astype(np.float32),
        "torso": detail["torso"],
        "number": np.where(probs.argmax(axis=1) == 0, -1, best).astype(np.int16),
        "p_number": conf.astype(np.float32),
        "p_illegible": probs[:, 0].astype(np.float32),
        "p0": list(detail["p0"]),
        "p1": list(detail["p1"]),
        "ocr_version": OCR_PERCROP_VERSION,
    })
    if "alpha" in detail:  # campaign v7 V3: the Dirichlet evidential arm keeps its concentrations
        alpha = detail["alpha"].astype(np.float32)
        out["alpha"] = list(alpha)
        out["u"] = (alpha.shape[1] / alpha.sum(axis=1)).astype(np.float32)
    return out


def read_sequence_percrop(
    seq_dir: Path, parquet: Path, dest: Path, detector, *, max_crops: int, force: bool = False,
    crop_scale: float = 1.0, parseq_ckpt: str | Path | None = None,
    edl_head: str | Path | None = None, leg_thresh: float = 0.5,
) -> pd.DataFrame:
    """Run the Koshkina chain over one sequence and persist EVERY crop's read (resumable by disk).

    Args:
        seq_dir: GSR sequence directory.
        parquet: The sequence's cached positions parquet.
        dest: Destination per-crop parquet; reused if present.
        detector: The football detector (built once, reused across sequences).
        max_crops: Crops sampled per track (the densification lever; the on-record run used 20).
        force: Recompute even if ``dest`` exists.
        crop_scale: Detector-box widening (see :func:`extract_track_crops`); stamped on every row.
        parseq_ckpt: Reader weights; ``None`` = the incumbent. The checkpoint stem is stamped on
            every row (``reader``) so a parquet names the model that produced it.
        edl_head: Dirichlet evidential head checkpoint (campaign v7 V3); ``None`` = incumbent.
        leg_thresh: External legibility floor; ``0.0`` sends every crop to the pose + reader stage.

    Returns:
        The per-crop frame (see :func:`percrop_frame`).
    """
    import shutil  # noqa: PLC0415

    if dest.exists() and not force:
        return pd.read_parquet(dest)
    df = pd.read_parquet(parquet)
    crop_dir = dest.parent / "_crops" / seq_dir.name
    shutil.rmtree(crop_dir, ignore_errors=True)
    paths_by_tid = extract_track_crops(seq_dir, df, detector, crop_dir, max_crops=max_crops,
                                       crop_scale=crop_scale)
    flat: list[Path] = []
    index: list[int] = []
    for tid, ps in paths_by_tid.items():
        for p in ps:
            flat.append(p)
            index.append(int(tid))
    from generator.jersey_id import NUM_CLASSES  # noqa: PLC0415

    ckpt = resolve_parseq_ckpt(parseq_ckpt)
    if flat:
        recog = _build_recognizer(parseq_ckpt=ckpt, edl_head=edl_head, leg_thresh=leg_thresh)
        probs, detail = recog.crop_reads(flat)
    else:
        probs = np.empty((0, NUM_CLASSES), np.float32)
        detail = {"leg": np.empty(0, np.float32), "torso": np.empty(0, bool),
                  "p0": np.empty((0, 11), np.float32), "p1": np.empty((0, 11), np.float32)}
    out = percrop_frame(paths_by_tid, probs, detail, index, flat).assign(
        crop_scale=float(crop_scale), reader=ckpt.stem)
    shutil.rmtree(crop_dir, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    return out


def run_percrop(data_dir: Path, out_dir: Path, *, max_crops: int, limit: int | None,
                crop_scale: float = 1.0, variant: str = "", only: list[str] | None = None,
                parseq_ckpt: str | Path | None = None,
                positions_subdir: str = "positions",
                edl_head: str | Path | None = None, leg_thresh: float = 0.5) -> None:
    """GPU pass: persist per-crop OCR evidence for every valid-split sequence (resumable).

    ``variant`` suffixes the output directory so a re-run at a different crop geometry -- or a
    different reader checkpoint (``parseq_ckpt``) -- lands beside the shipped cache instead of
    overwriting it (the ``tools.ocr_match`` pattern). ``positions_subdir`` selects which extraction
    the crops are recovered from; it MUST name the parquet set the active detector produced, since
    crop recovery re-detects and matches foot points.
    """
    import shutil  # noqa: PLC0415

    pos_dir = out_dir / positions_subdir
    dest_dir = out_dir / (PERCROP_SUBDIR + variant)
    seqs = sorted(p for p in data_dir.iterdir()
                  if p.is_dir() and (pos_dir / f"{p.name}.parquet").exists()
                  and (only is None or p.name in set(only)))
    if limit:
        seqs = seqs[:limit]
    detector = None
    for i, seq_dir in enumerate(seqs):
        dest = dest_dir / f"{seq_dir.name}.parquet"
        if dest.exists():
            continue
        if detector is None:
            import torch  # noqa: PLC0415

            from generator.extract import _build_detector  # noqa: PLC0415

            detector = _build_detector("cuda" if torch.cuda.is_available() else "cpu", "football")
        t0 = time.time()
        frame = read_sequence_percrop(seq_dir, pos_dir / f"{seq_dir.name}.parquet", dest, detector,
                                      max_crops=max_crops, crop_scale=crop_scale,
                                      parseq_ckpt=parseq_ckpt, edl_head=edl_head,
                                      leg_thresh=leg_thresh)
        n_read = int((frame["number"] > 0).sum())
        logger.info("[%d/%d] %s: %d crops, %d with a number, %d tracks (%.0fs)",
                    i + 1, len(seqs), seq_dir.name, len(frame), n_read,
                    frame["track_id"].nunique(), time.time() - t0)
        # Integrity guard: legible crops but NO torso RoI means the pose stage produced nothing for
        # the whole sequence, so PARSeq never ran -- the sequence then carries zero reads, the
        # GT-free self-roster is empty and the solver names nobody (measured: -34.7 GS-HOTA on
        # SNGS-056). Drop the artifact so the resumable pass rebuilds it instead of shipping it.
        n_leg = int((frame["legibility"] >= 0.5).sum())  # noqa: PLR2004 - the reader's own floor
        if n_leg > 0 and not bool(frame["torso"].any()):
            logger.error("%s: %d legible crops but ZERO torso RoIs -- discarding, re-run needed",
                         seq_dir.name, n_leg)
            dest.unlink(missing_ok=True)
    shutil.rmtree(koshkina_work_dir(), ignore_errors=True)  # this process's sidecar scratch


def patch_submission(base_json: Path, votes: dict[int, tuple[int, float]], dest: Path) -> dict:
    """Copy a baseline submission, filling ``attributes.jersey`` on player rows with the voted number.

    Args:
        base_json: Baseline submission (``{"predictions": [...]}``) with ``jersey = null``.
        votes: ``{track_id: (number, conf)}`` for this sequence (``number == -1`` -> stay null).
        dest: Destination submission path (created).

    Returns:
        ``{"n_player_preds", "n_attached"}`` bookkeeping for this sequence.
    """
    payload = json.loads(base_json.read_text(encoding="utf-8"))
    n_player = n_attached = 0
    for pred in payload["predictions"]:
        if pred["attributes"].get("role") != "player":
            continue
        n_player += 1
        num = votes.get(int(pred["track_id"]), (-1, 0.0))[0]
        if num >= 1:
            pred["attributes"]["jersey"] = str(num)  # GT stores the number as a string ("7")
            n_attached += 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload), encoding="utf-8")
    return {"n_player_preds": n_player, "n_attached": n_attached}


def run(
    data_dir: Path, out_dir: Path, results_dir: Path, *, limit: int | None, score_only: bool,
) -> dict:
    """Read jerseys (GPU, resumable), patch submissions, re-score, and write gsr_scores_koshkina.json.

    Args:
        data_dir: GSR ground-truth folder (sequence dirs with ``Labels-GameState.json``).
        out_dir: Pipeline output root (holds ``positions/`` and the baseline ``eval/`` submissions).
        results_dir: Where ``gsr_scores_koshkina.json`` is written (NEW file; baseline untouched).
        limit: Process only the first N sequences (pilot).
        score_only: Skip the GPU read; patch + score from existing vote checkpoints only.

    Returns:
        The written payload dict.
    """
    pos_dir = out_dir / "positions"
    base_data = out_dir / "eval" / "predictions" / "data"
    ck_data = out_dir / "eval_koshkina" / "predictions" / "data"
    vote_dir = out_dir / "koshkina_jersey"
    seqs = sorted(p for p in data_dir.iterdir()
                  if p.is_dir() and (pos_dir / f"{p.name}.parquet").exists()
                  and (base_data / f"{p.name}.json").exists())
    if limit:
        seqs = seqs[:limit]
    if not seqs:
        raise SystemExit(f"no GSR sequences with positions + baseline submissions under {out_dir}")
    logger.info("koshkina jersey rescoring: %d sequences (score_only=%s)", len(seqs), score_only)

    detector = None
    per_seq_read: dict[str, dict] = {}
    per_seq_attach: dict[str, dict] = {}
    for i, seq_dir in enumerate(seqs):
        name = seq_dir.name
        ckpt_json = vote_dir / f"{name}.json"
        if not score_only and (not ckpt_json.exists()):
            if detector is None:
                import torch  # noqa: PLC0415

                from generator.extract import _build_detector  # noqa: PLC0415

                device = "cuda" if torch.cuda.is_available() else "cpu"
                detector = _build_detector(device, "football")
            t0 = time.time()
            votes = read_sequence_jerseys(seq_dir, pos_dir / f"{name}.parquet", ckpt_json, detector)
            n_read = sum(1 for n, _ in votes.values() if n >= 1)
            logger.info("[%d/%d] %s: %d tracks, %d read a number (%.0fs)",
                        i + 1, len(seqs), name, len(votes), n_read, time.time() - t0)
        if not ckpt_json.exists():
            logger.info("[%d/%d] %s: no vote checkpoint, skipping", i + 1, len(seqs), name)
            continue
        blob = json.loads(ckpt_json.read_text(encoding="utf-8"))
        votes = {int(k): (int(v[0]), float(v[1])) for k, v in blob["votes"].items()}
        per_seq_read[name] = {"n_tracks": blob["n_tracks"], "n_read": blob["n_read"],
                              "n_crops": blob.get("n_crops", 0)}
        per_seq_attach[name] = patch_submission(
            base_data / f"{name}.json", votes, ck_data / f"{name}.json")

    scored = {name: 0 for name in per_seq_attach}
    if not scored:
        raise SystemExit("no sequences patched; run the GPU read first (drop --score-only)")
    results: dict[str, dict] = {}
    for cfg_name, cfg in EVAL_CONFIGS.items():
        logger.info("scoring koshkina config '%s' over %d sequences", cfg_name, len(scored))
        results[cfg_name] = gs_hota(out_dir / "eval_koshkina", data_dir, seq_info=scored, **cfg)

    baseline = json.loads((results_dir / "gsr_scores.json").read_text(encoding="utf-8"))["configs"]
    payload = _assemble_payload(results, baseline, per_seq_read, per_seq_attach, scored)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "gsr_scores_koshkina.json").write_text(json.dumps(payload, indent=2),
                                                          encoding="utf-8")
    _log_summary(payload)
    return payload


def _assemble_payload(
    results: dict, baseline: dict, per_seq_read: dict, per_seq_attach: dict, scored: dict,
) -> dict:
    """Assemble the koshkina rescore payload: arms, abstention, and per-seq HURT decomposition."""
    n_tracks = int(sum(v["n_tracks"] for v in per_seq_read.values()))
    n_read = int(sum(v["n_read"] for v in per_seq_read.values()))
    n_player = int(sum(v["n_player_preds"] for v in per_seq_attach.values()))
    n_attached = int(sum(v["n_attached"] for v in per_seq_attach.values()))
    per_seq = {}
    for name in scored:
        new = results["gs_hota_full"]["per_seq"].get(name, {})
        base = baseline["gs_hota_full"]["per_seq"].get(name, {})
        rd = per_seq_read.get(name, {})
        nt = max(rd.get("n_tracks", 0), 1)
        per_seq[name] = {
            "gs_hota_full_attach": new.get("GS-HOTA"),
            "gs_hota_full_abstain": base.get("GS-HOTA"),
            "delta": (None if new.get("GS-HOTA") is None or base.get("GS-HOTA") is None
                      else round(new["GS-HOTA"] - base["GS-HOTA"], 3)),
            "n_tracks": rd.get("n_tracks", 0),
            "n_read": rd.get("n_read", 0),
            "abstain_rate": round(1.0 - rd.get("n_read", 0) / nt, 3),
            "n_attached_player_rows": per_seq_attach.get(name, {}).get("n_attached", 0),
        }
    hurt = {n: d for n, d in per_seq.items() if d["delta"] is not None and d["delta"] < 0}
    return {
        "n_sequences": len(scored),
        "reader": "koshkina", "roster_mask": False, "min_conf": MIN_CONF,
        "max_crops_per_track": MAX_CROPS,
        "tracks_total": n_tracks, "tracks_read_number": n_read,
        "track_abstain_rate": round(1.0 - n_read / max(n_tracks, 1), 4),
        "player_rows_total": n_player, "player_rows_attached": n_attached,
        "arms": {
            "attach": {k: v["combined"] for k, v in results.items()},
            "abstain": {k: v["combined"] for k, v in baseline.items()},
        },
        "attach_per_seq": {k: v["per_seq"] for k, v in results.items()},
        "per_seq_hurt_decomposition": per_seq,
        "n_seqs_hurt": len(hurt),
        "seqs_hurt": {n: per_seq[n]["delta"] for n in hurt},
    }


def _log_summary(payload: dict) -> None:
    """ASCII summary of the two arms + abstention (cp1252-safe)."""
    a = payload["arms"]["attach"]["gs_hota_full"]
    b = payload["arms"]["abstain"]["gs_hota_full"]
    logger.info("== gs_hota_full  abstain(baseline) %.2f -> attach %.2f  (%+.2f)",
                b["GS-HOTA"], a["GS-HOTA"], a["GS-HOTA"] - b["GS-HOTA"])
    logger.info("   DetA %.2f -> %.2f | AssA %.2f -> %.2f | LocA %.2f -> %.2f | IDF1 %.2f -> %.2f",
                b["GS-DetA"], a["GS-DetA"], b["GS-AssA"], a["GS-AssA"],
                b["GS-LocA"], a["GS-LocA"], b["IDF1"], a["IDF1"])
    logger.info("   tracks %d, read a number %d, abstain rate %.1f%%; player rows attached %d/%d",
                payload["tracks_total"], payload["tracks_read_number"],
                100.0 * payload["track_abstain_rate"], payload["player_rows_attached"],
                payload["player_rows_total"])
    for cfg in ("no_jersey", "loc_assoc"):
        aa, bb = payload["arms"]["attach"][cfg], payload["arms"]["abstain"][cfg]
        logger.info("   invariant %-10s attach %.2f vs baseline %.2f (should match)",
                    cfg, aa["GS-HOTA"], bb["GS-HOTA"])
    logger.info("   sequences where jerseys HURT: %d -> %s",
                payload["n_seqs_hurt"], payload["seqs_hurt"])


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N sequences")
    ap.add_argument("--score-only", action="store_true",
                    help="patch + score from existing vote checkpoints; skip the GPU read")
    ap.add_argument("--percrop", action="store_true",
                    help="GPU pass persisting EVERY crop's read to outputs/gsr/koshkina_percrop/")
    ap.add_argument("--max-crops", type=int, default=MAX_CROPS,
                    help="crops sampled per track (on-record aggregated run used 20)")
    ap.add_argument("--crop-scale", type=float, default=1.0,
                    help="widen the detector box about its foot point; 1.0 is the shipped geometry")
    ap.add_argument("--variant", default="",
                    help="suffix on outputs/gsr/koshkina_percrop<variant>/ (e.g. _w125, _v6)")
    ap.add_argument("--parseq-ckpt", type=Path, default=None,
                    help="PARSeq reader weights; default = the incumbent SoccerNet fine-tune")
    ap.add_argument("--seqs", default=None, help="comma-separated sequence names")
    ap.add_argument("--positions", default="positions",
                    help="positions subdir the crops are recovered from")
    ap.add_argument("--edl-head", type=Path, default=None,
                    help="Dirichlet evidential head checkpoint (campaign v7 V3); adds per-crop "
                         "alpha/u columns and replaces the folded softmax with the Dirichlet mean")
    ap.add_argument("--leg-thresh", type=float, default=0.5,
                    help="external ResNet34 legibility floor; 0.0 disables the gate (5x crops)")
    args = ap.parse_args()
    if args.percrop:
        run_percrop(args.data_dir, args.out_dir, max_crops=args.max_crops, limit=args.limit,
                    crop_scale=args.crop_scale, variant=args.variant,
                    only=args.seqs.split(",") if args.seqs else None,
                    parseq_ckpt=args.parseq_ckpt, positions_subdir=args.positions,
                    edl_head=args.edl_head, leg_thresh=args.leg_thresh)
        return
    run(args.data_dir, args.out_dir, args.results_dir, limit=args.limit, score_only=args.score_only)


if __name__ == "__main__":
    main()
