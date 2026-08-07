"""Campaign v6 session S7: price the S4b detector fine-tune end-to-end on GS-HOTA.

`results/CLUSTER_SESSION_S4.md` §S4b passed a **box-level** gate (mAP@0.5 +20.66, role-correct
coverage +7.27, per-game player->referee leak <= 0.05, ball mAP >= control) and explicitly recorded
that **no GS-HOTA was measured with those weights**. Under GS-HOTA a detection only scores if its
team, jersey number and role are all right, so a recall gain can just as easily arrive as
unmatched anonymous boxes -- i.e. false positives that push GS-DetA *down*. This module runs the
whole cached chain a second time on the new detections so the question is answered by measurement.

Every GPU artifact the chain caches is keyed by ``(track_id, frame)``, and a re-extraction changes
every track id, so the detector cannot be swapped in isolation: positions, the calibration re-gate,
the per-crop OCR evidence, the per-detection box cache and the appearance embeddings all have to be
rebuilt. Each stage writes to a ``_v6det``-suffixed directory, so the on-record (detector-OUT)
artifacts are never touched and both arms stay scorable from disk.

Stages (all resumable by disk state; run in this order):

1. ``extract``    GPU, ~7 min/sequence -- detect + ByteTrack + PnLCalib -> ``positions_v6det``
2. ``gate``       CPU -- the frozen calibration re-gate + gap fill -> ``positions_gate_v6det``
3. ``percrop``    GPU, ~2 min/sequence -- the v6 PARSeq reader on the new boxes
4. ``boxes``      GPU -- per-detection box cache for the EIoU tracker
5. ``embed``      GPU -- CLIP per-detection embeddings
6. ``arm``        CPU -- the frozen EIoU + v6-reader arm on the new evidence, scored

CLI::

    python -m tools.gsr_v6det --stages all --split dev
    python -m tools.gsr_v6det --stages arm --split dev        # re-score without re-doing GPU work
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("gsr_v6det")

#: Suffix every artifact of the S4b-detector arm carries.
VARIANT = "_v6det"
#: The S4b detector weights (server origin ``~/runs/det/gsr_v3_ft_b/weights/last.pt``).
WEIGHTS = Path("models/gsr_det/gsr_v3_ft_b_last.pt")
WEIGHTS_MD5 = "2074d8741f97a6892a1322d0f808244c"
#: The v6 jersey reader (S2 arm 4t), unchanged from S3.
PARSEQ_CKPT = Path.home() / "jersey-number-pipeline" / "models" / "parseq_v6_arm4t.ckpt"
#: The frozen downstream point (results/gsr_eiou_frozen.json + GSR_S3_READER.md arm A).
TAU = 0.450
MAX_CROPS = 60
FILL_GAP = 10


#: The shipped (control) detector: the HF model every on-record number was extracted with.
CONTROL_WEIGHTS = Path.home() / "models" / "football_yolov8s_baseline.pt"
CONTROL_MD5 = "b3417a4c3228d7f932844e4e39ec5059"


def set_arm(detector: str) -> None:
    """Point the module at the S4b arm (``'s4b'``) or the shipped-detector control (``'control'``).

    The control arm exists for the same-machine pairing rule: when the GPU stages move to another
    host, an arm produced there may only be paired against a control produced there too, so the
    incumbent detector gets its own re-extraction rather than re-using laptop-cached artifacts.
    """
    global VARIANT, WEIGHTS, WEIGHTS_MD5  # noqa: PLW0603 - one switch for every stage in the module

    if detector == "s4b":
        VARIANT, WEIGHTS = "_v6det", Path("models/gsr_det/gsr_v3_ft_b_last.pt")
        WEIGHTS_MD5 = "2074d8741f97a6892a1322d0f808244c"
    elif detector == "control":
        # The shipped detector is a server-side copy of the laptop's HuggingFace snapshot (md5
        # verified equal on both machines); resolve whichever this host actually holds.
        from generator.extract import _FootballRoleDetector  # noqa: PLC0415

        VARIANT, WEIGHTS_MD5 = "_base", CONTROL_MD5
        WEIGHTS = (CONTROL_WEIGHTS if CONTROL_WEIGHTS.exists()
                   else Path(_FootballRoleDetector._download_default()))  # noqa: SLF001
    else:
        raise SystemExit(f"unknown detector arm {detector!r}")


def check_weights() -> str:
    """Verify the detector checkpoint against its parked md5, returning it."""
    import hashlib  # noqa: PLC0415

    got = hashlib.md5(WEIGHTS.read_bytes()).hexdigest()  # noqa: S324 - provenance check, not crypto
    if got != WEIGHTS_MD5:
        raise SystemExit(f"detector md5 {got} != the parked {WEIGHTS_MD5}")
    return got


def use_new_detector() -> None:
    """Point every detector construction in the process at this arm's weights."""
    import generator.extract as ex  # noqa: PLC0415

    ex.FOOTBALL_WEIGHTS = str(WEIGHTS)


def stage_extract(data_dir: Path, out_dir: Path, seqs: list[str]) -> dict:
    """GPU: re-extract positions with the S4b detector (resumable, one sequence per parquet)."""
    from generator.calibrate import PnLCalibCalibrator  # noqa: PLC0415
    from generator.extract import extract_positions  # noqa: PLC0415

    dest = out_dir / ("positions" + VARIANT)
    dest.mkdir(parents=True, exist_ok=True)
    todo = [s for s in seqs if not (dest / f"{s}.parquet").exists()]
    logger.info("extract: %d/%d cold", len(todo), len(seqs))
    if not todo:
        return {}
    calibrator = PnLCalibCalibrator()
    stats = {}
    for i, name in enumerate(todo):
        t0 = time.time()
        df = extract_positions(str(data_dir / name / "img1" / "%06d.jpg"),
                               dest / f"{name}.parquet", sample_every=1, calibrator=calibrator,
                               detector_name="football", detector_weights=str(WEIGHTS),
                               tracker_name="bytetrack", calib_period=1)
        stats[name] = {"rows": int(len(df)), "tracks": int(df["track_id"].nunique()),
                       "seconds": round(time.time() - t0, 1)}
        logger.info("[%d/%d] %s: %d rows, %d tracks (%.0fs)", i + 1, len(todo), name,
                    stats[name]["rows"], stats[name]["tracks"], stats[name]["seconds"])
    return stats


def stage_gate(out_dir: Path, seqs: list[str]) -> dict:
    """CPU: the frozen calibration re-gate + gap fill on the new extraction."""
    from tools.gsr_calibgate import apply_split  # noqa: PLC0415

    return apply_split(out_dir, seqs, src_dir=out_dir / ("positions" + VARIANT),
                       dest_dir=out_dir / ("positions_gate" + VARIANT), fill_gap=FILL_GAP)


def percrop_variant(reader: str) -> str:
    """The per-crop evidence directory suffix for one reader arm (pure).

    ``'v6'`` = the S2 arm-4t bundle reader; ``'incumbent'`` = the shipped SoccerNet fine-tune (the
    reader leave-one-out on the new boxes); ``'edl'`` = the campaign-v7 Dirichlet evidential head on
    the frozen arm-4t trunk, which lands in its own ``_v7e`` tree so nothing on record can mix.
    """
    return {"v6": "_v6", "incumbent": "", "edl": "_v7e"}[reader] + VARIANT


def stage_percrop(data_dir: Path, out_dir: Path, seqs: list[str], *, reader: str = "v6",
                  edl_head: Path | None = None, leg_thresh: float = 0.5) -> None:
    """GPU: a PARSeq reader over crops recovered from the NEW detector's boxes.

    Args:
        data_dir: GSR ground-truth folder (supplies the frames).
        out_dir: Pipeline output root.
        seqs: Sequences to read.
        reader: See :func:`percrop_variant`.
        edl_head: Dirichlet evidential head checkpoint; required when ``reader='edl'``.
        leg_thresh: External ResNet34 legibility floor; ``0.0`` sends every crop through pose +
            reader (~5x the crops, campaign v7 V3) and leaves the gate to an offline sweep.
    """
    from eval.gsr_jersey import run_percrop  # noqa: PLC0415

    if (reader == "edl") != (edl_head is not None):
        raise SystemExit("reader='edl' needs --edl-head, and --edl-head needs reader='edl'")
    use_new_detector()
    run_percrop(data_dir, out_dir, max_crops=MAX_CROPS, limit=None,
                variant=percrop_variant(reader), only=seqs,
                parseq_ckpt=PARSEQ_CKPT if reader in ("v6", "edl") else None,
                positions_subdir="positions" + VARIANT,
                edl_head=edl_head, leg_thresh=leg_thresh)


def stage_boxes(data_dir: Path, out_dir: Path, seqs: list[str]) -> dict:
    """GPU: per-detection box cache for the EIoU tracker, on the new detections."""
    import tools.gsr_eiou as eiou  # noqa: PLC0415

    use_new_detector()
    eiou.BOX_SUBDIR = "detbox_cache" + VARIANT
    return eiou.build_box_cache(data_dir, out_dir, seqs,
                                positions_subdir="positions_gate" + VARIANT,
                                params=eiou.EiouParams())


def stage_embed(data_dir: Path, out_dir: Path, seqs: list[str]) -> None:
    """GPU: CLIP per-detection embeddings on this arm's detections (resumable by disk state).

    Deliberately does NOT go through :func:`eval.gsr_gta.build_cache`: that helper filters its
    sequence list by the presence of the on-record ``positions`` parquet AND an ``eval_koshkina``
    baseline submission, neither of which a re-extracted variant has -- it silently embeds nothing.
    """
    import time  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from generator.extract import _build_detector  # noqa: PLC0415
    from generator.gta_link import GtaParams, load_or_build_det_embeddings  # noqa: PLC0415

    from tools.clip_embedder import ClipEmbedder  # noqa: PLC0415

    use_new_detector()
    cache = out_dir / ("detembed_cache_clip" + VARIANT)
    cache.mkdir(parents=True, exist_ok=True)
    cold = [s for s in seqs if not (cache / f"{s}.npz").exists()]
    logger.info("embed: %d sequences, %d cold", len(seqs), len(cold))
    if not cold:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder, detector = ClipEmbedder(device=device), _build_detector(device, "football")
    for i, name in enumerate(cold):
        t0 = time.time()
        df = pd.read_parquet(out_dir / ("positions" + VARIANT) / f"{name}.parquet")
        det = load_or_build_det_embeddings(data_dir / name, df, cache / f"{name}.npz",
                                           params=GtaParams(frame_stride=2), embedder=embedder,
                                           detector=detector)
        logger.info("[%d/%d] %s: %d tracks, %d detections embedded (%.0fs)", i + 1, len(cold),
                    name, len(det), sum(len(f) for f, _e in det.values()), time.time() - t0)


def stage_arm(data_dir: Path, out_dir: Path, results_dir: Path, seqs: list[str], split: str, *,
              reader: str = "v6") -> dict:
    """CPU: the frozen EIoU arm on the S4b detections, scored (``reader`` picks the OCR evidence)."""
    import tools.gsr_eiou as eiou  # noqa: PLC0415

    eiou.BOX_SUBDIR = "detbox_cache" + VARIANT
    pv = percrop_variant(reader)
    stem = f"{split}{VARIANT}{'' if reader == 'v6' else '_' + reader}"
    res = eiou.run_point(data_dir, out_dir, seqs, eiou.EiouParams(e=0.3, rounds=1, w_app=0.5,
                                                                 app_max=0.30),
                         embedder="clip" + VARIANT, tau=TAU,
                         tag=f"v6{stem}_clip_e0.3r1w0.5a0.3", percrop_variant=pv,
                         positions_subdir="positions_gate" + VARIANT)
    res["detector"] = {"weights": str(WEIGHTS), "md5": WEIGHTS_MD5, "arm_variant": VARIANT}
    results_dir.mkdir(parents=True, exist_ok=True)
    dest = results_dir / f"gsr_v6det_{stem}.json"
    dest.write_text(json.dumps({"split": split, "seqs": seqs, "arm": res}, indent=1, default=str),
                    encoding="utf-8")
    h = res["gs_hota"]
    print(f"v6det {split}: GS-HOTA {h['GS-HOTA']:.4f} DetA {h['GS-DetA']:.4f} "
          f"AssA {h['GS-AssA']:.4f} IDF1 {h['IDF1']:.4f}")
    print(f"wrote {dest}")
    return res


def _sha256(path: Path) -> str:
    """Hex sha256 of a file (pure IO)."""
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_v6(dev_arms: dict, *, detector_in: bool, dest: Path = Path("results/gsr_v6_frozen.json"),
              gate_t38: float = 40.0) -> dict:
    """Pre-declare the v6 bundle -- component manifest, hashes, DEV evidence -- before TEST-38.

    Args:
        dev_arms: ``{arm name: {"gs_hota": ..., "note": ...}}`` -- the leave-one-out DEV table the
            bundle was chosen on.
        detector_in: Whether the S4b detector is part of the frozen bundle.
        dest: Where the freeze is written.
        gate_t38: The pre-declared TEST-38 gate (STATUS 2026-08-06: 40.0).

    Returns:
        The written record.
    """
    import hashlib  # noqa: PLC0415
    from datetime import datetime, timezone  # noqa: PLC0415

    from generator.jersey_id import OCR_PERCROP_VERSION  # noqa: PLC0415
    from tools.gsr_v4 import rule_for  # noqa: PLC0415

    eiou_frozen = Path("results/gsr_eiou_frozen.json")
    v4_frozen = Path("results/gsr_v4_frozen.json")
    variant = ("_v6" + VARIANT if detector_in else "_v6") + "_eiou"
    record = {
        "declared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": "v6-1.0",
        "parents": {
            str(p): {"sha256": _sha256(p),
                     "declared_at": json.loads(p.read_text(encoding="utf-8"))["declared_at"]}
            for p in (eiou_frozen, v4_frozen) if p.exists()},
        "components": {
            "association": {
                "in_bundle": True, "name": "clean-room ExpansionIoU (S5)",
                "params": json.loads(eiou_frozen.read_text(encoding="utf-8"))["association"],
                "module_sha256": _sha256(Path("tools/gsr_eiou.py")),
                "embedder": "clip" + (VARIANT if detector_in else "")},
            "jersey_reader": {
                "in_bundle": True, "name": "PARSeq v6 arm 4t (S2)",
                "ckpt": str(PARSEQ_CKPT), "ckpt_md5": hashlib.md5(  # noqa: S324
                    PARSEQ_CKPT.read_bytes()).hexdigest(),
                "percrop_version": OCR_PERCROP_VERSION,
                "aggregation_rule": rule_for("0.80", variant),
                "aggregation_rule_source": "the incumbent 0.80-floor freeze (S3 arm A), NOT the "
                                           "v6 sweep's denser rule (S3 §6: measured worse)"},
            "detector": {
                "in_bundle": detector_in, "name": "YOLOv8s GSR+SoccerNet-v3 fine-tune (S4b)",
                "weights": str(WEIGHTS) if detector_in else
                           "uisikdag/yolo-v8-football-players-detection (shipped)",
                "md5": WEIGHTS_MD5 if detector_in else None},
        },
        "downstream": {
            "aggregation_floor": "0.80", "connector_tau": TAU, "jersey_compatible_merge": True,
            "confusion_prior_refit": False,
            "positions_source": ("positions_gate" + VARIANT if detector_in else "positions_gate")
                                + " -> re-associated (_eiou)",
            "percrop_variant": variant},
        "legitimacy": {"team_map": "free (geometric resolver)", "roster": "self (own reads)",
                       "standard": "results/GSR_DELEAK.md -- no GT read in the prediction chain"},
        "provenance": {
            "dev20_ablation": "laptop (RTX 3050, py3.14, supervision 0.28.x) -- every DEV arm and "
                              "its controls produced on that one stack",
            "test38_and_test49_gpu_stages": "a100server1 GPU 1 (py3.11 conda 'gsr', torch "
                                            "2.5.1+cu121, supervision 0.30.0), per the 2026-08-06 "
                                            "directive to free the laptop GPU",
            "test38_and_test49_cpu_stages": "laptop (re-association, connector, solver, official "
                                            "scorer)",
            "same_machine_rule": "every paired comparison uses arms whose GPU artifacts came from "
                                 "ONE machine; the detector-OUT control is therefore re-extracted "
                                 "server-side rather than read from the laptop cache",
            "cross_stack_check": {"sequence": "SNGS-021", "detector": "S4b",
                                  "paired_detections": 0.944, "pitch_disagreement_median_m": 0.0657,
                                  "pitch_disagreement_p90_m": 0.307, "role_agreement": 1.0,
                                  "team_agreement": 1.0},
            "eiou_module_drift": "tools/gsr_eiou.py sha256 differs from gsr_eiou_frozen.json "
                                 "(S3 added per-variant plumbing); AST-checked: expand_boxes, "
                                 "eiou_matrix, _match, associate, relink_sequence and attach_boxes "
                                 "are identical to the frozen module -- only clear_arm_caches, "
                                 "run_point, sweep and main changed",
        },
        "code_sha256": {str(p): _sha256(p) for p in (
            Path("tools/gsr_eiou.py"), Path("tools/gsr_v4.py"), Path("tools/gsr_v6det.py"),
            Path("eval/gsr_jersey.py"), Path("generator/jersey_id.py"))},
        "dev20": dev_arms,
        "chosen_on": "valid DEV-20 leave-one-out ablation (results/GSR_V6.md)",
        "pre_declared_gate": {"split": "valid TEST-38", "gs_hota_min": gate_t38,
                              "source": "STATUS.md 2026-08-06, raised from the plan's 37.5",
                              "controls": {"v4": 36.71, "eiou_v5": 39.54}},
        "note": "declared BEFORE any TEST-38 read of this bundle; the official test-49 split is "
                "untouched at declaration time.",
    }
    dest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("frozen v6 bundle -> %s", dest)
    return record


def stage_package(data_dir: Path, out_dir: Path, results_dir: Path, names: list[str]) -> dict:
    """The official test-49 tail: legitimacy audit, package, zip self-score (CPU).

    Runs only after :func:`stage_arm` has written the arm for ``names``; every check is the one
    ``tools.gsr_v4.run_testsplit`` applies to an on-record submission.
    """
    from tools.gsr_calibfill import verify_gtfree  # noqa: PLC0415
    from tools.gsr_calibgate import zip_selfscore  # noqa: PLC0415
    from tools.gsr_deleak import package_free  # noqa: PLC0415
    from tools.gsr_v4 import config_key  # noqa: PLC0415

    import tools.gsr_eiou as eiou  # noqa: PLC0415

    eiou.set_embedder("clip" + VARIANT + eiou.VARIANT)
    variant = "_v6" + VARIANT + eiou.VARIANT
    arm_dir = out_dir / f"deleak_v6test{VARIANT}_clip_e0.3r1w0.5a0.3"
    gta_dir = out_dir / ("eval_v4_gta_" + config_key("0.80", TAU, True, variant, False))
    frozen = json.loads(Path("results/gsr_v6_frozen.json").read_text(encoding="utf-8"))
    out: dict = {"legitimacy": verify_gtfree(arm_dir, gta_dir, data_dir,
                                             out_dir / ("positions_gate" + VARIANT), names)}
    out["manifest"] = package_free(arm_dir, names, frozen, stem="gtfree_v6")
    zip_path = Path(out["manifest"]["zip"])
    out["zip_selfscore"] = zip_selfscore(zip_path, data_dir, names,
                                         zip_path.parent / "zip_selfscore_gtfree_v6.json")[
                                             "combined"]
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "gsr_v6_testsplit_package.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    logger.info("legitimacy: %s", out["legitimacy"])
    logger.info("zip self-score: %s", out["zip_selfscore"])
    return out


STAGES = ("extract", "gate", "percrop", "boxes", "embed", "arm", "package")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gsr"))
    ap.add_argument("--results-dir", type=Path, default=Path("results/gsr_benchmark"))
    ap.add_argument("--split", choices=("dev", "t38", "test"), default="dev")
    ap.add_argument("--seqs", default=None, help="comma-separated sequence names")
    ap.add_argument("--stages", default="all", help=f"comma-separated subset of {STAGES} or 'all'")
    ap.add_argument("--reader", choices=("v6", "incumbent", "edl"), default="v6",
                    help="OCR evidence for the percrop/arm stages ('incumbent' = the reader "
                         "leave-one-out on the new detections; 'edl' = the v7 evidential head)")
    ap.add_argument("--edl-head", type=Path, default=None,
                    help="Dirichlet evidential head checkpoint (reader='edl')")
    ap.add_argument("--leg-thresh", type=float, default=0.5,
                    help="external legibility floor for the percrop stage; 0.0 disables the gate")
    ap.add_argument("--detector", choices=("s4b", "control"), default="s4b",
                    help="'control' re-extracts with the shipped detector (same-stack control)")
    args = ap.parse_args()
    set_arm(args.detector)

    from eval.gsr_identity import split_sequences  # noqa: PLC0415

    check_weights()
    if args.seqs:
        seqs = args.seqs.split(",")
    elif args.split == "test":
        from tools.gsr_deleak import split_names  # noqa: PLC0415

        seqs = split_names(args.data_dir, "test")
    else:
        dev, t38 = split_sequences(args.data_dir, args.out_dir)
        seqs = dev if args.split == "dev" else t38
    stages = STAGES if args.stages == "all" else tuple(args.stages.split(","))
    logger.info("v6det: %d sequences, stages %s", len(seqs), stages)
    for st in stages:
        t0 = time.time()
        if st == "extract":
            stage_extract(args.data_dir, args.out_dir, seqs)
        elif st == "gate":
            stage_gate(args.out_dir, seqs)
        elif st == "percrop":
            stage_percrop(args.data_dir, args.out_dir, seqs, reader=args.reader,
                          edl_head=args.edl_head, leg_thresh=args.leg_thresh)
        elif st == "boxes":
            stage_boxes(args.data_dir, args.out_dir, seqs)
        elif st == "embed":
            stage_embed(args.data_dir, args.out_dir, seqs)
        elif st == "arm":
            stage_arm(args.data_dir, args.out_dir, args.results_dir, seqs, args.split,
                      reader=args.reader)
        elif st == "package":
            stage_package(args.data_dir, args.out_dir, args.results_dir, seqs)
        else:
            raise SystemExit(f"unknown stage {st!r}; pick from {STAGES}")
        logger.info("stage %s done in %.0fs", st, time.time() - t0)


if __name__ == "__main__":
    main()
