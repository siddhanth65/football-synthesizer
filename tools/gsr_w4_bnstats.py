"""Campaign v8 session W4: per-sequence BatchNorm test-time adaptation, end to end on GS-HOTA.

The registered arm (`results/GSR_V8_W4.md` §1.2) recomputes the detector's BatchNorm running
statistics on each test sequence's OWN unlabeled frames -- no labels, no gradients, no loss -- and
then runs the frozen v6 chain on the adapted detections. Because every GPU stage below extraction
re-detects and matches back to the persisted foot points (§0.4), the adapted detector has to be
carried through **all** of them; that is what `generator.extract.BN_STATS` is for, and why this
runner drives the `tools.gsr_v6det` stages one sequence at a time with the module rebound.

The secondary arm (§1.3) does the same to the CLIP embedder's `BatchNorm1d(256)` BNNeck on the
shipped detections, which needs only one GPU stage.

CLI::

    python -m tools.gsr_w4_bnstats --stages repro                    # control reproducibility proof
    python -m tools.gsr_w4_bnstats --stages stats,extract,gate,percrop,boxes,embed,arm
    python -m tools.gsr_w4_bnstats --stages neck,neckarm             # the secondary arm
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("gsr_w4")

#: Artifact suffix of the registered (detector) arm.
VARIANT = "_w4bn"
#: Artifact suffix of the secondary (embedder BNNeck) arm.
NECK_VARIANT = "_w4neck"
#: The S4b detector: the shipped v6 bundle's detector, unchanged (only its BN buffers move).
WEIGHTS = Path("models/gsr_det/gsr_v3_ft_b_last.pt")
#: Frame stride of the detector adaptation pass -- REGISTERED a priori, never tuned (§1.2).
STRIDE = 5
#: Frame stride of the embedder BNNeck adaptation pass -- REGISTERED a priori (§1.3).
NECK_STRIDE = 10
#: The frozen downstream point (results/gsr_eiou_frozen.json + GSR_S3_READER.md arm A).
TAU = 0.450
#: The W2/W2b probe, unchanged.
PROBE = ["SNGS-024", "SNGS-027", "SNGS-039", "SNGS-042", "SNGS-045",
         "SNGS-048", "SNGS-051", "SNGS-054", "SNGS-057", "SNGS-078"]
CONTROL = Path("results/gsr_benchmark/gsr_v7_control_dev.json")
#: The shipped lineage the secondary arm and the reproducibility proof are measured against.
SHIPPED_VARIANT = "_v6det"


def set_lineage(name: str) -> None:
    """Point every stage at the adapted lineage (``'w4bn'``) or a same-stack control (``'w4off'``).

    The control lineage exists for the same-machine/same-stack pairing rule (`tools/gsr_v6det.py`
    ``--detector control``): if a flag-OFF re-extraction does not reproduce the on-record artifacts
    bit for bit, the arm may only be paired against a control re-extracted on THIS stack, so the
    incumbent detector gets its own full re-run rather than reusing the cached lineage.
    """
    global VARIANT  # noqa: PLW0603 - one switch for every stage in the module

    VARIANT = {"w4bn": "_w4bn", "w4off": "_w4off"}[name]


def stats_path(out_dir: Path, name: str) -> Path:
    """Per-sequence detector BN snapshot path (one file per sequence, never shared)."""
    return out_dir / ("bn_stats" + VARIANT) / f"{name}.pt"


def frames_of(seq_dir: Path, stride: int):
    """Yield every ``stride``-th frame of a sequence as RGB uint8, streamed (pure IO)."""
    import cv2  # noqa: PLC0415

    for path in sorted((seq_dir / "img1").glob("*.jpg"))[::stride]:
        bgr = cv2.imread(str(path))
        if bgr is not None:
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _shift_report(buffers: dict, base: dict) -> dict:
    """How far one sequence's adapted BN buffers sit from the checkpoint's (relative L2, pure)."""
    import torch  # noqa: PLC0415

    shift = {kind: sorted(float(torch.norm(buffers[k] - base[k]) / (torch.norm(base[k]) + 1e-9))
                          for k in base if k.endswith(kind))
             for kind in ("running_mean", "running_var")}
    return {"rel_shift_median": {k: round(v[len(v) // 2], 4) for k, v in shift.items()},
            "rel_shift_max": {k: round(v[-1], 4) for k, v in shift.items()},
            "rel_shift_min": {k: round(v[0], 4) for k, v in shift.items()}}


def stage_stats(data_dir: Path, out_dir: Path, results_dir: Path, seqs: list[str]) -> dict:
    """GPU: compute and cache each sequence's own BN statistics, then report how far they moved.

    Resumable by disk state: sequences whose snapshot exists are only re-read (CPU) for the report.
    """
    import torch  # noqa: PLC0415

    from generator.bn_adapt import adapt_yolo, stats_of  # noqa: PLC0415
    from generator.extract import _build_detector  # noqa: PLC0415

    cold = [s for s in seqs if not stats_path(out_dir, s).exists()]
    logger.info("stats: %d/%d cold", len(cold), len(seqs))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = stats_of(_build_detector("cpu", "football", weights=str(WEIGHTS)).yolo.model)
    out: dict[str, dict] = {}
    for i, name in enumerate(cold):
        t0 = time.time()
        det = _build_detector(device, "football", weights=str(WEIGHTS))
        snap = adapt_yolo(det.yolo, frames_of(data_dir / name, STRIDE))
        dest = stats_path(out_dir, name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v for k, v in snap.items()
                    if k.endswith(("running_mean", "running_var"))}, dest)
        logger.info("[%d/%d] %s: %d frames, %d BN layers (%.0fs)", i + 1, len(cold), name,
                    snap["n_frames"], snap["n_bn"], time.time() - t0)
        del det
        torch.cuda.empty_cache()
    for name in seqs:
        buffers = torch.load(stats_path(out_dir, name), map_location="cpu", weights_only=True)
        out[name] = {"n_bn": len(buffers) // 2, "stride": STRIDE, "momentum": None,
                     **_shift_report(buffers, base)}
        print(f"stats {name}: shift median mean/var "
              f"{out[name]['rel_shift_median']['running_mean']:.4f}/"
              f"{out[name]['rel_shift_median']['running_var']:.4f}  max "
              f"{out[name]['rel_shift_max']['running_mean']:.4f}/"
              f"{out[name]['rel_shift_max']['running_var']:.4f}")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "gsr_v8_w4_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def stage_extract(data_dir: Path, out_dir: Path, seqs: list[str]) -> dict:
    """GPU: re-extract positions, one sequence at a time, with that sequence's BN statistics.

    Under the ``w4off`` lineage the stage is OFF (``BN_STATS is None``) and this is the identical
    code path the shipped chain runs -- the control rebuild that proves the flag changes nothing
    when it is not set (§1.5 guard 4).

    Args:
        data_dir: GSR ground-truth folder.
        out_dir: Pipeline output root.
        seqs: Sequences to extract.

    Returns:
        Per-sequence row/track/second counts.
    """
    import generator.extract as ex  # noqa: PLC0415
    from generator.calibrate import PnLCalibCalibrator  # noqa: PLC0415

    adapt = VARIANT == "_w4bn"
    dest = out_dir / ("positions" + VARIANT)
    dest.mkdir(parents=True, exist_ok=True)
    todo = [s for s in seqs if not (dest / f"{s}.parquet").exists()]
    logger.info("extract: %d/%d cold (adapt=%s)", len(todo), len(seqs), adapt)
    calibrator, stats = PnLCalibCalibrator() if todo else None, {}
    for i, name in enumerate(todo):
        t0 = time.time()
        ex.BN_STATS = str(stats_path(out_dir, name)) if adapt else None
        try:
            df = ex.extract_positions(str(data_dir / name / "img1" / "%06d.jpg"),
                                      dest / f"{name}.parquet", sample_every=1,
                                      calibrator=calibrator, detector_name="football",
                                      detector_weights=str(WEIGHTS), tracker_name="bytetrack",
                                      calib_period=1)
        finally:
            ex.BN_STATS = None
        stats[name] = {"rows": int(len(df)), "tracks": int(df["track_id"].nunique()),
                       "seconds": round(time.time() - t0, 1)}
        logger.info("[%d/%d] %s: %d rows, %d tracks (%.0fs)", i + 1, len(todo), name,
                    stats[name]["rows"], stats[name]["tracks"], stats[name]["seconds"])
    return stats


def _per_sequence(fn, out_dir: Path, seqs: list[str], *args) -> None:
    """Run a ``tools.gsr_v6det`` GPU stage one sequence at a time with that sequence's BN stats.

    The v6det stages build their detector once and loop internally, which cannot express
    per-sequence statistics; calling them with a single-element sequence list can, at the cost of
    one model construction per sequence (~2 s).
    """
    import generator.extract as ex  # noqa: PLC0415

    for name in seqs:
        ex.BN_STATS = str(stats_path(out_dir, name)) if VARIANT == "_w4bn" else None
        try:
            fn(*args, [name])
        finally:
            ex.BN_STATS = None


def _v6det():
    """The v6det stage module, rebound to this arm's artifact suffix (its stages key off it)."""
    import tools.gsr_v6det as v6  # noqa: PLC0415

    v6.VARIANT, v6.WEIGHTS = VARIANT, WEIGHTS
    return v6


def stage_gate(out_dir: Path, seqs: list[str]) -> dict:
    """CPU: the frozen calibration re-gate + gap fill on this arm's extraction."""
    v6 = _v6det()
    return v6.stage_gate(out_dir, seqs)


def stage_percrop(data_dir: Path, out_dir: Path, seqs: list[str]) -> None:
    """GPU: the v6 PARSeq reader on this arm's boxes."""
    v6 = _v6det()
    _per_sequence(v6.stage_percrop, out_dir, seqs, data_dir, out_dir)


def stage_boxes(data_dir: Path, out_dir: Path, seqs: list[str]) -> None:
    """GPU: per-detection box cache for the EIoU tracker, on this arm's detections."""
    v6 = _v6det()
    _per_sequence(v6.stage_boxes, out_dir, seqs, data_dir, out_dir)


def stage_embed(data_dir: Path, out_dir: Path, seqs: list[str]) -> None:
    """GPU: CLIP per-detection embeddings on this arm's detections (shipped embedder weights)."""
    v6 = _v6det()
    _per_sequence(v6.stage_embed, out_dir, seqs, data_dir, out_dir)


def stage_neck(data_dir: Path, out_dir: Path, seqs: list[str]) -> dict:
    """GPU (secondary arm): CLIP embeddings whose BNNeck statistics are the sequence's own.

    Two passes over the same crops: pass 1 (stride :data:`NECK_STRIDE`) collects the statistics with
    the BNNeck in cumulative-averaging mode and throws its embeddings away; pass 2 embeds for real at
    the shipped stride with the adapted buffers. Detections are the SHIPPED ones, so this arm differs
    from the control in the embedder's 256 BN buffers alone.
    """
    import shutil  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from generator.bn_adapt import recomputing  # noqa: PLC0415
    from generator.extract import _build_detector  # noqa: PLC0415
    from generator.gta_link import GtaParams, load_or_build_det_embeddings  # noqa: PLC0415

    from tools.clip_embedder import ClipEmbedder  # noqa: PLC0415

    cache = out_dir / ("detembed_cache_clip" + NECK_VARIANT)
    scratch = out_dir / ("detembed_cache_clip" + NECK_VARIANT + "_adaptpass")
    cache.mkdir(parents=True, exist_ok=True)
    cold = [s for s in seqs if not (cache / f"{s}.npz").exists()]
    logger.info("neck: %d/%d cold", len(cold), len(seqs))
    if not cold:
        return {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = ClipEmbedder(device=device)
    detector = _build_detector(device, "football", weights=str(WEIGHTS))  # shipped, unadapted
    base = {k: v.clone() for k, v in embedder.bn.state_dict().items()}
    stats: dict[str, dict] = {}
    for i, name in enumerate(cold):
        t0 = time.time()
        embedder.bn.load_state_dict(base)  # reset per sequence: no cross-sequence leakage
        df = pd.read_parquet(out_dir / ("positions_gate" + SHIPPED_VARIANT) / f"{name}.parquet")
        shutil.rmtree(scratch, ignore_errors=True)
        with recomputing(embedder.bn), torch.no_grad():
            load_or_build_det_embeddings(data_dir / name, df, scratch / f"{name}.npz",
                                         params=GtaParams(frame_stride=NECK_STRIDE),
                                         embedder=embedder, detector=detector)
        shift = float(torch.norm(embedder.bn.running_mean - base["running_mean"])
                      / (torch.norm(base["running_mean"]) + 1e-9))
        det = load_or_build_det_embeddings(data_dir / name, df, cache / f"{name}.npz",
                                           params=GtaParams(frame_stride=2), embedder=embedder,
                                           detector=detector)
        stats[name] = {"tracks": len(det), "detections": sum(len(f) for f, _e in det.values()),
                       "n_batches": int(embedder.bn.num_batches_tracked),
                       "rel_shift_running_mean": round(shift, 4),
                       "seconds": round(time.time() - t0, 1)}
        logger.info("[%d/%d] %s: %d dets, %d adapt batches, mean shift %.4f (%.0fs)", i + 1,
                    len(cold), name, stats[name]["detections"], stats[name]["n_batches"],
                    shift, stats[name]["seconds"])
    shutil.rmtree(scratch, ignore_errors=True)
    embedder.bn.load_state_dict(base)
    return stats


def score_arm(data_dir: Path, out_dir: Path, results_dir: Path, seqs: list[str], *,
              arm: str) -> dict:
    """CPU: the frozen EIoU + v6-reader arm on one W4 arm's artifacts, scored and paired.

    Args:
        data_dir: GSR ground-truth folder.
        out_dir: Pipeline output root.
        results_dir: Where the raw JSON lands.
        seqs: The probe.
        arm: ``'w4bn'`` (registered, adapted detections), ``'w4off'`` (the same-stack control
            re-extraction) or ``'w4neck'`` (secondary, shipped detections + adapted BNNeck
            embeddings).

    Returns:
        The scored payload with its paired-vs-control statistics.
    """
    import tools.gsr_eiou as eiou  # noqa: PLC0415

    from tools.gsr_v7_edl import paired_vs_control  # noqa: PLC0415

    det_variant = SHIPPED_VARIANT if arm == "w4neck" else VARIANT
    embedder = "clip" + (NECK_VARIANT if arm == "w4neck" else VARIANT)
    eiou.BOX_SUBDIR = "detbox_cache" + det_variant
    t0 = time.time()
    res = eiou.run_point(data_dir, out_dir, seqs,
                         eiou.EiouParams(e=0.3, rounds=1, w_app=0.5, app_max=0.30),
                         embedder=embedder, tau=TAU, tag=f"v8{arm}_clip_e0.3r1w0.5a0.3",
                         percrop_variant="_v6" + det_variant,
                         positions_subdir="positions_gate" + det_variant)
    res["w4"] = {"arm": arm, "detector_variant": det_variant, "embedder": embedder,
                 "stride": NECK_STRIDE if arm == "w4neck" else STRIDE,
                 "momentum": None, "seconds": round(time.time() - t0, 1)}
    res["paired_vs_control"] = paired_vs_control(res, seqs, control=CONTROL)
    results_dir.mkdir(parents=True, exist_ok=True)
    dest = results_dir / f"gsr_v8_w4_{arm}_probe.json"
    dest.write_text(json.dumps({"probe": seqs, "arm": res}, indent=1, default=str),
                    encoding="utf-8")
    h, p = res["gs_hota"], res["paired_vs_control"]
    print(f"w4 {arm}: GS-HOTA {h['GS-HOTA']:.4f} (control {p['control_gs_hota']:.4f}) "
          f"DetA {h['GS-DetA']:.4f} AssA {h['GS-AssA']:.4f} LocA {h['GS-LocA']:.4f}")
    print(f"w4 {arm}: paired mean {p['mean']:+.4f} median {p['median']:+.4f} "
          f"helped {p['helped']} hurt {p['hurt']} worst {p['worst']:+.4f} best {p['best']:+.4f} "
          f"p={p['wilcoxon_p']:.3g}")
    print(f"wrote {dest}")
    return res


def _diff_positions(a_path: Path, b_path: Path) -> dict:
    """Compare two positions parquets row for row (pure IO + arithmetic)."""
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    a = pd.read_parquet(a_path).reset_index(drop=True)
    b = pd.read_parquet(b_path).reset_index(drop=True)
    rec: dict = {"rows_a": len(a), "rows_b": len(b), "tracks_a": int(a["track_id"].nunique()),
                 "tracks_b": int(b["track_id"].nunique()), "identical": bool(a.equals(b))}
    if a.shape == b.shape:
        num = ["pitch_x", "pitch_y", "image_x", "image_y", "conf"]
        diff = np.abs(a[num].to_numpy() - b[num].to_numpy())
        rec["max_abs_numeric"] = float(np.nanmax(diff)) if diff.size else 0.0
        for col in ("track_id", "role", "team"):
            rec[f"{col}_mismatches"] = int((a[col].to_numpy() != b[col].to_numpy()).sum())
    return rec


def stage_repro(out_dir: Path, results_dir: Path, seqs: list[str]) -> dict:
    """CPU: how reproducible is the extraction the pairing rests on?

    Two comparisons, both against ``positions_w4off`` (the flag-OFF re-extraction, i.e. the shipped
    chain run through the identical code path with ``BN_STATS is None``):

    * **cross-stack** vs the on-record ``positions_v6det``. A difference means the on-record control
      may not be paired against anything extracted today, and the arm needs the same-stack control
      this module's ``w4off`` lineage builds.
    * **run-to-run** vs ``positions_w4off_rerun`` when present -- the same code, same stack, same
      sequence, run twice. This prices the noise floor the BN delta has to beat.
    """
    out: dict[str, dict] = {}
    for name in seqs:
        path = out_dir / "positions_w4off" / f"{name}.parquet"
        if not path.exists():
            continue
        rec = {"vs_onrecord": _diff_positions(
            path, out_dir / ("positions" + SHIPPED_VARIANT) / f"{name}.parquet")}
        rerun = out_dir / "positions_w4off_rerun" / f"{name}.parquet"
        if rerun.exists():
            rec["vs_rerun_same_stack"] = _diff_positions(path, rerun)
        out[name] = rec
        for kind, d in rec.items():
            print(f"repro {name} {kind}: identical={d['identical']} "
                  f"rows {d['rows_a']}/{d['rows_b']} tracks {d['tracks_a']}/{d['tracks_b']} "
                  f"max_abs={d.get('max_abs_numeric')}")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "gsr_v8_w4_repro.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def stage_summary(results_dir: Path, seqs: list[str], *, arm: str, control: str) -> dict:
    """CPU: pair two scored W4 arms per sequence and apply the registered bars (§1.4).

    Args:
        results_dir: Where the arm JSONs live.
        seqs: The probe.
        arm: Arm name whose JSON is the treatment.
        control: Arm name whose JSON is the control, or ``'onrecord'`` for
            :data:`CONTROL` (the on-record v7 DEV control).

    Returns:
        The paired record, also written to ``gsr_v8_w4_summary_<arm>_vs_<control>.json``.
    """
    from eval.gsr_identity import paired_stats  # noqa: PLC0415

    def load(name: str) -> dict:
        if name == "onrecord":
            return json.loads(CONTROL.read_text(encoding="utf-8"))["arm"]
        return json.loads((results_dir / f"gsr_v8_w4_{name}_probe.json").read_text(
            encoding="utf-8"))["arm"]

    a, c = load(arm), load(control)
    st = paired_stats(c["gs_hota_per_seq"], a["gs_hota_per_seq"], seqs)
    rec = {"arm": arm, "control": control, "n": len(seqs),
           "arm_gs_hota": a["gs_hota"]["GS-HOTA"], "control_gs_hota": c["gs_hota"]["GS-HOTA"],
           "pooled_delta": a["gs_hota"]["GS-HOTA"] - c["gs_hota"]["GS-HOTA"], "paired": st,
           "per_seq": {n: {"control": c["gs_hota_per_seq"][n], "arm": a["gs_hota_per_seq"][n],
                           "delta": a["gs_hota_per_seq"][n] - c["gs_hota_per_seq"][n]}
                       for n in seqs},
           "kill_bar": {"threshold": 0.10, "measured": st["mean"],
                        "verdict": "PASS" if st["mean"] >= 0.10 else "REGISTERED FAIL"}}  # noqa: PLR2004
    dest = results_dir / f"gsr_v8_w4_summary_{arm}_vs_{control}.json"
    dest.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"{'sequence':<12}{'control':>10}{'arm':>10}{'delta':>10}")
    for n in seqs:
        p = rec["per_seq"][n]
        print(f"{n:<12}{p['control']:>10.3f}{p['arm']:>10.3f}{p['delta']:>+10.3f}")
    print(f"pooled {rec['control_gs_hota']:.4f} -> {rec['arm_gs_hota']:.4f} "
          f"({rec['pooled_delta']:+.4f})")
    print(f"paired mean {st['mean']:+.4f} median {st['median']:+.4f} helped {st['helped']} "
          f"hurt {st['hurt']} worst {st['worst']:+.4f} best {st['best']:+.4f} "
          f"p={st['wilcoxon_p']:.3g}")
    print(f"kill bar >= +0.10: {rec['kill_bar']['verdict']}")
    print(f"wrote {dest}")
    return rec


def _demo() -> None:
    """Self-check: artifact naming is arm-separated and the probe is the registered one."""
    assert VARIANT != NECK_VARIANT != SHIPPED_VARIANT
    assert stats_path(Path("o"), "SNGS-024") == Path("o/bn_stats_w4bn/SNGS-024.pt")
    assert len(PROBE) == 10 and len(set(PROBE)) == 10
    assert PROBE[0] == "SNGS-024" and PROBE[-1] == "SNGS-078"
    assert STRIDE == 5 and NECK_STRIDE == 10, "registered strides must not drift"
    v6 = _v6det()
    assert v6.VARIANT == VARIANT and v6.percrop_variant("v6") == "_v6" + VARIANT
    set_lineage("w4off")
    assert VARIANT == "_w4off" and _v6det().VARIANT == "_w4off"
    set_lineage("w4bn")
    assert VARIANT == "_w4bn" and _v6det().VARIANT == "_w4bn"
    print("gsr_w4_bnstats self-check OK")


STAGES = ("stats", "extract", "gate", "percrop", "boxes", "embed", "arm", "neck", "neckarm",
          "repro", "summary", "demo")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gsr"))
    ap.add_argument("--results-dir", type=Path, default=Path("results/gsr_benchmark"))
    ap.add_argument("--seqs", default=None, help="comma-separated sequence names (default: probe)")
    ap.add_argument("--stages", default="demo", help=f"comma-separated subset of {STAGES}")
    ap.add_argument("--lineage", choices=("w4bn", "w4off"), default="w4bn",
                    help="'w4off' runs the identical stages with the BN stage OFF (same-stack "
                         "control re-extraction)")
    ap.add_argument("--summary", default="w4bn:onrecord",
                    help="'<arm>:<control>' for --stages summary")
    args = ap.parse_args()
    set_lineage(args.lineage)
    seqs = args.seqs.split(",") if args.seqs else list(PROBE)
    logger.info("w4: %d sequences, lineage %s, stages %s", len(seqs), args.lineage, args.stages)
    for st in args.stages.split(","):
        t0 = time.time()
        if st == "stats":
            stage_stats(args.data_dir, args.out_dir, args.results_dir, seqs)
        elif st == "extract":
            stage_extract(args.data_dir, args.out_dir, seqs)
        elif st == "gate":
            stage_gate(args.out_dir, seqs)
        elif st == "percrop":
            stage_percrop(args.data_dir, args.out_dir, seqs)
        elif st == "boxes":
            stage_boxes(args.data_dir, args.out_dir, seqs)
        elif st == "embed":
            stage_embed(args.data_dir, args.out_dir, seqs)
        elif st == "arm":
            score_arm(args.data_dir, args.out_dir, args.results_dir, seqs,
                      arm=VARIANT.lstrip("_"))
        elif st == "neck":
            stage_neck(args.data_dir, args.out_dir, seqs)
        elif st == "neckarm":
            score_arm(args.data_dir, args.out_dir, args.results_dir, seqs, arm="w4neck")
        elif st == "repro":
            stage_repro(args.out_dir, args.results_dir, seqs)
        elif st == "summary":
            arm, _, control = args.summary.partition(":")
            stage_summary(args.results_dir, seqs, arm=arm, control=control)
        elif st == "demo":
            _demo()
        else:
            raise SystemExit(f"unknown stage {st!r}; pick from {STAGES}")
        logger.info("stage %s done in %.0fs", st, time.time() - t0)


if __name__ == "__main__":
    main()
