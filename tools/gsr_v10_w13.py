"""v10-W13: price the incumbent detector at inference ``imgsz`` 1280, end to end, registered.

Registration: ``results/gsr_v10_w13_registered.json`` (written before any GS-HOTA of either arm).

v10-W12 killed the modern-detector swap and left one candidate behind: **our own S4b weights at
inference resolution 1280**, which dominate their own 640 curve on every component axis measured
(kb v10-w12-003: recall +1.077 points at better precision, median box IoU +0.012, foot error -7.4%,
post-ByteTrack recall +0.71, role accuracy +0.79, player->referee rows -23%, tracks +2%, at ~1.5x
detector inference). Those are components, not GS-HOTA. A detector-side change invalidates every
cached artifact keyed on ``(track_id, frame)``, so pricing it needs a two-arm re-extraction of the
whole chain -- and both arms must sit in ONE extraction window (kb v10-frz-003: ~1.1 GS-HOTA of
run-to-run drift between windows), with the pairing taken only inside it.

The 640 arm is therefore not "the on-record control": it is a freshly extracted same-window twin.

Stages (per detector arm; GPU ones run server-side, sharded):

* ``--stages extract,gate,percrop,boxes,embed`` -- the frozen v6 chain's GPU/CPU artifact build
  (:mod:`tools.gsr_v6det`) with :data:`generator.extract.IMGSZ` bound for the whole process, so
  extraction, crop recovery, box caching and embedding all detect at the same resolution.
* ``--positions`` -- CPU. The geometry riders (``D`` = the shipped v10 bundle, ``A`` = W8's adaptive
  stiffness, the freeze-#2 candidate) replayed from the FROZEN s4 candidate cache onto each arm's
  own gated positions. The candidate cache is detector-independent (PnLCalib heatmaps), so both
  detector arms ride the identical hypothesis pool; only the gate's foot-point plausibility input
  and the reprojected rows differ.
* ``--score`` -- CPU, foreground. The frozen v6 chain per (detector arm x geometry arm), then the
  v9-W7 paired scorer 1280-vs-640 WITHIN each geometry arm, flags ON and OFF.

CLI::

    python -m tools.gsr_v10_w13 --det i1280 --stages extract,gate --seqs SNGS-021,SNGS-024
    python -m tools.gsr_v10_w13 --positions --score
    python -m tools.gsr_v10_w13 --demo
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("gsr_v10_w13")

#: The two detector arms: artifact key -> ultralytics inference ``imgsz``. ``i640`` reproduces the
#: shipped operating point (the S4b checkpoint's own stored ``imgsz`` is 640), freshly extracted.
DET_ARMS: dict[str, int] = {"i640": 640, "i1280": 1280}
#: Geometry riders replayed on both detector arms: ``D`` = the v10 freeze bundle (S then B),
#: ``A`` = v10-W8's adaptive-stiffness variant, the freeze-#2 candidate rider.
GEO_ARMS: tuple[str, ...] = ("D", "A")
#: The FROZEN half-cell (s4) candidate cache -- in the v10 bundle, and detector-independent.
CACHE_S4 = Path("outputs/gsr/v10_w2_full/s4")
#: Frozen temporal fill gap (``results/gsr_calibgate_frozen.json``).
FILL_GAP = 10


def variant_of(det: str) -> str:
    """Artifact suffix for one detector arm (pure). Never a prefix of another arm's suffix."""
    if det not in DET_ARMS:
        raise SystemExit(f"unknown detector arm {det!r}; pick from {sorted(DET_ARMS)}")
    return f"_w13{det}"


def positions_subdir(det: str, geo: str) -> str:
    """Positions directory of one (detector arm, geometry arm) cell (pure)."""
    return f"positions_v10w13{geo}{variant_of(det)}"


def arm_tag(det: str, geo: str) -> str:
    """Chain tag of one cell; names ``deleak_<tag>`` (pure)."""
    return f"v10w13{geo}{variant_of(det)}"


@contextmanager
def detector_at(det: str):
    """Bind the process-wide detector weights + inference resolution for one arm, then restore."""
    import generator.extract as ex  # noqa: PLC0415

    import tools.gsr_v6det as v6  # noqa: PLC0415

    v6.set_arm("s4b")
    v6.set_variant(variant_of(det))
    v6.check_weights()
    v6.use_new_detector()
    before = ex.IMGSZ
    ex.IMGSZ = DET_ARMS[det]
    logger.info("arm %s: variant %s, imgsz %d, weights %s", det, variant_of(det), ex.IMGSZ,
                v6.WEIGHTS)
    try:
        yield v6
    finally:
        ex.IMGSZ = before


def _assert_nonempty(dest: Path, seqs: list[str]) -> None:
    """Fail loudly on a silently empty extraction.

    ``generator.extract.extract_positions`` writes an EMPTY parquet and returns 0 when the OpenCV
    capture reports no frames -- which happened transiently to SNGS-096 in both arms of this
    session's first pass, under 8-way parallel load, on a sequence whose 750 frames are present and
    readable. Left alone it survives every downstream stage (the gate copies it, the box cache and
    the OCR write empty files) and only surfaces as a missing embedding cache. Since
    :func:`tools.gsr_v6det.stage_extract` is resumable by disk state, the empty parquet must be
    DELETED before a re-run; this check names the files to delete rather than deleting them.
    """
    empty = [s for s in seqs
             if not (dest / f"{s}.parquet").exists() or len(pd.read_parquet(dest / f"{s}.parquet"))
             == 0]
    if empty:
        raise SystemExit(f"empty extraction for {empty} in {dest} -- delete those parquets "
                         "(and every downstream artifact of the same sequences) and re-run")


def run_stages(det: str, stages: list[str], data_dir: Path, out_dir: Path, seqs: list[str]) -> dict:
    """Build one detector arm's chain artifacts (``extract``/``gate``/``percrop``/``boxes``/``embed``)."""
    stats: dict[str, object] = {}
    with detector_at(det) as v6:
        for st in stages:
            t0 = time.time()
            if st == "extract":
                stats[st] = v6.stage_extract(data_dir, out_dir, seqs)
                _assert_nonempty(out_dir / ("positions" + variant_of(det)), seqs)
            elif st == "gate":
                stats[st] = v6.stage_gate(out_dir, seqs)
            elif st == "percrop":
                v6.stage_percrop(data_dir, out_dir, seqs, reader="v6")
            elif st == "boxes":
                stats[st] = v6.stage_boxes(data_dir, out_dir, seqs)
            elif st == "embed":
                v6.stage_embed(data_dir, out_dir, seqs)
            else:
                raise SystemExit(f"unknown stage {st!r}")
            logger.info("[%s] stage %s done in %.0f s", det, st, time.time() - t0)
    return stats


def build_positions(out_dir: Path, cache_dir: Path, seqs: list[str], dets: list[str],
                    geos: list[str]) -> dict:
    """Geometry rider -> reprojected positions, per (detector arm, geometry arm)."""
    from generator.postprocess import fill_calibration_gaps  # noqa: PLC0415

    import tools.gsr_v10_w5 as w5  # noqa: PLC0415
    from tools.gsr_v10_w8 import arm_homographies  # noqa: PLC0415
    from tools.gsr_w2_calibswap import swap_positions  # noqa: PLC0415

    stats: dict[str, dict] = {}
    for det in dets:
        base = "positions_gate" + variant_of(det)
        w5.POSITIONS_SUBDIR = base  # read by load_seq inside arm_homographies
        for seq in seqs:
            df = pd.read_parquet(out_dir / base / f"{seq}.parquet")
            for geo in geos:
                homs, st = arm_homographies(geo, seq, out_dir, cache_dir)
                tab, sw = swap_positions(df, homs, origin=(0.0, 0.0), mode="t1")
                tab = fill_calibration_gaps(tab, max_gap=FILL_GAP)
                sub = out_dir / positions_subdir(det, geo)
                sub.mkdir(parents=True, exist_ok=True)
                tab.to_parquet(sub / f"{seq}.parquet", index=False)
                st.update({k: sw[k] for k in ("pitch_rows_control", "pitch_rows_final")})
                st["pitch_rows_after_fill"] = int(np.isfinite(tab["pitch_x"]).sum())
                stats.setdefault(f"{det}/{geo}", {})[seq] = st
            logger.info("%s %s: %s", det, seq,
                        {g: stats[f"{det}/{g}"][seq]["pitch_rows_after_fill"] for g in geos})
    return stats


def score(data_dir: Path, out_dir: Path, seqs: list[str], dets: list[str], geos: list[str]) -> dict:
    """The frozen v6 chain per cell, then the paired 1280-vs-640 scorer inside each geometry arm."""
    import tools.gsr_eiou as eiou  # noqa: PLC0415

    from tools.gsr_v6det import TAU  # noqa: PLC0415
    from tools.gsr_v9_w7 import score_pair  # noqa: PLC0415

    raw: dict[str, dict] = {}
    for det in dets:
        eiou.BOX_SUBDIR = "detbox_cache" + variant_of(det)
        for geo in geos:
            res = eiou.run_point(data_dir, out_dir, seqs,
                                 eiou.EiouParams(e=0.3, rounds=1, w_app=0.5, app_max=0.30),
                                 embedder="clip" + variant_of(det), tau=TAU,
                                 tag=arm_tag(det, geo),
                                 percrop_variant="_v6" + variant_of(det),
                                 positions_subdir=positions_subdir(det, geo))
            h = res["gs_hota"]
            raw[f"{det}/{geo}"] = {"gs_hota": h, "gs_hota_per_seq": res["gs_hota_per_seq"],
                                   "eiou_tracks": (res["eiou"]["n_tracks_before"],
                                                   res["eiou"]["n_tracks_after"])}
            print(f"{det}/{geo:<6} GS-HOTA {h['GS-HOTA']:7.4f} DetA {h['GS-DetA']:7.4f} "
                  f"AssA {h['GS-AssA']:7.4f} LocA {h['GS-LocA']:7.4f}", flush=True)
    paired = {}
    for geo in geos:
        if not {"i640", "i1280"} <= set(dets):
            continue
        ctrl = out_dir / f"deleak_{arm_tag('i640', geo)}" / "predictions" / "data"
        arm = out_dir / f"deleak_{arm_tag('i1280', geo)}" / "predictions" / "data"
        paired[geo] = score_pair(ctrl, arm, data_dir, out_dir / "v10_w13" / "pair" / geo, seqs)
        for state in ("off", "on"):
            p = paired[geo][state]["paired"]["GS-HOTA"]
            print(f"paired {geo} flags-{state}: mean {p['mean']:+.4f} "
                  f"helped {p['helped']}/{p['helped'] + p['hurt']} p={p['wilcoxon_p']:.3g} "
                  f"| DetA {paired[geo][state]['paired']['GS-DetA']['mean']:+.4f}", flush=True)
    return {"cells": raw, "paired_1280_vs_640": paired}


def _demo() -> None:
    """Assert the artifact-name map is collision-free and the IMGSZ binding is scoped."""
    import generator.extract as ex  # noqa: PLC0415

    names = [variant_of(d) for d in DET_ARMS]
    assert len(set(names)) == len(names), names
    assert not any(a != b and a.startswith(b) for a in names for b in names), names
    cells = [positions_subdir(d, g) for d in DET_ARMS for g in GEO_ARMS]
    cells += [arm_tag(d, g) for d in DET_ARMS for g in GEO_ARMS]
    assert len(set(cells)) == len(cells), cells

    assert ex.IMGSZ is None, "the shipped default must be untouched before an arm is bound"
    try:
        with detector_at("i1280"):
            assert ex.IMGSZ == 1280
            raise RuntimeError("boom")  # the binding must be restored even on failure
    except RuntimeError:
        pass
    assert ex.IMGSZ is None, ex.IMGSZ
    print("gsr_v10_w13 demo OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/gsr"))
    ap.add_argument("--cache-dir", type=Path, default=CACHE_S4)
    ap.add_argument("--results", type=Path,
                    default=Path("results/gsr_benchmark/gsr_v10_w13.json"))
    ap.add_argument("--seqs", default=None, help="comma-separated; default = DEV-20")
    ap.add_argument("--det", default=",".join(DET_ARMS), help=f"subset of {sorted(DET_ARMS)}")
    ap.add_argument("--geo", default=",".join(GEO_ARMS), help=f"subset of {GEO_ARMS}")
    ap.add_argument("--stages", default=None,
                    help="comma-separated: extract,gate,percrop,boxes,embed")
    ap.add_argument("--positions", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return

    from tools.gsr_v10_w5 import DEV20  # noqa: PLC0415

    seqs = args.seqs.split(",") if args.seqs else DEV20
    dets = [d for d in args.det.split(",") if d in DET_ARMS]
    geos = [g for g in args.geo.split(",") if g in GEO_ARMS]
    payload: dict = {"seqs": seqs, "det_arms": {d: DET_ARMS[d] for d in dets}, "geo_arms": geos,
                     "cache_dir": str(args.cache_dir)}
    if args.stages:
        for det in dets:
            payload.setdefault("stages", {})[det] = run_stages(
                det, args.stages.split(","), args.data_dir, args.out_dir, seqs)
    if args.positions:
        payload["positions"] = build_positions(args.out_dir, args.cache_dir, seqs, dets, geos)
    if args.score:
        payload["score"] = score(args.data_dir, args.out_dir, seqs, dets, geos)
    if args.positions or args.score:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        prev = (json.loads(args.results.read_text(encoding="utf-8"))
                if args.results.exists() else {})
        args.results.write_text(json.dumps({**prev, **payload}, indent=1, default=str),
                                encoding="utf-8")
        print(f"wrote {args.results}")


if __name__ == "__main__":
    main()
