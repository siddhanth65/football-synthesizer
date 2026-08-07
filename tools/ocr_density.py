"""OCR densification: measure read density ``d`` and read precision from per-crop OCR evidence.

Build item (1) of the re-ranked order in ``docs/ATTRIBUTION_RESEARCH_PLAN.md`` ("The measured law").
``results/EVIDENCE_DENSITY_LAW.md`` measured that event attribution needs a jersey-read density
``d = 0.347`` where the pipeline delivers ``0.087``, and that *volume* -- not read purity -- is the
binding constraint. This module grades candidate aggregation rules over the per-crop evidence
persisted by ``python -m eval.gsr_jersey --percrop`` (:data:`eval.gsr_jersey.PERCROP_SUBDIR`):

* ``d`` = fraction of ORIGINAL player/GK tracks carrying >= 1 read -- the same denominator as the
  on-record ``425 / 4,870 = 0.0873`` (``results/gsr_benchmark/GSR_RESCORE_KOSHKINA.md``).
* read precision = fraction of emitted ``(track, number)`` reads equal to the dominant GT jersey of
  that track, over reads on tracks whose GT player carries a number (the ``328 / 375 = 0.875``
  definition of ``results/IDENTITY_SOLVER_STAGE2.md`` §2).

The DEV/TEST partition is imported unchanged from :mod:`eval.gsr_identity` (``DEV = sorted[::3]``,
20 sequences; TEST = the other 38). Rules are swept on DEV, one is frozen to
``results/ocr_density_rule.json``, and TEST is measured once from that file. CPU only.

CLI::

    python -m tools.ocr_density --gt-cache      # CPU: per-track GT jersey cache
    python -m tools.ocr_density --sweep         # DEV rule sweep -> frozen rule
    python -m tools.ocr_density --measure test  # the pre-declared gate, one run
    python -m tools.ocr_density --emit-votes    # densified votes JSONs for the frozen solver
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from eval.gsr_identity import _gt_attributes, split_sequences
from eval.gsr_jersey import PERCROP_SUBDIR
from eval.gsr_score import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, DEFAULT_RESULTS_DIR
from generator.jersey_id import OCR_PERCROP_VERSION, percrop_votes

logger = logging.getLogger("ocr_density")

RULE_PATH = Path("results/ocr_density_rule.json")
GT_CACHE = "percrop_gt_tracks.json"


def rule_path(variant: str = "") -> Path:
    """Frozen-rule file for one per-crop evidence variant (pure).

    A sweep over a *different* evidence variant (widened crop, retrained reader) must never land on
    the on-record freeze, so ``variant`` suffixes the filename exactly the way it suffixes the
    evidence directory.
    """
    return RULE_PATH if not variant else RULE_PATH.with_name(f"ocr_density_rule{variant}.json")
#: Densified per-track votes, consumed by eval.gsr_identity in place of koshkina_jersey/*.json.
VOTES_SUBDIR = "koshkina_percrop_votes"


def track_gt(seq_dir: Path, out_dir: Path, positions_subdir: str = "positions",
             ) -> dict[int, str | None]:
    """Dominant GT jersey per prediction track (``None`` = GT player carries no number).

    Args:
        seq_dir: GSR sequence directory (holds ``Labels-GameState.json``).
        out_dir: Pipeline output root (holds ``positions/``).
        positions_subdir: Which extraction the track ids come from. **Load-bearing:** a
            re-extraction (the ``_v6det`` detector arm) renumbers every track, so grading its reads
            against the original extraction's ids silently scores noise. Defaults to the on-record
            ``positions`` so every existing cache entry is unchanged.

    Returns:
        ``{track_id: jersey_string | None}``; tracks with no auditable GT row are absent.
    """
    from eval.gsr_gta import gt_rows  # noqa: PLC0415
    from generator.track_relink import load_gt_ids_by_frame  # noqa: PLC0415

    df = pd.read_parquet(out_dir / positions_subdir / f"{seq_dir.name}.parquet")
    pre = gt_rows(df, load_gt_ids_by_frame(seq_dir))
    attrs = _gt_attributes(seq_dir)
    tally: dict[int, Counter] = {}
    for (tid, _frame), gid in pre.items():
        if gid not in attrs:
            continue
        tally.setdefault(int(tid), Counter())[attrs[gid][1]] += 1
    return {t: c.most_common(1)[0][0] for t, c in tally.items()}


def load_gt_cache(data_dir: Path, out_dir: Path, names: list[str],
                  positions_subdir: str = "positions") -> dict[str, dict[int, str | None]]:
    """Load (building once) the per-sequence per-track GT jersey cache.

    The cache file is namespaced by ``positions_subdir`` so a re-extraction's ids can never be
    served from (or written into) the on-record cache.
    """
    suffix = "" if positions_subdir == "positions" else positions_subdir[len("positions"):]
    path = out_dir / PERCROP_SUBDIR / GT_CACHE.replace(".json", f"{suffix}.json")
    cache: dict[str, dict] = {}
    if path.exists():
        cache = json.loads(path.read_text(encoding="utf-8"))
    missing = [n for n in names if n not in cache]
    for i, name in enumerate(missing):
        cache[name] = {str(k): v for k, v in
                       track_gt(data_dir / name, out_dir, positions_subdir).items()}
        logger.info("[%d/%d] GT cache %s: %d auditable tracks", i + 1, len(missing), name,
                    len(cache[name]))
    if missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache), encoding="utf-8")
    return {n: {int(k): v for k, v in cache[n].items()} for n in names}


def crop_alpha(df: pd.DataFrame) -> np.ndarray | None:
    """The persisted Dirichlet concentrations for one tracklet, or ``None`` for a pre-v7 frame.

    Rows with no read carry an all-``NaN`` alpha; they are returned as the zero-evidence Dirichlet
    ``alpha = 1``, whose mean is uniform and whose uncertainty is exactly 1 -- i.e. "nothing was
    observed", which is what a crop the reader never saw actually means.
    """
    if "alpha" not in df.columns:
        return None
    from generator.jersey_id import NUM_CLASSES  # noqa: PLC0415

    out = np.ones((len(df), NUM_CLASSES), dtype=np.float32)
    for i, a in enumerate(df["alpha"].to_numpy()):
        arr = np.asarray(a, dtype=np.float32)
        if arr.size == NUM_CLASSES and np.isfinite(arr).all():
            out[i] = arr
    return out


def crop_probs_from_percrop(df: pd.DataFrame) -> np.ndarray:
    """Rebuild the ``[n, NUM_CLASSES]`` per-crop distributions from a persisted frame.

    An evidential frame (campaign v7 V3, ``alpha`` column present) yields the Dirichlet MEAN
    ``alpha / sum(alpha)``; a pre-v7 frame yields the folded PARSeq positional product. Both land in
    the identical ``[NUM_CLASSES]`` layout, so every downstream aggregation is reader-agnostic.

    Args:
        df: A per-crop frame (:func:`eval.gsr_jersey.percrop_frame`) for one tracklet.

    Returns:
        ``[n_crops, NUM_CLASSES]`` rows; a crop with no PARSeq read is an ``ILLEGIBLE`` one-hot
        (pre-v7) or the uniform zero-evidence Dirichlet mean (v7).
    """
    from generator.jersey_id import ILLEGIBLE, NUM_CLASSES, parseq_positions_to_probs  # noqa: PLC0415

    alpha = crop_alpha(df)
    if alpha is not None:
        return alpha / alpha.sum(axis=1, keepdims=True)
    out = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)
    out[:, ILLEGIBLE] = 1.0
    for i, (p0, p1) in enumerate(zip(df["p0"].to_numpy(), df["p1"].to_numpy())):
        a0 = np.asarray(p0, dtype=np.float64)
        if a0.size == 11 and np.isfinite(a0).all():
            out[i] = parseq_positions_to_probs(a0, np.asarray(p1, dtype=np.float64))
    return out


def reads_for_sequence(
    df: pd.DataFrame, rule: dict, *, max_crops: int | None = None,
) -> dict[int, list[tuple[int, float]]]:
    """Apply an aggregation rule to one sequence's per-crop frame -> ``{track_id: [(num, conf)]}``.

    Args:
        df: The sequence's per-crop frame.
        rule: ``{"min_crop_conf", "min_votes", "emit_all", "min_legibility"}`` -- the first three go
            to :func:`generator.jersey_id.percrop_votes`; ``min_legibility`` raises the ResNet34
            legibility floor above the reader's shipped 0.5 *offline*, which is only possible
            because the per-crop pass persists the score. Campaign v7 V3 adds three optional keys,
            all inert unless present: ``max_u`` / ``max_p_none`` drop per-crop rows the evidential
            head is uncertain about or calls empty, and ``fuse: true`` swaps the Koshkina vote for
            :func:`generator.evidential_jersey.fuse_tracklet` (additive Dirichlet evidence). With
            ``fuse`` absent, the SAME evidential rows go through the incumbent voting machinery --
            that is the Koshkina voting control the fusion arm has to beat.
        max_crops: Optionally sub-sample each track's crops to this many evenly-spread rows -- the
            matched-volume control that separates "more crops" from "better aggregation".

    Returns:
        ``{track_id: [(number, confidence), ...]}``, empty lists omitted.
    """
    out: dict[int, list[tuple[int, float]]] = {}
    min_leg = float(rule.get("min_legibility", 0.0))
    max_u = float(rule.get("max_u", 1.01))
    max_p_none = float(rule.get("max_p_none", 1.01))
    fuse = bool(rule.get("fuse", False))
    for tid, grp in df.groupby("track_id"):
        grp = grp.sort_values("frame")
        if max_crops is not None and len(grp) > max_crops:
            idx = np.linspace(0, len(grp) - 1, max_crops).round().astype(int)
            grp = grp.iloc[sorted(set(idx.tolist()))]
        if min_leg > 0.0:
            grp = grp[grp["legibility"].to_numpy() >= min_leg]
            if grp.empty:
                continue
        if fuse:
            from generator.evidential_jersey import fuse_tracklet  # noqa: PLC0415

            alpha = crop_alpha(grp)
            if alpha is None:
                raise SystemExit("rule asks for evidential fusion but the frame has no 'alpha'")
            votes = fuse_tracklet(alpha, max_u=max_u, max_p_none=max_p_none,
                                  min_conf=float(rule["min_crop_conf"]),
                                  min_crops=int(rule["min_votes"]))
        else:
            probs = crop_probs_from_percrop(grp)
            if max_u < 1.0 or max_p_none < 1.0:
                alpha = crop_alpha(grp)
                if alpha is None:
                    raise SystemExit("rule asks for an evidential filter but there is no 'alpha'")
                s = alpha.sum(1)
                keep = (alpha.shape[1] / s < max_u) & (alpha[:, 0] / s < max_p_none)
                probs = probs[keep]
            votes = percrop_votes(
                probs, min_crop_conf=float(rule["min_crop_conf"]),
                min_votes=int(rule["min_votes"]), emit_all=bool(rule["emit_all"]))
        if votes:
            out[int(tid)] = votes
    return out


def measure(
    per_seq: dict[str, pd.DataFrame], gt: dict[str, dict[int, str | None]], rule: dict,
    *, max_crops: int | None = None,
) -> dict:
    """Read density and read precision for one rule over a set of sequences (pure).

    Args:
        per_seq: ``{sequence: per-crop frame}``.
        gt: ``{sequence: {track_id: GT jersey string or None}}``.
        rule: The aggregation rule (see :func:`reads_for_sequence`).
        max_crops: Matched-volume crop budget per track, or ``None`` for all persisted crops.

    Returns:
        ``d``, ``read_precision`` (per read), ``track_precision`` (per read-carrying track, the
        0.875-comparable figure), and every denominator.
    """
    n_tracks = n_read = 0
    reads_ok = reads_aud = trk_ok = trk_aud = n_reads = 0
    for name, df in per_seq.items():
        g = gt.get(name, {})
        tracks = set(df["track_id"].astype(int).tolist())
        n_tracks += len(tracks)
        reads = reads_for_sequence(df, rule, max_crops=max_crops)
        n_read += len(reads)
        for tid, votes in reads.items():
            n_reads += len(votes)
            true = g.get(tid, "missing")
            if true in (None, "missing"):
                continue  # GT player carries no number (or track unauditable): not gradable
            reads_aud += len(votes)
            reads_ok += sum(1 for num, _c in votes if str(num) == true)
            trk_aud += 1
            trk_ok += int(str(votes[0][0]) == true)
    return {
        "rule": rule, "max_crops": max_crops,
        "n_tracks": n_tracks, "n_tracks_read": n_read,
        "d": n_read / max(n_tracks, 1),
        "n_reads": n_reads, "n_reads_auditable": reads_aud,
        "read_precision": reads_ok / max(reads_aud, 1),
        "n_tracks_auditable": trk_aud,
        "track_precision": trk_ok / max(trk_aud, 1),
    }


def load_percrop(out_dir: Path, names: list[str], variant: str = "") -> dict[str, pd.DataFrame]:
    """Load the persisted per-crop frames for ``names`` (missing sequences are skipped).

    ``variant`` selects a re-run at a different crop geometry (e.g. ``"_w125"``).
    """
    out: dict[str, pd.DataFrame] = {}
    for n in names:
        p = out_dir / (PERCROP_SUBDIR + variant) / f"{n}.parquet"
        if p.exists():
            out[n] = pd.read_parquet(p)
    return out


#: DEV sweep grid. Coarse on purpose: 20 sequences cannot resolve a fine one.
GRID_CONF = [0.50, 0.90, 0.99]
GRID_VOTES = [1, 2, 3, 5]
GRID_LEG = [0.50, 0.90, 0.99, 0.999]
#: DEV read-precision floors, one frozen rule each. ``0.80`` is the gate's own floor and supplies
#: the PRIMARY arm; ``0.85`` and ``0.87`` (the on-record reader's precision) are declared alongside
#: because the measured DEV frontier is steep -- see results/OCR_DENSIFICATION.md section 3.
DEV_PRECISION_FLOORS = (0.80, 0.85, 0.87)
DEV_PRECISION_FLOOR = DEV_PRECISION_FLOORS[0]


def sweep(per_seq: dict, gt: dict, *, max_crops: int | None = None) -> list[dict]:
    """Grade every ``(min_crop_conf, min_votes, min_legibility, emit_all)`` point on DEV."""
    rows = []
    for conf in GRID_CONF:
        for votes in GRID_VOTES:
            for leg in GRID_LEG:
                for emit_all in (False, True):
                    rule = {"min_crop_conf": conf, "min_votes": votes, "min_legibility": leg,
                            "emit_all": emit_all}
                    rows.append(measure(per_seq, gt, rule, max_crops=max_crops))
    return rows


def pick_rule(rows: list[dict], floor: float = DEV_PRECISION_FLOOR) -> dict:
    """Highest-``d`` rule whose DEV read precision clears ``floor`` (pure)."""
    ok = [r for r in rows if r["read_precision"] >= floor]
    if not ok:
        raise SystemExit(f"no DEV rule reaches read precision {floor}")
    return max(ok, key=lambda r: r["d"])


def emit_votes(out_dir: Path, names: list[str], rule: dict, *, votes_subdir: str = VOTES_SUBDIR,
               variant: str = "") -> None:
    """Write densified ``{track_id: [[num, conf], ...]}`` per sequence for the identity solver.

    Mirrors the ``outputs/gsr/koshkina_jersey/<seq>.json`` shape but allows MANY reads per track
    (Stage 2's design intent: a track whose crops disagree keeps every vote and the confusion prior
    arbitrates), so :mod:`eval.gsr_identity` can consume it without a schema change.

    Args:
        out_dir: Pipeline output root.
        names: Sequences to emit.
        rule: The aggregation rule (see :func:`reads_for_sequence`).
        votes_subdir: Destination directory, so an alternative rule never overwrites the shipped
            votes of the on-record recipe.
        variant: Per-crop source variant (see :func:`load_percrop`).
    """
    dest = out_dir / votes_subdir
    dest.mkdir(parents=True, exist_ok=True)
    for name, df in load_percrop(out_dir, names, variant).items():
        reads = reads_for_sequence(df, rule)
        n_tracks = int(df["track_id"].nunique())
        (dest / f"{name}.json").write_text(json.dumps({
            "seq": name, "n_tracks": n_tracks, "n_read": len(reads),
            "ocr_version": OCR_PERCROP_VERSION, "rule": rule,
            "votes": {str(t): [[int(n), round(float(c), 4)] for n, c in v]
                      for t, v in reads.items()},
        }), encoding="utf-8")
    logger.info("densified votes -> %s (%d sequences, rule %s)", dest, len(names), rule)


def _report(tag: str, res: dict) -> None:
    """ASCII one-liner per measured arm (cp1252-safe)."""
    logger.info("%-28s d %.4f (%d/%d tracks) | read prec %.4f (%d/%d) | track prec %.4f (%d)",
                tag, res["d"], res["n_tracks_read"], res["n_tracks"], res["read_precision"],
                round(res["read_precision"] * res["n_reads_auditable"]), res["n_reads_auditable"],
                res["track_precision"], res["n_tracks_auditable"])


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--gt-cache", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="DEV rule sweep -> frozen rule")
    ap.add_argument("--measure", choices=["dev", "test"], help="measure the frozen rule once")
    ap.add_argument("--emit-votes", action="store_true")
    ap.add_argument("--rule-floor", default=None,
                    help="which frozen DEV floor's rule to emit votes for (default: primary)")
    ap.add_argument("--variant", default="",
                    help="per-crop evidence variant (e.g. _v6); suffixes the rule + results files")
    args = ap.parse_args()

    dev, test = split_sequences(args.data_dir, args.out_dir)
    gt = load_gt_cache(args.data_dir, args.out_dir, dev + test)
    if args.gt_cache:
        return
    rpath = rule_path(args.variant)
    if args.sweep:
        per_seq = load_percrop(args.out_dir, dev, args.variant)
        if not per_seq:
            raise SystemExit(f"no per-crop evidence under koshkina_percrop{args.variant}")
        logger.info("DEV per-crop frames: %d sequences, %d crops",
                    len(per_seq), sum(len(v) for v in per_seq.values()))
        rows = sweep(per_seq, gt)
        rows20 = sweep(per_seq, gt, max_crops=20)
        for r in sorted(rows, key=lambda r: -r["d"]):
            _report(f"conf{r['rule']['min_crop_conf']:.2f} v{r['rule']['min_votes']} "
                    f"leg{r['rule']['min_legibility']:.3f} "
                    f"{'all' if r['rule']['emit_all'] else 'top'}", r)
        picks = {f"{f:.2f}": pick_rule(rows, f) for f in DEV_PRECISION_FLOORS}
        args.results_dir.mkdir(parents=True, exist_ok=True)
        (args.results_dir / f"gsr_ocr_density_dev{args.variant}.json").write_text(json.dumps(
            {"version": OCR_PERCROP_VERSION, "variant": args.variant, "dev": dev,
             "floors": DEV_PRECISION_FLOORS, "rows": rows, "rows_max20": rows20,
             "picked": picks}, indent=2), encoding="utf-8")
        rpath.write_text(json.dumps(
            {"version": OCR_PERCROP_VERSION, "variant": args.variant,
             "floors": list(DEV_PRECISION_FLOORS), "primary_floor": DEV_PRECISION_FLOOR,
             "rules": {k: v["rule"] for k, v in picks.items()},
             **picks[f"{DEV_PRECISION_FLOOR:.2f}"]["rule"]}, indent=2), encoding="utf-8")
        for k, v in picks.items():
            _report(f"DEV PICK floor {k}", v)
        return
    frozen = json.loads(rpath.read_text(encoding="utf-8"))
    rule = {k: v for k, v in frozen.items()
            if k in {"min_crop_conf", "min_votes", "emit_all", "min_legibility"}}
    if args.emit_votes:
        if args.rule_floor:
            rule = frozen["rules"][args.rule_floor]
        logger.info("emitting votes for rule %s", rule)
        emit_votes(args.out_dir, dev + test, rule, votes_subdir=VOTES_SUBDIR + args.variant,
                   variant=args.variant)
        return
    if args.measure:
        names = dev if args.measure == "dev" else test
        per_seq = load_percrop(args.out_dir, names, args.variant)
        arms = {}
        for floor, r in frozen["rules"].items():
            arms[floor] = {"full": measure(per_seq, gt, r),
                           "matched_volume_20": measure(per_seq, gt, r, max_crops=20)}
            _report(f"{args.measure.upper()} floor {floor}", arms[floor]["full"])
            _report(f"{args.measure.upper()} floor {floor} @20 crops",
                    arms[floor]["matched_volume_20"])
        args.results_dir.mkdir(parents=True, exist_ok=True)
        (args.results_dir / f"gsr_ocr_density_{args.measure}{args.variant}.json").write_text(
            json.dumps({"version": OCR_PERCROP_VERSION, "variant": args.variant,
                        "sequences": names, "primary_floor": f"{DEV_PRECISION_FLOOR:.2f}",
                        "arms": arms}, indent=2), encoding="utf-8")
        return
    ap.error("choose one of --gt-cache / --sweep / --measure / --emit-votes")


if __name__ == "__main__":
    main()
