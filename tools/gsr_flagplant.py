"""S2 flag-plant: run the FROZEN identity recipe on the official SoccerNet-GSR **test** split.

The recipe is the one that measured GS-HOTA 33.20 on our internal TEST-38 subset of the public
*valid* split (``results/OCR_DENSIFICATION.md`` section 6): BoT-SORT/ByteTrack positions + PnLCalib,
PRTreID per-detection embeddings, GTA connector at ``tau = 0.040`` with the splitter OFF, per-crop
OCR at 60 crops/track aggregated by the frozen DEV-0.85-floor rule, and the frozen Stage-2 identity
solver carrying the DEV-refit digit-confusion prior. **Nothing here is tuned**: every knob is read
off a file that predates this run, and the resolved set is hashed to
``results/gsr_flagplant_frozen.json`` before the first sequence is touched.

Codabench competition 4365 is the *Test Phase* of the 2025 GSR challenge -- it scores the same 49
sequences whose labels we hold locally, so the local official-scorer number and the leaderboard
number measure the same thing (see :func:`package` for the submission format, taken from the
official example zip).

Everything runs in its own subprocess, one stage at a time, so a 4 GB GPU never holds two models;
every stage is resumable from the per-sequence artifacts the underlying drivers already checkpoint.

CLI::

    python -m tools.gsr_flagplant --freeze     # hash + timestamp the recipe (do this FIRST)
    python -m tools.gsr_flagplant --run        # every stage in order, resumable
    python -m tools.gsr_flagplant --stage ocr  # one named stage
    python -m tools.gsr_flagplant --package    # build the codabench zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("gsr_flagplant")

#: Split we plant the flag on, and where its artifacts live (never mixed with the valid-split run).
SPLIT = "test"
DATA_DIR = Path("data/soccernet/gamestate-2024")
OUT_DIR = Path("outputs/gsr_test")
RESULTS_DIR = Path("results/gsr_benchmark/testsplit")
FROZEN_PATH = Path("results/gsr_flagplant_frozen.json")
TIMES_PATH = RESULTS_DIR / "gsr_flagplant_times.json"
SUBMISSION_DIR = Path("results/gsr_submission")

#: Frozen recipe files (written days before this run; hashed, never edited here).
SOLVER_CONFIG = Path("results/identity_solver_config_percrop.json")
OCR_RULE = Path("results/ocr_density_rule.json")
#: Which frozen DEV precision floor's aggregation rule the 33.20 arm used.
OCR_RULE_FLOOR = "0.85"
#: Crops per track in the per-crop OCR pass (the on-record densified run).
MAX_CROPS = 60
#: GTA connector arm: splitter OFF, tau 0.040 (the only arm meeting the 80% merge-precision bar).
GTA_TAU = 0.040
#: Result tag; the solver writes gsr_identity_<TAG>.json and its arm dir under OUT_DIR.
TAG = "testsplit_percrop"
#: Top-level folder inside the codabench zip -- mirrors the official example submission exactly
#: (github.com/SoccerNet/sn-gamestate/blob/main/examples_predictions/SoccerNetGS-test.zip).
ZIP_TRACKER_DIR = "tracklab"


def test_sequences(data_dir: Path = DATA_DIR) -> list[str]:
    """The 49 official test-split sequence names, from the dataset's own ``sequences_info.json``."""
    info = json.loads((data_dir / "sequences_info.json").read_text(encoding="utf-8"))
    return sorted(s["name"] for s in info[SPLIT])


# === Freeze ======================================================================================
def _sha256(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(data_dir: Path = DATA_DIR, path: Path = FROZEN_PATH) -> dict:
    """Resolve + hash the recipe and write it to ``path`` BEFORE any test data is read.

    Args:
        data_dir: GSR dataset root (for the split list).
        path: Where to write the record. The self-check passes a scratch path so re-running it can
            never overwrite the provenance of a completed run.

    Returns:
        The frozen record, including ``recipe_hash`` (sha256 over the canonical record).
    """
    from generator.gta_link import GTA_VERSION  # noqa: PLC0415
    from generator.identity_solve import SOLVER_VERSION  # noqa: PLC0415
    from generator.jersey_id import OCR_PERCROP_VERSION  # noqa: PLC0415

    rule = json.loads(OCR_RULE.read_text(encoding="utf-8"))["rules"][OCR_RULE_FLOOR]
    solver = json.loads(SOLVER_CONFIG.read_text(encoding="utf-8"))
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=False).stdout.strip()
    record = {
        "split": SPLIT,
        "sequences": test_sequences(data_dir),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": head,
        "versions": {"solver": SOLVER_VERSION, "gta": GTA_VERSION, "ocr": OCR_PERCROP_VERSION},
        "detect_track": {"detector": "football", "tracker": "bytetrack", "calib_period": 1,
                         "sample_every": 1},
        "embeddings": {"model": "prtreid", "frame_stride": 2},
        "ocr": {"max_crops_per_track": MAX_CROPS, "crop_scale": 1.0, "rule_floor": OCR_RULE_FLOOR,
                "rule": rule},
        "gta": {"tau": GTA_TAU, "splitter": False},
        "solver": {k: v for k, v in solver.items() if k != "confusion"},
        "solver_confusion_fit": {k: solver["confusion"].get(k)
                                 for k in ("n", "n_wrong", "n_aligned")},
        "file_hashes": {str(SOLVER_CONFIG): _sha256(SOLVER_CONFIG), str(OCR_RULE): _sha256(OCR_RULE)},
        "reference_score": {"internal_TEST38_valid_split_GS_HOTA": 33.20,
                            "source": "results/OCR_DENSIFICATION.md section 6"},
    }
    blob = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    record["recipe_hash"] = hashlib.sha256(blob).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("frozen recipe %s -> %s (%d sequences)", record["recipe_hash"][:12], path,
                len(record["sequences"]))
    return record


# === Stages ======================================================================================
def _seed_base_arm(out_dir: Path) -> dict:
    """Seed ``eval_koshkina`` (the GTA arm's template) from the jersey-null baseline submissions.

    The aggregated 20-crop Koshkina reader is DELIBERATELY not run on test. Its only downstream
    effect is ``attributes.jersey`` on ``role == "player"`` rows of the GTA arm, and
    :func:`eval.gsr_identity.write_solver_submissions` overwrites that field on **every** player row
    unconditionally -- so it cannot reach the scored artifact. Everything the template does supply
    (pitch geometry, image box, role, team, track id, confidence) is byte-identical to ``eval/``.
    """
    src = out_dir / "eval" / "predictions" / "data"
    dst = out_dir / "eval_koshkina" / "predictions" / "data"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.glob("*.json")):
        q = dst / p.name
        if not q.exists():
            shutil.copyfile(p, q)
            n += 1
    logger.info("base arm seeded: %d copied, %d present", n, len(list(dst.glob("*.json"))))
    return {"copied": n, "total": len(list(dst.glob("*.json")))}


def _solve(data_dir: Path, out_dir: Path, results_dir: Path) -> dict:
    """Run the frozen Stage-2 solver over ALL test sequences and score GS-HOTA (CPU, in-process)."""
    from eval.gsr_identity import run_test  # noqa: PLC0415
    from generator.identity_solve import SolverConfig  # noqa: PLC0415

    names = [n for n in test_sequences(data_dir)
             if (out_dir / "eval_gta_tau0.040_nosplit" / "predictions" / "data"
                 / f"{n}.json").exists()]
    cfg = SolverConfig.load(SOLVER_CONFIG)
    logger.info("solving %d sequences with %s", len(names), SOLVER_CONFIG)
    return run_test(data_dir, out_dir, results_dir, names, cfg, tag=TAG, score_hota=True,
                    votes_subdir="koshkina_percrop_votes", cache_subdir="identity_bundles_percrop")


def _cmd(stage: str, data_dir: Path, out_dir: Path, results_dir: Path) -> list[str]:
    """Subprocess argv for one shell-out stage (a fresh process per stage frees the GPU)."""
    common = ["--data-dir", str(data_dir), "--out-dir", str(out_dir),
              "--results-dir", str(results_dir)]
    if stage == "extract":
        return [sys.executable, "-m", "eval.gsr_score", *common,
                "--seqs", ",".join(test_sequences(data_dir))]
    if stage == "embed":
        return [sys.executable, "-m", "eval.gsr_gta", "--build-cache", *common,
                "--tau", str(GTA_TAU)]
    if stage == "ocr":
        return [sys.executable, "-m", "eval.gsr_jersey", "--percrop",
                "--max-crops", str(MAX_CROPS), *common]
    if stage == "connect":
        return [sys.executable, "-m", "eval.gsr_gta", "--tau", str(GTA_TAU), "--no-split", *common]
    if stage == "votes":
        return [sys.executable, "-m", "tools.ocr_density", "--emit-votes",
                "--rule-floor", OCR_RULE_FLOOR, *common]
    raise ValueError(f"no subprocess for stage {stage}")


#: Ordered pipeline. ``gpu`` marks the stages that must never overlap.
STAGES: tuple[tuple[str, bool], ...] = (
    ("extract", True),    # detect + track + calibrate + team -> positions + jersey-null submissions
    ("base_arm", False),  # seed the GTA template (see _seed_base_arm)
    ("embed", True),      # PRTreID per-detection embeddings (stride 2)
    ("ocr", True),        # per-crop OCR evidence, 60 crops/track
    ("connect", False),   # GTA connector, tau 0.040, splitter OFF
    ("votes", False),     # densified per-track votes at the frozen 0.85-floor rule
    ("solve", False),     # frozen Stage-2 identity solver + official GS-HOTA
)


def run_stage(stage: str, data_dir: Path, out_dir: Path, results_dir: Path) -> dict:
    """Run one stage, timing it and appending the wall time to :data:`TIMES_PATH`."""
    if not FROZEN_PATH.exists():
        raise SystemExit(f"{FROZEN_PATH} missing -- run --freeze before touching test data")
    results_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logger.info("=== stage %s starting", stage)
    if stage == "base_arm":
        _seed_base_arm(out_dir)
    elif stage == "solve":
        _solve(data_dir, out_dir, results_dir)
    else:
        argv = _cmd(stage, data_dir, out_dir, results_dir)
        proc = subprocess.run(argv, check=False)
        if proc.returncode != 0:
            raise SystemExit(f"stage {stage} failed with exit code {proc.returncode}")
    dt = time.time() - t0
    times = json.loads(TIMES_PATH.read_text(encoding="utf-8")) if TIMES_PATH.exists() else {}
    times[stage] = round(dt, 1)
    TIMES_PATH.write_text(json.dumps(times, indent=2), encoding="utf-8")
    logger.info("=== stage %s done in %.0fs (%.2f h)", stage, dt, dt / 3600.0)
    return {"stage": stage, "seconds": dt}


# === Submission package ==========================================================================
def package(out_dir: Path = OUT_DIR, dest_dir: Path = SUBMISSION_DIR) -> dict:
    """Zip the solver arm's per-sequence prediction JSONs into a codabench-ready archive.

    Format (verified against the official example submission
    ``sn-gamestate/examples_predictions/SoccerNetGS-test.zip``, 49 entries): a single top-level
    folder whose name is the tracker name, holding one ``<SEQ>.json`` per sequence, each
    ``{"predictions": [...]}`` in the Labels-GameState prediction schema. No metadata file.

    Returns:
        Package manifest (path, size, entry count, per-sequence prediction counts).
    """
    src = out_dir / "eval_identity_solver_percrop" / "predictions" / "data"
    names = test_sequences()
    missing = [n for n in names if not (src / f"{n}.json").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} test sequences have no prediction JSON: {missing[:5]}")
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"gsr_testphase_{frozen['recipe_hash'][:8]}.zip"
    counts: dict[str, int] = {}
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for n in names:
            blob = (src / f"{n}.json").read_bytes()
            counts[n] = len(json.loads(blob)["predictions"])
            z.writestr(f"{ZIP_TRACKER_DIR}/{n}.json", blob)
    manifest = {
        "zip": str(zip_path), "bytes": zip_path.stat().st_size, "n_sequences": len(names),
        "zip_layout": f"{ZIP_TRACKER_DIR}/<SEQ>.json", "recipe_hash": frozen["recipe_hash"],
        "competition": "codabench 4365 (2025 SoccerNet GSR -- Test Phase)",
        "source_arm": str(src), "predictions_per_sequence": counts,
        "n_predictions_total": int(sum(counts.values())),
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "DO_NOT_UPLOAD": (
            "This package reads the test-split ground truth twice: eval.gsr_score.resolve_team_map "
            "picks the KMeans-cluster -> left/right permutation by agreement with GT, and "
            "eval.gsr_identity._roster hands the solver the exact set of (team, jersey) pairs "
            "present in the sequence's labels. Codabench 4365 scores these same 49 labelled "
            "sequences, so uploading this would post a GT-informed number to a public leaderboard. "
            "Both inputs must be replaced by label-free substitutes before any submission -- see "
            "results/GSR_TEST_FLAGPLANT.md section 6."),
    }
    (dest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("submission %s (%.1f MB, %d sequences, %d predictions)", zip_path,
                manifest["bytes"] / 1e6, manifest["n_sequences"], manifest["n_predictions_total"])
    return manifest


def _demo() -> None:
    """Self-check: the frozen recipe resolves, hashes stably, and names exactly 49 sequences."""
    names = test_sequences()
    assert len(names) == 49, len(names)
    assert names == sorted(set(names))
    rule = json.loads(OCR_RULE.read_text(encoding="utf-8"))["rules"][OCR_RULE_FLOOR]
    assert rule == {"min_crop_conf": 0.99, "min_votes": 5, "min_legibility": 0.5,
                    "emit_all": False}, rule
    import tempfile  # noqa: PLC0415

    scratch = Path(tempfile.gettempdir()) / "gsr_flagplant_freeze_selfcheck.json"
    a, b = freeze(path=scratch), freeze(path=scratch)
    assert a["sequences"] == names
    scratch.unlink(missing_ok=True)
    del a["frozen_at_utc"], b["frozen_at_utc"], a["recipe_hash"], b["recipe_hash"]
    assert a == b, "recipe must be deterministic apart from its timestamp"
    print(f"OK: {len(names)} test sequences, rule {rule}, recipe deterministic")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--freeze", action="store_true", help="write the frozen recipe hash + exit")
    ap.add_argument("--run", action="store_true", help="every stage in order (resumable)")
    ap.add_argument("--stage", choices=[s for s, _ in STAGES], default=None)
    ap.add_argument("--from-stage", default=None, help="with --run: start at this stage")
    ap.add_argument("--package", action="store_true", help="build the codabench zip")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return
    if args.freeze:
        freeze(args.data_dir)
        return
    if args.package:
        package(args.out_dir)
        return
    if args.stage:
        run_stage(args.stage, args.data_dir, args.out_dir, args.results_dir)
        return
    if args.run:
        order = [s for s, _ in STAGES]
        start = order.index(args.from_stage) if args.from_stage else 0
        for stage in order[start:]:
            run_stage(stage, args.data_dir, args.out_dir, args.results_dir)
        package(args.out_dir)
        return
    ap.error("choose one of --freeze / --run / --stage / --package / --demo")


if __name__ == "__main__":
    main()
