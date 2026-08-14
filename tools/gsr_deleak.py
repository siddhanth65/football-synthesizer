"""De-leak the SoccerNet-GSR pipeline: measure and replace the two ground-truth reads.

``results/GSR_TEST_FLAGPLANT.md`` §6 found the scored chain consumes the evaluation labels twice:

1. :func:`eval.gsr_score.resolve_team_map` picks the KMeans-cluster -> ``left``/``right``
   permutation by agreement with GT positions.
2. :func:`eval.gsr_identity._roster` hands the identity solver the exact ``(team, jersey)`` slot set
   of the sequence's ``Labels-GameState.json`` (~15-20 slots instead of the 198 of a free roster).

This module measures what each leak is worth in GS-HOTA and supplies GT-free replacements:

* **team side** -- :func:`eval.gsr_score.resolve_team_map_free` (the deeper cluster mean is ``left``),
  graded here against the leaky map on the *train* split (never used for anything else) and on valid.
* **roster** -- ``full`` (1..99 x both sides) or ``self`` (only numbers this sequence's own OCR chain
  actually read), both built from the cached evidence bundle without re-reading any label file.

Every ablation is a pure in-memory transform of the cached bundles (identities and the gallery
similarity tensor are recomputed from the tracklets the bundle already carries), so the whole matrix
is CPU-only and needs no re-extraction.

CLI::

    python -m tools.gsr_deleak --train-probe        # GPU, ~1 h: light positions for the 57 train seqs
    python -m tools.gsr_deleak --teamside           # CPU: resolver accuracy, train + valid
    python -m tools.gsr_deleak --dev                # CPU: roster arms on the valid DEV-20 partition
    python -m tools.gsr_deleak --freeze             # write the chosen GT-free config (pre-declare)
    python -m tools.gsr_deleak --matrix             # CPU: the leak-cost matrix on valid TEST-38
    python -m tools.gsr_deleak --testsplit          # CPU: frozen GT-free recipe on the test-49 split
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from eval.gsr_identity import (
    MAX_TOPK,
    coverage_curve,
    load_bundles,
    paired_stats,
    score_assignment,
    solve_bundle_scored,
    split_sequences,
)
from eval.gsr_score import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    EVAL_CONFIGS,
    VOTE_TRACK_ATTRS,
    gs_hota,
    load_gt_people_by_frame,
    resolve_team_map,
    resolve_team_map_free,
    vote_track_attributes,
)
from generator.identity_solve import Identity, SolverConfig

logger = logging.getLogger("gsr_deleak")

#: Positions for the train split (team-side resolver evidence only, never scored).
TRAIN_PROBE_DIR = Path("outputs/gsr_train_probe")
#: Frame stride of the train probe. Must stay at 1: measured on valid, the resolver needs the whole
#: clip (96.6% over 750 frames, 93.1% over 375, 86.2% over 250, 79.3% over 150 -- a short window is
#: one attacking phase), and ByteTrack drops most detections at stride >= 5 (4.7 people/frame
#: against the 8.6-18.4 of the stride-1 valid tables).
PROBE_STRIDE = 1
#: Where the pre-declared GT-free configuration lands.
FROZEN_PATH = Path("results/gsr_deleak_frozen.json")
#: Results directory for this study.
RESULTS_DIR = Path("results/gsr_benchmark")
#: The frozen 33.20 solver calibration (unchanged by this work).
SOLVER_CONFIG = Path("results/identity_solver_config_percrop.json")
#: Base arm whose rows every solver arm rewrites (jersey, and now team).
BASE_ARM = "eval_gta_tau0.040_nosplit"


# === Split bookkeeping ===========================================================================
def split_names(data_dir: Path, split: str) -> list[str]:
    """Sequence names of one official split, from the dataset's own ``sequences_info.json``."""
    key = {"train": "train", "valid": "validation", "test": "test"}[split]
    blob = json.loads((data_dir / "sequences_info.json").read_text(encoding="utf-8"))
    return sorted(s["name"] for s in blob[key])


# === Team-side resolver grading ==================================================================
def grade_teamside(data_dir: Path, pos_dir: Path, names: list[str]) -> pd.DataFrame:
    """Grade :func:`resolve_team_map_free` against the GT-agreement map, one row per sequence.

    Args:
        data_dir: GSR ground-truth folder.
        pos_dir: Folder of ``<seq>.parquet`` positions tables.
        names: Sequences to grade (skipped when no parquet exists).

    Returns:
        Frame with ``seq, gt, free, margin, ok, purity`` -- ``purity`` is the GT map's own
        agreement fraction, i.e. how decidable the sequence is at all.
    """
    rows = []
    for name in names:
        path = pos_dir / f"{name}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        gt_map = resolve_team_map(df, load_gt_people_by_frame(data_dir / name))
        free_map, margin = resolve_team_map_free(df)
        rows.append({"seq": name, "gt": gt_map[0], "free": free_map[0], "margin": margin,
                     "ok": gt_map[0] == free_map[0]})
    return pd.DataFrame(rows)


def train_probe(data_dir: Path, names: list[str]) -> None:
    """Extract light positions (stride :data:`PROBE_STRIDE`) for the train split. GPU, resumable."""
    from generator.calibrate import PnLCalibCalibrator  # noqa: PLC0415
    from generator.extract import extract_positions  # noqa: PLC0415

    pos_dir = TRAIN_PROBE_DIR / "positions"
    pos_dir.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not (pos_dir / f"{n}.parquet").exists()]
    logger.info("train probe: %d/%d sequences to extract at stride %d", len(todo), len(names),
                PROBE_STRIDE)
    if not todo:
        return
    calibrator = PnLCalibCalibrator()
    for i, name in enumerate(todo):
        t0 = time.time()
        df = extract_positions(str(data_dir / name / "img1" / "%06d.jpg"),
                               pos_dir / f"{name}.parquet", sample_every=PROBE_STRIDE,
                               calibrator=calibrator, detector_name="football",
                               tracker_name="bytetrack", calib_period=1)
        logger.info("[%d/%d] %s: %d rows (%.0fs)", i + 1, len(todo), name, len(df),
                    time.time() - t0)


# === Roster construction (GT-free) ===============================================================
def roster_full(sides: dict[int, str], numbers: range = range(1, 100)) -> list[Identity]:
    """Every shirt number on both sides: the unconstrained 198-slot roster (no labels read)."""
    return [Identity((sides[t], n), t, n, "player") for t in (0, 1) for n in numbers]


def roster_self(bundle: dict, sides: dict[int, str], *, both_teams: bool = False) -> list[Identity]:
    """Roster built from the sequence's OWN confident OCR reads (the self-roster).

    A slot exists for ``(team of the reading tracklet, number read)``. With ``both_teams`` the number
    is opened on both sides, which costs nothing when the reader's team label is wrong.
    """
    seen: set[tuple[int, int]] = set()
    for trk in bundle["tracklets"]:
        if trk.team is None:
            continue
        for num, _conf in trk.reads:
            seen.add((0, int(num)))
            seen.add((1, int(num)))
            if not both_teams:
                seen.discard((1 - int(trk.team), int(num)))
    return [Identity((sides[t], n), t, n, "player") for t, n in sorted(seen)]


def rebuild_top(bundle: dict, identities: list[Identity]) -> np.ndarray:
    """Recompute the leave-one-out gallery similarity tensor for a new roster (pure, in memory).

    Mirrors :func:`eval.gsr_identity.build_bundle`: identity ``i`` is anchored by the tracklets whose
    own dominant read is ``i.number`` on ``i.team``; a tracklet is scored against the anchors that
    are not itself.
    """
    tracklets = bundle["tracklets"]
    anchors: dict[int, list[int]] = defaultdict(list)
    by_num: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, ident in enumerate(identities):
        if ident.number is not None:
            by_num[(ident.team, int(ident.number))].append(i)
    for k, trk in enumerate(tracklets):
        if not trk.reads or trk.team is None:
            continue
        num = Counter(n for n, _c in trk.reads).most_common(1)[0][0]
        for i in by_num.get((int(trk.team), int(num)), []):
            anchors[i].append(k)
    top = np.full((len(tracklets), len(identities), MAX_TOPK), np.nan, np.float32)
    for i, members in anchors.items():
        for k, trk in enumerate(tracklets):
            gal = [tracklets[m].emb for m in members if m != k and tracklets[m].emb.size]
            if not gal or trk.emb.size == 0:
                continue
            sims = np.sort((trk.emb @ np.concatenate(gal).T).ravel())[-MAX_TOPK:]
            top[k, i, MAX_TOPK - len(sims):] = sims
    return top


def retarget(bundle: dict, *, roster: str, sides: dict[int, str]) -> dict:
    """Return a copy of ``bundle`` with a new roster and/or a new cluster -> side map.

    Args:
        bundle: A cached bundle from :func:`eval.gsr_identity.build_bundle`.
        roster: ``"gt"`` (keep the label-derived slots), ``"full"`` or ``"self"``.
        sides: The cluster -> side map to grade and submit under.

    Returns:
        A shallow copy with ``identities``, ``top``, ``rowcounts`` and ``team_map`` replaced. The
        row counts' *predicted* side is re-derived from ``sides``, so team accuracy is graded under
        exactly the map the submission will carry.
    """
    old = bundle["team_map"]
    flip = old[0] != sides[0]
    out = dict(bundle)
    out["team_map"] = dict(sides)
    out["team_map_src"] = dict(old)  # the map the cached submissions were written under
    if flip:
        swap = {"left": "right", "right": "left"}
        out["rowcounts"] = [Counter({(swap.get(ps, ps), gs, gj): c
                                     for (ps, gs, gj), c in rc.items()})
                            for rc in bundle["rowcounts"]]
    if roster == "gt":
        # The label-derived slots are keyed by GT *side*, so a flipped map moves each slot to the
        # other cluster -- exactly what rebuilding the bundle under that map would have produced.
        if flip:
            out["identities"] = [Identity(i.key, 1 - i.team, i.number, i.role, i.window)
                                 for i in bundle["identities"]]
            out["top"] = rebuild_top(bundle, out["identities"])
        return out
    out["identities"] = (roster_full(sides) if roster == "full"
                         else roster_self(bundle, sides, both_teams=roster == "selfboth"))
    out["top"] = rebuild_top(bundle, out["identities"])
    return out


# === Arm runner ==================================================================================
def _team_maps(data_dir: Path, out_dir: Path, names: list[str], source: str,
               positions_subdir: str = "positions") -> dict[str, dict]:
    """Cluster -> side map per sequence under ``gt`` (leaky), ``free`` (geometric) or ``fixed``."""
    maps: dict[str, dict] = {}
    for name in names:
        df = pd.read_parquet(out_dir / positions_subdir / f"{name}.parquet")
        if source == "gt":
            maps[name] = resolve_team_map(df, load_gt_people_by_frame(data_dir / name))
        elif source == "free":
            maps[name] = resolve_team_map_free(df)[0]
        else:
            maps[name] = {0: "left", 1: "right"}
    return maps


def write_arm(bundles: dict, assigns: dict, maps: dict, out_dir: Path, arm_dir: Path,
              base_arm: str = BASE_ARM) -> None:
    """Write the arm's submissions: base arm rows with our jersey AND our cluster -> side map.

    The base arm's ``attributes.team`` was written under the leaky GT map, so it is flipped whenever
    this arm's map disagrees with the one the cache was built under (``team_map_src``).
    """
    dest = arm_dir / "predictions" / "data"
    dest.mkdir(parents=True, exist_ok=True)
    swap = {"left": "right", "right": "left"}
    for name, b in bundles.items():
        number = {t.track_id: a for t, a in zip(b["tracklets"], assigns[name])}
        flip = b["team_map_src"][0] != maps[name][0]
        payload = json.loads((out_dir / base_arm / "predictions" / "data" / f"{name}.json")
                             .read_text(encoding="utf-8"))
        for p in payload["predictions"]:
            attrs = p["attributes"]
            if flip and attrs.get("team") in swap:
                attrs["team"] = swap[attrs["team"]]
            if attrs.get("role") == "player":
                num = number.get(int(p["track_id"]))
                attrs["jersey"] = None if num is None else str(num)
        # Registered v9-W3 component, default OFF (VOTE_TRACK_ATTRS = ()): a GT identity carries one
        # role and one team for the whole clip, so within-track disagreement is guaranteed error.
        vote_track_attributes(payload["predictions"], VOTE_TRACK_ATTRS)
        (dest / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def run_arm(bundles: dict, cfg: SolverConfig, data_dir: Path, out_dir: Path, names: list[str], *,
            team_src: str, roster: str, tag: str, score_hota: bool,
            positions_subdir: str = "positions", base_arm: str = BASE_ARM) -> dict:
    """Solve + grade one (team-map source, roster) arm. Returns its payload.

    ``positions_subdir`` and ``base_arm`` move the whole arm onto a repaired positions source (the
    calibration-gap fill); their defaults reproduce every on-record number unchanged.
    """
    maps = _team_maps(data_dir, out_dir, names, team_src, positions_subdir)
    t0 = time.time()
    arms = {n: retarget(bundles[n], roster=roster, sides=maps[n]) for n in names}
    scored = {n: solve_bundle_scored(b, cfg) for n, b in arms.items()}
    assigns = {n: v[0] for n, v in scored.items()}
    confs = {n: v[1] for n, v in scored.items()}
    tot = Counter()
    per_seq = {}
    for name in names:
        n, ok, okj = score_assignment(arms[name], assigns[name])
        per_seq[name] = {"n_rows": n, "identity": ok / max(n, 1), "jersey": okj / max(n, 1),
                         "n_named": int(sum(1 for a in assigns[name] if a is not None)),
                         "n_identities": len(arms[name]["identities"])}
        tot["n"] += n
        tot["ok"] += ok
        tot["okj"] += okj
        tot["named"] += per_seq[name]["n_named"]
        tot["trk"] += len(arms[name]["tracklets"])
    payload = {
        "tag": tag, "team_src": team_src, "roster": roster, "n_sequences": len(names),
        "solve_seconds": round(time.time() - t0, 1),
        "pooled": {"n_rows": tot["n"], "identity": tot["ok"] / max(tot["n"], 1),
                   "jersey": tot["okj"] / max(tot["n"], 1),
                   "tracklets": tot["trk"], "tracklets_named": tot["named"],
                   "mean_identities": float(np.mean([p["n_identities"]
                                                     for p in per_seq.values()]))},
        "coverage_curve": coverage_curve(arms, assigns, confs), "per_seq": per_seq,
    }
    if score_hota:
        arm_dir = out_dir / f"deleak_{tag}"
        write_arm(arms, assigns, maps, out_dir, arm_dir, base_arm)
        res = gs_hota(arm_dir, data_dir, seq_info={n: 0 for n in names},
                      **EVAL_CONFIGS["gs_hota_full"])
        payload["gs_hota"] = res["combined"]
        payload["gs_hota_per_seq"] = {n: v["GS-HOTA"] for n, v in res["per_seq"].items()}
    logger.info("%-22s identity %.4f  named %d/%d  slots %.0f%s", tag,
                payload["pooled"]["identity"], tot["named"], tot["trk"],
                payload["pooled"]["mean_identities"],
                "  GS-HOTA %.2f (DetA %.2f AssA %.2f)" % (
                    payload["gs_hota"]["GS-HOTA"], payload["gs_hota"]["GS-DetA"],
                    payload["gs_hota"]["GS-AssA"]) if score_hota else "")
    return payload


# === Official test split (score-once, then repackage) ============================================
#: Where the test-split artifacts live, and where the codabench zip is built.
TEST_OUT_DIR = Path("outputs/gsr_test")
TEST_RESULTS_DIR = Path("results/gsr_benchmark/testsplit")
SUBMISSION_DIR = Path("results/gsr_submission")
#: Top-level folder inside the codabench zip (mirrors the official example submission).
ZIP_TRACKER_DIR = "tracklab"


def package_free(arm_dir: Path, names: list[str], frozen: dict, stem: str = "gtfree") -> dict:
    """Zip a GT-free arm's per-sequence prediction JSONs into a codabench-ready archive.

    Same layout as :func:`tools.gsr_flagplant.package` (one top-level ``tracklab/`` folder, one
    ``<SEQ>.json`` each, no metadata file) but the manifest carries no ``DO_NOT_UPLOAD`` field: this
    package reads no label file. ``stem`` names the package family (``gsr_testphase_<stem>_<hash>``
    plus ``manifest_<stem>.json``) so a new recipe never overwrites a prior record.
    """
    src = arm_dir / "predictions" / "data"
    missing = [n for n in names if not (src / f"{n}.json").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} test sequences have no prediction JSON: {missing[:5]}")
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(frozen, sort_keys=True).encode("utf-8")
    recipe_hash = hashlib.sha256(payload).hexdigest()
    zip_path = SUBMISSION_DIR / f"gsr_testphase_{stem}_{recipe_hash[:8]}.zip"
    counts: dict[str, int] = {}
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for n in names:
            blob = (src / f"{n}.json").read_bytes()
            counts[n] = len(json.loads(blob)["predictions"])
            z.writestr(f"{ZIP_TRACKER_DIR}/{n}.json", blob)
    manifest = {
        "zip": str(zip_path), "bytes": zip_path.stat().st_size, "n_sequences": len(names),
        "zip_layout": f"{ZIP_TRACKER_DIR}/<SEQ>.json", "recipe_hash": recipe_hash,
        "competition": "codabench 4365 (2025 SoccerNet GSR -- Test Phase)",
        "source_arm": str(src), "predictions_per_sequence": counts,
        "n_predictions_total": int(sum(counts.values())),
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gt_free": frozen,
        "label_reads": ("none. The cluster -> side map comes from resolve_team_map_free (geometry "
                        "only) and the solver's roster from the sequence's own OCR reads. No "
                        "Labels-GameState.json is opened anywhere in the prediction chain."),
    }
    (SUBMISSION_DIR / f"manifest_{stem}.json").write_text(json.dumps(manifest, indent=2),
                                                          encoding="utf-8")
    logger.info("submission %s (%.1f MB, %d sequences, %d predictions)", zip_path,
                manifest["bytes"] / 1e6, manifest["n_sequences"], manifest["n_predictions_total"])
    return manifest


def run_testsplit(data_dir: Path) -> dict:
    """Score the frozen GT-free recipe once on the official 49-sequence test split and repackage."""
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    names = split_names(data_dir, "test")
    cfg = SolverConfig.load(SOLVER_CONFIG)
    bundles = load_bundles(data_dir, TEST_OUT_DIR, names,
                           votes_subdir="koshkina_percrop_votes",
                           cache_subdir="identity_bundles_percrop")
    arm = run_arm(bundles, cfg, data_dir, TEST_OUT_DIR, names,
                  team_src="free", roster="self", tag="t49_free_self", score_hota=True)
    grade = grade_teamside(data_dir, TEST_OUT_DIR / "positions", names)
    arm["teamside_diagnostic"] = {
        "n": len(grade), "correct": int(grade["ok"].sum()),
        "accuracy": float(grade["ok"].mean()), "wrong": grade.loc[~grade["ok"], "seq"].tolist(),
        "note": "local diagnostic only -- the submission itself uses the free map unconditionally",
    }
    manifest = package_free(TEST_OUT_DIR / "deleak_t49_free_self", names, frozen)
    arm["manifest"] = manifest
    _write(TEST_RESULTS_DIR / "gsr_deleak_testsplit.json", arm)
    return arm


# === Self-check ==================================================================================
def _demo() -> None:
    """Self-check: the free map reads the geometry, and a flip moves slots AND graded sides."""
    from generator.identity_solve import Tracklet  # noqa: PLC0415

    # Cluster 0 sits deep in the left half, cluster 1 in the right -> cluster 0 is 'left'.
    df = pd.DataFrame({
        "role": ["player"] * 6 + ["goalkeeper", "referee"],
        "team": [0, 0, 0, 1, 1, 1, 0, -1],
        # the GK of cluster 0 is parked on the RIGHT touchline: a kit-clustered GK must not vote
        "pitch_x": [20.0, 25.0, 30.0, 70.0, 75.0, 80.0, 104.0, 50.0],
    })
    sides, margin = resolve_team_map_free(df)
    assert sides == {0: "left", 1: "right"}, sides
    assert abs(margin - 50.0) < 1e-6, margin
    flipped, _ = resolve_team_map_free(df.assign(team=1 - df["team"]))
    assert flipped == {0: "right", 1: "left"}, flipped

    trk = Tracklet(7, 10, 0, 1.0, {"player": 1.0}, ((9, 0.9),), np.zeros((0, 4), np.float32), (0, 9))
    bundle = {
        "tracklets": [trk], "identities": [Identity(("left", 9), 0, 9, "player")],
        "top": np.full((1, 1, MAX_TOPK), np.nan, np.float32), "groups": [],
        "rowcounts": [Counter({("left", "left", "9"): 10})], "team_map": {0: "left", 1: "right"},
    }
    # Self-roster: the read opens #9 on the reader's own cluster only.
    keep = retarget(bundle, roster="self", sides={0: "left", 1: "right"})
    assert [(i.team, i.number) for i in keep["identities"]] == [(0, 9)], keep["identities"]
    assert roster_self(bundle, {0: "left", 1: "right"}, both_teams=True).__len__() == 2
    assert len(roster_full({0: "left", 1: "right"})) == 198

    # Flipping the map re-sides the graded rows and moves the label-derived slot to the other cluster.
    flip = retarget(bundle, roster="gt", sides={0: "right", 1: "left"})
    assert list(flip["rowcounts"][0]) == [("right", "left", "9")], flip["rowcounts"][0]
    assert flip["identities"][0].team == 1, flip["identities"][0]
    assert bundle["rowcounts"][0][("left", "left", "9")] == 10, "source bundle was mutated"
    assert flip["team_map_src"] == {0: "left", 1: "right"}, flip["team_map_src"]
    print("gsr_deleak demo OK: free map ignores GKs, self-roster stays on its own cluster, "
          "a flip moves both the slots and the graded sides")


# === CLI =========================================================================================
def _write(path: Path, payload: dict) -> None:
    """Write a JSON payload (UTF-8, indented) and log the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", path)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--train-probe", action="store_true", help="GPU: light train-split positions")
    ap.add_argument("--teamside", action="store_true", help="grade the GT-free team resolver")
    ap.add_argument("--dev", action="store_true", help="roster arms on the valid DEV-20 partition")
    ap.add_argument("--freeze", action="store_true", help="pre-declare the GT-free config")
    ap.add_argument("--matrix", action="store_true", help="leak-cost matrix on valid TEST-38")
    ap.add_argument("--testsplit", action="store_true", help="frozen GT-free recipe on test-49")
    ap.add_argument("--demo", action="store_true", help="runnable self-check (no data needed)")
    args = ap.parse_args()

    if args.demo:
        _demo()
        return

    if args.train_probe:
        train_probe(args.data_dir, split_names(args.data_dir, "train"))
        return

    if args.teamside:
        payload = {}
        for split, pos in (("train", TRAIN_PROBE_DIR / "positions"),
                           ("valid", args.out_dir / "positions")):
            g = grade_teamside(args.data_dir, pos, split_names(args.data_dir, split))
            payload[split] = {
                "n": len(g), "correct": int(g["ok"].sum()),
                "accuracy": float(g["ok"].mean()) if len(g) else 0.0,
                "wrong": g.loc[~g["ok"], "seq"].tolist(),
                "margin_correct_median": float(g.loc[g["ok"], "margin"].median()) if len(g) else 0,
                "margin_wrong": g.loc[~g["ok"], "margin"].round(3).tolist(),
                "per_seq": g.to_dict("records"),
            }
            logger.info("%s: %d/%d = %.4f  wrong %s", split, payload[split]["correct"],
                        payload[split]["n"], payload[split]["accuracy"], payload[split]["wrong"])
        _write(RESULTS_DIR / "gsr_deleak_teamside.json", payload)
        return

    cfg = SolverConfig.load(SOLVER_CONFIG)
    dev, test38 = split_sequences(args.data_dir, args.out_dir)
    kw = {"votes_subdir": "koshkina_percrop_votes", "cache_subdir": "identity_bundles_percrop"}

    if args.dev:
        bundles = load_bundles(args.data_dir, args.out_dir, dev, **kw)
        arms = {}
        for team_src in ("gt", "free"):
            for roster in ("gt", "full", "self"):
                tag = f"dev_{team_src}_{roster}"
                arms[tag] = run_arm(bundles, cfg, args.data_dir, args.out_dir, dev,
                                    team_src=team_src, roster=roster, tag=tag, score_hota=True)
        _write(RESULTS_DIR / "gsr_deleak_dev.json", {"dev": dev, "arms": arms})
        return

    if args.matrix:
        bundles = load_bundles(args.data_dir, args.out_dir, test38, **kw)
        arms = {}
        for team_src, roster in (("gt", "gt"), ("free", "gt"), ("fixed", "gt"),
                                 ("gt", "full"), ("gt", "self"),
                                 ("free", "full"), ("free", "self")):
            tag = f"t38_{team_src}_{roster}"
            arms[tag] = run_arm(bundles, cfg, args.data_dir, args.out_dir, test38,
                                team_src=team_src, roster=roster, tag=tag, score_hota=True)
        base = arms["t38_gt_gt"]["gs_hota_per_seq"]
        for tag, a in arms.items():
            a["paired_vs_leaky"] = paired_stats(base, a["gs_hota_per_seq"], test38)
        _write(RESULTS_DIR / "gsr_deleak_matrix.json", {"test38": test38, "arms": arms})
        return

    if args.testsplit:
        run_testsplit(args.data_dir)
        return

    if args.freeze:
        blob = json.loads(FROZEN_PATH.read_text(encoding="utf-8")) if FROZEN_PATH.exists() else {}
        blob = {"declared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "team_map": "resolve_team_map_free (deeper cluster mean pitch_x = left)",
                "roster": blob.get("roster", "TBD"),
                "solver_config": str(SOLVER_CONFIG),
                "solver_config_sha256": hashlib.sha256(SOLVER_CONFIG.read_bytes()).hexdigest(),
                "note": "chosen on train (team side) and valid DEV-20 (roster); "
                        "valid TEST-38 and the official test split are verification only"}
        _write(FROZEN_PATH, blob)
        return

    ap.error("choose one of --train-probe / --teamside / --dev / --freeze / --matrix / --testsplit")


if __name__ == "__main__":
    main()
