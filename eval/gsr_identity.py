"""Stage-2 gate driver: the global identity solve on the SoccerNet-GSR public split.

Runs :mod:`generator.identity_solve` over the same tracklet partition the Stage-1 connector produced
at ``tau = 0.040`` (the only arm meeting the 80% merge-precision bar, ``results/GTA_LINK_STAGE1.md``)
and grades it against the appearance-only arm that partition already ships:

* **Gate 1 (primary)** -- per-row identity accuracy ``(team AND jersey)`` over every prediction row
  matched to a GT player within 5 m, solver vs baseline, paired per sequence (Wilcoxon) with a
  row-level McNemar as a secondary.
* **Gate 2** -- official GS-HOTA on the same submissions (only ``attributes.jersey`` of ``player``
  rows differs), paired per sequence against the 23.53 baseline.

Discipline. The 58 sequences on disk are the GSR **valid** split; no train/test split is held
locally (see ``results/IDENTITY_SOLVER_STAGE2.md``). A deterministic DEV/TEST partition is therefore
declared here *in code* -- ``DEV = sorted(seqs)[::3]`` (20 sequences), ``TEST`` = the other 38 -- the
solver's weights are fitted on DEV only, written to ``results/identity_solver_config.json``, and the
TEST arm is run from that file.

CLI::

    python -m eval.gsr_identity --build-bundles     # CPU: per-sequence evidence cache
    python -m eval.gsr_identity --fit               # CPU: DEV grid search -> frozen config
    python -m eval.gsr_identity --test              # CPU: TEST arm, gates 1 + 2, ablations
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from eval.gsr_gta import gt_rows, repair_sequence
from eval.gsr_score import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_RESULTS_DIR,
    EVAL_CONFIGS,
    gs_hota,
    load_gt_people_by_frame,
    resolve_team_map,
)
from generator.gta_link import GtaParams, load_or_build_det_embeddings
from generator.identity_solve import (
    SOLVER_VERSION,
    Identity,
    SolverConfig,
    Tracklet,
    digit_confusion_prior,
    exclusion_groups,
    posterior,
    solve_assignment,
)
from generator.track_relink import load_gt_ids_by_frame

logger = logging.getLogger("gsr_identity")

#: Frozen Stage-1 partition consumed by the solver (80.1% GT-audited merge precision).
PARTITION = GtaParams(tau=0.040)
#: Baseline arm directory (GTA connector + unanimous jersey propagation, GS-HOTA 23.53).
BASE_ARM = "eval_gta_tau0.040_nosplit"
#: Solver arm directory.
SOLVER_ARM = "eval_identity_solver"
#: Bundle cache and frozen-config locations.
BUNDLE_SUBDIR = "identity_bundles"
CONFIG_PATH = Path("results/identity_solver_config.json")
#: Number of best crop pairs kept per (tracklet, identity) so ``topk`` can be swept from the cache.
MAX_TOPK = 5


# === Bundle construction (one CPU pass per sequence) =============================================
def _load_votes(path: Path) -> dict[int, list[tuple[int, float]]]:
    """Read a per-track jersey-vote cache, accepting one vote (`[n, c]`) or many (`[[n, c], ...]`).

    The on-record aggregated reader writes exactly one vote per track; the densified per-crop
    evidence (:func:`tools.ocr_density.emit_votes`) may write several, which the solver's soft-vote
    term already supports.
    """
    blob = json.loads(path.read_text(encoding="utf-8"))["votes"]
    out: dict[int, list[tuple[int, float]]] = {}
    for k, v in blob.items():
        pairs = v if isinstance(v[0], (list, tuple)) else [v]
        out[int(k)] = [(int(n), float(c)) for n, c in pairs]
    return out


def _tracklet_reads(
    votes: dict[int, list[tuple[int, float]]], remap: dict[int, int]
) -> dict[int, list]:
    """Group original-tracklet Koshkina votes onto their merged tracklet id."""
    out: dict[int, list] = defaultdict(list)
    for tid, pairs in votes.items():
        for num, conf in pairs:
            if num >= 1:
                out[int(remap.get(int(tid), int(tid)))].append((int(num), float(conf)))
    return out


def _roster(seq_dir: Path, team_map: dict[int, str]) -> list[Identity]:
    """Numbered GT player identities of a sequence as roster slots (the lineup-sheet analogue)."""
    gt = json.loads((seq_dir / "Labels-GameState.json").read_text(encoding="utf-8"))
    side_to_team = {v: k for k, v in team_map.items()}
    seen: dict[tuple, Identity] = {}
    for ann in gt["annotations"]:
        a = ann.get("attributes") or {}
        if a.get("role") != "player" or ann.get("track_id") is None:
            continue
        j, side = a.get("jersey"), a.get("team")
        if j in (None, "") or side not in side_to_team:
            continue
        key = (side, int(j))
        seen.setdefault(key, Identity(key, side_to_team[side], int(j), "player"))
    return [seen[k] for k in sorted(seen)]


def build_bundle(seq_dir: Path, out_dir: Path, *, votes_subdir: str = "koshkina_jersey",
                 positions_subdir: str = "positions") -> dict:
    """Summarise one sequence into everything the solver and the grader need (CPU, cached).

    Args:
        seq_dir: GSR sequence directory.
        out_dir: Pipeline output root.
        votes_subdir: Which per-track jersey-vote cache to consume. ``koshkina_jersey`` is the
            on-record aggregated reader (one vote per track); ``koshkina_percrop_votes`` is the
            densified evidence of :mod:`tools.ocr_density` (many votes per track allowed).
        positions_subdir: Which positions table to bundle. ``positions`` is the on-record
            calibrator output; ``positions_filled`` is the calibration-gap repair of
            :func:`generator.postprocess.fill_calibration_gaps` (``tools.gsr_calibfill``).

    Returns a dict with the merged tracklets, their evidence, the roster, leave-one-out gallery
    similarities (top ``MAX_TOPK`` per pair), the mutual-exclusion groups and the GT row counts that
    turn any assignment into a per-row accuracy without re-reading the annotations.
    """
    name = seq_dir.name
    df = pd.read_parquet(out_dir / positions_subdir / f"{name}.parquet")
    det = load_or_build_det_embeddings(
        seq_dir, df, out_dir / "detembed_cache_prtreid" / f"{name}.npz", params=PARTITION)
    _sdf, _lookup, remap, _st = repair_sequence(df, det, PARTITION, do_split=False)
    team_map = resolve_team_map(df, load_gt_people_by_frame(seq_dir))
    identities = _roster(seq_dir, team_map)

    people = df[df["role"].isin(["player", "goalkeeper"])].copy()
    people["mid"] = people["track_id"].map(lambda t: int(remap.get(int(t), int(t))))
    emb: dict[int, list] = defaultdict(list)
    for tid, (_f, e) in det.items():
        emb[int(remap.get(int(tid), int(tid)))].append(e)
    reads = _tracklet_reads(_load_votes(out_dir / votes_subdir / f"{name}.json"), remap)

    order = sorted(people["mid"].unique().tolist())
    index = {m: i for i, m in enumerate(order)}
    tracklets: list[Tracklet] = []
    for mid in order:
        grp = people[people["mid"] == mid]
        team_counts = Counter(int(v) for v in grp["team"] if np.isfinite(v))
        team, team_frac = (None, 0.0)
        if team_counts:
            team, cnt = team_counts.most_common(1)[0]
            team_frac = cnt / sum(team_counts.values())
        role_counts = Counter(str(v) for v in grp["role"])
        n = sum(role_counts.values())
        tracklets.append(Tracklet(
            track_id=int(mid), n_rows=int(len(grp)), team=team, team_frac=float(team_frac),
            role_frac={k: v / n for k, v in role_counts.items()},
            reads=tuple(reads.get(int(mid), [])),
            emb=np.concatenate(emb[mid]) if emb.get(mid) else np.zeros((0, 256), np.float32),
            span=(int(grp["frame"].min()), int(grp["frame"].max()))))

    # Leave-one-out galleries: identity i is anchored by tracklets whose own read is i.number.
    anchors: dict[int, list[int]] = defaultdict(list)
    for k, trk in enumerate(tracklets):
        if not trk.reads:
            continue
        num = Counter(n for n, _c in trk.reads).most_common(1)[0][0]
        for i, ident in enumerate(identities):
            if ident.number == num and ident.team == trk.team:
                anchors[i].append(k)
    top = np.full((len(tracklets), len(identities), MAX_TOPK), np.nan, np.float32)
    for i, members in anchors.items():
        for k, trk in enumerate(tracklets):
            gal = [tracklets[m].emb for m in members if m != k and tracklets[m].emb.size]
            if not gal or trk.emb.size == 0:
                continue
            sims = np.sort((trk.emb @ np.concatenate(gal).T).ravel())[-MAX_TOPK:]
            top[k, i, MAX_TOPK - len(sims):] = sims

    alive = [[index[int(m)] for m in g["mid"].unique()] for _f, g in people.groupby("frame")]
    groups = exclusion_groups(alive)

    # GT row counts per tracklet: (predicted side, GT side, GT jersey) -> rows.
    pre = gt_rows(df, load_gt_ids_by_frame(seq_dir))
    gt_attr = _gt_attributes(seq_dir)
    rowcounts: list[Counter] = [Counter() for _ in tracklets]
    for r in people.itertuples():
        gid = pre.get((int(r.track_id), int(r.frame)))
        if gid is None or gid not in gt_attr:
            continue
        gside, gjersey = gt_attr[gid]
        pside = team_map.get(int(r.team)) if np.isfinite(r.team) else None
        rowcounts[index[int(r.mid)]][(pside, gside, gjersey)] += 1
    return {
        "seq": name, "tracklets": tracklets, "identities": identities, "top": top,
        "groups": groups, "rowcounts": rowcounts, "team_map": team_map,
        "n_rows": int(len(people)), "n_auditable": int(sum(sum(c.values()) for c in rowcounts)),
    }


def _gt_attributes(seq_dir: Path) -> dict[int, tuple[str | None, str | None]]:
    """``GT track_id -> (team side, jersey string or None)`` for players and goalkeepers."""
    gt = json.loads((seq_dir / "Labels-GameState.json").read_text(encoding="utf-8"))
    out: dict[int, tuple[str | None, str | None]] = {}
    for ann in gt["annotations"]:
        a = ann.get("attributes") or {}
        if a.get("role") not in {"player", "goalkeeper"} or ann.get("track_id") is None:
            continue
        j = a.get("jersey")
        out[int(ann["track_id"])] = (a.get("team"), None if j in (None, "") else str(j))
    return out


def load_bundles(data_dir: Path, out_dir: Path, names: list[str], *, rebuild: bool = False,
                 votes_subdir: str = "koshkina_jersey", cache_subdir: str = BUNDLE_SUBDIR,
                 positions_subdir: str = "positions") -> dict:
    """Load (building and caching if needed) the evidence bundles for ``names``.

    Args:
        data_dir: GSR ground-truth folder.
        out_dir: Pipeline output root.
        names: Sequence names.
        rebuild: Rebuild even when a cached bundle exists.
        votes_subdir: Jersey-vote cache to consume (see :func:`build_bundle`).
        cache_subdir: Bundle cache directory, so a densified arm never overwrites the on-record one.
        positions_subdir: Positions table to bundle (see :func:`build_bundle`). Give a repaired
            source its own ``cache_subdir`` or it will read the other one's pickles.
    """
    cache = out_dir / cache_subdir
    cache.mkdir(parents=True, exist_ok=True)
    bundles: dict[str, dict] = {}
    for i, name in enumerate(names):
        path = cache / f"{name}.pkl"
        if path.exists() and not rebuild:
            bundles[name] = pickle.loads(path.read_bytes())
            continue
        t0 = time.time()
        bundles[name] = build_bundle(data_dir / name, out_dir, votes_subdir=votes_subdir,
                                     positions_subdir=positions_subdir)
        path.write_bytes(pickle.dumps(bundles[name]))
        b = bundles[name]
        logger.info("[%d/%d] %s: %d tracklets, %d identities, %d groups, %d auditable rows (%.1fs)",
                    i + 1, len(names), name, len(b["tracklets"]), len(b["identities"]),
                    len(b["groups"]), b["n_auditable"], time.time() - t0)
    return bundles


# === Solve + score ===============================================================================
def sims_at(top: np.ndarray, topk: int) -> np.ndarray:
    """Mean of the best ``topk`` cached similarities per (tracklet, identity) pair (pure)."""
    if top.size == 0:
        return np.zeros(top.shape[:2])
    sel = top[:, :, MAX_TOPK - min(topk, MAX_TOPK):]
    with np.errstate(invalid="ignore"):
        return np.nanmean(sel, axis=2)


def solve_bundle_scored(bundle: dict, cfg: SolverConfig) -> tuple[list[int | None], list[float]]:
    """Solve one bundle -> ``(assigned jersey number or None, posterior of that pick)`` per tracklet.

    The posterior is the dial `results/EVIDENCE_DENSITY_LAW.md` sweeps to trace coverage at a
    precision floor; abstentions carry ``0.0``.
    """
    tracklets, identities = bundle["tracklets"], bundle["identities"]
    if not tracklets or not identities:
        return [None] * len(tracklets), [0.0] * len(tracklets)
    sims = sims_at(bundle["top"], cfg.topk)
    probs = np.stack([posterior(t, identities, sims[k], cfg) for k, t in enumerate(tracklets)])
    probs[:, -1] = cfg.r_abstain  # the abstain payoff is the calibrated floor, not the unknown mass
    weights = np.array([t.n_rows for t in tracklets], float)
    groups = bundle["groups"] if cfg.use_mutex else []
    pick = solve_assignment(probs, weights, groups, [i.team for i in identities],
                            max_concurrent=cfg.max_concurrent)
    return ([None if p < 0 else identities[p].number for p in pick],
            [0.0 if p < 0 else float(probs[k, p]) for k, p in enumerate(pick)])


def solve_bundle(bundle: dict, cfg: SolverConfig) -> list[int | None]:
    """Solve one bundle -> assigned jersey number (or ``None``) per tracklet."""
    return solve_bundle_scored(bundle, cfg)[0]


def coverage_curve(
    bundles: dict, assigns: dict, confs: dict, floors: tuple[float, ...] = (0.85, 0.60),
) -> dict:
    """Row-weighted coverage reachable at each jersey-precision floor by dialling the posterior.

    Args:
        bundles: The evidence bundles (for ``rowcounts``).
        assigns: ``{sequence: [jersey number or None per tracklet]}``.
        confs: ``{sequence: [posterior per tracklet]}``.
        floors: Precision floors to report coverage at.

    Returns:
        ``{"n_rows", "at_<floor>": {"coverage", "precision", "threshold"}, "curve": [...]}``.
    """
    items: list[tuple[float, int, int]] = []  # (posterior, correct rows, named rows)
    n_rows = 0
    for name, b in bundles.items():
        for rc, num, c in zip(b["rowcounts"], assigns[name], confs[name]):
            tot = sum(rc.values())
            n_rows += tot
            if num is None:
                continue
            ok = sum(cnt for (_ps, _gs, gj), cnt in rc.items() if gj == str(num))
            items.append((float(c), int(ok), int(tot)))
    items.sort(key=lambda r: -r[0])
    curve, ok_cum, named_cum = [], 0, 0
    for conf, ok, tot in items:
        ok_cum += ok
        named_cum += tot
        curve.append({"threshold": conf, "coverage": named_cum / max(n_rows, 1),
                      "precision": ok_cum / max(named_cum, 1)})
    out: dict = {"n_rows": n_rows, "n_named_tracklets": len(items),
                 "full_coverage": curve[-1] if curve else None}
    for f in floors:
        best = max((p for p in curve if p["precision"] >= f),
                   key=lambda p: p["coverage"], default=None)
        out[f"at_{f}"] = best or {"coverage": 0.0, "precision": 0.0, "threshold": None}
    return out


def score_assignment(bundle: dict, assign: list[int | None]) -> tuple[int, int, int]:
    """``(n_auditable_rows, rows with team+jersey right, rows with jersey right)`` (pure)."""
    n = ok = okj = 0
    for rc, num in zip(bundle["rowcounts"], assign):
        target = None if num is None else str(num)
        for (pside, gside, gjersey), cnt in rc.items():
            n += cnt
            if gjersey == target:
                okj += cnt
                if pside == gside:
                    ok += cnt
    return n, ok, okj


def baseline_assignments(bundles: dict, out_dir: Path) -> dict[str, list[int | None]]:
    """The appearance-only baseline's jersey per merged tracklet, read from its own submissions."""
    out: dict[str, list[int | None]] = {}
    for name, b in bundles.items():
        preds = json.loads((out_dir / BASE_ARM / "predictions" / "data" /
                            f"{name}.json").read_text(encoding="utf-8"))["predictions"]
        jersey: dict[int, Counter] = defaultdict(Counter)
        for p in preds:
            j = p["attributes"].get("jersey")
            jersey[int(p["track_id"])][None if j in (None, "") else str(j)] += 1
        out[name] = [int(jersey[t.track_id].most_common(1)[0][0])
                     if jersey.get(t.track_id) and jersey[t.track_id].most_common(1)[0][0]
                     else None for t in b["tracklets"]]
    return out


def mcnemar_rows(bundle: dict, base: list[int | None], arm: list[int | None]) -> tuple[int, int]:
    """Discordant row counts ``(base-only-right, arm-only-right)`` for one sequence (pure).

    When the two arms give a tracklet different targets no row can be right under both, so the
    discordance is exactly each arm's own correct-row count on that tracklet.
    """
    b_only = a_only = 0
    for rc, x, y in zip(bundle["rowcounts"], base, arm):
        if x == y:
            continue
        tx = None if x is None else str(x)
        ty = None if y is None else str(y)
        for (pside, gside, gjersey), cnt in rc.items():
            if pside != gside:
                continue
            b_only += cnt if gjersey == tx else 0
            a_only += cnt if gjersey == ty else 0
    return b_only, a_only


# === DEV fit =====================================================================================
def fit_confusion(bundles: dict, data_dir: Path) -> tuple[float, dict]:
    """Measure OCR read accuracy and the digit-confusion prior on the DEV bundles only."""
    pairs: list[tuple[int, int]] = []
    for b in bundles.values():
        for trk, rc in zip(b["tracklets"], b["rowcounts"]):
            if not trk.reads or not rc:
                continue
            # dominant GT jersey of this tracklet's auditable rows
            tally: Counter = Counter()
            for (_ps, _gs, gj), cnt in rc.items():
                tally[gj] += cnt
            true = tally.most_common(1)[0][0]
            if true is None:
                continue
            for num, _conf in trk.reads:
                pairs.append((int(true), int(num)))
    n_ok = sum(1 for t, r in pairs if t == r)
    return (n_ok / len(pairs) if pairs else 0.875), digit_confusion_prior(pairs)


#: DEV grid. Deliberately coarse -- 20 sequences cannot resolve a fine one.
GRID = {
    "pi_none": [0.05, 0.30, 0.70],
    "r_abstain": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
    "app_gain": [10.0, 40.0, 160.0],
    "sim_none": [0.85, 0.92],
    "topk": [1, 3, 5],
}


def fit_config(bundles: dict, data_dir: Path) -> tuple[SolverConfig, list[dict]]:
    """Grid-search the DEV bundles for the config with the best per-row identity accuracy."""
    p_correct, confusion = fit_confusion(bundles, data_dir)
    logger.info("DEV OCR: p_correct %.4f over %d reads (%d wrong, %d aligned digit obs)",
                p_correct, confusion["n"], confusion["n_wrong"], confusion["n_aligned"])
    base = SolverConfig(p_correct=p_correct, confusion=confusion)
    rows: list[dict] = []
    best, best_cfg = -1.0, base
    for topk in GRID["topk"]:
        for gain in GRID["app_gain"]:
            for sim_none in GRID["sim_none"]:
                for pi in GRID["pi_none"]:
                    for r in GRID["r_abstain"]:
                        cfg = replace(base, topk=topk, app_gain=gain, sim_none=sim_none,
                                      pi_none=pi, r_abstain=r)
                        n = ok = okj = named = 0
                        for b in bundles.values():
                            assign = solve_bundle(b, cfg)
                            a, c, cj = score_assignment(b, assign)
                            n += a
                            ok += c
                            okj += cj
                            named += sum(1 for x in assign if x is not None)
                        acc = ok / max(n, 1)
                        rows.append({"topk": topk, "app_gain": gain, "sim_none": sim_none,
                                     "pi_none": pi, "r_abstain": r, "identity": acc,
                                     "jersey": okj / max(n, 1), "n_named": named})
                        if acc > best:
                            best, best_cfg = acc, cfg
        logger.info("grid topk=%d done, best identity accuracy so far %.4f", topk, best)
    return best_cfg, rows


# === Submissions + GS-HOTA =======================================================================
def write_solver_submissions(bundles: dict, assigns: dict, arm_dir: Path, data_dir: Path) -> None:
    """Rewrite only ``attributes.jersey`` on player rows of the baseline arm -> ``arm_dir``."""
    dest = arm_dir / "predictions" / "data"
    dest.mkdir(parents=True, exist_ok=True)
    for name, b in bundles.items():
        number = {t.track_id: a for t, a in zip(b["tracklets"], assigns[name])}
        src = arm_dir.parent / BASE_ARM / "predictions" / "data" / f"{name}.json"
        payload = json.loads(src.read_text(encoding="utf-8"))
        for p in payload["predictions"]:
            if p["attributes"].get("role") != "player":
                continue
            num = number.get(int(p["track_id"]))
            p["attributes"]["jersey"] = None if num is None else str(num)
        (dest / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def hota_per_seq(arm_root: Path, data_dir: Path, names: list[str]) -> dict[str, float]:
    """Official GS-HOTA (``gs_hota_full``) per sequence for one arm."""
    res = gs_hota(arm_root, data_dir, seq_info={n: 0 for n in names},
                  **EVAL_CONFIGS["gs_hota_full"])
    return {n: v["GS-HOTA"] for n, v in res["per_seq"].items()} | {
        "COMBINED": res["combined"]["GS-HOTA"]}


# === Reporting ===================================================================================
def paired_stats(base: dict[str, float], arm: dict[str, float], names: list[str]) -> dict:
    """Wilcoxon signed-rank on the per-sequence paired deltas (plus helped/hurt counts)."""
    from scipy.stats import wilcoxon  # noqa: PLC0415

    d = np.array([arm[n] - base[n] for n in names if n in arm and n in base])
    stat = {"n": int(d.size), "mean": float(d.mean()) if d.size else 0.0,
            "median": float(np.median(d)) if d.size else 0.0,
            "helped": int((d > 0).sum()), "hurt": int((d < 0).sum()),
            "worst": float(d.min()) if d.size else 0.0,
            "best": float(d.max()) if d.size else 0.0}
    if d.size and np.any(d != 0):
        w = wilcoxon(d)
        stat["wilcoxon_p"] = float(w.pvalue)
    else:
        stat["wilcoxon_p"] = 1.0
    return stat


def run_test(data_dir: Path, out_dir: Path, results_dir: Path, names: list[str],
             cfg: SolverConfig, *, tag: str, score_hota: bool = True,
             votes_subdir: str = "koshkina_jersey", cache_subdir: str = BUNDLE_SUBDIR) -> dict:
    """Solve, grade gate 1, optionally gate 2, and return the payload for ``names``."""
    bundles = load_bundles(data_dir, out_dir, names, votes_subdir=votes_subdir,
                           cache_subdir=cache_subdir)
    scored = {n: solve_bundle_scored(b, cfg) for n, b in bundles.items()}
    assigns = {n: v[0] for n, v in scored.items()}
    confs = {n: v[1] for n, v in scored.items()}
    base_assign = baseline_assignments(bundles, out_dir)
    per_seq: dict[str, dict] = {}
    tot = Counter()
    for name, b in bundles.items():
        n, ok, okj = score_assignment(b, assigns[name])
        bn, bok, bokj = score_assignment(b, base_assign[name])
        b_only, a_only = mcnemar_rows(b, base_assign[name], assigns[name])
        tot["mcnemar_base_only"] += b_only
        tot["mcnemar_arm_only"] += a_only
        per_seq[name] = {
            "n_rows": n, "solver_identity": ok / max(n, 1), "base_identity": bok / max(bn, 1),
            "solver_jersey": okj / max(n, 1), "base_jersey": bokj / max(bn, 1),
            "n_named": int(sum(1 for a in assigns[name] if a is not None)),
            "n_tracklets": len(b["tracklets"]),
        }
        tot["n"] += n
        tot["solver_ok"] += ok
        tot["solver_okj"] += okj
        tot["base_ok"] += bok
        tot["base_okj"] += bokj
        tot["named"] += per_seq[name]["n_named"]
        tot["tracklets"] += len(b["tracklets"])
    gate1 = paired_stats({k: v["base_identity"] for k, v in per_seq.items()},
                         {k: v["solver_identity"] for k, v in per_seq.items()}, names)
    from scipy.stats import binomtest  # noqa: PLC0415

    b_only, a_only = tot["mcnemar_base_only"], tot["mcnemar_arm_only"]
    gate1["mcnemar"] = {
        "base_only_right_rows": int(b_only), "arm_only_right_rows": int(a_only),
        "p": float(binomtest(int(a_only), int(a_only + b_only)).pvalue) if a_only + b_only else 1.0,
    }
    payload = {
        "version": SOLVER_VERSION, "tag": tag, "n_sequences": len(names), "sequences": names,
        "config": {k: v for k, v in vars(cfg).items() if k != "confusion"},
        "pooled": {
            "n_rows": tot["n"],
            "base_identity": tot["base_ok"] / max(tot["n"], 1),
            "solver_identity": tot["solver_ok"] / max(tot["n"], 1),
            "base_jersey": tot["base_okj"] / max(tot["n"], 1),
            "solver_jersey": tot["solver_okj"] / max(tot["n"], 1),
            "tracklets": tot["tracklets"], "tracklets_named": tot["named"],
        },
        "gate1_paired": gate1, "per_seq": per_seq,
        "coverage_curve": coverage_curve(bundles, assigns, confs),
        "votes_subdir": votes_subdir,
    }
    if score_hota:
        arm = SOLVER_ARM if votes_subdir == "koshkina_jersey" else f"{SOLVER_ARM}_percrop"
        write_solver_submissions(bundles, assigns, out_dir / arm, data_dir)
        b_h = hota_per_seq(out_dir / BASE_ARM, data_dir, names)
        s_h = hota_per_seq(out_dir / arm, data_dir, names)
        payload["gs_hota"] = {"base_combined": b_h["COMBINED"], "solver_combined": s_h["COMBINED"],
                              "paired": paired_stats(b_h, s_h, names),
                              "per_seq": {n: {"base": b_h.get(n), "solver": s_h.get(n)}
                                          for n in names}}
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"gsr_identity_{tag}.json").write_text(json.dumps(payload, indent=2),
                                                          encoding="utf-8")
    _log(payload)
    return payload


def _log(p: dict) -> None:
    """ASCII summary (cp1252-safe)."""
    pl, g = p["pooled"], p["gate1_paired"]
    logger.info("== %s over %d sequences, %d auditable rows", p["tag"], p["n_sequences"],
                pl["n_rows"])
    logger.info("   identity (team+jersey): base %.4f -> solver %.4f (%+.4f)",
                pl["base_identity"], pl["solver_identity"],
                pl["solver_identity"] - pl["base_identity"])
    logger.info("   jersey only:            base %.4f -> solver %.4f", pl["base_jersey"],
                pl["solver_jersey"])
    logger.info("   tracklets named %d / %d", pl["tracklets_named"], pl["tracklets"])
    logger.info("   paired per-seq delta mean %+.4f median %+.4f helped %d hurt %d Wilcoxon p=%.3g",
                g["mean"], g["median"], g["helped"], g["hurt"], g["wilcoxon_p"])
    m = g.get("mcnemar", {})
    if m:
        logger.info("   row-level McNemar: base-only-right %d, solver-only-right %d, p=%.3g",
                    m["base_only_right_rows"], m["arm_only_right_rows"], m["p"])
    if "gs_hota" in p:
        h = p["gs_hota"]
        logger.info("   GS-HOTA base %.2f -> solver %.2f; paired mean %+.3f helped %d hurt %d "
                    "p=%.3g", h["base_combined"], h["solver_combined"], h["paired"]["mean"],
                    h["paired"]["helped"], h["paired"]["hurt"], h["paired"]["wilcoxon_p"])


def split_sequences(data_dir: Path, out_dir: Path) -> tuple[list[str], list[str]]:
    """Declared DEV/TEST partition of the local 58-sequence split (every 3rd sequence is DEV)."""
    names = sorted(p.name for p in data_dir.iterdir() if p.is_dir()
                   and (p / "Labels-GameState.json").exists()
                   and (out_dir / BASE_ARM / "predictions" / "data" / f"{p.name}.json").exists())
    dev = names[::3]
    return dev, [n for n in names if n not in set(dev)]


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--build-bundles", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--fit", action="store_true", help="DEV grid search -> frozen config")
    ap.add_argument("--test", action="store_true", help="run the frozen config on TEST")
    ap.add_argument("--dev-report", action="store_true", help="re-score DEV with the frozen config")
    ap.add_argument("--ablate", action="store_true", help="TEST ablations (no OCR / no appearance)")
    ap.add_argument("--percrop", action="store_true",
                    help="consume the densified per-crop votes (tools.ocr_density --emit-votes); "
                         "the frozen config is reused with ONLY the digit-confusion prior refit")
    ap.add_argument("--refit-p-correct", action="store_true",
                    help="with --percrop, also refit p_correct on DEV (declared second arm)")
    ap.add_argument("--dump-config", type=Path, default=None,
                    help="with --percrop: write the frozen config carrying the DEV-refit digit "
                         "prior to this path and exit (no TEST run)")
    args = ap.parse_args()

    votes_subdir = "koshkina_percrop_votes" if args.percrop else "koshkina_jersey"
    cache_subdir = BUNDLE_SUBDIR + ("_percrop" if args.percrop else "")
    suffix = ("_percrop" if args.percrop else "") + ("_pfit" if args.refit_p_correct else "")
    kw = {"votes_subdir": votes_subdir, "cache_subdir": cache_subdir}

    dev, test = split_sequences(args.data_dir, args.out_dir)
    logger.info("DEV %d sequences, TEST %d sequences", len(dev), len(test))
    if args.build_bundles:
        load_bundles(args.data_dir, args.out_dir, dev + test, rebuild=args.rebuild, **kw)
        return
    if args.percrop:
        cfg = SolverConfig.load(args.config)
        dev_bundles = load_bundles(args.data_dir, args.out_dir, dev, **kw)
        p_correct, confusion = fit_confusion(dev_bundles, args.data_dir)
        logger.info("DEV per-crop OCR: p_correct %.4f over %d reads (%d wrong, %d digit obs)",
                    p_correct, confusion["n"], confusion["n_wrong"], confusion["n_aligned"])
        cfg = replace(cfg, confusion=confusion,
                      p_correct=p_correct if args.refit_p_correct else cfg.p_correct)
        if args.dump_config:
            cfg.save(args.dump_config)
            logger.info("wrote %s", args.dump_config)
            return
        for tag, names in (("dev" + suffix, dev), ("test" + suffix, test)):
            run_test(args.data_dir, args.out_dir, args.results_dir, names, cfg, tag=tag,
                     score_hota=(tag.startswith("test")), **kw)
        return
    if args.fit:
        bundles = load_bundles(args.data_dir, args.out_dir, dev)
        cfg, rows = fit_config(bundles, args.data_dir)
        cfg.save(args.config)
        args.results_dir.mkdir(parents=True, exist_ok=True)
        (args.results_dir / "gsr_identity_devgrid.json").write_text(
            json.dumps({"version": SOLVER_VERSION, "dev": dev, "rows": rows}, indent=2),
            encoding="utf-8")
        logger.info("frozen config -> %s", args.config)
        run_test(args.data_dir, args.out_dir, args.results_dir, dev, cfg, tag="dev",
                 score_hota=False)
        return
    cfg = SolverConfig.load(args.config)
    if args.dev_report:
        run_test(args.data_dir, args.out_dir, args.results_dir, dev, cfg, tag="dev_frozen",
                 score_hota=False)
        return
    if args.ablate:
        for tag, over in (("test_no_ocr", {"use_ocr": False}),
                          ("test_no_app", {"use_app": False}),
                          ("test_no_confusion", {"use_confusion": False}),
                          ("test_no_mutex", {"use_mutex": False})):
            run_test(args.data_dir, args.out_dir, args.results_dir, test,
                     replace(cfg, **over), tag=tag, score_hota=False)
        return
    if args.test:
        run_test(args.data_dir, args.out_dir, args.results_dir, test, cfg, tag="test")
        return
    ap.error("choose one of --build-bundles / --fit / --test / --dev-report / --ablate")


if __name__ == "__main__":
    main()
