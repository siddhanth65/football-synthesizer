"""v9 W6: jersey NAME-BORROWING -- propagate a number between fragments, never merge them.

W3's census put the biggest live prize on the jersey axis: the coverage oracle is worth +16.36
GS-DetA and the bound that only propagates numbers **our own chain already emitted** is +8.15. The
mechanism here is deliberately the weakest one that can reach it: for an unnamed track, find a named
track that is plausibly the same person and **copy the number**. No association changes, so GS-AssA
is untouched and a wrong guess cannot corrupt a trajectory.

The cost asymmetry (W3, on record) shapes every filter:

* naming a row whose GT identity IS numbered but with the wrong number costs nothing vs ``null``
  (under GS-DetA a wrong number and a missing number are both simply "not a match");
* naming a row whose GT identity is UNNUMBERED destroys a ``null == null`` match -- a real loss;
* GT never numbers goalkeepers (v8-W3: 0/156), and referees carry no number either.

So goalkeeper- and referee-majority tracks are **hard-excluded** on both sides, and the operating
point is chosen on the train-split ratio of newly-correct to over-named rows. Hard filters are
correct here because this is a copy, not a merge (W1's "features not filters" finding was about
merges, where a wrong gate destroys a trajectory).

Two arms, registered before any DEV contact:

* ``twix`` -- the W2 :class:`generator.assoc_twix.TwixMetric` checkpoints (frozen, seed-ensemble
  mean logit) ranking the (named, unnamed) candidate pairs.
* ``phys`` -- the learned-model-free fallback: rank by required speed between the two endpoints.

Both use the same candidate set, the same one-borrow rule and a threshold fitted **only** on
GSR-TRAIN.

CLI::

    python -m tools.gsr_v9_w6 --traincal        # stage 1: GSR-TRAIN census + threshold curves
    python -m tools.gsr_v9_w6 --arms            # stage 2: DEV-20 scoring + row audit
    python -m tools.gsr_v9_w6 --demo            # self-check
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from eval.gsr_score import DEFAULT_DATA_DIR, EVAL_CONFIGS, gs_hota
from tools.gsr_v9_deta import DEFAULT_ARM, load_gt, match_positions, similarity
from tools.gsr_v9_factory import FactoryParams, build_pairs, summarise_tracklets
from tools.gsr_v9_w5 import control_predictions, row_key

logger = logging.getLogger("gsr_v9_w6")

#: Where W6 writes its artifacts.
RESULTS_DIR = Path("results/gsr_benchmark")
#: Scratch tree (one process per work dir -- W4's corruption lesson).
WORK_DIR = Path("outputs/gsr/v9_w6")
#: W1 pair/tracklet dataset (GSR-TRAIN only) used for the threshold calibration.
DATASET_DIR = Path("outputs/gsr/v9_assoc/v1")
#: W2 checkpoints; the registered scorer is the mean logit over these three seeds.
CKPT_DIR = Path("outputs/gsr/v9_assoc/ckpts")
CKPT_SEEDS = (0, 1, 2)
#: Thresholds swept on GSR-TRAIN (logit for ``twix``, ``-req_speed_mps`` for ``phys``).
TWIX_GRID = tuple(float(x) for x in np.arange(-6.0, 8.01, 0.5))
PHYS_GRID = tuple(-float(x) for x in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0))


# === tracks of a submission ======================================================================
def frame_of(image_id: str) -> int:
    """Frame index of a GSR ``image_id`` (its last six digits are the 1-based frame number)."""
    return int(str(image_id)[-6:]) - 1


def submission_rows(preds: list[dict]) -> tuple[pd.DataFrame, dict[int, dict]]:
    """Turn a control-state submission into the factory's row table + per-track jersey evidence.

    The factory was written for the pipeline's positions parquet; a submission carries the same
    quantities (pitch bottom-middle point, role, team side, confidence) plus the solver's jersey
    decision, which is what the borrow copies.

    Args:
        preds: One sequence's ``predictions`` list, already voted (so each track's attributes are
            internally constant).

    Returns:
        ``(rows, jersey)`` where ``rows`` has the columns
        :func:`tools.gsr_v9_factory.summarise_tracklets` needs and ``jersey`` maps track id to a
        ``{"number", "mass", "share", "n_reads"}`` record for every track the solver named.
    """
    side = {"left": 0, "right": 1}
    recs, jersey = [], {}
    for p in preds:
        a = p.get("attributes") or {}
        if a.get("role") == "ball":
            continue
        tid = int(p["track_id"])
        bp = p["bbox_pitch"]
        recs.append({"track_id": tid, "frame": frame_of(p["image_id"]),
                     "pitch_x": float(bp["x_bottom_middle"]), "pitch_y": float(bp["y_bottom_middle"]),
                     "role": str(a.get("role")), "team": side.get(a.get("team"), -1),
                     "conf": float(p.get("confidence", 1.0)), "calib_error_m": 0.0, "gt_id": -1})
        j = a.get("jersey")
        if j not in (None, "") and str(j).lstrip("-").isdigit():
            jersey[tid] = {"number": int(j), "mass": 1.0, "share": 1.0, "n_reads": 1}
    rows = pd.DataFrame(recs).sort_values(["track_id", "frame"], ignore_index=True)
    return rows, jersey


# === candidates ==================================================================================
def candidate_mask(pairs: pd.DataFrame) -> np.ndarray:
    """Rows of a pair table that are legal name-borrow candidates (pure).

    Hard filters, all justified by the copy semantics:

    * exactly one side carries a number (nothing to borrow otherwise, and a jersey-conflict guard is
      inapplicable when one side is silent);
    * **both** sides are player-majority -- GT numbers no goalkeeper and no referee, so a borrow that
      touches one can only over-name;
    * both sides carry a known team and the same one.

    Args:
        pairs: Output of :func:`tools.gsr_v9_factory.build_pairs`.

    Returns:
        Boolean mask over ``pairs`` rows.
    """
    if pairs.empty:
        return np.zeros(0, bool)
    na = pairs["num_a"].to_numpy() >= 0
    nb = pairs["num_b"].to_numpy() >= 0
    player = (pairs["role_a"].to_numpy() == "player") & (pairs["role_b"].to_numpy() == "player")
    ta, tb = pairs["team_a"].to_numpy(), pairs["team_b"].to_numpy()
    return (na ^ nb) & player & (ta == tb) & (ta >= 0) & (tb >= 0)


def borrow_plan(pairs: pd.DataFrame, scores: np.ndarray, mask: np.ndarray,
                thr: float) -> list[dict]:
    """One borrow per unnamed track: its best-scoring named partner above ``thr`` (pure).

    A named track may lend to several borrowers -- each of those pairs cleared the bar on its own
    evidence -- but a borrower takes exactly one number, so the result never depends on order.

    Args:
        pairs: Candidate pair table.
        scores: One score per pair row (higher = more likely the same person).
        mask: :func:`candidate_mask` output.
        thr: Score threshold.

    Returns:
        One record per borrowing track, sorted by track id.
    """
    best: dict[int, dict] = {}
    idx = np.flatnonzero(mask & (scores >= thr))
    # Stable, order-independent: strictly-better score wins, ties go to the smaller lender id.
    for i in idx:
        r = pairs.iloc[int(i)]
        named, unnamed = ("a", "b") if int(r["num_a"]) >= 0 else ("b", "a")
        tid = int(r[f"tid_{unnamed}"])
        lender = int(r[f"tid_{named}"])
        cur = best.get(tid)
        s = float(scores[i])
        if cur is None or (s, -lender) > (cur["score"], -cur["lender"]):
            best[tid] = {"track_id": tid, "lender": lender, "number": int(r[f"num_{named}"]),
                         "score": s, "gap_frames": int(r["gap_frames"]),
                         "dist_m": float(r["dist_m"])}
    return [best[t] for t in sorted(best)]


def apply_borrow(preds: list[dict], plan: list[dict]) -> dict[str, int]:
    """Write the borrowed numbers onto every row of each borrowing track, in place.

    Returns:
        ``{"tracks": borrowers, "rows": rows written}``.
    """
    num = {p["track_id"]: str(p["number"]) for p in plan}
    n = 0
    for p in preds:
        v = num.get(int(p["track_id"]))
        a = p.get("attributes") or {}
        if v is None or a.get("role") != "player":
            continue
        a["jersey"] = v
        n += 1
    return {"tracks": len(plan), "rows": n}


# === scorers =====================================================================================
def load_models(ckpt_dir: Path, seeds: tuple[int, ...]):
    """Load the frozen W2 TwixMetric checkpoints in eval mode (CPU)."""
    import torch  # noqa: PLC0415

    from generator.assoc_twix import TwixConfig, TwixMetric  # noqa: PLC0415

    out = []
    for s in seeds:
        ck = torch.load(ckpt_dir / f"twixm_side_s{s}.pt", map_location="cpu", weights_only=False)
        m = TwixMetric(TwixConfig(**ck["cfg"]))
        m.load_state_dict(ck["state_dict"])
        out.append(m.eval())
    return out


def twix_scores(models, rows: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    """Seed-ensemble mean merge logit for every pair row (CPU, deterministic).

    ``clip_cos_dist`` is forced missing: the DEV-20 detection-embedding cache joins the shipped
    submission on only 790/896 tracks and is broken on 11 of 20 sequences (kb v9-w1-006 /
    v9-w5-002), so the appearance channel is unusable there. The train-side calibration therefore
    runs with the same channel switched off, and application and calibration see identical features.
    """
    import torch  # noqa: PLC0415

    from tools.gsr_v9_train import predict, sequence_windows  # noqa: PLC0415

    if pairs.empty:
        return np.zeros(0, np.float32)
    w = sequence_windows(rows, pairs.assign(clip_cos_dist=np.nan))
    w["group"] = w["win"].astype(np.int32)
    per = [predict(m, w, torch.device("cpu")) for m in models]
    return np.mean(per, axis=0)


def phys_scores(pairs: pd.DataFrame) -> np.ndarray:
    """The model-free fallback ranking: minus the speed the gap would require (pure)."""
    if pairs.empty:
        return np.zeros(0, np.float32)
    return -pairs["req_speed_mps"].to_numpy(np.float64)


# === stage 1: GSR-TRAIN calibration ==============================================================
def gt_jersey(seq_dir: Path) -> dict[int, str | None]:
    """GT number of every GT identity (``None`` = the annotators left it unnumbered).

    GSR carries the jersey per annotation but constant per identity (verified on the train split:
    every identity is either numbered on all its rows or on none), so a per-identity map is exactly
    the row-level truth GS-DetA compares against.
    """
    gt = json.loads((seq_dir / "Labels-GameState.json").read_text(encoding="utf-8"))
    tally: dict[int, Counter] = defaultdict(Counter)
    for ann in gt["annotations"]:
        if ann.get("supercategory") != "object" or ann.get("track_id") is None:
            continue
        a = ann.get("attributes") or {}
        if a.get("role") == "ball":
            continue
        j = a.get("jersey")
        tally[int(ann["track_id"])][None if j in (None, "") else str(int(j))] += 1
    return {t: c.most_common(1)[0][0] for t, c in tally.items()}


def outcome(rec: dict, tr_by_id: dict[int, pd.Series], gtj: dict[int, str | None]) -> str:
    """Label one planned borrow against GT: ``correct`` / ``neutral`` / ``overnamed`` / ``unknown``.

    ``neutral`` means the borrower's GT identity IS numbered but we copied a different number --
    DetA-neutral, because a wrong number scores exactly what ``null`` scores.
    """
    t = tr_by_id.get(rec["track_id"])
    gid = int(t["gt_id"]) if t is not None else -1
    if gid < 0:
        return "unknown"
    want = gtj.get(gid, None)
    if want is None:
        return "overnamed"
    return "correct" if str(rec["number"]) == want else "neutral"


def calibrate(dest: Path, data_dir: Path, ckpt_dir: Path, grids: dict) -> dict:
    """Stage 1: the GSR-TRAIN census and the precision-at-threshold curve of both arms."""
    models = load_models(ckpt_dir, CKPT_SEEDS)
    names = sorted(p.stem for p in (dest / "pairs").glob("*.parquet"))
    census: Counter = Counter()
    curves: dict[str, dict[float, Counter]] = {
        a: {t: Counter() for t in g} for a, g in grids.items()}
    for i, name in enumerate(names):
        rows = pd.read_parquet(dest / "tracklets" / f"{name}.parquet")
        pairs = pd.read_parquet(dest / "pairs" / f"{name}.parquet")
        tr = summarise_tracklets(rows, _jersey_from_pairs(pairs), FactoryParams())
        if tr.empty or pairs.empty:
            continue
        tr_by_id = {int(r["track_id"]): dict(r) for _k, r in tr.iterrows()}
        gtj = gt_jersey(data_dir / name)
        _census(census, tr, gtj)
        mask = candidate_mask(pairs)
        sc = {"twix": lambda: twix_scores(models, rows, pairs), "phys": lambda: phys_scores(pairs)}
        for arm, grid in grids.items():
            scores = sc[arm]()
            for thr in grid:
                for rec in borrow_plan(pairs, scores, mask, thr):
                    t = tr_by_id[rec["track_id"]]
                    c = curves[arm][thr]
                    c["tracks"] += 1
                    c["rows"] += int(t["n_rows"])
                    o = outcome(rec, tr_by_id, gtj)
                    c[o] += 1
                    c[f"rows_{o}"] += int(t["n_rows"])
        logger.info("[%d/%d] %s: %d tracklets, %d pairs, %d candidates", i + 1, len(names), name,
                    len(tr), len(pairs), int(mask.sum()))
    return {"n_sequences": len(names), "census": dict(census),
            "curves": {a: [{"threshold": t, **_curve_row(c)} for t, c in sorted(g.items())]
                       for a, g in curves.items()}}


def _jersey_from_pairs(pairs: pd.DataFrame) -> dict[int, dict]:
    """Per-tracklet jersey evidence recovered from a pair table (both sides carry it)."""
    out: dict[int, dict] = {}
    for s in ("a", "b"):
        for tid, num, mass in zip(pairs[f"tid_{s}"], pairs[f"num_{s}"], pairs[f"num_mass_{s}"]):
            if int(num) >= 0:
                out[int(tid)] = {"number": int(num), "mass": float(mass), "share": 1.0,
                                 "n_reads": 1}
    return out


def _census(c: Counter, tr: pd.DataFrame, gtj: dict[int, str | None]) -> None:
    """Accumulate the unnamed-player-tracklet census that calibrates the over-naming cost."""
    for r in tr.itertuples():
        if r.role != "player" or r.jersey_number >= 0:
            continue
        c["unnamed_player_tracks"] += 1
        c["unnamed_player_rows"] += int(r.n_rows)
        if int(r.gt_id) < 0:
            c["gt_unknown_tracks"] += 1
            c["gt_unknown_rows"] += int(r.n_rows)
        elif gtj.get(int(r.gt_id)) is None:
            c["gt_unnumbered_tracks"] += 1
            c["gt_unnumbered_rows"] += int(r.n_rows)
        else:
            c["gt_numbered_tracks"] += 1
            c["gt_numbered_rows"] += int(r.n_rows)


def _curve_row(c: Counter) -> dict:
    """Derived rates of one threshold's counters."""
    rows = max(c["rows"], 1)
    known = max(c["rows_correct"] + c["rows_overnamed"], 1)
    return {**{k: int(v) for k, v in c.items()},
            "overname_share_of_touched": c["rows_overnamed"] / rows,
            "row_precision_vs_overname": c["rows_correct"] / known,
            "net_rows": int(c["rows_correct"] - c["rows_overnamed"])}


# === stage 2: DEV-20 arms ========================================================================
def gt_row_jersey(seq_dir: Path, preds: list[dict]) -> dict[tuple, str | None]:
    """Map each prediction row to the GT jersey of the GT row it is position-matched to.

    Uses the W3/W4/W5 attribute-blind Hungarian matcher, so the audit populations are the same ones
    every previous stage-2 audit reported.
    """
    ts, gt_rows = load_gt(seq_dir)
    by_ts: dict[int, list[dict]] = defaultdict(list)
    for p in preds:
        if (p.get("attributes") or {}).get("role") == "ball" or p["image_id"] not in ts:
            continue
        by_ts[ts[p["image_id"]]].append(p)
    out: dict[tuple, str | None] = {}
    for t, gts in enumerate(gt_rows):
        prs = by_ts.get(t, [])
        gxy = np.array([[g["bbox_pitch"]["x_bottom_middle"], g["bbox_pitch"]["y_bottom_middle"]]
                        for g in gts], float).reshape(-1, 2)
        pxy = np.array([[p["bbox_pitch"]["x_bottom_middle"], p["bbox_pitch"]["y_bottom_middle"]]
                        for p in prs], float).reshape(-1, 2)
        for i, j in match_positions(similarity(gxy, pxy)):
            a = gts[i].get("attributes") or {}
            j_gt = a.get("jersey")
            out[row_key(prs[j])] = (None if a.get("role") != "player" or j_gt in (None, "")
                                    else str(int(j_gt)))
    return out


def audit_rows(ctrl: list[dict], arm: list[dict], gtj: dict[tuple, str | None]) -> dict:
    """Row-level audit of the rows the borrow changed (only jersey can change here)."""
    before = {row_key(p): (p.get("attributes") or {}).get("jersey") for p in ctrl}
    out: Counter = Counter()
    for p in arm:
        k = row_key(p)
        now = (p.get("attributes") or {}).get("jersey")
        if k not in before or before[k] == now:
            continue
        out["touched"] += 1
        if k not in gtj:
            out["unmatched"] += 1
        elif gtj[k] is None:
            out["overnamed"] += 1
        elif str(now) == gtj[k]:
            out["newly_correct"] += 1
        else:
            out["newly_wrong_named"] += 1
    return dict(out)


def dev_census(arm_dir: Path, data_dir: Path, seqs: list[str], ckpt_dir: Path) -> dict:
    """Post-mortem: what the DEV-20 candidate pool is actually made of.

    Two tables the train-side calibration could not produce, because on GSR-TRAIN the "named" side
    is the READER's dominant number on the EIoU partition while on DEV it is the SOLVER's decision
    on post-connector tracks:

    * the track census -- named / unnamed x our role x whether the owning GT identity is numbered;
    * the candidate ceiling -- the outcome of every one of the 165 (named, unnamed) candidates if it
      fired, so the best any threshold on this candidate set could ever do is bounded directly.
    """
    from tools.gsr_v9_w5 import owners  # noqa: PLC0415

    models = load_models(ckpt_dir, CKPT_SEEDS)
    src = arm_dir / "predictions" / "data"
    tracks: Counter = Counter()
    cand: Counter = Counter()
    recs: list[dict] = []
    for s in seqs:
        preds = control_predictions(src / f"{s}.json")
        rows, jersey = submission_rows(preds)
        tr = summarise_tracklets(rows, jersey, FactoryParams())
        own = owners(data_dir / s, preds)
        gtj = gt_jersey(data_dir / s)
        state = {}
        for r in tr.itertuples():
            o = own.get(int(r.track_id), {})
            gid = o.get("gt_id")
            want = gtj.get(gid) if gid is not None else None
            key = ("named" if r.jersey_number >= 0 else "unnamed", r.role,
                   "gt_unknown" if gid is None else
                   ("gt_numbered" if want is not None else "gt_unnumbered"))
            tracks[key] += 1
            tracks[(*key, "rows")] += int(r.n_rows)
            state[int(r.track_id)] = {"want": want, "gt_role": o.get("gt_role"),
                                      "n_rows": int(r.n_rows), "gid": gid}
        pairs = build_pairs(tr, FactoryParams())
        if pairs.empty:
            continue
        mask = candidate_mask(pairs)
        scores = twix_scores(models, rows, pairs)
        for i in np.flatnonzero(mask):
            r = pairs.iloc[int(i)]
            named, unnamed = ("a", "b") if int(r["num_a"]) >= 0 else ("b", "a")
            bt = state.get(int(r[f"tid_{unnamed}"]), {})
            lt = state.get(int(r[f"tid_{named}"]), {})
            got = str(int(r[f"num_{named}"]))
            out = ("unknown" if bt.get("gid") is None else
                   "overnamed" if bt.get("want") is None else
                   "correct" if got == bt["want"] else "neutral")
            cand[out] += 1
            cand[f"rows_{out}"] += bt.get("n_rows", 0)
            cand["lender_overnamed"] += int(lt.get("want") is None and lt.get("gid") is not None)
            recs.append({"seq": s, "borrower": int(r[f"tid_{unnamed}"]),
                         "lender": int(r[f"tid_{named}"]), "number": got,
                         "score": float(scores[i]), "gap_frames": int(r["gap_frames"]),
                         "dist_m": float(r["dist_m"]), "outcome": out,
                         "borrower_gt_role": bt.get("gt_role"), "lender_gt_role": lt.get("gt_role"),
                         "borrower_gt_jersey": bt.get("want"), "lender_gt_jersey": lt.get("want"),
                         "borrower_rows": bt.get("n_rows")})
        logger.info("census %s done", s)
    return {"tracks": {"|".join(str(x) for x in k): v for k, v in sorted(tracks.items(),
                                                                        key=lambda kv: str(kv[0]))},
            "candidates": dict(cand), "candidate_records": recs}


def run_arms(arm_dir: Path, data_dir: Path, seqs: list[str], work: Path, arms: dict,
             ckpt_dir: Path) -> dict:
    """Score each registered borrow arm against the shipped control, with paired stats + audit."""
    from scipy.stats import wilcoxon  # noqa: PLC0415

    src = arm_dir / "predictions" / "data"
    ctrl_dir = work / "control"
    (ctrl_dir / "predictions" / "data").mkdir(parents=True, exist_ok=True)
    ctrl, gtj, built = {}, {}, {}
    models = load_models(ckpt_dir, CKPT_SEEDS) if any(a["arm"] == "twix" for a in arms.values()) \
        else []
    for s in seqs:
        ctrl[s] = control_predictions(src / f"{s}.json")
        gtj[s] = gt_row_jersey(data_dir / s, ctrl[s])
        (ctrl_dir / "predictions" / "data" / f"{s}.json").write_text(
            json.dumps({"predictions": ctrl[s]}), encoding="utf-8")
        rows, jersey = submission_rows(ctrl[s])
        tr = summarise_tracklets(rows, jersey, FactoryParams())
        pairs = build_pairs(tr, FactoryParams())
        mask = candidate_mask(pairs) if len(pairs) else np.zeros(0, bool)
        built[s] = {"rows": rows, "tr": tr, "pairs": pairs, "mask": mask,
                    "twix": twix_scores(models, rows, pairs) if models and len(pairs)
                    else np.zeros(len(pairs)),
                    "phys": phys_scores(pairs)}
        logger.info("%s: %d tracks (%d named), %d pairs, %d candidates", s, len(tr),
                    int((tr["jersey_number"] >= 0).sum()) if len(tr) else 0, len(pairs),
                    int(mask.sum()))
    base = gs_hota(ctrl_dir, data_dir, seq_info={s: 0 for s in seqs}, **EVAL_CONFIGS["gs_hota_full"])
    out = {"control": {"combined": base["combined"], "per_seq": base["per_seq"]},
           "candidates": {s: int(built[s]["mask"].sum()) for s in seqs}}
    logger.info("control GS-DetA %.4f GS-HOTA %.4f", base["combined"]["GS-DetA"],
                base["combined"]["GS-HOTA"])
    for name, spec in arms.items():
        t0 = time.time()
        dest = work / "arm" / "predictions" / "data"
        dest.mkdir(parents=True, exist_ok=True)
        changed: Counter = Counter()
        audit: Counter = Counter()
        plans: list[dict] = []
        for s in seqs:
            preds = control_predictions(src / f"{s}.json")
            b = built[s]
            plan = borrow_plan(b["pairs"], b[spec["arm"]], b["mask"], spec["threshold"]) \
                if len(b["pairs"]) else []
            changed.update(apply_borrow(preds, plan))
            audit.update(audit_rows(ctrl[s], preds, gtj[s]))
            plans += [{"seq": s, **p} for p in plan]
            (dest / f"{s}.json").write_text(json.dumps({"predictions": preds}), encoding="utf-8")
        res = gs_hota(work / "arm", data_dir, seq_info={s: 0 for s in seqs},
                      **EVAL_CONFIGS["gs_hota_full"])
        dh = np.array([res["per_seq"][s]["GS-HOTA"] - base["per_seq"][s]["GS-HOTA"] for s in seqs])
        dd = np.array([res["per_seq"][s]["GS-DetA"] - base["per_seq"][s]["GS-DetA"] for s in seqs])
        p = float(wilcoxon(dh).pvalue) if np.any(dh != 0) else 1.0
        out[name] = {
            "spec": spec, "combined": res["combined"], "per_seq": res["per_seq"],
            "changed": dict(changed), "audit_rows": dict(audit), "plan": plans,
            "paired_hota": {"mean": float(dh.mean()), "helped": int((dh > 0).sum()),
                            "hurt": int((dh < 0).sum()), "wilcoxon_p": p},
            "paired_deta": {"mean": float(dd.mean()), "helped": int((dd > 0).sum()),
                            "hurt": int((dd < 0).sum())}}
        logger.info("%-10s GS-DetA %.4f (%+.4f) GS-HOTA %.4f (%+.4f) %s audit %s [%.0fs]", name,
                    res["combined"]["GS-DetA"],
                    res["combined"]["GS-DetA"] - base["combined"]["GS-DetA"],
                    res["combined"]["GS-HOTA"],
                    res["combined"]["GS-HOTA"] - base["combined"]["GS-HOTA"], dict(changed),
                    dict(audit), time.time() - t0)
        shutil.rmtree(work / "arm", ignore_errors=True)
    return out


# === self-check ==================================================================================
def demo() -> None:
    """Assert the candidate filter, the one-borrow rule, the writer and the audit."""
    cols = dict(num_mass_a=0.0, num_mass_b=0.0, gap_frames=10, dist_m=1.0)
    pairs = pd.DataFrame([
        {"tid_a": 1, "tid_b": 2, "num_a": 7, "num_b": -1, "role_a": "player", "role_b": "player",
         "team_a": 0, "team_b": 0, **cols},                                    # ok
        {"tid_a": 3, "tid_b": 2, "num_a": 9, "num_b": -1, "role_a": "player", "role_b": "player",
         "team_a": 0, "team_b": 0, **cols},                                    # ok, competes
        {"tid_a": 4, "tid_b": 5, "num_a": 7, "num_b": -1, "role_a": "player",
         "role_b": "goalkeeper", "team_a": 0, "team_b": 0, **cols},            # GK borrower
        {"tid_a": 6, "tid_b": 7, "num_a": 7, "num_b": -1, "role_a": "player", "role_b": "player",
         "team_a": 0, "team_b": 1, **cols},                                    # team mismatch
        {"tid_a": 8, "tid_b": 9, "num_a": 7, "num_b": 7, "role_a": "player", "role_b": "player",
         "team_a": 0, "team_b": 0, **cols},                                    # both named
        {"tid_a": 10, "tid_b": 11, "num_a": -1, "num_b": -1, "role_a": "player",
         "role_b": "player", "team_a": 0, "team_b": 0, **cols},                # neither named
    ])
    m = candidate_mask(pairs)
    assert list(m) == [True, True, False, False, False, False], m
    plan = borrow_plan(pairs, np.array([1.0, 2.0, 9.0, 9.0, 9.0, 9.0]), m, 0.5)
    assert plan == [{"track_id": 2, "lender": 3, "number": 9, "score": 2.0, "gap_frames": 10,
                     "dist_m": 1.0}], plan
    assert borrow_plan(pairs, np.array([1.0, 2.0, 9.0, 9.0, 9.0, 9.0]), m, 5.0) == []

    def row(img, tid, role, jersey):
        return {"image_id": img, "track_id": tid,
                "attributes": {"role": role, "team": "left", "jersey": jersey}}

    ctrl = [row("a", 2, "player", None), row("b", 2, "player", None),
            row("a", 5, "goalkeeper", None)]
    arm = [dict(r, attributes=dict(r["attributes"])) for r in ctrl]
    assert apply_borrow(arm, plan) == {"tracks": 1, "rows": 2}, apply_borrow(arm, plan)
    assert arm[2]["attributes"]["jersey"] is None, "a goalkeeper row must never be written"
    aud = audit_rows(ctrl, arm, {("a", 2): "9", ("b", 2): None})
    assert aud == {"touched": 2, "newly_correct": 1, "overnamed": 1}, aud

    assert frame_of("2021000004") == 3, frame_of("2021000004")
    rows, jersey = submission_rows([
        {"image_id": "2021000004", "track_id": 1, "confidence": 0.9,
         "attributes": {"role": "player", "team": "left", "jersey": "19"},
         "bbox_pitch": {"x_bottom_middle": 1.0, "y_bottom_middle": 2.0}},
        {"image_id": "2021000005", "track_id": 1, "confidence": 0.9,
         "attributes": {"role": "player", "team": "left", "jersey": "19"},
         "bbox_pitch": {"x_bottom_middle": 1.5, "y_bottom_middle": 2.0}},
        {"image_id": "2021000005", "track_id": 2, "confidence": 0.9,
         "attributes": {"role": "ball", "team": None, "jersey": None},
         "bbox_pitch": {"x_bottom_middle": 0.0, "y_bottom_middle": 0.0}}])
    assert len(rows) == 2 and list(rows["frame"]) == [3, 4], rows
    assert jersey == {1: {"number": 19, "mass": 1.0, "share": 1.0, "n_reads": 1}}, jersey
    assert rows["team"].tolist() == [0, 0]
    print("gsr_v9_w6 demo: OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-dir", type=Path, default=DEFAULT_ARM)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--ckpts", type=Path, default=CKPT_DIR)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    ap.add_argument("--work", type=Path, default=WORK_DIR)
    ap.add_argument("--traincal", action="store_true")
    ap.add_argument("--devcensus", action="store_true")
    ap.add_argument("--arms", action="store_true")
    ap.add_argument("--registered", type=Path, default=Path("results/gsr_v9_w6_registered.json"))
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        demo()
        return
    args.out.mkdir(parents=True, exist_ok=True)
    if args.traincal:
        payload = calibrate(args.dataset, args.data_dir, args.ckpts,
                            {"twix": TWIX_GRID, "phys": PHYS_GRID})
        (args.out / "gsr_v9_w6_traincal.json").write_text(json.dumps(payload, indent=1),
                                                          encoding="utf-8")
        logger.info("census %s", json.dumps(payload["census"], indent=1))
        return
    seqs = sorted(p.stem for p in (args.arm_dir / "predictions" / "data").glob("*.json"))
    if args.devcensus:
        payload = dev_census(args.arm_dir, args.data_dir, seqs, args.ckpts)
        (args.out / "gsr_v9_w6_devcensus.json").write_text(json.dumps(payload, indent=1),
                                                           encoding="utf-8")
        logger.info("candidates %s", json.dumps(payload["candidates"], indent=1))
        return
    if args.arms:
        arms = json.loads(args.registered.read_text(encoding="utf-8"))["arms"]
        payload = run_arms(args.arm_dir, args.data_dir, seqs, args.work,
                           {a["name"]: a for a in arms}, args.ckpts)
        (args.out / "gsr_v9_w6_arms.json").write_text(json.dumps(payload, indent=2),
                                                      encoding="utf-8")
        return
    ap.error("choose --traincal, --arms or --demo")


if __name__ == "__main__":
    main()
