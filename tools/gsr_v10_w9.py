"""v10-W9: a GT-free duplicate-concurrent-track absorber, fit on GSR-TRAIN and priced on DEV-20.

Registered in ``results/gsr_v10_w9_registered.json`` before any held-out or DEV number existed.

kb v10-w7-002 measured the largest unclaimed association lever on the arm-D substrate: 185 of the
265 same-identity predicted-track pairs OVERLAP IN TIME, and an oracle that absorbs them -- drops the
duplicate's overlapping rows AND relinks its disjoint remainder -- is worth +4.4999 GS-HOTA flags ON,
2.5x the sum of merging alone (+1.2976) and deduping alone (+0.5261). This module builds the GT-free
version of that stage (:func:`eval.gsr_score.dedup_absorb`), fits its thresholds on the v9 factory
GSR-TRAIN split and prices it end to end.

CLI::

    python -m tools.gsr_v10_w9 --fit     # CPU: fit on 45 train clips, read the 12 held-out ONCE
    python -m tools.gsr_v10_w9 --dev     # CPU: DEV-20 end to end, flags ON and OFF, paired vs arm D
    python -m tools.gsr_v10_w9 --audit   # CPU: re-derive the test-49 GT-free audit, flags OFF
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from itertools import product
from pathlib import Path

import pandas as pd

from eval.gsr_score import (
    DEDUP_MIN_TRACK_ROWS,
    accepts_pair,
    dedup_absorb,
    pair_features,
    track_mean_embeddings,
)
from tools.gsr_v10_w5 import DEV20
from tools.gsr_v10_w7 import ARM_D_PRED, MIN_AUDITED

logger = logging.getLogger("gsr_v10_w9")

#: The v9 factory dataset (GSR-TRAIN): tracklets, per-row GT ownership and a CLIP cache.
V9_ASSOC = Path("outputs/gsr/v9_assoc/v1")
#: Per-detection CLIP cache of the arm-D lineage (v6det detections, EIoU partition).
DEV_EMBED_DIR = Path("outputs/gsr/detembed_cache_clip_v6det_eiou")
#: Session output roots (never the freeze or W8 directories).
WORK = Path("outputs/gsr/v10_w9")
RESULTS = Path("results/gsr_benchmark/gsr_v10_w9.json")
FROZEN = Path("results/gsr_v10_w9_frozen.json")

#: Registered fit grid (``results/gsr_v10_w9_registered.json`` -> fit_procedure.grid).
GRID_D = (0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
GRID_K = (1, 3, 5, 10, 25)
GRID_C = (0.70, 0.75, 0.80)
#: Fit-split precision the operating point must clear (0.05 margin over the held-out bar).
FIT_MIN_PRECISION = 0.85
#: Held-out component gate.
GATE_PRECISION, GATE_RECALL = 0.80, 0.30


# === pair tables ==================================================================================
def factory_info(seq: str, root: Path = V9_ASSOC) -> tuple[dict[int, dict], dict[int, int]]:
    """``info`` (the :func:`eval.gsr_score.pair_features` shape) and GT ownership for one clip.

    Ownership is TRACK level, the W7 convention: dominant GT id by row vote over audited rows,
    trusted only at >= :data:`tools.gsr_v10_w7.MIN_AUDITED` audited rows, else -1. Row-level labels
    are never used (kb v10-w7-005: the labeller flickers between players under a metre apart).
    """
    tr = pd.read_parquet(root / "tracklets" / f"{seq}.parquet")
    info: dict[int, dict] = {}
    own: dict[int, int] = {}
    for tid, g in tr.groupby("track_id"):
        tid = int(tid)
        info[tid] = {
            "pos": dict(zip(g["frame"].astype(int), zip(g["pitch_x"], g["pitch_y"]))),
            "team": str(g["team"].mode().iloc[0]),
        }
        aud = [int(v) for v in g["gt_id"] if v >= 0]
        votes = Counter(aud)
        own[tid] = int(votes.most_common(1)[0][0]) if len(aud) >= MIN_AUDITED else -1
    return info, own


def factory_pairs(seqs: list[str], root: Path = V9_ASSOC) -> pd.DataFrame:
    """Labelled overlapping-pair table over factory sequences (one row per candidate pair)."""
    out = []
    for seq in seqs:
        info, own = factory_info(seq, root)
        embs = track_mean_embeddings(root / "detembed", seq)
        for f in pair_features(info, embs, DEDUP_MIN_TRACK_ROWS):
            ga, gb = own[f["a"]], own[f["b"]]
            out.append({**f, "seq": seq, "auditable": int(ga >= 0 and gb >= 0),
                        "label": int(ga >= 0 and gb >= 0 and ga == gb),
                        "n_rows_a": len(info[f["a"]]["pos"]), "n_rows_b": len(info[f["b"]]["pos"])})
        logger.info("%s: %d candidate pairs", seq, sum(1 for r in out if r["seq"] == seq))
    return pd.DataFrame(out)


def score_rule(table: pd.DataFrame, params: dict) -> dict:
    """Precision / recall of one parameter set against the track-level pair labels (pure)."""
    fired = table.apply(lambda r: accepts_pair(r.to_dict(), params), axis=1).to_numpy(dtype=bool)
    aud = table["auditable"].to_numpy(dtype=bool)
    lab = table["label"].to_numpy(dtype=bool)
    tp = int((fired & aud & lab).sum())
    fp = int((fired & aud & ~lab).sum())
    return {"fired": int(fired.sum()), "fired_auditable": int((fired & aud).sum()),
            "fired_unauditable": int((fired & ~aud).sum()), "tp": tp, "fp": fp,
            "positives": int((aud & lab).sum()),
            "precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
            "recall": float(tp / max(int((aud & lab).sum()), 1))}


def fit_arm(fit: pd.DataFrame, *, appearance: bool) -> dict:
    """Choose one arm's operating point on the FIT split under the registered selection rule."""
    grid = []
    for d, k, c, t in product(GRID_D, GRID_K, GRID_C if appearance else (None,), (False, True)):
        params = {"n_ov": k, "d_med": d, "cos": c, "same_team": t,
                  "min_track_rows": DEDUP_MIN_TRACK_ROWS}
        s = score_rule(fit, params)
        if s["fired_auditable"] >= 10:
            grid.append({"params": params, **s})
    ok = [g for g in grid if g["precision"] >= FIT_MIN_PRECISION]
    if ok:
        pick = max(ok, key=lambda g: (round(g["recall"], 4), round(g["precision"], 4),
                                      -g["params"]["d_med"]))
        expected = "pass"
    else:  # registered fallback: highest precision at recall >= 0.10, declared expected-to-fail
        cand = [g for g in grid if g["recall"] >= 0.10] or grid
        pick = max(cand, key=lambda g: (round(g["precision"], 4), round(g["recall"], 4)))
        expected = "fail"
    return {"params": pick["params"], "fit": {k: v for k, v in pick.items() if k != "params"},
            "expectation": expected,
            "fit_grid_points": len(grid),
            "fit_best_precision_at_recall_30": max(
                (g["precision"] for g in grid if g["recall"] >= GATE_RECALL), default=None)}


def run_fit(root: Path = V9_ASSOC) -> dict:
    """Fit both arms on the 45 train clips, then read the 12 held-out clips ONCE."""
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    fit_seqs, held_seqs = list(man["split"]["train_slice"]), list(man["split"]["val_slice"])
    fit = factory_pairs(fit_seqs, root)
    held = factory_pairs(held_seqs, root)
    out: dict = {"split": {"fit": fit_seqs, "held_out": held_seqs},
                 "pairs": {"fit": len(fit), "held_out": len(held),
                           "fit_auditable": int(fit["auditable"].sum()),
                           "held_out_auditable": int(held["auditable"].sum()),
                           "fit_positive": int(fit["label"].sum()),
                           "held_out_positive": int(held["label"].sum())},
                 "min_track_rows": DEDUP_MIN_TRACK_ROWS, "arms": {}}
    for name, app in (("A_geom", False), ("B_geom_app", True)):
        arm = fit_arm(fit, appearance=app)
        arm["held_out"] = score_rule(held, arm["params"])
        arm["gate_pass"] = bool(arm["held_out"]["precision"] >= GATE_PRECISION
                                and arm["held_out"]["recall"] >= GATE_RECALL)
        out["arms"][name] = arm
        logger.info("%s: params %s | fit P %.3f R %.3f | held-out P %.3f R %.3f -> %s", name,
                    arm["params"], arm["fit"]["precision"], arm["fit"]["recall"],
                    arm["held_out"]["precision"], arm["held_out"]["recall"],
                    "PASS" if arm["gate_pass"] else "FAIL")
    # Reported alongside, SELECTS NOTHING (the W7 splitter convention): the whole grid on the
    # held-out slice, so the reader can see the rule family's ceiling rather than one point of it.
    out["held_out_curve_information_only"] = [
        {"params": g["params"], "precision": g["precision"], "recall": g["recall"],
         "tp": g["tp"], "fp": g["fp"], "fired": g["fired"]}
        for g in (dict(score_rule(held, p), params=p) for p in _grid_params())
        if g["fired_auditable"] >= 5]
    frozen = {"written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "registration": "results/gsr_v10_w9_registered.json",
              "arms": {k: v["params"] for k, v in out["arms"].items()},
              "gate": {k: v["gate_pass"] for k, v in out["arms"].items()}}
    FROZEN.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    return out


def _grid_params() -> list[dict]:
    """Every registered grid point, both arms (pure)."""
    return [{"n_ov": k, "d_med": d, "cos": c, "same_team": t,
             "min_track_rows": DEDUP_MIN_TRACK_ROWS}
            for d, k, c, t in product(GRID_D, GRID_K, (None, *GRID_C), (False, True))]


# === DEV-20 end to end ============================================================================
def apply_arm(src: Path, dest: Path, seqs: list[str], params: dict) -> dict:
    """Write the absorbed submission and return per-sequence stage counters."""
    dest.mkdir(parents=True, exist_ok=True)
    stats = {}
    for seq in seqs:
        payload = json.loads((src / f"{seq}.json").read_text(encoding="utf-8"))
        st = dedup_absorb(payload["predictions"], params, seq)
        st["rows_in"] = st["rows_dropped"] + len(payload["predictions"])
        st["rows_out"] = len(payload["predictions"])
        (dest / f"{seq}.json").write_text(json.dumps(payload), encoding="utf-8")
        stats[seq] = st
        logger.info("%s: %d pairs -> %d components, %d tracks absorbed, %d rows relabelled, "
                    "%d rows dropped", seq, st["pairs"], st["components"], st["tracks_absorbed"],
                    st["rows_relabelled"], st["rows_dropped"])
    return stats


def audit_pairs(seqs: list[str], params: dict, data_dir: Path) -> dict:
    """GT audit of the accepted DEV pairs: how many are genuinely one person (measure-only).

    Uses the same track-level ownership as W7, on arm D's own submission rows, so the number is
    directly comparable to kb v10-w7-002's 265 same-identity pairs / 185 overlapping ones.
    """
    from tools.gsr_v9_factory import label_rows  # noqa: PLC0415
    from tools.gsr_v10_w7 import ownership, submission_rows  # noqa: PLC0415

    tp = fp = unaud = 0
    true_overlapping = 0
    caught = 0
    per_seq = {}
    for seq in seqs:
        rows = submission_rows(ARM_D_PRED / f"{seq}.json", data_dir / seq)
        own = ownership(label_rows(rows, data_dir / seq))
        owner = dict(zip(own["track_id"].astype(int), own["gt_id"].astype(int)))
        info = {int(t): {"pos": dict(zip(g["frame"].astype(int), zip(g["pitch_x"], g["pitch_y"]))),
                         "team": Counter(str(v) for v in g["team"]).most_common(1)[0][0]}
                for t, g in rows.groupby("track_id")}
        embs = track_mean_embeddings(params.get("embed_dir"), seq)
        feats = pair_features(info, embs, int(params.get("min_track_rows", DEDUP_MIN_TRACK_ROWS)))
        truth = {(f["a"], f["b"]) for f in feats
                 if owner.get(f["a"], -1) >= 0 and owner[f["a"]] == owner.get(f["b"], -2)}
        got = {(f["a"], f["b"]) for f in feats if accepts_pair(f, params)}
        a = sum(1 for p in got if owner.get(p[0], -1) >= 0 and owner.get(p[1], -1) >= 0)
        per_seq[seq] = {"candidates": len(feats), "true_overlapping": len(truth),
                        "accepted": len(got), "accepted_true": len(got & truth),
                        "accepted_auditable": a}
        tp += len(got & truth)
        fp += a - len(got & truth)
        unaud += len(got) - a
        true_overlapping += len(truth)
        caught += len(got & truth)
    return {"per_seq": per_seq, "accepted": tp + fp + unaud, "tp": tp, "fp": fp,
            "unauditable": unaud,
            "precision_auditable": float(tp / (tp + fp)) if tp + fp else float("nan"),
            "true_overlapping_pairs": true_overlapping,
            "recall": float(caught / max(true_overlapping, 1))}


def run_dev(name: str, params: dict, data_dir: Path, seqs: list[str]) -> dict:
    """Price one arm end to end on DEV-20 against arm D, flags ON and OFF."""
    from tools.gsr_v9_w7 import score_pair  # noqa: PLC0415

    dest = WORK / name / "predictions" / "data"
    stats = apply_arm(ARM_D_PRED, dest, seqs, params)
    scored = score_pair(ARM_D_PRED, dest, data_dir, WORK / "pair" / name, seqs)
    tot = {k: int(sum(v[k] for v in stats.values())) for k in next(iter(stats.values()))}
    delta = {s: scored["on"]["arm"]["per_seq"][s]["GS-HOTA"]
             - scored["on"]["control"]["per_seq"][s]["GS-HOTA"] for s in seqs}
    worst = min(delta, key=delta.get)
    return {"params": params, "per_seq_stage": stats, "totals": tot, "score": scored,
            "per_clip_gs_hota_delta": delta,
            "worst_clip": {"seq": worst, "delta": delta[worst]},
            "best_clip": {"seq": max(delta, key=delta.get), "delta": max(delta.values())},
            "gate": _dev_gate(scored, delta)}


def _dev_gate(scored: dict, delta: dict[str, float]) -> dict:
    """Registered DEV bars, all required (pure)."""
    on = scored["on"]
    bars = {
        "gs_hota_ge_1.0": on["delta"]["GS-HOTA"] >= 1.0,
        "deta_not_worse": on["delta"]["GS-DetA"] >= 0.0,
        "helped_gt_hurt_p_lt_0.05": (on["paired"]["GS-HOTA"]["helped"]
                                     > on["paired"]["GS-HOTA"]["hurt"]
                                     and on["paired"]["GS-HOTA"]["wilcoxon_p"] < 0.05),
        "no_clip_worse_than_-1.0": min(delta.values()) >= -1.0,
    }
    return {**bars, "pass": all(bars.values()),
            "gs_hota_delta": on["delta"]["GS-HOTA"], "deta_delta": on["delta"]["GS-DetA"]}


# === self-check ===================================================================================
def _demo() -> None:
    """Assert the absorb mechanics and the fit/apply seam on a synthetic two-track duplicate."""
    def row(tid: int, f: int, x: float, conf: float = 0.9) -> dict:
        return {"image_id": f"{f:06d}", "track_id": tid, "confidence": conf,
                "attributes": {"role": "player", "team": "left", "jersey": None},
                "bbox_pitch": {"x_bottom_middle": x, "y_bottom_middle": 0.0}}

    # track 1 runs 0..39, track 2 shadows it over 20..39 at 0.2 m and continues alone to 59.
    preds = ([row(1, f, 0.0) for f in range(40)]
             + [row(2, f, 0.2, 0.8) for f in range(20, 60)])
    params = {"n_ov": 5, "d_med": 1.0, "cos": None, "same_team": False, "min_track_rows": 25}
    st = dedup_absorb(preds, params)
    assert st["pairs"] == 1 and st["components"] == 1, st
    assert st["rows_dropped"] == 20, st            # the loser's overlapping rows go
    assert st["rows_relabelled"] == 20, st         # its disjoint remainder is relinked
    # equal row counts -> the higher mean confidence wins, and track 1 keeps the overlap
    assert len(preds) == 60 and {p["track_id"] for p in preds} == {1}, preds[:2]
    assert len({p["image_id"] for p in preds}) == 60, "one row per timestep after absorb"
    # row count outranks confidence: the longer track wins even at a lower confidence
    preds2 = ([row(1, f, 0.0, 0.5) for f in range(60)]
              + [row(2, f, 0.2, 0.99) for f in range(20, 50)])
    st2 = dedup_absorb(preds2, params)
    assert {p["track_id"] for p in preds2} == {1}, "the longer track wins the overlap"
    assert len(preds2) == 60 and st2["rows_dropped"] == 30, st2
    # a 3 m separation is not a duplicate, and a short track is not a candidate
    far = [row(1, f, 0.0) for f in range(40)] + [row(2, f, 3.0) for f in range(40)]
    assert dedup_absorb(far, params)["pairs"] == 0
    short = [row(1, f, 0.0) for f in range(40)] + [row(2, f, 0.1) for f in range(10)]
    assert dedup_absorb(short, params)["pairs"] == 0
    assert dedup_absorb(far, None) == {}, "OFF is a no-op"
    # the appearance term binds only when both sides carry an embedding
    f_no = {"a": 1, "b": 2, "n_ov": 30, "d_med": 0.3, "cos": float("nan"), "same_team": 1}
    strict = {"n_ov": 5, "d_med": 1.0, "cos": 0.8}
    assert accepts_pair(f_no, strict), "missing embedding must fall back to geometry"
    assert not accepts_pair({**f_no, "cos": 0.5}, strict)
    assert accepts_pair({**f_no, "cos": 0.9}, strict)
    print("gsr_v10_w9 demo OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--seqs", default=None)
    ap.add_argument("--fit", action="store_true", help="fit on GSR-TRAIN, read the held-out once")
    ap.add_argument("--dev", action="store_true", help="DEV-20 end to end for every gated arm")
    ap.add_argument("--arms", default=None, help="comma-separated arm names (default: gated ones)")
    ap.add_argument("--audit", action="store_true", help="re-derive the test-49 GT-free audit")
    ap.add_argument("--stage", action="store_true",
                    help="measure-only: write the absorbed DEV submissions and report the stage "
                         "counters (rows touched). No GS-HOTA is read.")
    ap.add_argument("--dev-pairs", action="store_true",
                    help="measure-only: pair-level precision/recall of the FROZEN rules on the "
                         "DEV-20 arm-D population. No GS-HOTA is read and no threshold moves.")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return
    seqs = args.seqs.split(",") if args.seqs else DEV20
    payload: dict = {}
    t0 = time.time()
    if args.fit:
        payload["fit"] = run_fit()
        for k, v in payload["fit"]["arms"].items():
            print(f"{k}: {v['params']} fit P {v['fit']['precision']:.4f} R {v['fit']['recall']:.4f}"
                  f" | held-out P {v['held_out']['precision']:.4f} "
                  f"R {v['held_out']['recall']:.4f} -> {'PASS' if v['gate_pass'] else 'FAIL'}")
    if args.dev:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        names = (args.arms.split(",") if args.arms
                 else [k for k, ok in frozen["gate"].items() if ok])
        payload["dev"] = {}
        for name in names:
            params = dict(frozen["arms"][name], embed_dir=str(DEV_EMBED_DIR))
            payload["dev"][name] = run_dev(name, params, args.data_dir, seqs)
            payload["dev"][name]["gt_audit"] = audit_pairs(seqs, params, args.data_dir)
            d = payload["dev"][name]
            for f in ("on", "off"):
                s = d["score"][f]
                print(f"{name} flags {f}: arm D {s['control']['combined']['GS-HOTA']:.4f} -> "
                      f"{s['arm']['combined']['GS-HOTA']:.4f} "
                      f"({s['delta']['GS-HOTA']:+.4f}) DetA {s['delta']['GS-DetA']:+.4f} "
                      f"AssA {s['delta']['GS-AssA']:+.4f}")
            print(f"{name} gate:", d["gate"], "worst clip", d["worst_clip"])
    if args.stage:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        payload["stage"] = {}
        for name, params in frozen["arms"].items():
            stats = apply_arm(ARM_D_PRED, WORK / name / "predictions" / "data", seqs,
                              dict(params, embed_dir=str(DEV_EMBED_DIR)))
            tot = {k: int(sum(v[k] for v in stats.values())) for k in next(iter(stats.values()))}
            payload["stage"][name] = {"params": params, "per_seq": stats, "totals": tot}
            print(name, tot)
    if args.dev_pairs:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        payload["dev_pairs"] = {}
        for name, params in frozen["arms"].items():
            got = audit_pairs(seqs, dict(params, embed_dir=str(DEV_EMBED_DIR)), args.data_dir)
            payload["dev_pairs"][name] = got
            print(f"{name}: accepted {got['accepted']} (tp {got['tp']} fp {got['fp']} "
                  f"unauditable {got['unauditable']}) precision {got['precision_auditable']:.4f} "
                  f"of {got['true_overlapping_pairs']} true overlapping pairs, "
                  f"recall {got['recall']:.4f}")
    if args.audit:
        payload["audit"] = run_audit()
        print(json.dumps(payload["audit"], indent=1))
    if payload:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        prev = (json.loads(args.results.read_text(encoding="utf-8"))
                if args.results.exists() else {})
        prev.update(payload)
        args.results.write_text(json.dumps(prev, indent=1, default=str), encoding="utf-8")
        print(f"wrote {args.results} ({time.time() - t0:.0f} s)")


#: The on-record test-49 package and its inputs (``results/gsr_benchmark/gsr_v6_legitimacy_audit``).
SRVTEST = Path("outputs/gsr_srvtest")
AUDIT_ARM = SRVTEST / "deleak_v6test_v6det_clip_e0.3r1w0.5a0.3"
AUDIT_BASE = SRVTEST / "eval_v4_gta_f080_v6_v6det_eiou_t0.450_jg_clip_v6det_eiou"
AUDIT_POS = SRVTEST / "positions_gate_v6det"


def run_audit(data_dir: Path = Path("data/soccernet/gamestate-2024")) -> dict:
    """Re-derive the on-record test-49 GT-free audit, flags OFF, through the extended checker.

    Must reproduce 0 violations / 467,425 rows / flip set {SNGS-126, -130, -131, -197} exactly
    (``results/gsr_benchmark/gsr_v6_legitimacy_audit.json``, kb v9-w4-004 / v9-w5-005). No metric is
    read on any test sequence -- this compares our own rows against our own expectation.
    """
    from tools.gsr_calibfill import verify_gtfree  # noqa: PLC0415
    from tools.gsr_deleak import split_names  # noqa: PLC0415

    names = split_names(data_dir, "test")
    got = verify_gtfree(AUDIT_ARM, AUDIT_BASE, data_dir, AUDIT_POS, names)
    on_record = {"n_sequences": 49, "n_rows": 467425, "violations": 0, "n_flipped": 4,
                 "flipped_sequences": ["SNGS-126", "SNGS-130", "SNGS-131", "SNGS-197"]}
    return {"got": got, "on_record": on_record,
            "identical": all(got[k] == v for k, v in on_record.items())}


if __name__ == "__main__":
    main()
