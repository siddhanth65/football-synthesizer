"""v9 W5: the referee axis (measured and refused) and the fragmented-keeper repair.

W4 shipped ``vote_track_attributes`` + ``gk_side_repair(("team",))`` -- DEV-20 GS-DetA 43.3757 /
GS-HOTA 56.2601. Its anatomy left two live attribute buckets: referee<->player (oracle +1.63 GS-DetA)
and keepers we call players (+0.7597, blocked because the tracker fragments a keeper and W4's guard
required a half with *no* goalkeeper track at all).

This module carries the stage-1 instruments for both, and the stage-2 arms for the one that has a
mechanism. Everything runs on top of the W4 control and reads only our own submission JSON.

CLI::

    python -m tools.gsr_v9_w5 --refanatomy   # stage 1: geometry / role-flicker / kit distinctness
    python -m tools.gsr_v9_w5 --arms         # stage 2: the registered arms, scored + audited
    python -m tools.gsr_v9_w5 --demo         # runnable self-check
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

from eval.gsr_score import (
    DEFAULT_DATA_DIR,
    EVAL_CONFIGS,
    GK_MIN_ABSX_M,
    GK_MIN_PEN_FRAC,
    gk_side_repair,
    gs_hota,
    vote_track_attributes,
)
from tools.gsr_v9_deta import DEFAULT_ARM, load_gt, match_positions, similarity
from tools.gsr_v9_w4 import VOTE_KEYS, classify_track, gt_track_attrs

logger = logging.getLogger("gsr_v9_w5")

#: Where W5 writes its artifacts.
RESULTS_DIR = Path("results/gsr_benchmark")
#: Scratch tree for the arm submissions (one process per work dir -- W4's corruption lesson).
WORK_DIR = Path("outputs/gsr/v9_w5")
#: W5's control: the shipped DEV-20 arm, voted, with the W4 keeper-side team repair.
CONTROL_STEPS: tuple[str, ...] = ("team",)
#: The registered stage-2 arms (``results/gsr_v9_w5_registered.json``), composed AFTER the control.
ARMS: dict[str, tuple[str, ...]] = {
    "gk_second": ("role2", "team"),
    "gk_second_dom": ("role2dom", "team"),
}


# === control =====================================================================================
def control_predictions(pred_path: Path) -> list[dict]:
    """Load one submission in W5's control state (W3 vote + W4 keeper-side team repair)."""
    payload = json.loads(pred_path.read_text(encoding="utf-8"))
    vote_track_attributes(payload["predictions"], VOTE_KEYS)
    gk_side_repair(payload["predictions"], CONTROL_STEPS)
    return payload["predictions"]


def row_key(p: dict) -> tuple:
    """Identity of a prediction row inside a submission (``image_id`` + track)."""
    return (p["image_id"], int(p["track_id"]))


# === stage 1: the referee axis ===================================================================
def track_features(preds: list[dict], unvoted: list[dict]) -> dict[int, dict]:
    """Per-track pitch geometry (control state) and per-row role tally (unvoted state)."""
    by: dict[int, list[dict]] = defaultdict(list)
    for p in preds:
        if (p.get("attributes") or {}).get("role") == "ball":
            continue
        by[int(p["track_id"])].append(p)
    tally: dict[int, Counter] = defaultdict(Counter)
    for p in unvoted:
        a = p.get("attributes") or {}
        if a.get("role") == "ball":
            continue
        tally[int(p["track_id"])][str(a.get("role"))] += 1
    out = {}
    for tid, rows in by.items():
        x = np.array([r["bbox_pitch"]["x_bottom_middle"] for r in rows], float)
        y = np.array([r["bbox_pitch"]["y_bottom_middle"] for r in rows], float)
        c = tally.get(tid, Counter())
        n = max(sum(c.values()), 1)
        out[tid] = {
            "n_rows": len(rows),
            "our_role": Counter(str((r["attributes"] or {}).get("role"))
                                for r in rows).most_common(1)[0][0],
            "our_team": (rows[0]["attributes"] or {}).get("team"),
            "med_absy": float(np.median(np.abs(y))), "med_absx": float(np.median(np.abs(x))),
            "s_ref": c["referee"] / n, "s_gk": c["goalkeeper"] / n,
        }
    return out


def owners(seq_dir: Path, preds: list[dict]) -> dict[int, dict]:
    """Attribute-blind Hungarian ownership of each prediction track by a GT identity (W4's matcher)."""
    ts, gt_rows = load_gt(seq_dir)
    gt_attrs = gt_track_attrs(gt_rows)
    by_ts: dict[int, list[dict]] = defaultdict(list)
    for p in preds:
        if (p.get("attributes") or {}).get("role") == "ball" or p["image_id"] not in ts:
            continue
        by_ts[ts[p["image_id"]]].append(p)
    own: dict[int, Counter] = defaultdict(Counter)
    for t, gts in enumerate(gt_rows):
        prs = by_ts.get(t, [])
        gxy = np.array([[g["bbox_pitch"]["x_bottom_middle"], g["bbox_pitch"]["y_bottom_middle"]]
                        for g in gts], float).reshape(-1, 2)
        pxy = np.array([[p["bbox_pitch"]["x_bottom_middle"], p["bbox_pitch"]["y_bottom_middle"]]
                        for p in prs], float).reshape(-1, 2)
        for i, j in match_positions(similarity(gxy, pxy)):
            own[int(prs[j]["track_id"])][int(gts[i]["track_id"])] += 1
    out = {}
    for tid, c in own.items():
        gid, n_own = c.most_common(1)[0]
        ga = gt_attrs.get(gid, {})
        out[tid] = {"gt_id": gid, "n_matched": int(sum(c.values())),
                    "purity": n_own / max(sum(c.values()), 1),
                    "gt_role": ga.get("role"), "gt_team": ga.get("team")}
    return out


def run_refanatomy(arm_dir: Path, data_dir: Path, seqs: list[str]) -> dict:
    """Stage 1: per-track geometry + role flicker + GT ownership, and the bucket roll-up."""
    src = arm_dir / "predictions" / "data"
    recs: list[dict] = []
    for i, s in enumerate(seqs):
        preds = control_predictions(src / f"{s}.json")
        unvoted = json.loads((src / f"{s}.json").read_text(encoding="utf-8"))["predictions"]
        feats = track_features(preds, unvoted)
        own = owners(data_dir / s, preds)
        for tid, f in feats.items():
            o = own.get(tid, {"gt_id": None, "n_matched": 0, "purity": 0.0,
                              "gt_role": None, "gt_team": None})
            rec = {"seq": s, "track_id": tid, **f, **o}
            rec["bucket"] = classify_track(rec)
            recs.append(rec)
        logger.info("[%d/%d] %s %d tracks", i + 1, len(seqs), s, len(feats))
    return {"arm": str(arm_dir), "control_steps": list(CONTROL_STEPS), "tracks": recs,
            "summary": refanatomy_summary(recs)}


def refanatomy_summary(recs: list[dict]) -> dict:
    """The stage-1 tables the referee decision rests on."""
    def block(sel: list[dict]) -> dict:
        if not sel:
            return {"tracks": 0, "rows": 0}
        return {"tracks": len(sel), "rows": int(sum(r["n_rows"] for r in sel)),
                "med_absy": round(float(np.median([r["med_absy"] for r in sel])), 2),
                "med_s_ref": round(float(np.median([r["s_ref"] for r in sel])), 3)}

    pool = [r for r in recs if r["purity"] >= 0.7 and r["n_matched"] > 0]
    out = {"buckets": {b: block([r for r in recs if r["bucket"] == b])
                       for b in sorted({r["bucket"] for r in recs})},
           "our_referee_tracks": {
               "central_med_absy_lt_25": block([r for r in recs if r["our_role"] == "referee"
                                                and r["med_absy"] < 25]),
               "touchline_med_absy_ge_25": block([r for r in recs if r["our_role"] == "referee"
                                                  and r["med_absy"] >= 25])},
           "role_flicker": {}}
    for role in ("player", "goalkeeper", "referee"):
        v = np.array([r["s_ref"] for r in pool if r["gt_role"] == role])
        if len(v):
            out["role_flicker"][role] = {
                "n": int(len(v)), "median": round(float(np.median(v)), 3),
                "frac_gt_0": round(float(np.mean(v > 0)), 3),
                "frac_ge_0.1": round(float(np.mean(v >= 0.1)), 3)}
    sweep = []
    cand = [r for r in pool if r["our_role"] == "player"]
    for t in (0.05, 0.10, 0.20, 0.30, 0.40):
        fire = [r for r in cand if r["s_ref"] >= t]
        tp = [r for r in fire if r["gt_role"] == "referee"]
        fp = [r for r in fire if r["gt_role"] != "referee"]
        sweep.append({"s_ref_threshold": t, "fire_tracks": len(fire),
                      "tp_tracks": len(tp), "tp_rows": int(sum(r["n_rows"] for r in tp)),
                      "fp_tracks": len(fp), "fp_rows": int(sum(r["n_rows"] for r in fp))})
    out["role_flicker_sweep"] = sweep
    return out


# === stage 2: the arms ===========================================================================
def audit_rows(ctrl: list[dict], arm: list[dict], gt_by_key: dict[tuple, dict]) -> dict:
    """Row-level sub-population audit of the rows an arm changed.

    A row is *right* when its role matches its owning GT row's role, and (for player/goalkeeper)
    its team matches too -- the evaluator's own attribute preprocessing.
    """
    def ok(p: dict, g: dict) -> bool:
        a = p.get("attributes") or {}
        if a.get("role") != g["role"]:
            return False
        return a.get("role") not in {"player", "goalkeeper"} or a.get("team") == g["team"]

    before = {row_key(p): p for p in ctrl}
    out = Counter()
    for p in arm:
        k = row_key(p)
        b = before.get(k)
        g = gt_by_key.get(k)
        if b is None or (b.get("attributes") or {}) == (p.get("attributes") or {}):
            continue
        if g is None:
            out["changed_unmatched"] += 1
            continue
        out[f"{'right' if ok(b, g) else 'wrong'}2{'right' if ok(p, g) else 'wrong'}"] += 1
    return dict(out)


def gt_row_map(seq_dir: Path, preds: list[dict]) -> dict[tuple, dict]:
    """Map each prediction row to the attributes of the GT row it is position-matched to."""
    ts, gt_rows = load_gt(seq_dir)
    by_ts: dict[int, list[dict]] = defaultdict(list)
    for p in preds:
        if (p.get("attributes") or {}).get("role") == "ball" or p["image_id"] not in ts:
            continue
        by_ts[ts[p["image_id"]]].append(p)
    out: dict[tuple, dict] = {}
    for t, gts in enumerate(gt_rows):
        prs = by_ts.get(t, [])
        gxy = np.array([[g["bbox_pitch"]["x_bottom_middle"], g["bbox_pitch"]["y_bottom_middle"]]
                        for g in gts], float).reshape(-1, 2)
        pxy = np.array([[p["bbox_pitch"]["x_bottom_middle"], p["bbox_pitch"]["y_bottom_middle"]]
                        for p in prs], float).reshape(-1, 2)
        for i, j in match_positions(similarity(gxy, pxy)):
            a = gts[i].get("attributes") or {}
            out[row_key(prs[j])] = {"role": a.get("role"), "team": a.get("team")}
    return out


def run_arms(arm_dir: Path, data_dir: Path, seqs: list[str], work: Path, names: list[str]) -> dict:
    """Score each registered arm against the W4 control, with paired stats and the row audit."""
    from scipy.stats import wilcoxon  # noqa: PLC0415

    src = arm_dir / "predictions" / "data"
    ctrl_dir = work / "control"
    (ctrl_dir / "predictions" / "data").mkdir(parents=True, exist_ok=True)
    ctrl, gtmap = {}, {}
    for s in seqs:
        ctrl[s] = control_predictions(src / f"{s}.json")
        gtmap[s] = gt_row_map(data_dir / s, ctrl[s])
        (ctrl_dir / "predictions" / "data" / f"{s}.json").write_text(
            json.dumps({"predictions": ctrl[s]}), encoding="utf-8")
    base = gs_hota(ctrl_dir, data_dir, seq_info={s: 0 for s in seqs}, **EVAL_CONFIGS["gs_hota_full"])
    out = {"control": {"combined": base["combined"], "per_seq": base["per_seq"]}}
    logger.info("control GS-DetA %.4f GS-HOTA %.4f", base["combined"]["GS-DetA"],
                base["combined"]["GS-HOTA"])
    for name in names:
        t0 = time.time()
        dest = work / "arm" / "predictions" / "data"
        dest.mkdir(parents=True, exist_ok=True)
        changed: Counter = Counter()
        audit: Counter = Counter()
        promoted: list[dict] = []
        for s in seqs:
            preds = control_predictions(src / f"{s}.json")
            changed.update(gk_side_repair(preds, ARMS[name]))
            audit.update(audit_rows(ctrl[s], preds, gtmap[s]))
            promoted += promoted_tracks(ctrl[s], preds, gtmap[s], s)
            (dest / f"{s}.json").write_text(json.dumps({"predictions": preds}), encoding="utf-8")
        res = gs_hota(work / "arm", data_dir, seq_info={s: 0 for s in seqs},
                      **EVAL_CONFIGS["gs_hota_full"])
        dh = np.array([res["per_seq"][s]["GS-HOTA"] - base["per_seq"][s]["GS-HOTA"] for s in seqs])
        dd = np.array([res["per_seq"][s]["GS-DetA"] - base["per_seq"][s]["GS-DetA"] for s in seqs])
        p = float(wilcoxon(dh).pvalue) if np.any(dh != 0) else 1.0
        out[name] = {
            "combined": res["combined"], "per_seq": res["per_seq"], "changed": dict(changed),
            "audit_rows": dict(audit), "promoted_tracks": promoted,
            "paired_hota": {"mean": float(dh.mean()), "helped": int((dh > 0).sum()),
                            "hurt": int((dh < 0).sum()), "wilcoxon_p": p},
            "paired_deta": {"mean": float(dd.mean()), "helped": int((dd > 0).sum()),
                            "hurt": int((dd < 0).sum())}}
        logger.info("%-14s GS-DetA %.4f (%+.4f) GS-HOTA %.4f (%+.4f) %s audit %s [%.0fs]", name,
                    res["combined"]["GS-DetA"],
                    res["combined"]["GS-DetA"] - base["combined"]["GS-DetA"],
                    res["combined"]["GS-HOTA"],
                    res["combined"]["GS-HOTA"] - base["combined"]["GS-HOTA"], dict(changed),
                    dict(audit), time.time() - t0)
        shutil.rmtree(work / "arm", ignore_errors=True)
    return out


def promoted_tracks(ctrl: list[dict], arm: list[dict], gtmap: dict[tuple, dict],
                    seq: str) -> list[dict]:
    """One record per track the arm relabelled, with the GT role of the identity that owns it."""
    before = {int(p["track_id"]): (p.get("attributes") or {}).get("role") for p in ctrl}
    after: dict[int, str] = {}
    rows: dict[int, list[dict]] = defaultdict(list)
    for p in arm:
        tid = int(p["track_id"])
        after[tid] = (p.get("attributes") or {}).get("role")
        rows[tid].append(p)
    out = []
    for tid, role in after.items():
        if before.get(tid) == role:
            continue
        gt = Counter(g["role"] for k, g in ((row_key(p), gtmap.get(row_key(p)))
                                            for p in rows[tid]) if g is not None and k)
        out.append({"seq": seq, "track_id": tid, "n_rows": len(rows[tid]),
                    "from": before.get(tid), "to": role,
                    "gt_role": gt.most_common(1)[0][0] if gt else None,
                    "gt_matched_rows": int(sum(gt.values()))})
    return out


# === self-check ==================================================================================
def demo() -> None:
    """Assert the second-keeper rule and the row audit on hand-built inputs."""
    from eval.gsr_score import _second_keepers  # noqa: PLC0415

    deep, shallow = -(GK_MIN_ABSX_M + 10), -(GK_MIN_ABSX_M + 2)
    info = {
        1: {"role": "goalkeeper", "med_x": shallow, "pen": 1.0, "frames": {"a", "b"}},
        2: {"role": "player", "med_x": deep, "pen": 1.0, "frames": {"c", "d"}},      # no overlap
        3: {"role": "player", "med_x": deep, "pen": 1.0, "frames": {"a"}},           # overlaps, deeper
        4: {"role": "player", "med_x": shallow, "pen": GK_MIN_PEN_FRAC - 0.01,
            "frames": {"e"}},                                                        # fails pen gate
        5: {"role": "player", "med_x": -(GK_MIN_ABSX_M - 1), "pen": 1.0, "frames": {"f"}},
        6: {"role": "player", "med_x": 40.0, "pen": 1.0, "frames": {"g"}},           # other half
    }
    assert _second_keepers(info, dominance=False) == [2, 6], _second_keepers(info, dominance=False)
    assert _second_keepers(info, dominance=True) == [2, 3, 6], _second_keepers(info, dominance=True)

    def row(img, tid, role, team):
        return {"image_id": img, "track_id": tid,
                "attributes": {"role": role, "team": team, "jersey": None},
                "bbox_pitch": {"x_bottom_middle": deep, "y_bottom_middle": 0.0}}

    ctrl = [row("a", 7, "player", "left"), row("b", 8, "player", "left")]
    arm = [row("a", 7, "goalkeeper", "right"), row("b", 8, "goalkeeper", "left")]
    gtm = {("a", 7): {"role": "goalkeeper", "team": "right"},
           ("b", 8): {"role": "player", "team": "left"}}
    assert audit_rows(ctrl, arm, gtm) == {"wrong2right": 1, "right2wrong": 1}, \
        audit_rows(ctrl, arm, gtm)
    assert not audit_rows(ctrl, ctrl, gtm)
    print("gsr_v9_w5 demo: OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-dir", type=Path, default=DEFAULT_ARM)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    ap.add_argument("--work", type=Path, default=WORK_DIR)
    ap.add_argument("--refanatomy", action="store_true")
    ap.add_argument("--arms", nargs="*", default=None, help=f"one or more of {list(ARMS)}")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        demo()
        return
    seqs = sorted(p.stem for p in (args.arm_dir / "predictions" / "data").glob("*.json"))
    if not seqs:
        raise SystemExit(f"no submissions under {args.arm_dir}")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.refanatomy:
        payload = run_refanatomy(args.arm_dir, args.data_dir, seqs)
        (args.out / "gsr_v9_w5_refanatomy.json").write_text(json.dumps(payload), encoding="utf-8")
        logger.info("%s", json.dumps(payload["summary"], indent=1))
        return
    if args.arms is not None:
        names = args.arms or list(ARMS)
        bad = [a for a in names if a not in ARMS]
        if bad:
            raise SystemExit(f"unknown arms {bad}; known: {list(ARMS)}")
        payload = run_arms(args.arm_dir, args.data_dir, seqs, args.work, names)
        (args.out / "gsr_v9_w5_arms.json").write_text(json.dumps(payload, indent=2),
                                                      encoding="utf-8")
        return
    ap.error("choose --refanatomy, --arms or --demo")


if __name__ == "__main__":
    main()
