"""v9 W4: error anatomy of the tracks whose *majority* role/team is wrong, and the repairs.

W3 shipped ``eval.gsr_score.vote_track_attributes`` (+1.7999 GS-DetA on DEV-20). Voting can only
turn a partly-right track into an all-right one; it cannot touch a track whose majority attribute is
wrong. Oracle role+team is worth +6.15 DetA and voting captured +0.75 of it, so the rest lives in
majority-wrong tracks. This module measures where they are and what each sub-bucket is worth.

Instruments (all on top of the **voted** submission, which is W4's control):

* ``--anatomy`` -- attribute-blind Hungarian match (the W3 census matcher, imported) of every
  prediction row against GT, aggregated **per prediction track**: which GT identity owns the track,
  how pure that ownership is, what role/team the track carries vs what its GT identity carries, and
  where on the pitch the track lives. Writes one record per track.
* ``--oracles`` -- track-level relabel counterfactuals: for the tracks of one sub-bucket, force the
  attribute of **every** row onto the dominant GT identity's value, then score with the official
  evaluator. That is exactly what a perfect track-level classifier would do, so the delta is the
  sub-bucket's reachable DetA value.
* ``--arms`` -- the registered stage-2 arms (GT-free), scored against the voted control.

CLI::

    python -m tools.gsr_v9_w4 --anatomy
    python -m tools.gsr_v9_w4 --oracles
    python -m tools.gsr_v9_w4 --arms
    python -m tools.gsr_v9_w4 --demo
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
    gs_hota,
    penalty_frac,
    vote_track_attributes,
)
from tools.gsr_v9_deta import DEFAULT_ARM, load_gt, match_positions, similarity

logger = logging.getLogger("gsr_v9_w4")

#: Where W4 writes its artifacts.
RESULTS_DIR = Path("results/gsr_benchmark")
#: Scratch tree for counterfactual / arm submissions.
WORK_DIR = Path("outputs/gsr/v9_w4")
#: W4's control: the shipped DEV-20 arm with the W3 vote applied.
VOTE_KEYS = ("role", "team", "jersey")
#: A track whose matched rows are owned by its dominant GT identity below this share is called
#: CONTAMINATED: no relabelling can fix it, only a split. Fixed a priori.
PURITY_FLOOR = 0.70


# === pure helpers =================================================================================
def voted_predictions(pred_path: Path) -> list[dict]:
    """Load one submission and apply the W3 vote (W4's control state)."""
    payload = json.loads(pred_path.read_text(encoding="utf-8"))
    vote_track_attributes(payload["predictions"], VOTE_KEYS)
    return payload["predictions"]


def gt_track_attrs(gt_rows: list[list[dict]]) -> dict[int, dict]:
    """Majority ``role``/``team`` of every GT track id (a GT identity carries one of each)."""
    roles: dict[int, Counter] = defaultdict(Counter)
    teams: dict[int, Counter] = defaultdict(Counter)
    for frame in gt_rows:
        for ann in frame:
            a = ann.get("attributes") or {}
            tid = int(ann["track_id"])
            roles[tid][a.get("role")] += 1
            teams[tid][str(a.get("team"))] += 1
    out: dict[int, dict] = {}
    for tid, rc in roles.items():
        team = teams[tid].most_common(1)[0][0]
        out[tid] = {"role": rc.most_common(1)[0][0],
                    "team": None if team == "None" else team,
                    "n": int(sum(rc.values()))}
    return out


def classify_track(rec: dict) -> str:
    """Name a track's role/team error bucket (pure; ``rec`` is one ``--anatomy`` record).

    Buckets, checked in order: unmatched (no GT row ever within 5 m) -> contaminated (dominant GT
    identity owns < :data:`PURITY_FLOOR` of the matched rows) -> role error named by the GT->ours
    confusion -> team error -> ok.
    """
    if not rec["n_matched"]:
        return "unmatched"
    if rec["purity"] < PURITY_FLOOR:
        return "contaminated"
    if rec["gt_role"] != rec["our_role"]:
        return f"role:{rec['gt_role']}->{rec['our_role']}"
    if rec["our_role"] in {"player", "goalkeeper"} and rec["gt_team"] != rec["our_team"]:
        return f"team:{rec['gt_role']}"
    return "ok"


# === anatomy ======================================================================================
def anatomy_sequence(seq_dir: Path, preds: list[dict]) -> list[dict]:
    """One record per prediction track: GT ownership, attributes, pitch geometry."""
    ts, gt_rows = load_gt(seq_dir)
    gt_attrs = gt_track_attrs(gt_rows)
    by_ts: dict[int, list[dict]] = defaultdict(list)
    for p in preds:
        if (p.get("attributes") or {}).get("role") == "ball" or p["image_id"] not in ts:
            continue
        by_ts[ts[p["image_id"]]].append(p)
    owner: dict[int, Counter] = defaultdict(Counter)
    rows: dict[int, list[dict]] = defaultdict(list)
    for t, gts in enumerate(gt_rows):
        prs = by_ts.get(t, [])
        for p in prs:
            rows[int(p["track_id"])].append(p)
        gxy = np.array([[g["bbox_pitch"]["x_bottom_middle"], g["bbox_pitch"]["y_bottom_middle"]]
                        for g in gts], float).reshape(-1, 2)
        pxy = np.array([[p["bbox_pitch"]["x_bottom_middle"], p["bbox_pitch"]["y_bottom_middle"]]
                        for p in prs], float).reshape(-1, 2)
        sim = similarity(gxy, pxy)
        for i, j in match_positions(sim):
            owner[int(prs[j]["track_id"])][int(gts[i]["track_id"])] += 1
    out: list[dict] = []
    for tid, prs in rows.items():
        x = np.array([p["bbox_pitch"]["x_bottom_middle"] for p in prs], float)
        y = np.array([p["bbox_pitch"]["y_bottom_middle"] for p in prs], float)
        own = owner.get(tid, Counter())
        n_matched = int(sum(own.values()))
        gid, n_own = own.most_common(1)[0] if own else (None, 0)
        ga = gt_attrs.get(gid, {}) if gid is not None else {}
        a = prs[0]["attributes"]
        out.append({
            "track_id": tid, "n_rows": len(prs), "n_matched": n_matched,
            "gt_id": gid, "purity": (n_own / n_matched) if n_matched else 0.0,
            "gt_role": ga.get("role"), "gt_team": ga.get("team"),
            "our_role": a.get("role"), "our_team": a.get("team"),
            "mean_x": float(x.mean()), "mean_y": float(y.mean()),
            "med_x": float(np.median(x)), "med_absx": float(np.median(np.abs(x))),
            "pen_frac": float(penalty_frac(x, y)),
        })
    return out


def run_anatomy(arm_dir: Path, data_dir: Path, seqs: list[str]) -> dict:
    """Build the per-track anatomy of the voted submission for every sequence."""
    src = arm_dir / "predictions" / "data"
    per_seq: dict[str, list[dict]] = {}
    for i, s in enumerate(seqs):
        per_seq[s] = anatomy_sequence(data_dir / s, voted_predictions(src / f"{s}.json"))
        logger.info("[%d/%d] %s %d tracks", i + 1, len(seqs), s, len(per_seq[s]))
    return {"arm": str(arm_dir), "vote_keys": list(VOTE_KEYS), "purity_floor": PURITY_FLOOR,
            "per_seq": per_seq}


def summarise_anatomy(payload: dict) -> dict:
    """Aggregate the per-track records into the stage-1 bucket table."""
    buckets: Counter = Counter()
    rows: Counter = Counter()
    matched: Counter = Counter()
    for recs in payload["per_seq"].values():
        for rec in recs:
            b = classify_track(rec)
            buckets[b] += 1
            rows[b] += rec["n_rows"]
            matched[b] += rec["n_matched"]
    return {"tracks": dict(buckets), "rows": dict(rows), "matched_rows": dict(matched)}


# === sub-bucket oracles ===========================================================================
def _relabel(preds: list[dict], recs: list[dict], want: set[str], attrs: tuple[str, ...]) -> int:
    """Force ``attrs`` of every row of the selected tracks onto the dominant GT identity's value."""
    fix = {r["track_id"]: r for r in recs if classify_track(r) in want}
    n = 0
    for p in preds:
        rec = fix.get(int(p["track_id"]))
        if rec is None:
            continue
        a = p["attributes"]
        for key in attrs:
            val = rec[f"gt_{key}"]
            if key == "team" and a.get("role") not in {"player", "goalkeeper"}:
                val = None
            if a.get(key) != val:
                a[key] = val
                n += 1
    return n


def bucket_arms(payload: dict) -> dict[str, tuple[set[str], tuple[str, ...]]]:
    """The stage-1 counterfactual arms: which buckets to relabel, and on which attributes."""
    seen = set(summarise_anatomy(payload)["tracks"])
    role_buckets = {b for b in seen if b.startswith("role:")}
    team_buckets = {b for b in seen if b.startswith("team:")}
    gk_buckets = {b for b in seen if b.endswith("goalkeeper") or b == "role:goalkeeper->player"}
    arms: dict[str, tuple[set[str], tuple[str, ...]]] = {
        "role_all": (role_buckets, ("role", "team")),
        "team_all": (team_buckets, ("team",)),
        "gk_all": (gk_buckets, ("role", "team")),
        "contaminated": ({"contaminated"}, ("role", "team")),
    }
    for b in sorted(role_buckets | team_buckets):
        arms[b.replace(":", "_").replace("->", "2")] = ({b}, ("role", "team"))
    return arms


def run_oracles(arm_dir: Path, data_dir: Path, anat: dict, seqs: list[str], work: Path) -> dict:
    """Score every sub-bucket relabel counterfactual with the official evaluator."""
    src = arm_dir / "predictions" / "data"
    ctrl = work / "control"
    (ctrl / "predictions" / "data").mkdir(parents=True, exist_ok=True)
    for s in seqs:
        (ctrl / "predictions" / "data" / f"{s}.json").write_text(
            json.dumps({"predictions": voted_predictions(src / f"{s}.json")}), encoding="utf-8")
    base = gs_hota(ctrl, data_dir, seq_info={s: 0 for s in seqs}, **EVAL_CONFIGS["gs_hota_full"])
    out = {"control": {"combined": base["combined"], "per_seq": base["per_seq"], "changed": 0}}
    logger.info("control (voted) GS-DetA %.4f GS-HOTA %.4f", base["combined"]["GS-DetA"],
                base["combined"]["GS-HOTA"])
    for name, (want, attrs) in bucket_arms(anat).items():
        t0 = time.time()
        dest = work / "orc" / "predictions" / "data"
        dest.mkdir(parents=True, exist_ok=True)
        changed = 0
        for s in seqs:
            preds = voted_predictions(src / f"{s}.json")
            changed += _relabel(preds, anat["per_seq"][s], want, attrs)
            (dest / f"{s}.json").write_text(json.dumps({"predictions": preds}), encoding="utf-8")
        res = gs_hota(work / "orc", data_dir, seq_info={s: 0 for s in seqs},
                      **EVAL_CONFIGS["gs_hota_full"])
        out[name] = {"combined": res["combined"], "changed": changed}
        logger.info("%-28s GS-DetA %.4f (%+.4f) GS-HOTA %.4f (%+.4f) rows %d [%.0fs]", name,
                    res["combined"]["GS-DetA"],
                    res["combined"]["GS-DetA"] - base["combined"]["GS-DetA"],
                    res["combined"]["GS-HOTA"],
                    res["combined"]["GS-HOTA"] - base["combined"]["GS-HOTA"], changed,
                    time.time() - t0)
        shutil.rmtree(work / "orc", ignore_errors=True)
    return out


# === stage 2 arms =================================================================================
#: The registered stage-2 arms (all GT-free, all deterministic, all post-processing on our own JSON).
ARMS: dict[str, tuple[str, ...]] = {
    "gk_side": ("team",),
    "gk_side_role": ("role", "team"),
}


def apply_arm(preds: list[dict], steps: tuple[str, ...]) -> dict[str, int]:
    """Apply one registered stage-2 arm to a voted submission, in place."""
    from eval.gsr_score import gk_side_repair  # noqa: PLC0415

    return gk_side_repair(preds, steps)


def run_arms(arm_dir: Path, data_dir: Path, seqs: list[str], work: Path,
             names: list[str]) -> dict:
    """Score each registered arm against the voted control, with the paired per-sequence stats."""
    from scipy.stats import wilcoxon  # noqa: PLC0415

    src = arm_dir / "predictions" / "data"
    ctrl = work / "control"
    (ctrl / "predictions" / "data").mkdir(parents=True, exist_ok=True)
    for s in seqs:
        (ctrl / "predictions" / "data" / f"{s}.json").write_text(
            json.dumps({"predictions": voted_predictions(src / f"{s}.json")}), encoding="utf-8")
    base = gs_hota(ctrl, data_dir, seq_info={s: 0 for s in seqs}, **EVAL_CONFIGS["gs_hota_full"])
    out = {"control": {"combined": base["combined"], "per_seq": base["per_seq"]}}
    logger.info("control (voted) GS-DetA %.4f GS-HOTA %.4f", base["combined"]["GS-DetA"],
                base["combined"]["GS-HOTA"])
    for name in names:
        dest = work / "arm" / "predictions" / "data"
        dest.mkdir(parents=True, exist_ok=True)
        changed: Counter = Counter()
        for s in seqs:
            preds = voted_predictions(src / f"{s}.json")
            changed.update(apply_arm(preds, ARMS[name]))
            (dest / f"{s}.json").write_text(json.dumps({"predictions": preds}), encoding="utf-8")
        res = gs_hota(work / "arm", data_dir, seq_info={s: 0 for s in seqs},
                      **EVAL_CONFIGS["gs_hota_full"])
        dh = np.array([res["per_seq"][s]["GS-HOTA"] - base["per_seq"][s]["GS-HOTA"] for s in seqs])
        dd = np.array([res["per_seq"][s]["GS-DetA"] - base["per_seq"][s]["GS-DetA"] for s in seqs])
        p = float(wilcoxon(dh).pvalue) if np.any(dh != 0) else 1.0
        out[name] = {"combined": res["combined"], "per_seq": res["per_seq"],
                     "changed": dict(changed),
                     "paired_hota": {"mean": float(dh.mean()), "helped": int((dh > 0).sum()),
                                     "hurt": int((dh < 0).sum()), "wilcoxon_p": p},
                     "paired_deta": {"mean": float(dd.mean()), "helped": int((dd > 0).sum()),
                                     "hurt": int((dd < 0).sum())}}
        logger.info("%-10s GS-DetA %.4f (%+.4f) GS-HOTA %.4f (%+.4f) paired %+.4f %d/%d p=%.3g %s",
                    name, res["combined"]["GS-DetA"],
                    res["combined"]["GS-DetA"] - base["combined"]["GS-DetA"],
                    res["combined"]["GS-HOTA"],
                    res["combined"]["GS-HOTA"] - base["combined"]["GS-HOTA"], dh.mean(),
                    int((dh > 0).sum()), int((dh < 0).sum()), p, dict(changed))
        shutil.rmtree(work / "arm", ignore_errors=True)
    return out


# === self-check ===================================================================================
def demo() -> None:
    """Assert the pure seams: GT majority attributes, bucket naming, track relabelling."""
    gt_rows = [[{"track_id": 1, "attributes": {"role": "player", "team": "left"}},
                {"track_id": 2, "attributes": {"role": "goalkeeper", "team": "right"}}],
               [{"track_id": 1, "attributes": {"role": "player", "team": "left"}},
                {"track_id": 2, "attributes": {"role": "player", "team": "right"}}]]
    ga = gt_track_attrs(gt_rows)
    assert ga[1] == {"role": "player", "team": "left", "n": 2}, ga
    assert ga[2]["role"] == "goalkeeper" and ga[2]["n"] == 2, ga  # tie -> first seen
    rec = {"n_matched": 0, "purity": 0.0, "gt_role": None, "our_role": "player",
           "gt_team": None, "our_team": "left"}
    assert classify_track(rec) == "unmatched"
    rec = {"n_matched": 10, "purity": 0.5, "gt_role": "player", "our_role": "player",
           "gt_team": "left", "our_team": "left"}
    assert classify_track(rec) == "contaminated"
    rec = {"n_matched": 10, "purity": 1.0, "gt_role": "goalkeeper", "our_role": "player",
           "gt_team": "left", "our_team": "left"}
    assert classify_track(rec) == "role:goalkeeper->player"
    rec = {"n_matched": 10, "purity": 1.0, "gt_role": "player", "our_role": "player",
           "gt_team": "left", "our_team": "right"}
    assert classify_track(rec) == "team:player"
    rec["our_team"] = "left"
    assert classify_track(rec) == "ok"
    preds = [{"track_id": 5, "attributes": {"role": "player", "team": "right", "jersey": "9"}},
             {"track_id": 6, "attributes": {"role": "player", "team": "right", "jersey": "9"}}]
    recs = [{"track_id": 5, "n_matched": 9, "purity": 1.0, "gt_role": "goalkeeper",
             "our_role": "player", "gt_team": "left", "our_team": "right"}]
    n = _relabel(preds, recs, {"role:goalkeeper->player"}, ("role", "team"))
    assert n == 2 and preds[0]["attributes"] == {"role": "goalkeeper", "team": "left",
                                                 "jersey": "9"}, (n, preds)
    assert preds[1]["attributes"]["role"] == "player", preds
    print("gsr_v9_w4 demo: OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-dir", type=Path, default=DEFAULT_ARM)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    ap.add_argument("--work", type=Path, default=WORK_DIR)
    ap.add_argument("--anatomy", action="store_true")
    ap.add_argument("--oracles", action="store_true")
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
    anat_path = args.out / "gsr_v9_w4_anatomy.json"
    if args.anatomy:
        payload = run_anatomy(args.arm_dir, args.data_dir, seqs)
        anat_path.write_text(json.dumps(payload), encoding="utf-8")
        logger.info("wrote %s", anat_path)
        logger.info("%s", json.dumps(summarise_anatomy(payload), indent=1))
        return
    if args.oracles:
        anat = json.loads(anat_path.read_text(encoding="utf-8"))
        payload = run_oracles(args.arm_dir, args.data_dir, anat, seqs, args.work)
        (args.out / "gsr_v9_w4_oracles.json").write_text(json.dumps(payload, indent=2),
                                                         encoding="utf-8")
        return
    if args.arms is not None:
        names = args.arms or list(ARMS)
        bad = [a for a in names if a not in ARMS]
        if bad:
            raise SystemExit(f"unknown arms {bad}; known: {list(ARMS)}")
        payload = run_arms(args.arm_dir, args.data_dir, seqs, args.work, names)
        (args.out / "gsr_v9_w4_arms.json").write_text(json.dumps(payload, indent=2),
                                                      encoding="utf-8")
        return
    ap.error("choose --anatomy, --oracles, --arms or --demo")


if __name__ == "__main__":
    main()
