"""v9 W3 stage 1: decompose the GS-DetA loss of the shipped chain on DEV-20.

GS-DetA gates a matched detection on **position** (gaussian on the pitch bottom-middle point, sigma
from a 5 m tolerance) AND **role** AND **team** (players/GKs only) AND **jersey** (players only): any
attribute mismatch zeroes the similarity, so the pair is never a TP at any alpha
(``trackeval.datasets.SoccerNetGS.get_raw_seq_data``).

Two instruments, deliberately separate:

* **census** -- an attribute-blind, per-frame Hungarian match of our submission against GT on
  position alone (same sigma, same 5 m cut). Every GT row and every prediction row lands in exactly
  one bucket: detection miss / role wrong / team wrong / jersey wrong (named-wrong, coverage loss,
  over-naming) / correct-but-lost-to-the-alpha-sweep / false positive. This is a row census, not a
  metric.
* **oracles** -- counterfactual submissions that fix exactly one bucket and are then scored with the
  official evaluator (``eval.gsr_score.gs_hota``). The DetA delta of each arm is that bucket's point
  cost. Buckets are measured one at a time, so their deltas are NOT additive.

CLI::

    python -m tools.gsr_v9_deta --census
    python -m tools.gsr_v9_deta --oracles role team jersey ...
    python -m tools.gsr_v9_deta --demo        # self-check, no data needed
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from eval.gsr_score import DEFAULT_DATA_DIR, EVAL_CONFIGS, gs_hota, vote_track_attributes

logger = logging.getLogger("gsr_v9_deta")

#: Shipped v6 bundle, DEV-20 arm (results/GSR_V6.md section 2: GS-DetA 39.3083).
DEFAULT_ARM = Path("outputs/gsr/deleak_v6detdev_clip_e0.3r1w0.5a0.3")
#: Alpha sweep of trackeval's HOTA (``np.arange(0.05, 0.99, 0.05)``).
ALPHAS = np.arange(0.05, 0.99, 0.05)
#: Gaussian sigma for a 5 m tolerance (``soccernet_gs.calculate_sigma``).
SIGMA = float(np.sqrt(5.0**2 / (-2 * np.log(0.05))))
#: Track-id offset for oracle-inserted rows (must not collide with real prediction ids).
INSERT_ID_OFFSET = 1_000_000
#: Every oracle arm this module knows how to build.
ARMS = ("role", "team", "jersey", "jersey_namedwrong", "jersey_coverage", "jersey_overname",
        "attrs", "nofp", "nomiss", "nomiss_deadframes", "loc", "all",
        "attrs_no_role", "attrs_no_team", "attrs_no_jersey", "jersey_coverage_roster")


# === pure helpers =================================================================================
def similarity(gt_xy: np.ndarray, pr_xy: np.ndarray) -> np.ndarray:
    """Gaussian pitch similarity matrix, the evaluator's own formula (pure).

    Args:
        gt_xy: ``(N, 2)`` GT bottom-middle points, centred metres.
        pr_xy: ``(M, 2)`` prediction bottom-middle points, centred metres.

    Returns:
        ``(N, M)`` similarities in ``[0, 1]``.
    """
    if len(gt_xy) == 0 or len(pr_xy) == 0:
        return np.zeros((len(gt_xy), len(pr_xy)))
    d = np.linalg.norm(gt_xy[:, None, :] - pr_xy[None, :, :], axis=-1)
    return np.exp(-0.5 * (d / SIGMA) ** 2)


def match_positions(sim: np.ndarray, floor: float = 0.05) -> list[tuple[int, int]]:
    """Attribute-blind one-to-one match maximising total similarity, keeping pairs >= ``floor``.

    ``floor`` 0.05 is the evaluator's smallest alpha, i.e. exactly 5 m.
    """
    if sim.size == 0:
        return []
    rows, cols = linear_sum_assignment(-sim)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if sim[r, c] >= floor - 1e-12]


def effective_attrs(role: str | None, team: str | None, jersey) -> tuple:
    """Attributes as the evaluator compares them under the full config (pure).

    Team is only carried by players/goalkeepers and jersey only by players; everything else is
    ``None`` on both sides, so ``None == None`` matches.
    """
    t = team if role in {"player", "goalkeeper"} else None
    j = jersey if role == "player" else None
    if j is not None and str(j) != "":
        j = str(int(j)) if str(j).lstrip("-").isdigit() else str(j)
    else:
        j = None
    return role, t, j


def alphas_passed(sim: float) -> int:
    """How many of the 19 HOTA alpha thresholds a matched pair's similarity clears (pure)."""
    return int(np.sum(sim >= ALPHAS - np.finfo(float).eps))


def box_iou(a: dict, b: dict) -> float:
    """IoU of two GSR ``bbox_image`` dicts (pure)."""
    ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return float(inter / union) if union > 0 else 0.0


# === loading ======================================================================================
def load_gt(seq_dir: Path) -> tuple[dict[str, int], list[list[dict]]]:
    """Load a GSR sequence's GT the way the evaluator does.

    Returns:
        ``(image_id -> timestep, per-timestep list of person annotations)``. Only images with
        ``has_labeled_{person,pitch,camera}`` count, sorted by frame number; the ball is dropped.
    """
    gt = json.loads((seq_dir / "Labels-GameState.json").read_text(encoding="utf-8"))
    images = [i for i in gt["images"]
              if i["has_labeled_pitch"] and i["has_labeled_camera"] and i["has_labeled_person"]]
    images.sort(key=lambda i: int(i["image_id"].split("_")[-1]))
    ts = {im["image_id"]: k for k, im in enumerate(images)}
    rows: list[list[dict]] = [[] for _ in images]
    for ann in gt["annotations"]:
        if ann.get("supercategory") != "object":
            continue
        attrs = ann.get("attributes") or {}
        if attrs.get("role") == "ball" or ann["image_id"] not in ts:
            continue
        rows[ts[ann["image_id"]]].append(ann)
    return ts, rows


def load_pred(path: Path, ts: dict[str, int], n: int) -> list[list[dict]]:
    """Load one prediction JSON into per-timestep lists (ball dropped, unknown images dropped)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[list[dict]] = [[] for _ in range(n)]
    for p in payload["predictions"]:
        if p.get("supercategory") != "object":
            continue
        if (p.get("attributes") or {}).get("role") == "ball" or p["image_id"] not in ts:
            continue
        rows[ts[p["image_id"]]].append(p)
    return rows


def _xy(rows: list[dict]) -> np.ndarray:
    """Bottom-middle pitch points of a row list as ``(N, 2)`` (pure)."""
    if not rows:
        return np.zeros((0, 2))
    return np.array([[r["bbox_pitch"]["x_bottom_middle"], r["bbox_pitch"]["y_bottom_middle"]]
                     for r in rows], float)


# === census =======================================================================================
def census_sequence(seq_dir: Path, pred_path: Path) -> dict:
    """Bucket every GT row and every prediction row of one sequence (attribute-blind matching)."""
    ts, gt_rows = load_gt(seq_dir)
    pr_rows = load_pred(pred_path, ts, len(gt_rows))
    c: Counter = Counter()
    heights: dict[str, list[float]] = {"miss": [], "hit": []}
    crowd: dict[str, list[float]] = {"miss": [], "hit": []}
    alpha_units = 0.0          # sum over correct rows of alphas cleared / 19
    n_correct = 0
    fp_roles: Counter = Counter()
    for t, (gts, prs) in enumerate(zip(gt_rows, pr_rows)):
        sim = similarity(_xy(gts), _xy(prs))
        pairs = match_positions(sim)
        gt_hit = {i: j for i, j in pairs}
        pr_hit = {j: i for i, j in pairs}
        dead = len(prs) == 0
        # crowding proxy: best image-box IoU with another GT box in the same frame
        for i, g in enumerate(gts):
            other = max((box_iou(g["bbox_image"], o["bbox_image"])
                         for k, o in enumerate(gts) if k != i), default=0.0)
            h = float(g["bbox_image"]["h"])
            if i not in gt_hit:
                c["miss_deadframe" if dead else "miss_live"] += 1
                heights["miss"].append(h)
                crowd["miss"].append(other)
                continue
            heights["hit"].append(h)
            crowd["hit"].append(other)
            p = prs[gt_hit[i]]
            gr, gt_team, gj = effective_attrs(g["attributes"].get("role"),
                                              g["attributes"].get("team"),
                                              g["attributes"].get("jersey"))
            pr, pr_team, pj = effective_attrs(p["attributes"].get("role"),
                                              p["attributes"].get("team"),
                                              p["attributes"].get("jersey"))
            if gr != pr:
                c["role_wrong"] += 1
                c[f"role_wrong::{gr}->{pr}"] += 1
            elif gt_team != pr_team:
                c["team_wrong"] += 1
            elif gj != pj:
                c["jersey_wrong"] += 1
                if gj is not None and pj is not None:
                    c["jersey_named_wrong"] += 1
                elif gj is not None:
                    c["jersey_coverage_loss"] += 1
                else:
                    c["jersey_overnamed"] += 1
            else:
                c["correct"] += 1
                n_correct += 1
                alpha_units += alphas_passed(float(sim[i, gt_hit[i]])) / len(ALPHAS)
        for j, p in enumerate(prs):
            if j in pr_hit:
                continue
            near = bool(len(gts)) and bool(np.any(sim[:, j] >= 0.05 - 1e-12))
            c["fp_duplicate" if near else "fp_phantom"] += 1
            fp_roles[str((p["attributes"] or {}).get("role"))] += 1
    c["n_gt"] = sum(len(g) for g in gt_rows)
    c["n_pred"] = sum(len(p) for p in pr_rows)
    c["n_frames"] = len(gt_rows)
    c["n_dead_frames"] = sum(1 for p in pr_rows if not p)
    return {"counts": dict(c), "fp_roles": dict(fp_roles),
            "alpha_units_correct": alpha_units, "n_correct": n_correct,
            "heights": {k: v for k, v in heights.items()}, "crowd": crowd}


# === oracle arms ==================================================================================
def _fix_attrs(p: dict, g: dict, *, role: bool, team: bool, jersey: bool) -> None:
    """Copy the selected GT attributes onto a prediction, in place."""
    a, ga = p["attributes"], g["attributes"]
    if role:
        a["role"] = ga.get("role")
    if team:
        a["team"] = ga.get("team") if a.get("role") in {"player", "goalkeeper"} else None
    if jersey:
        j = ga.get("jersey")
        a["jersey"] = None if (j is None or str(j) == "" or a.get("role") != "player") else str(j)


def _inserted(g: dict, image_id: str) -> dict:
    """A prediction row placed exactly on a GT row (oracle detection), with GT attributes."""
    a = g["attributes"]
    role = a.get("role")
    return {
        "id": f"ins{image_id}{int(g['track_id']):04d}", "image_id": image_id,
        "track_id": INSERT_ID_OFFSET + int(g["track_id"]), "supercategory": "object",
        "category_id": g.get("category_id", 1),
        "attributes": {
            "role": role,
            "team": a.get("team") if role in {"player", "goalkeeper"} else None,
            "jersey": (str(a["jersey"]) if role == "player" and a.get("jersey") not in (None, "")
                       else None),
        },
        "bbox_pitch": dict(g["bbox_pitch"]),
        "bbox_image": dict(g["bbox_image"]),
        "confidence": 1.0,
    }


def build_arm(seq_dir: Path, pred_path: Path, out_path: Path, arm: str) -> dict:
    """Write one counterfactual submission for a sequence; returns how many rows it touched."""
    ts, gt_rows = load_gt(seq_dir)
    payload = json.loads(pred_path.read_text(encoding="utf-8"))
    by_ts: dict[int, list[dict]] = {}
    for p in payload["predictions"]:
        if (p.get("attributes") or {}).get("role") == "ball" or p["image_id"] not in ts:
            continue
        by_ts.setdefault(ts[p["image_id"]], []).append(p)
    id_of_ts = {v: k for k, v in ts.items()}
    # Our own roster: the (team side, number) pairs this sequence's submission emits anywhere. It
    # bounds what a better ABSTENTION POLICY could ever name, since the shipped chain is GT-free and
    # the solver may only pick numbers its own reader produced.
    roster = {(p["attributes"].get("team"), str(p["attributes"].get("jersey")))
              for p in payload["predictions"] if p["attributes"].get("jersey") not in (None, "")}
    stats: Counter = Counter()
    drop: set[int] = set()
    add: list[dict] = []
    for t, gts in enumerate(gt_rows):
        prs = by_ts.get(t, [])
        sim = similarity(_xy(gts), _xy(prs))
        pairs = match_positions(sim)
        matched_pr = {j for _i, j in pairs}
        for i, j in pairs:
            g, p = gts[i], prs[j]
            gr, gteam, gj = effective_attrs(g["attributes"].get("role"),
                                            g["attributes"].get("team"),
                                            g["attributes"].get("jersey"))
            pr, pteam, pj = effective_attrs(p["attributes"].get("role"),
                                            p["attributes"].get("team"),
                                            p["attributes"].get("jersey"))
            if arm in {"role", "attrs", "all", "attrs_no_team", "attrs_no_jersey"}:
                _fix_attrs(p, g, role=True, team=False, jersey=False)
                stats["role"] += gr != pr
            if arm in {"team", "attrs", "all", "attrs_no_role", "attrs_no_jersey"}:
                _fix_attrs(p, g, role=False, team=True, jersey=False)
                stats["team"] += gteam != pteam
            if arm in {"jersey", "attrs", "all", "attrs_no_role", "attrs_no_team"}:
                _fix_attrs(p, g, role=False, team=False, jersey=True)
                stats["jersey"] += gj != pj
            if (arm == "jersey_coverage_roster" and gj is not None and pj is None
                    and (gteam, gj) in roster):
                _fix_attrs(p, g, role=False, team=False, jersey=True)
                stats["jersey"] += 1
            if arm == "jersey_namedwrong" and gj is not None and pj is not None and gj != pj:
                _fix_attrs(p, g, role=False, team=False, jersey=True)
                stats["jersey"] += 1
            if arm == "jersey_coverage" and gj is not None and pj is None:
                _fix_attrs(p, g, role=False, team=False, jersey=True)
                stats["jersey"] += 1
            if arm == "jersey_overname" and gj is None and pj is not None:
                p["attributes"]["jersey"] = None
                stats["jersey"] += 1
            if arm in {"loc", "all"}:
                p["bbox_pitch"] = dict(g["bbox_pitch"])
                stats["loc"] += 1
        if arm in {"nofp", "all"}:
            for j, p in enumerate(prs):
                if j not in matched_pr:
                    drop.add(id(p))
                    stats["fp"] += 1
        if arm in {"nomiss", "all", "nomiss_deadframes"}:
            if arm != "nomiss_deadframes" or not prs:
                hit = {i for i, _j in pairs}
                for i, g in enumerate(gts):
                    if i not in hit:
                        add.append(_inserted(g, id_of_ts[t]))
                        stats["insert"] += 1
    kept = [p for p in payload["predictions"] if id(p) not in drop]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"predictions": kept + add}), encoding="utf-8")
    return dict(stats)


def run_oracles(arms: list[str], arm_dir: Path, data_dir: Path, work: Path,
                seqs: list[str]) -> dict:
    """Build and officially score each counterfactual arm; returns the ranked table payload."""
    data = arm_dir / "predictions" / "data"
    out: dict[str, dict] = {}
    base = gs_hota(arm_dir, data_dir, seq_info={s: 0 for s in seqs},
                   **EVAL_CONFIGS["gs_hota_full"])
    out["baseline"] = {"combined": base["combined"], "touched": {}}
    logger.info("baseline GS-DetA %.4f  GS-HOTA %.4f", base["combined"]["GS-DetA"],
                base["combined"]["GS-HOTA"])
    for arm in arms:
        t0 = time.time()
        dest = work / arm / "predictions" / "data"
        touched: Counter = Counter()
        for s in seqs:
            touched.update(build_arm(data_dir / s, data / f"{s}.json", dest / f"{s}.json", arm))
        res = gs_hota(work / arm, data_dir, seq_info={s: 0 for s in seqs},
                      **EVAL_CONFIGS["gs_hota_full"])
        out[arm] = {"combined": res["combined"], "touched": dict(touched)}
        logger.info("%-18s GS-DetA %.4f (%+.4f)  GS-HOTA %.4f (%+.4f)  rows %s  [%.0fs]", arm,
                    res["combined"]["GS-DetA"],
                    res["combined"]["GS-DetA"] - base["combined"]["GS-DetA"],
                    res["combined"]["GS-HOTA"],
                    res["combined"]["GS-HOTA"] - base["combined"]["GS-HOTA"],
                    dict(touched), time.time() - t0)
        shutil.rmtree(work / arm, ignore_errors=True)
    return out


# === stage 2: the per-track attribute vote ========================================================
#: The four arms registered a priori in ``results/GSR_V9_W3.md`` section 1.3.
VOTE_ARMS: dict[str, tuple[str, ...]] = {
    "vote_role": ("role",),
    "vote_team": ("team",),
    "vote_roleteam": ("role", "team"),
    "vote_all": ("role", "team", "jersey"),
}


def run_vote(arm_dir: Path, data_dir: Path, work: Path, seqs: list[str]) -> dict:
    """Score every registered vote arm against the untouched shipped arm (same-stack by design)."""
    src = arm_dir / "predictions" / "data"
    base = gs_hota(arm_dir, data_dir, seq_info={s: 0 for s in seqs}, **EVAL_CONFIGS["gs_hota_full"])
    out: dict[str, dict] = {"baseline": {"combined": base["combined"],
                                         "per_seq": base["per_seq"], "changed": {}}}
    logger.info("control GS-DetA %.4f  GS-HOTA %.4f", base["combined"]["GS-DetA"],
                base["combined"]["GS-HOTA"])
    for arm, keys in VOTE_ARMS.items():
        dest = work / arm / "predictions" / "data"
        dest.mkdir(parents=True, exist_ok=True)
        changed: Counter = Counter()
        for s in seqs:
            payload = json.loads((src / f"{s}.json").read_text(encoding="utf-8"))
            changed.update(vote_track_attributes(payload["predictions"], keys))
            (dest / f"{s}.json").write_text(json.dumps(payload), encoding="utf-8")
        res = gs_hota(work / arm, data_dir, seq_info={s: 0 for s in seqs},
                      **EVAL_CONFIGS["gs_hota_full"])
        d = np.array([res["per_seq"][s]["GS-HOTA"] - base["per_seq"][s]["GS-HOTA"] for s in seqs])
        from scipy.stats import wilcoxon  # noqa: PLC0415

        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        out[arm] = {"combined": res["combined"], "per_seq": res["per_seq"],
                    "changed": dict(changed),
                    "paired_hota": {"mean": float(d.mean()), "helped": int((d > 0).sum()),
                                    "hurt": int((d < 0).sum()), "wilcoxon_p": p}}
        logger.info("%-14s GS-DetA %.4f (%+.4f)  GS-HOTA %.4f (%+.4f)  paired %+.4f "
                    "%d helped / %d hurt p=%.4g  rows %s", arm, res["combined"]["GS-DetA"],
                    res["combined"]["GS-DetA"] - base["combined"]["GS-DetA"],
                    res["combined"]["GS-HOTA"],
                    res["combined"]["GS-HOTA"] - base["combined"]["GS-HOTA"],
                    d.mean(), int((d > 0).sum()), int((d < 0).sum()), p, dict(changed))
        shutil.rmtree(work / arm, ignore_errors=True)
    return out


# === self-check ===================================================================================
def demo() -> None:
    """Assert the pure seams behave: sigma, alpha counting, matching, attribute preprocessing."""
    assert abs(np.exp(-0.5 * (5.0 / SIGMA) ** 2) - 0.05) < 1e-12, "sigma is not the 5 m tolerance"
    assert len(ALPHAS) == 19
    assert alphas_passed(1.0) == 19
    assert alphas_passed(0.049) == 0
    assert alphas_passed(0.5) == 10, alphas_passed(0.5)
    gt = np.array([[0.0, 0.0], [10.0, 0.0]])
    pr = np.array([[0.3, 0.0], [50.0, 0.0]])
    pairs = match_positions(similarity(gt, pr))
    assert pairs == [(0, 0)], pairs
    assert effective_attrs("referee", "left", "7") == ("referee", None, None)
    assert effective_attrs("goalkeeper", "left", "7") == ("goalkeeper", "left", None)
    assert effective_attrs("player", "left", 7) == ("player", "left", "7")
    assert effective_attrs("player", "left", "") == ("player", "left", None)
    a = {"x": 0, "y": 0, "w": 10, "h": 10}
    assert abs(box_iou(a, a) - 1.0) < 1e-12
    assert box_iou(a, {"x": 20, "y": 0, "w": 10, "h": 10}) == 0.0
    preds = [{"track_id": 1, "attributes": {"role": "player", "team": "left", "jersey": "7"}},
             {"track_id": 1, "attributes": {"role": "player", "team": "right", "jersey": "7"}},
             {"track_id": 1, "attributes": {"role": "referee", "team": None, "jersey": None}},
             {"track_id": 2, "attributes": {"role": "ball", "team": None, "jersey": None}}]
    changed = vote_track_attributes(preds, ("role", "team", "jersey"))
    assert [p["attributes"]["role"] for p in preds] == ["player"] * 3 + ["ball"], preds
    assert [p["attributes"]["team"] for p in preds[:3]] == ["left"] * 3, preds
    assert [p["attributes"]["jersey"] for p in preds[:3]] == ["7"] * 3, preds
    assert changed == {"role": 1, "team": 2, "jersey": 1}, changed
    assert vote_track_attributes(preds, ()) == {}
    print("gsr_v9_deta demo: OK")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-dir", type=Path, default=DEFAULT_ARM)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", type=Path, default=Path("results/gsr_benchmark"))
    ap.add_argument("--work", type=Path, default=Path("outputs/gsr/v9_deta"))
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--oracles", nargs="*", default=None, help=f"one or more of {ARMS}")
    ap.add_argument("--vote", action="store_true", help="stage 2: the registered vote arms")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        demo()
        return
    seqs = sorted(p.stem for p in (args.arm_dir / "predictions" / "data").glob("*.json"))
    if not seqs:
        raise SystemExit(f"no submissions under {args.arm_dir}")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.census:
        per_seq = {}
        for i, s in enumerate(seqs):
            per_seq[s] = census_sequence(args.data_dir / s,
                                         args.arm_dir / "predictions" / "data" / f"{s}.json")
            logger.info("[%d/%d] %s %s", i + 1, len(seqs), s, per_seq[s]["counts"])
        (args.out / "gsr_v9_w3_census.json").write_text(
            json.dumps({"arm": str(args.arm_dir), "seqs": seqs, "per_seq": per_seq}),
            encoding="utf-8")
        logger.info("wrote %s", args.out / "gsr_v9_w3_census.json")
        return
    if args.vote:
        payload = run_vote(args.arm_dir, args.data_dir, args.work, seqs)
        (args.out / "gsr_v9_w3_vote.json").write_text(json.dumps(payload, indent=2),
                                                      encoding="utf-8")
        logger.info("wrote %s", args.out / "gsr_v9_w3_vote.json")
        return
    if args.oracles is not None:
        arms = args.oracles or list(ARMS)
        bad = [a for a in arms if a not in ARMS]
        if bad:
            raise SystemExit(f"unknown arms {bad}; known: {ARMS}")
        payload = run_oracles(arms, args.arm_dir, args.data_dir, args.work, seqs)
        (args.out / "gsr_v9_w3_oracles.json").write_text(json.dumps(payload, indent=2),
                                                         encoding="utf-8")
        logger.info("wrote %s", args.out / "gsr_v9_w3_oracles.json")
        return
    ap.error("choose --census, --oracles or --demo")


if __name__ == "__main__":
    main()
