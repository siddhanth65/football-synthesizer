"""Row-weighted read-transition diff between two jersey vote caches.

Two arms of the same GSR pipeline write their per-crop reads and their per-track votes into
separate cache directories, but each arm re-runs the association into the *same* positions
directory, and that re-run is only stable up to a track-id relabelling (measured: 96 of ~100
crop blocks identical, ids shifted by two).  Comparing the two vote files by raw track id
therefore compares different tracks and produces nonsense.

This module aligns the two arms by the one thing that is stable -- the partition of crops into
tracks, keyed by the frame set of each track -- then reports, per sequence, how many *positions
rows* moved between {correct read, wrong read, abstention}.  Rows are the unit that GS-HOTA's
jersey term actually integrates over, so a read lost on a 450-row tracklet and a read gained on a
40-row tracklet are not treated as cancelling.

Ground truth for a track is the majority GT track within 3 m on the pitch, from
``Labels-GameState.json``; tracks whose GT player carries no jersey label are skipped.

Example:
    python -m tools.gsr_vote_delta \\
        --base outputs/gsr_srv/koshkina_percrop_v6_v6det_eiou \\
        --arm outputs/gsr_srv/koshkina_percrop_v7e_v6det_eiou \\
        --positions outputs/gsr_srv/positions_gate_v6det_eiou \\
        --out results/gsr_benchmark/gsr_vote_delta.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PITCH_HALF_LENGTH_M = 52.5
PITCH_HALF_WIDTH_M = 34.0
GT_RADIUS_M = 3.0
PLAYER_CATEGORIES = (1, 2)


def crop_partition(df: pd.DataFrame) -> dict[frozenset[int], int]:
    """Map each track's frame set to its track id.

    Args:
        df: A per-crop table with ``track_id`` and ``frame`` columns.

    Returns:
        Frame set -> track id, one entry per track.
    """
    return {frozenset(int(f) for f in g["frame"]): int(t) for t, g in df.groupby("track_id")}


def id_permutation(base: pd.DataFrame, arm: pd.DataFrame) -> dict[int, int]:
    """Track ids of ``base`` expressed in ``arm``'s numbering, for blocks present in both."""
    pb, pa = crop_partition(base), crop_partition(arm)
    return {pb[k]: pa[k] for k in pb.keys() & pa.keys()}


def gt_jerseys(labels: Path, positions: pd.DataFrame) -> dict[int, tuple[str | None, str | None]]:
    """Attribute every predicted track to a GT player and return its (jersey, team).

    Args:
        labels: Path to the sequence's ``Labels-GameState.json``.
        positions: The arm's positions table (``frame``, ``track_id``, ``pitch_x``, ``pitch_y``).

    Returns:
        Predicted track id -> (GT jersey or ``None``, GT team or ``None``).
    """
    doc = json.loads(labels.read_text(encoding="utf-8"))
    by_frame: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    meta: dict[int, tuple[str | None, str | None]] = {}
    for ann in doc["annotations"]:
        if ann["category_id"] not in PLAYER_CATEGORIES or not ann.get("bbox_pitch"):
            continue
        box = ann["bbox_pitch"]
        by_frame[int(ann["image_id"][-6:])].append((
            ann["track_id"],
            box["x_bottom_middle"] + PITCH_HALF_LENGTH_M,
            box["y_bottom_middle"] + PITCH_HALF_WIDTH_M,
        ))
        meta[ann["track_id"]] = (ann["attributes"].get("jersey"), ann["attributes"].get("team"))

    votes: dict[int, Counter] = defaultdict(Counter)
    for frame, tid, px, py in zip(positions["frame"], positions["track_id"],
                                  positions["pitch_x"], positions["pitch_y"], strict=True):
        best, best_d = None, GT_RADIUS_M
        for gid, gx, gy in by_frame.get(int(frame), ()):
            dist = math.hypot(px - gx, py - gy)
            if dist < best_d:
                best_d, best = dist, gid
        votes[int(tid)][best] += 1
    return {tid: meta.get(c.most_common(1)[0][0], (None, None)) for tid, c in votes.items()}


def load_votes(path: Path) -> dict[int, int]:
    """Track id -> winning number from one ``koshkina_percrop_votes_*`` sequence file."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): int(v[0][0]) for k, v in doc["votes"].items()}


def classify(number: int | None, truth: str | None) -> str:
    """Bucket one track's read as ``abst``, ``ok`` or ``wrong`` against its GT jersey."""
    if number is None:
        return "abst"
    return "ok" if str(number) == str(truth) else "wrong"


def votes_for(cache: Path, seq: str) -> Path:
    """The vote file that belongs to a per-crop cache directory."""
    name = cache.name.replace("koshkina_percrop_", "koshkina_percrop_votes_f080_")
    return cache.parent / name / f"{seq}.json"


def scan(base_dir: Path, arm_dir: Path, pos_dir: Path, data_dir: Path) -> list[dict]:
    """Run the row-weighted diff over every sequence present in both caches."""
    rows: list[dict] = []
    for path in sorted(base_dir.glob("SNGS-*.parquet")):
        seq = path.stem
        if not (arm_dir / f"{seq}.parquet").exists():
            continue
        base_crops = pd.read_parquet(path)
        arm_crops = pd.read_parquet(arm_dir / f"{seq}.parquet")
        perm = id_permutation(base_crops, arm_crops)
        base_v = {perm[k]: v for k, v in load_votes(votes_for(base_dir, seq)).items() if k in perm}
        arm_v = load_votes(votes_for(arm_dir, seq))
        positions = pd.read_parquet(pos_dir / f"{seq}.parquet")
        truth = gt_jerseys(data_dir / seq / "Labels-GameState.json", positions)
        lengths = Counter(int(t) for t in positions["track_id"])

        moved: Counter = Counter()
        for tid, (jersey, _team) in truth.items():
            if jersey is None:
                continue
            moved[(classify(base_v.get(tid), jersey),
                   classify(arm_v.get(tid), jersey))] += lengths[tid]
        total = len(positions)
        rows.append({
            "seq": seq, "rows": total, "aligned_tracks": len(perm),
            "unaligned_base_tracks": len(crop_partition(base_crops)) - len(perm),
            "ok_to_abst": moved[("ok", "abst")] / total,
            "ok_to_wrong": moved[("ok", "wrong")] / total,
            "abst_to_ok": moved[("abst", "ok")] / total,
            "abst_to_wrong": moved[("abst", "wrong")] / total,
            "wrong_to_ok": moved[("wrong", "ok")] / total,
            "net_ok": (moved[("abst", "ok")] + moved[("wrong", "ok")]
                       - moved[("ok", "abst")] - moved[("ok", "wrong")]) / total,
        })
        logger.info("%s: net_ok %+.3f (lost %.3f, gained %.3f)", seq, rows[-1]["net_ok"],
                    rows[-1]["ok_to_abst"] + rows[-1]["ok_to_wrong"],
                    rows[-1]["abst_to_ok"] + rows[-1]["wrong_to_ok"])
    return rows


def demo() -> None:
    """Self-check: the permutation aligner and the row-weighted buckets, on synthetic input."""
    base = pd.DataFrame({"track_id": [1, 1, 2, 2], "frame": [1, 2, 5, 6]})
    arm = pd.DataFrame({"track_id": [3, 3, 4, 4], "frame": [1, 2, 5, 6]})
    assert id_permutation(base, arm) == {1: 3, 2: 4}
    shifted = pd.DataFrame({"track_id": [3, 3, 4, 4], "frame": [1, 2, 5, 7]})
    assert id_permutation(base, shifted) == {1: 3}
    assert classify(None, "7") == "abst"
    assert classify(7, "7") == "ok"
    assert classify(7, "8") == "wrong"
    assert classify(7, None) == "wrong"
    print("gsr_vote_delta demo OK")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, help="baseline per-crop cache directory")
    ap.add_argument("--arm", type=Path, help="arm per-crop cache directory")
    ap.add_argument("--positions", type=Path, help="positions directory the arm ran on")
    ap.add_argument("--data", type=Path, default=Path("data/soccernet/gamestate-2024"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.demo:
        demo()
        return
    rows = scan(args.base, args.arm, args.positions, args.data)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"base": str(args.base), "arm": str(args.arm),
                                        "positions": str(args.positions), "per_seq": rows},
                                       indent=1), encoding="utf-8")
        print(f"wrote {args.out} ({len(rows)} sequences)")


if __name__ == "__main__":
    main()
