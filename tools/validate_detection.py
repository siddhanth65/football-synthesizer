"""Validate detection: person recall/precision, role accuracy and ball recall on held-out frames.

Two ground-truth modes (the review's "manually labelled holdouts", with an automated stand-in):

* **Consensus proxy (automatic, runs now):** a person box that >= ``MIN_AGREE`` of the detectors agree
  on (IoU match) is treated as a real person. Reports each detector's person recall/precision against
  that consensus and the consensus head-count per frame. Caveat: it is biased toward agreement and
  cannot judge *roles* (COCO/RF-DETR have no GK/referee classes) -- it is a screen, not truth.
* **Manual labels (true numbers):** if a YOLO-format label dir is passed via ``--labels``, the football
  detector is scored against it for person recall/precision, **role accuracy** and **ball recall**.

It also exports the held-out frames + the football detector's predictions as *draft* YOLO labels
(``--export``), so you can correct them in LabelImg/CVAT and feed them back via ``--labels``.

Run: ``python tools/validate_detection.py --export``  then, after correcting the drafts,
     ``python tools/validate_detection.py --labels outputs/validation/annotate``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.benchmark_detectors import _hf_pt, _norm_role, _rfdetr, _ultra  # noqa: E402

VIDEO = r"C:\Users\siddh_ygv5bws\OneDrive\Desktop\cv-football\chunks\chunk_000.mp4"
# Held-out frames spanning both segments (kept distinct from benchmark_detectors' set where possible).
FRAMES = [13030, 13040, 13055, 13070, 21725, 21735, 21745, 21760, 21775, 13062]
MIN_AGREE = 2
IOU_THR = 0.5
OUT = Path("outputs/validation")
_CLASSES = ["player", "goalkeeper", "referee", "ball"]  # YOLO class order for draft/GT labels
_PERSON_ROLES = ("player", "goalkeeper", "referee")


# ---- pure box geometry (unit-tested) -------------------------------------------------------------
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in ``a`` and every box in ``b`` (both ``(N, 4)`` xyxy)."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def greedy_match(pred: np.ndarray, gt: np.ndarray, thr: float = IOU_THR):
    """Greedily match predictions to ground truth by descending IoU. Returns (pairs, n_fp, n_fn)."""
    m = iou_matrix(pred, gt)
    pairs, used_p, used_g = [], set(), set()
    while m.size and m.max() >= thr:
        i, j = np.unravel_index(int(np.argmax(m)), m.shape)
        pairs.append((i, j))
        used_p.add(i)
        used_g.add(j)
        m[i, :] = -1
        m[:, j] = -1
    return pairs, len(pred) - len(used_p), len(gt) - len(used_g)


def score(pred: np.ndarray, gt: np.ndarray, thr: float = IOU_THR) -> dict:
    """Recall/precision of ``pred`` boxes against ``gt`` boxes."""
    pairs, fp, fn = greedy_match(pred, gt, thr)
    tp = len(pairs)
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "recall": rec, "precision": prec}


def consensus_boxes(boxsets: list[np.ndarray], min_agree: int = MIN_AGREE, thr: float = IOU_THR):
    """Cluster boxes pooled across detectors; keep clusters spanning >= ``min_agree`` detectors.

    Returns the consensus boxes (mean of each kept cluster). Greedy single-link clustering by IoU.
    """
    pooled = [(det, box) for det, boxes in enumerate(boxsets) for box in boxes]
    used = [False] * len(pooled)
    out = []
    for seed in range(len(pooled)):
        if used[seed]:
            continue
        cluster = [seed]
        used[seed] = True
        for k in range(len(pooled)):
            if used[k]:
                continue
            if float(iou_matrix(pooled[seed][1][None], pooled[k][1][None])[0, 0]) >= thr:
                cluster.append(k)
                used[k] = True
        dets = {pooled[c][0] for c in cluster}
        if len(dets) >= min_agree:
            out.append(np.mean([pooled[c][1] for c in cluster], axis=0))
    return np.array(out) if out else np.zeros((0, 4))


# ---- prediction extraction -----------------------------------------------------------------------
def _run_all(frame_bgr, runners):
    """Return ``{name: (person_boxes, ball_boxes, role_per_person)}`` for each detector runner."""
    out = {}
    for name, run in runners.items():
        dets = run(frame_bgr)  # list of (role, box)
        persons = [(r, b) for r, b in dets if r in _PERSON_ROLES]
        balls = [b for r, b in dets if r == "ball"]
        out[name] = (
            np.array([b for _, b in persons]) if persons else np.zeros((0, 4)),
            np.array(balls) if balls else np.zeros((0, 4)),
            [r for r, _ in persons],
        )
    return out


def _read_yolo_gt(label_dir: Path, frame: int, w: int, h: int):
    """Read a YOLO-format GT file -> (person_boxes, person_roles, ball_boxes). Missing file -> None."""
    p = label_dir / f"frame_{frame:05d}.txt"
    if not p.exists():
        return None
    persons, roles, balls = [], [], []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = int(parts[0]), *(float(x) for x in parts[1:])
        box = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
        if _CLASSES[cls] == "ball":
            balls.append(box)
        else:
            persons.append(box)
            roles.append(_CLASSES[cls])
    return (np.array(persons) if persons else np.zeros((0, 4)), roles,
            np.array(balls) if balls else np.zeros((0, 4)))


def _runners():
    runners = {}
    for name, factory in [
        ("football", lambda: _ultra(_hf_pt("uisikdag/yolo-v8-football-players-detection"))),
        ("coco", lambda: _ultra("yolov8s.pt")),
        ("rfdetr", _rfdetr),
    ]:
        try:
            runners[name] = factory()
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP {name}: {exc}")
    return runners


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default=None, help="dir of corrected YOLO-format GT (.txt per frame)")
    ap.add_argument("--export", action="store_true", help="write frames + draft labels for annotation")
    args = ap.parse_args()

    runners = _runners()
    cap = cv2.VideoCapture(VIDEO)
    frames = {}
    for fr in FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if ok:
            frames[fr] = img
    cap.release()

    label_dir = Path(args.labels) if args.labels else None
    annotate = OUT / "annotate"
    if args.export:
        annotate.mkdir(parents=True, exist_ok=True)
        (annotate / "classes.txt").write_text("\n".join(_CLASSES) + "\n")

    consensus_recall = {n: [] for n in runners}
    consensus_prec = {n: [] for n in runners}
    head_counts = []
    gt_rows = []  # (frame, person_recall, person_prec, role_acc, ball_recall) when labels present

    for fr, img in frames.items():
        h, w = img.shape[:2]
        res = _run_all(img, runners)
        # consensus over person boxes
        cons = consensus_boxes([res[n][0] for n in runners])
        head_counts.append(len(cons))
        for n in runners:
            s = score(res[n][0], cons)
            consensus_recall[n].append(s["recall"])
            consensus_prec[n].append(s["precision"])

        if args.export and "football" in res:
            cv2.imwrite(str(annotate / f"frame_{fr:05d}.jpg"), img)
            lines = []
            fb_boxes, fb_balls, fb_roles = res["football"]
            for role, b in [*zip(fb_roles, fb_boxes), *(("ball", bb) for bb in fb_balls)]:
                cid = _CLASSES.index(role)
                cx, cy = (b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h
                bw, bh = (b[2] - b[0]) / w, (b[3] - b[1]) / h
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (annotate / f"frame_{fr:05d}.txt").write_text("\n".join(lines) + "\n")

        if label_dir is not None and "football" in res:
            gt = _read_yolo_gt(label_dir, fr, w, h)
            if gt is not None:
                gt_persons, gt_roles, gt_balls = gt
                fb_boxes, fb_balls, fb_roles = res["football"]
                s = score(fb_boxes, gt_persons)
                pairs, _, _ = greedy_match(fb_boxes, gt_persons)
                role_ok = sum(1 for i, j in pairs if fb_roles[i] == gt_roles[j])
                role_acc = role_ok / len(pairs) if pairs else float("nan")
                ball_rec = score(fb_balls, gt_balls)["recall"] if len(gt_balls) else float("nan")
                gt_rows.append((fr, s["recall"], s["precision"], role_acc, ball_rec))

    nfr = len(frames)
    print(f"\n=== consensus proxy ({nfr} frames, >= {MIN_AGREE}/{len(runners)} detectors agree) ===")
    print(f"consensus head-count/frame: mean {np.mean(head_counts):.1f} (min {min(head_counts)}, "
          f"max {max(head_counts)})")
    print(f"{'detector':10}{'person recall':>15}{'person precision':>18}")
    for n in runners:
        print(f"{n:10}{np.mean(consensus_recall[n]):>15.3f}{np.mean(consensus_prec[n]):>18.3f}")

    if gt_rows:
        g = np.array([r[1:] for r in gt_rows], float)
        print(f"\n=== football detector vs MANUAL labels ({len(gt_rows)} frames) ===")
        print(f"person recall   : {np.nanmean(g[:, 0]):.3f}")
        print(f"person precision: {np.nanmean(g[:, 1]):.3f}")
        print(f"role accuracy   : {np.nanmean(g[:, 2]):.3f}")
        print(f"ball recall     : {np.nanmean(g[:, 3]):.3f}")
    elif label_dir is not None:
        print(f"\nno GT .txt files found in {label_dir} -- correct the exported drafts first.")

    if args.export:
        (annotate / "README.txt").write_text(
            "Correct these YOLO-format drafts (frame_XXXXX.txt) in LabelImg/CVAT against the .jpg, then:\n"
            "  python tools/validate_detection.py --labels outputs/validation/annotate\n"
            f"classes (id order): {_CLASSES}\n"
            "Each .txt line: <class_id> <cx> <cy> <w> <h>  (all normalised 0-1).\n"
            "The drafts are the football detector's own predictions -- fix wrong roles, add missed\n"
            "people/ball, delete false boxes. Only then do the numbers mean anything.\n"
        )
        print(f"\nexported {nfr} frames + draft labels to {annotate}/ (see README.txt to annotate)")


if __name__ == "__main__":
    main()
