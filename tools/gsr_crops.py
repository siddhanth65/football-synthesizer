"""GSR ground-truth crop set: the cluster session-2 crop law, laptop-local.

Reimplements ``~/work/build_crops.py`` (cluster session 2 section 2) from the contract recorded in
``docs/SOCCERNET_REPO_SWEEP.md`` section 1.1, so the trained jersey head can be evaluated on the
same crop distribution it was trained on without a server round-trip.

The law: drop ``w`` or ``h <= 30 px``; uniformly subsample each tracklet to 15 crops (first and
last always kept); drop identities with fewer than 4 crops; resize anything above 256x128 down.
The ``visibility >= 0.3`` rule is inapplicable -- GSR annotations carry no visibility field.

``plan_split`` is label-only (no image is opened) and reproduces the published train counts
(20,067 crops / 1,343 identities / 57 videos); ``cut_crops`` materialises the JPEGs.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

GSR_ROOT = Path("data/soccernet/gamestate-2024")
MIN_SIZE = 30
MAX_PER_ID = 15
MIN_PER_ID = 4
MAX_HW = (256, 128)
#: Object categories kept by the re-ID crop set (players, GKs, referees, ball, other).
_PERSON_CATS = {1, 2, 3, 4, 7}


@dataclass(frozen=True)
class Crop:
    """One planned crop.

    Attributes:
        seq: Sequence name, e.g. ``SNGS-060``.
        frame: Image file stem, e.g. ``000001``.
        track_id: Tracklet id within the sequence.
        xywh: Integer image box.
        role: ``player`` / ``goalkeeper`` / ``referee`` / ``other``.
        team: ``left`` / ``right`` / ``None``.
        jersey: Jersey string, or ``None`` when the annotation carries no number.
    """

    seq: str
    frame: str
    track_id: int
    xywh: tuple[int, int, int, int]
    role: str
    team: str | None
    jersey: str | None

    @property
    def name(self) -> str:
        """Stable on-disk file name for this crop."""
        return f"{self.seq}_{self.track_id:04d}_{self.frame}.jpg"


def split_sequences(split: str) -> list[str]:
    """Return the sequence names of a GSR split (``train`` / ``validation`` / ``test``)."""
    info = json.loads((GSR_ROOT / "sequences_info.json").read_text(encoding="utf-8"))
    return [s["name"] for s in info[split]]


def plan_sequence(seq: str) -> list[Crop]:
    """Apply the crop law to one sequence's labels. No image is opened."""
    labels = json.loads((GSR_ROOT / seq / "Labels-GameState.json").read_text(encoding="utf-8"))
    frame_of = {im["image_id"]: Path(im["file_name"]).stem for im in labels["images"]}
    by_track: dict[int, list[Crop]] = {}
    for ann in labels["annotations"]:
        if ann.get("category_id") not in _PERSON_CATS or "bbox_image" not in ann:
            continue
        box = ann["bbox_image"]
        w, h = int(box["w"]), int(box["h"])
        if w <= MIN_SIZE or h <= MIN_SIZE:
            continue
        attrs = ann.get("attributes") or {}
        tid = int(ann["track_id"])
        by_track.setdefault(tid, []).append(Crop(
            seq=seq, frame=frame_of[ann["image_id"]], track_id=tid,
            xywh=(int(box["x"]), int(box["y"]), w, h),
            role=attrs.get("role") or "other", team=attrs.get("team"),
            jersey=attrs.get("jersey"),
        ))
    out: list[Crop] = []
    for tid in sorted(by_track):
        crops = sorted(by_track[tid], key=lambda c: c.frame)
        if len(crops) < MIN_PER_ID:
            continue
        out.extend(_subsample(crops, MAX_PER_ID))
    return out


def _subsample(crops: list[Crop], k: int) -> list[Crop]:
    """Uniformly keep at most ``k`` crops, always keeping the first and the last."""
    n = len(crops)
    if n <= k:
        return crops
    idx = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    return [crops[i] for i in sorted(set(idx))]


def plan_split(split: str) -> list[Crop]:
    """Plan every sequence of a split."""
    out: list[Crop] = []
    for seq in split_sequences(split):
        out.extend(plan_sequence(seq))
    return out


def cut_crops(crops: list[Crop], out_dir: Path) -> list[Path]:
    """Materialise planned crops as JPEGs, one image read per frame. Resumable."""
    import cv2  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / c.name for c in crops]
    todo: dict[tuple[str, str], list[tuple[Crop, Path]]] = {}
    for crop, path in zip(crops, paths, strict=True):
        if not path.exists():
            todo.setdefault((crop.seq, crop.frame), []).append((crop, path))
    for done, (seq, frame) in enumerate(sorted(todo), 1):
        img = cv2.imread(str(GSR_ROOT / seq / "img1" / f"{frame}.jpg"))
        if img is None:
            logger.warning("unreadable frame %s/%s", seq, frame)
            continue
        for crop, path in todo[(seq, frame)]:
            x, y, w, h = crop.xywh
            sub = img[max(y, 0):y + h, max(x, 0):x + w]
            if sub.size == 0:
                continue
            if sub.shape[0] > MAX_HW[0] or sub.shape[1] > MAX_HW[1]:
                sub = cv2.resize(sub, MAX_HW[::-1], interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(path), sub)
        if done % 2000 == 0:
            logger.info("cut %d / %d frames", done, len(todo))
    return paths


def _demo() -> None:
    """Self-check: the law's own invariants plus the published train counts."""
    fake = [Crop("S", f"{i:06d}", 1, (0, 0, 40, 90), "player", "left", "7") for i in range(100)]
    kept = _subsample(fake, MAX_PER_ID)
    assert len(kept) == MAX_PER_ID, len(kept)
    assert kept[0] is fake[0] and kept[-1] is fake[-1], "first/last must survive"
    assert _subsample(fake[:3], MAX_PER_ID) == fake[:3]
    plan = plan_split("train")
    ids = {(c.seq, c.track_id) for c in plan}
    print(f"train crops={len(plan)} ids={len(ids)} videos={len({c.seq for c in plan})}")
    assert (len(plan), len(ids)) == (20067, 1343), "does not reproduce the session-2 crop set"
    print("gsr_crops demo OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=None, help="train / validation / test")
    ap.add_argument("--out", default="outputs/gsr/gt_crops")
    args = ap.parse_args()
    if args.split is None:
        _demo()
    else:
        planned = plan_split(args.split)
        print(f"{args.split}: {len(planned)} crops")
        cut_crops(planned, Path(args.out) / args.split)
