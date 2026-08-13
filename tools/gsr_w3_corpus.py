"""W3 prep: glyph-only corpus assembly + Sid's tracklet annotation queue.

Campaign v8, session W3 (data track). Four CPU/short-GPU steps, no training:

1. ``--legibility SPLIT`` -- score every GT crop of a GSR split with the shipped ResNet34
   legibility classifier (the same instrument, at its shipped weights, that
   ``CLUSTER_SESSION_S2.md`` section 3.2 used at threshold 0.7). Short GPU pass.
2. ``--j2023-sample N`` -- the same score on a seeded sample of the jersey-2023 train corpus, to
   measure its pass rate and size the full pass rather than guess it.
3. ``--manifest`` -- the glyph-only corpus manifest: one row per candidate ``(crop, number)`` pair
   with its source, its legibility score and whether the glyph-only rule admits it.
4. ``--queue`` / ``--ingest CSV`` -- build (and later consume) Sid's tracklet annotation queue.

Split hygiene, enforced in code (:func:`_assert_train_only`): label sources are GSR **train** and
jersey-2023 only. GSR ``validation`` (= DEV-20 + TEST-38) and ``test`` never enter a manifest or a
queue.

Glyph-only rule: GSR-train and jersey-2023 numbers are both *identity-carried* (measured: 0 of 1,224
GSR-train player/GK tracks mixes numbered and un-numbered boxes, so a number is a tracklet property,
not a per-frame observation), so a carried number is admitted for a crop only when the shipped
legibility model scores that crop above ``LEG_ADMIT`` (0.7). Human labels from ``--ingest`` are
admitted on the annotator's word alone -- that is the point of the evenings.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tools.gsr_crops import GSR_ROOT, Crop, plan_split, split_sequences

logger = logging.getLogger("gsr_w3_corpus")

CROP_ROOT = Path("outputs/gsr/gt_crops")
J2023_ROOT = Path("data/soccernet/jersey-2023/train")
OUT_ROOT = Path("outputs/gsr/w3_annotation")
LEG_WEIGHTS = Path.home() / "jersey-number-pipeline" / "models" / (
    "legibility_resnet34_soccer_20240215.pth")
#: Legibility floor a carried (identity-level) number must clear to enter the corpus. The S2 value.
LEG_ADMIT = 0.7
#: Splits that may supply labels. Everything else is evaluation-only.
LABEL_SPLITS = ("train",)
#: Contact-sheet geometry: 12 cells, 4 x 3.
SHEET_COLS, SHEET_ROWS = 4, 3
CELL_W, CELL_H = 170, 250
#: Tier-A queue = every GSR-train player/GK tracklet with no GT number at all.
#: Tier-B queue size (per-crop glyph-visibility grids on numbered tracklets).
TIER_B_N = 210
#: A tracklet needs at least this many GT boxes to be worth a contact sheet.
MIN_QUEUE_FRAMES = 12
#: Minimum box height a contact-sheet view must have, in pixels. Measured, not chosen: it is the
#: 10th percentile of the height of the GSR-train crops that pass legibility ``LEG_ADMIT``, and
#: below 60 px the shipped legibility model fires on 0.4% of crops. Showing an annotator a 46 px
#: box wastes his evening and produces a "none" that means "too small", not "no number".
MIN_VIEW_H = 89


def _assert_train_only(split: str) -> None:
    """Refuse to build label artifacts from an evaluation split."""
    if split not in LABEL_SPLITS:
        raise SystemExit(
            f"split '{split}' is not a label source; W3 may only label {LABEL_SPLITS}. "
            "GSR validation is DEV-20 + TEST-38 and GSR test is test-49.")


# --------------------------------------------------------------------------- legibility


def _scorer(device: str | None = None):  # noqa: ANN202
    """Load the shipped ResNet34 legibility classifier and its transform."""
    import torch  # noqa: PLC0415
    from torchvision import transforms  # noqa: PLC0415

    from generator.jersey_id import _MEAN, _STD, _Legibility34  # noqa: PLC0415

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _Legibility34().to(dev).eval()
    sd = torch.load(LEG_WEIGHTS, map_location=dev)
    if hasattr(sd, "_metadata"):
        del sd._metadata
    model.load_state_dict(sd)
    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                             transforms.Normalize(_MEAN, _STD)])
    return model, tf, dev


def score_paths(paths: list[Path], batch: int = 128, device: str | None = None) -> np.ndarray:
    """Legibility sigmoid per crop path (``NaN`` where the file will not decode)."""
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    model, tf, dev = _scorer(device)
    out = np.full(len(paths), np.nan, np.float32)
    buf: list[torch.Tensor] = []
    idx: list[int] = []

    def flush() -> None:
        if not buf:
            return
        with torch.no_grad(), torch.amp.autocast(dev, enabled=dev == "cuda"):
            sc = model(torch.stack(buf).to(dev)).float().squeeze(1).cpu().numpy()
        for j, s in enumerate(sc):
            out[idx[j]] = float(s)
        buf.clear()
        idx.clear()

    for i, p in enumerate(paths):
        try:
            buf.append(tf(Image.open(p).convert("RGB")))
        except (OSError, ValueError):
            continue
        idx.append(i)
        if len(buf) >= batch:
            flush()
        if i and i % 5000 == 0:
            logger.info("scored %d / %d", i, len(paths))
    flush()
    return out


def leg_path(tag: str) -> Path:
    """Where a legibility pass persists its scores."""
    return OUT_ROOT / f"legibility_{tag}.parquet"


#: Shard size for the full jersey-2023 pass -- small enough that an interrupted run only
#: re-scores one shard's worth of crops, large enough not to spend GPU time on I/O churn.
J2023_SHARD = 20_000


def _j2023_numbered_rows() -> list[tuple[str, str, int]]:
    """Every ``(tracklet, file, label)`` row of jersey-2023 train whose label is a real number."""
    gt = json.loads((J2023_ROOT / "train_gt.json").read_text(encoding="utf-8"))
    rows: list[tuple[str, str, int]] = []
    for tid, label in sorted(gt.items()):
        if int(label) < 0:
            continue
        d = J2023_ROOT / "images" / tid
        if not d.is_dir():
            continue
        rows.extend((tid, p.name, int(label)) for p in sorted(d.iterdir()))
    return rows


def run_j2023_full(batch: int = 128, shard: int = J2023_SHARD) -> Path:
    """Score every jersey-2023 train crop of a numbered tracklet (checkpointed per shard).

    Resumable: each shard's scores persist as their own parquet before the next shard starts, so an
    interrupted pass only re-does its current shard. The illegible pool (label -1) is not scored
    here -- it is admitted to the abstention class on the human label alone, per the glyph-only rule.
    """
    rows = _j2023_numbered_rows()
    shard_dir = OUT_ROOT / "j2023_full_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    n_shards = (len(rows) + shard - 1) // shard
    for s in range(n_shards):
        sp = shard_dir / f"shard{s:04d}.parquet"
        if sp.exists():
            continue
        chunk = rows[s * shard: (s + 1) * shard]
        paths = [J2023_ROOT / "images" / t / f for t, f, _ in chunk]
        scores = score_paths(paths, batch=batch)
        pd.DataFrame({"tracklet": [t for t, _, _ in chunk], "file": [f for _, f, _ in chunk],
                     "label": [lab for _, _, lab in chunk], "legibility": scores}).to_parquet(
            sp, index=False)
        logger.info("shard %d/%d done (%d rows) -> %s", s + 1, n_shards, len(chunk), sp)
    df = pd.concat([pd.read_parquet(shard_dir / f"shard{s:04d}.parquet")
                    for s in range(n_shards)], ignore_index=True)
    df.to_parquet(leg_path("j2023_full"), index=False)
    ok = df["legibility"].notna()
    passed = ok & (df["legibility"] > LEG_ADMIT)
    print(f"jersey-2023 train (full, numbered only): {len(rows)} crops, {int(ok.sum())} scored")
    print(f"  pass@{LEG_ADMIT} {int(passed.sum())} ({passed.sum() / len(rows):.4f})")
    return leg_path("j2023_full")


def build_j2023_full_manifest() -> Path:
    """Glyph-only manifest rows for the full jersey-2023 numbered-crop pass.

    Same columns as :func:`build_manifest`'s ``corpus_manifest.parquet`` so a training session can
    ``pd.concat`` the two. Kept as a sibling file (not merged into ``corpus_manifest.parquet``
    directly) because the two sources have different identity spaces (GSR sequence+tracklet vs.
    jersey-2023 tracklet folder) and merging is the training session's call.
    """
    lp = leg_path("j2023_full")
    if not lp.exists():
        raise SystemExit(f"missing {lp} -- run --full-jersey23 first")
    df = pd.read_parquet(lp)
    df["source"] = "jersey2023-train"
    df["crop_path"] = df.apply(
        lambda r: str(J2023_ROOT / "images" / r["tracklet"] / r["file"]), axis=1)
    df["label_provenance"] = "identity-carried (tracklet-level GT)"
    admitted = df["legibility"].notna() & (df["legibility"] > LEG_ADMIT)
    df["admitted"] = admitted
    df["reason"] = np.where(
        df["legibility"].isna(), "unscored (crop failed to decode)",
        np.where(admitted, f"legibility > {LEG_ADMIT}", f"legibility <= {LEG_ADMIT}"))
    df = df.rename(columns={"tracklet": "tracklet_id", "file": "frame"})
    df["sequence"] = "jersey2023-train"
    df["role"] = "jersey2023"
    cols = ["crop_path", "label", "source", "legibility", "sequence", "tracklet_id", "frame",
            "role", "label_provenance", "admitted", "reason"]
    out = OUT_ROOT / "jersey23_full_manifest.parquet"
    df[cols].to_parquet(out, index=False)
    n_tracks = df.loc[admitted, "tracklet_id"].nunique()
    print(f"jersey23_full manifest rows {len(df)} -> {out}")
    print(f"  admitted (glyph-only) {int(admitted.sum())} over {n_tracks} tracklets")
    print(f"  rejected (legibility <= {LEG_ADMIT} or unscored) {int((~admitted).sum())}")
    return out


def run_legibility(split: str, batch: int = 128) -> Path:
    """Score every materialised GT crop of ``split``; persist ``name, legibility``."""
    _assert_train_only(split)
    root = CROP_ROOT / split
    names = sorted(p.name for p in root.glob("*.jpg"))
    scores = score_paths([root / n for n in names], batch=batch)
    df = pd.DataFrame({"name": names, "legibility": scores})
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(leg_path(split), index=False)
    ok = np.isfinite(scores)
    print(f"{split}: {len(names)} crops, {int(ok.sum())} scored, "
          f"pass@{LEG_ADMIT} {int((scores[ok] > LEG_ADMIT).sum())} "
          f"({(scores[ok] > LEG_ADMIT).mean():.4f}) -> {leg_path(split)}")
    return leg_path(split)


def run_j2023_sample(n: int, seed: int = 0, batch: int = 128) -> Path:
    """Score a seeded sample of jersey-2023 train crops (the corpus is ~700k crops)."""
    gt = json.loads((J2023_ROOT / "train_gt.json").read_text(encoding="utf-8"))
    rng = random.Random(seed)
    rows: list[tuple[str, str, int]] = []
    for tid, label in sorted(gt.items()):
        d = J2023_ROOT / "images" / tid
        if not d.is_dir():
            continue
        rows.extend((tid, p.name, int(label)) for p in d.iterdir())
    rng.shuffle(rows)
    take = rows[:n]
    paths = [J2023_ROOT / "images" / t / f for t, f, _ in take]
    scores = score_paths(paths, batch=batch)
    df = pd.DataFrame({"tracklet": [t for t, _, _ in take], "file": [f for _, f, _ in take],
                       "label": [lab for _, _, lab in take], "legibility": scores})
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(leg_path("j2023_sample"), index=False)
    num = df[df["label"] >= 0]
    print(f"jersey-2023 train: {len(rows)} crops total, sampled {len(take)}")
    print(f"  numbered-tracklet crops {len(num)}, pass@{LEG_ADMIT} "
          f"{(num['legibility'] > LEG_ADMIT).mean():.4f}")
    print(f"  illegible(-1) crops {len(df) - len(num)}, "
          f"pass@{LEG_ADMIT} {(df[df['label'] < 0]['legibility'] > LEG_ADMIT).mean():.4f}")
    return leg_path("j2023_sample")


# --------------------------------------------------------------------------- manifest


def _gsr_rows(split: str) -> pd.DataFrame:
    """Planned GT crops of ``split`` joined to their legibility score."""
    crops = [c for c in plan_split(split) if c.role in ("player", "goalkeeper")]
    df = pd.DataFrame({
        "name": [c.name for c in crops], "sequence": [c.seq for c in crops],
        "tracklet_id": [c.track_id for c in crops], "frame": [c.frame for c in crops],
        "role": [c.role for c in crops], "label": [c.jersey for c in crops],
    })
    leg = pd.read_parquet(leg_path(split)) if leg_path(split).exists() else None
    if leg is None:
        raise SystemExit(f"missing {leg_path(split)} -- run --legibility {split} first")
    return df.merge(leg, on="name", how="left")


def build_manifest(split: str = "train") -> Path:
    """Write the glyph-only corpus manifest for the locally-assemblable sources."""
    _assert_train_only(split)
    df = _gsr_rows(split)
    df["source"] = f"gsr-{split}"
    df["crop_path"] = df["name"].map(lambda n: str(CROP_ROOT / split / n))
    df["label_provenance"] = "identity-carried (tracklet-level GT)"
    admitted = df["label"].notna() & (df["legibility"] > LEG_ADMIT)
    df["admitted"] = admitted
    df["reason"] = np.where(
        df["label"].isna(), "no GT number (unnamed pool -> Sid queue)",
        np.where(admitted, f"legibility > {LEG_ADMIT}", f"legibility <= {LEG_ADMIT}"))
    cols = ["crop_path", "label", "source", "legibility", "sequence", "tracklet_id", "frame",
            "role", "label_provenance", "admitted", "reason"]
    out = OUT_ROOT / "corpus_manifest.parquet"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df[cols].to_parquet(out, index=False)

    leaked = set(df["sequence"]) & set(split_sequences("validation") + split_sequences("test"))
    print(f"manifest rows {len(df)} -> {out}")
    n_tracks = df.loc[admitted, ["sequence", "tracklet_id"]].drop_duplicates().shape[0]
    print(f"  admitted (glyph-only) {int(admitted.sum())} over {n_tracks} tracklets")
    print(f"  rejected: carried-but-illegible {int((df['label'].notna() & ~admitted).sum())}, "
          f"no GT number {int(df['label'].isna().sum())}")
    print(f"  leakage check (validation/test sequences present): {sorted(leaked) or 'NONE'}")
    return out


# --------------------------------------------------------------------------- queue


@dataclass(frozen=True)
class QueueItem:
    """One tracklet Sid is asked about."""

    queue_id: str
    tier: str
    seq: str
    track_id: int
    n_frames: int
    role: str
    gt_number: str | None
    picks: list[Crop]


def _all_boxes(seq: str) -> dict[int, list[Crop]]:
    """Every player/GK GT box of a sequence, grouped by track (no crop-law subsampling)."""
    labels = json.loads((GSR_ROOT / seq / "Labels-GameState.json").read_text(encoding="utf-8"))
    frame_of = {im["image_id"]: Path(im["file_name"]).stem for im in labels["images"]}
    by: dict[int, list[Crop]] = {}
    for ann in labels["annotations"]:
        if ann.get("category_id") not in (1, 2) or "bbox_image" not in ann:
            continue
        b = ann["bbox_image"]
        attrs = ann.get("attributes") or {}
        by.setdefault(int(ann["track_id"]), []).append(Crop(
            seq=seq, frame=frame_of[ann["image_id"]], track_id=int(ann["track_id"]),
            xywh=(int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])),
            role=attrs.get("role") or "other", team=attrs.get("team"),
            jersey=attrs.get("jersey")))
    return by


def _pick_views(crops: list[Crop], k: int = SHEET_COLS * SHEET_ROWS) -> list[Crop]:
    """Pick ``k`` views: the tallest box in each of ``k`` equal time windows of the tracklet.

    Box height is a camera-distance proxy and needs no model. Stratifying by time first is what
    makes the abstention label defensible: the sheet shows the closest view in each of ``k`` windows
    spanning the whole tracklet, so "none" means "no number in the best view of any window", not
    "no number in one lucky close-up stretch".
    """
    ordered = sorted(crops, key=lambda c: c.frame)
    pool = [c for c in ordered if c.xywh[3] >= MIN_VIEW_H]
    if len(pool) < k:  # too few readable-size boxes: fall back to this tracklet's k tallest
        pool = sorted(sorted(ordered, key=lambda c: -c.xywh[3])[:k], key=lambda c: c.frame)
    if len(pool) <= k:
        return pool
    n = len(pool)
    return [max(pool[n * i // k: n * (i + 1) // k], key=lambda c: c.xywh[3]) for i in range(k)]


def _sheet(item: QueueItem) -> "np.ndarray":
    """Render one contact sheet: ``SHEET_ROWS x SHEET_COLS`` numbered cells."""
    import cv2  # noqa: PLC0415

    sheet = np.full((SHEET_ROWS * CELL_H, SHEET_COLS * CELL_W, 3), 32, np.uint8)
    by_frame: dict[str, list[Crop]] = {}
    for c in item.picks:
        by_frame.setdefault(c.frame, []).append(c)
    cell_of: dict[tuple[str, int], int] = {
        (c.frame, i): i for i, c in enumerate(item.picks)}
    del cell_of
    order = {id(c): i for i, c in enumerate(item.picks)}
    for frame, group in by_frame.items():
        img = cv2.imread(str(GSR_ROOT / item.seq / "img1" / f"{frame}.jpg"))
        if img is None:
            continue
        for c in group:
            i = order[id(c)]
            x, y, w, h = c.xywh
            sub = img[max(y, 0): y + h, max(x, 0): x + w]
            if sub.size == 0:
                continue
            scale = min((CELL_H - 24) / sub.shape[0], (CELL_W - 8) / sub.shape[1])
            sub = cv2.resize(sub, (max(1, int(sub.shape[1] * scale)),
                                   max(1, int(sub.shape[0] * scale))),
                             interpolation=cv2.INTER_CUBIC)
            r, cidx = divmod(i, SHEET_COLS)
            y0 = r * CELL_H + 22 + (CELL_H - 24 - sub.shape[0]) // 2
            x0 = cidx * CELL_W + (CELL_W - sub.shape[1]) // 2
            sheet[y0: y0 + sub.shape[0], x0: x0 + sub.shape[1]] = sub
    for i in range(len(item.picks)):
        r, cidx = divmod(i, SHEET_COLS)
        cv2.putText(sheet, str(i + 1), (cidx * CELL_W + 6, r * CELL_H + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
    return sheet


def _rank_tier_b(split: str, exclude: set[tuple[str, int]]) -> list[tuple[str, int, float]]:
    """Numbered tracklets ranked by legibility-gate uncertainty (mean |leg - 0.5| ascending).

    Model-derived *selection*, never a model-derived label: it only decides which tracklets a human
    is shown. The tracklets whose crops sit closest to the gate's decision boundary are the ones
    whose per-crop visibility labels most constrain a legibility retrain.
    """
    df = _gsr_rows(split)
    df = df[df["label"].notna() & df["legibility"].notna()]
    g = df.groupby(["sequence", "tracklet_id"])["legibility"]
    unc = (g.apply(lambda s: float(np.abs(s - 0.5).mean())).rename("unc").reset_index())
    unc = unc[~unc.apply(lambda r: (r["sequence"], int(r["tracklet_id"])) in exclude, axis=1)]
    return [(r.sequence, int(r.tracklet_id), float(r.unc))
            for r in unc.sort_values("unc").itertuples(index=False)]


def build_queue(split: str = "train", tier_b: int = TIER_B_N) -> Path:
    """Build the contact sheets + labelling manifest for an annotation evening."""
    _assert_train_only(split)
    import cv2  # noqa: PLC0415

    boxes = {seq: _all_boxes(seq) for seq in split_sequences(split)}
    items: list[QueueItem] = []
    unnamed: list[tuple[str, int, list[Crop]]] = []
    for seq, tracks in boxes.items():
        for tid, crops in tracks.items():
            if len(crops) < MIN_QUEUE_FRAMES or crops[0].role not in ("player", "goalkeeper"):
                continue
            if not any(c.jersey for c in crops):
                unnamed.append((seq, tid, crops))
    # Leverage = frames a label could actually be admitted on, i.e. readable-size boxes -- not raw
    # tracklet length. A 750-frame tracklet filmed at 45 px is unlabelable and un-trainable.
    unnamed.sort(key=lambda t: -sum(1 for c in t[2] if c.xywh[3] >= MIN_VIEW_H))
    for n, (seq, tid, crops) in enumerate(unnamed, 1):
        items.append(QueueItem(f"A{n:04d}", "A", seq, tid, len(crops), crops[0].role, None,
                               _pick_views(crops)))

    done = {(seq, tid) for _, _, seq, tid, *_ in
            ((i.queue_id, i.tier, i.seq, i.track_id) for i in items)}
    ranked = _rank_tier_b(split, done)
    for n, (seq, tid, _unc) in enumerate(ranked[:tier_b], 1):
        crops = boxes[seq][tid]
        gt = next((c.jersey for c in crops if c.jersey), None)
        items.append(QueueItem(f"B{n:04d}", "B", seq, tid, len(crops), crops[0].role, gt,
                               _pick_views(crops)))

    sheets = OUT_ROOT / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    rows = []
    for it in items:
        fname = f"{it.queue_id}_{it.seq}_t{it.track_id:04d}.jpg"
        if not (sheets / fname).exists():  # resumable: re-running only fills the gaps
            cv2.imwrite(str(sheets / fname), _sheet(it))
        heights = sorted(c.xywh[3] for c in it.picks)
        rows.append({
            "queue_id": it.queue_id, "tier": it.tier, "sheet": fname, "sequence": it.seq,
            "tracklet_id": it.track_id, "n_frames": it.n_frames,
            "n_readable_frames": sum(1 for c in boxes[it.seq][it.track_id]
                                     if c.xywh[3] >= MIN_VIEW_H),
            "role": it.role, "gt_number": it.gt_number or "", "n_cells": len(it.picks),
            "median_view_h": heights[len(heights) // 2], "max_view_h": heights[-1],
            "cell_frames": ";".join(c.frame for c in it.picks), "label": "", "note": "",
        })
    man = OUT_ROOT / "labelling_manifest.csv"
    with man.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    a = [r for r in rows if r["tier"] == "A"]
    b = [r for r in rows if r["tier"] == "B"]
    print(f"queue: {len(rows)} tracklets ({len(a)} tier A, {len(b)} tier B) -> {man}")
    print(f"  tier A frames behind the labels: {sum(r['n_frames'] for r in a)} "
          f"({sum(r['n_readable_frames'] for r in a)} at >= {MIN_VIEW_H} px)")
    print(f"  tier A sheets whose median view clears {MIN_VIEW_H} px: "
          f"{sum(1 for r in a if r['median_view_h'] >= MIN_VIEW_H)} / {len(a)}")
    print(f"  tier B per-crop cells offered:   {sum(r['n_cells'] for r in b)}")
    print(f"  sheets -> {sheets}")
    return man


# --------------------------------------------------------------------------- ingest


def _parse_a(label: str) -> tuple[str | None, str]:
    """Tier-A answer -> ``(number or None, verdict)``. Raises ``ValueError`` on a bad cell."""
    s = label.strip().lower()
    if s in ("none", "unsure", ""):
        return None, s or "blank"
    if not s.isdigit() or not 1 <= int(s) <= 99:  # noqa: PLR2004
        raise ValueError(f"tier-A label must be 1-99 / none / unsure, got {label!r}")
    return str(int(s)), "number"


def _parse_b(label: str, n_cells: int) -> list[int]:
    """Tier-B answer -> zero-based visible-cell indices. ``all`` / ``none`` accepted."""
    s = label.strip().lower()
    if s in ("", "unsure"):
        return []
    if s == "none":
        return []
    if s == "all":
        return list(range(n_cells))
    out = []
    for part in s.replace(" ", "").split(","):
        if not part.isdigit() or not 1 <= int(part) <= n_cells:
            raise ValueError(f"tier-B cell {part!r} out of range 1..{n_cells}")
        out.append(int(part) - 1)
    return out


def _admit(leg: dict[str, float], name: str, *, carried: bool) -> tuple[bool, str]:
    """Glyph-only admission for one crop: ``(admitted, reason)``.

    A human verdict about a crop he saw needs no gate. A number *carried* from one view to the rest
    of the tracklet does, and cannot be admitted at all until that crop has a legibility score --
    ``--legibility train`` only covers the 15-crop-per-tracklet re-ID crop set, so full-density
    crops come back ``unscored`` rather than silently rejected.
    """
    if not carried:
        return True, "human looked at this exact crop"
    score = leg.get(name)
    if score is None:
        return False, "unscored (needs a full-density legibility pass)"
    return (score > LEG_ADMIT, f"legibility {'>' if score > LEG_ADMIT else '<='} {LEG_ADMIT}")


def ingest(csv_path: Path, split: str = "train") -> Path:
    """Turn Sid's filled manifest into corpus rows.

    Propagation is exact and needs no ReID attachment: a GSR GT ``track_id`` already groups every
    crop of the tracklet, so a tier-A number propagates to the tracklet's whole box set (through the
    same legibility gate the carried GT numbers pass) and a tier-A ``none`` labels the shown cells
    as human-verified no-number crops. Tier-B answers are per-cell and propagate to nothing.
    """
    _assert_train_only(split)
    boxes = {seq: _all_boxes(seq) for seq in split_sequences(split)}
    leg = (pd.read_parquet(leg_path(split)).set_index("name")["legibility"].to_dict()
           if leg_path(split).exists() else {})
    rows = []
    stats = {"A_number": 0, "A_none": 0, "A_unsure": 0, "B_visible": 0, "B_hidden": 0, "B_unsure": 0}
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for rec in csv.DictReader(fh):
            crops = boxes.get(rec["sequence"], {}).get(int(rec["tracklet_id"]), [])
            cells = rec["cell_frames"].split(";") if rec["cell_frames"] else []
            if rec["tier"] == "A":
                num, verdict = _parse_a(rec["label"])
                if verdict in ("unsure", "blank"):
                    stats["A_unsure"] += 1
                    continue
                stats["A_number" if num else "A_none"] += 1
                # A number is an identity property -> carry it to the whole tracklet and let the
                # legibility gate admit only the crops that show the glyph. A "none" is a statement
                # about the 12 views he was shown, and is not carried past them.
                targets = crops if num else [c for c in crops if c.frame in set(cells)]
                for c in targets:
                    ok, why = _admit(leg, c.name, carried=num is not None)
                    rows.append({
                        "crop_name": c.name, "sequence": c.seq, "tracklet_id": c.track_id,
                        "frame": c.frame, "label": num or "<none>", "source": "sid-tierA",
                        "label_provenance": ("human tracklet number (glyph-gated)" if num
                                             else "human no-number verdict on shown views"),
                        "legibility": leg.get(c.name, float("nan")),
                        "admitted": ok, "reason": why,
                    })
            else:
                if rec["label"].strip().lower() in ("", "unsure"):
                    stats["B_unsure"] += 1
                    continue
                vis = set(_parse_b(rec["label"], int(rec["n_cells"])))
                by_frame = {c.frame: c for c in crops}
                for i, frame in enumerate(cells):
                    c = by_frame.get(frame)
                    if c is None:
                        continue
                    seen = i in vis
                    stats["B_visible" if seen else "B_hidden"] += 1
                    rows.append({
                        "crop_name": c.name, "sequence": c.seq, "tracklet_id": c.track_id,
                        "frame": c.frame,
                        "label": (rec["gt_number"] or "?") if seen else "<none>",
                        "source": "sid-tierB",
                        "label_provenance": "human per-crop glyph visibility",
                        "legibility": leg.get(c.name, float("nan")), "admitted": True,
                        "reason": "human looked at this exact crop",
                    })
    out = OUT_ROOT / "sid_labels.parquet"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"ingested {csv_path} -> {out}: {len(rows)} crop rows; {stats}")
    return out


# --------------------------------------------------------------------------- self-check


def _demo() -> None:
    """Assert the parsers, the view picker and the split guard."""
    assert _parse_a("7") == ("7", "number")
    assert _parse_a(" 07 ") == ("7", "number")
    assert _parse_a("none") == (None, "none")
    assert _parse_a("") == (None, "blank")
    for bad in ("0", "100", "abc", "-1"):
        try:
            _parse_a(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted bad tier-A label {bad!r}")
    assert _parse_b("1,3,12", 12) == [0, 2, 11]
    assert _parse_b("all", 3) == [0, 1, 2]
    assert _parse_b("none", 3) == []
    try:
        _parse_b("13", 12)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted out-of-range tier-B cell")

    k = SHEET_COLS * SHEET_ROWS
    # Heights rise with time (40..439), so only the tail clears MIN_VIEW_H and the picks must be
    # k time-ordered, readable-size views ending at the tallest box of all.
    ramp = [Crop("S", f"{i:06d}", 1, (0, 0, 40, 40 + i), "player", "left", None) for i in range(400)]
    picks = _pick_views(ramp)
    assert len(picks) == k, len(picks)
    assert [p.frame for p in picks] == sorted(p.frame for p in picks), "views must be time-ordered"
    assert min(p.xywh[3] for p in picks) >= MIN_VIEW_H, "views must clear the readable-size floor"
    assert picks[-1] is ramp[-1], "the tallest box must always be shown"
    # Alternating heights: every window holds a tall box, so every pick must be a tall one.
    alt = [Crop("S", f"{i:06d}", 1, (0, 0, 40, MIN_VIEW_H + (100 if i % 2 else 0)), "player",
                None, None) for i in range(400)]
    assert {p.xywh[3] for p in _pick_views(alt)} == {MIN_VIEW_H + 100}, "tallest per window"
    # Whole tracklet below the floor -> fall back to its k tallest, still time-ordered.
    tiny = [Crop("S", f"{i:06d}", 1, (0, 0, 40, 40 + i % 50), "player", None, None)
            for i in range(400)]
    fb = _pick_views(tiny)
    assert len(fb) == k and max(p.xywh[3] for p in fb) == 89, fb  # noqa: PLR2004
    assert [p.frame for p in fb] == sorted(p.frame for p in fb), "fallback must stay time-ordered"
    small = [Crop("S", f"{i:06d}", 1, (0, 0, 40, 90), "player", None, None) for i in range(5)]
    assert _pick_views(small) == small

    leg = {"hi.jpg": 0.9, "lo.jpg": 0.1}
    assert _admit(leg, "hi.jpg", carried=True)[0] is True
    assert _admit(leg, "lo.jpg", carried=True)[0] is False
    assert _admit(leg, "gone.jpg", carried=True) == (
        False, "unscored (needs a full-density legibility pass)")
    assert _admit(leg, "gone.jpg", carried=False)[0] is True, "a human verdict needs no gate"

    for split in ("validation", "test"):
        try:
            _assert_train_only(split)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"split guard let {split} through")
    print("gsr_w3_corpus demo OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legibility", default=None, help="score a split's GT crops (GPU, short)")
    ap.add_argument("--j2023-sample", type=int, default=0, help="score N jersey-2023 train crops")
    ap.add_argument("--full-jersey23", action="store_true",
                    help="score every jersey-2023 numbered-tracklet crop (checkpointed, GPU)")
    ap.add_argument("--j2023-full-manifest", action="store_true",
                    help="write the jersey23_full_manifest.parquet from a completed full pass")
    ap.add_argument("--manifest", action="store_true", help="write the glyph-only corpus manifest")
    ap.add_argument("--queue", action="store_true", help="build contact sheets + labelling CSV")
    ap.add_argument("--tier-b", type=int, default=TIER_B_N, help="tier-B queue size")
    ap.add_argument("--ingest", default=None, help="consume a filled labelling manifest CSV")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        _demo()
    if a.legibility:
        run_legibility(a.legibility)
    if a.j2023_sample:
        run_j2023_sample(a.j2023_sample)
    if a.full_jersey23:
        run_j2023_full()
    if a.j2023_full_manifest:
        build_j2023_full_manifest()
    if a.manifest:
        build_manifest()
    if a.queue:
        build_queue(tier_b=a.tier_b)
    if a.ingest:
        ingest(Path(a.ingest))
