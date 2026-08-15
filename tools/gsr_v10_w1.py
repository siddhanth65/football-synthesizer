"""v10-W1: open-weights VLM jersey reading, scored against Sid's 499-tracklet human annotation.

The shipped PARSeq + legibility chain abstains on most of the GSR-train unnamed pool, and the W3
human audit (kb ``v8-w3-007``) says 94.8% of that pool is *genuinely* unreadable at contact-sheet
resolution by a human. This script asks whether a modern open-weights VLM beats that read, on the
only per-tracklet human ground truth that exists.

Deliberately dependency-light (torch + transformers + pandas + opencv + PIL) and import-free of this
repo: it is copied to the cluster and run there, where the repo checkout may be stale.

Stages::

    python gsr_v10_w1.py --stage cells  --manifest ... --gsr-root ... --out-root ...   # CPU
    python gsr_v10_w1.py --stage read   --regime sheet|multi12|zoom1|percrop ...       # GPU
    python gsr_v10_w1.py --stage score  --manifest ... --reads r1.parquet r2.parquet   # CPU

Split discipline: every stage refuses any sequence outside the train list handed to ``--train-list``.
No DEV/TEST/challenge frame is opened at any point.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

#: Contact-sheet geometry of the annotation the human actually saw (tools/gsr_w3_corpus.py).
SHEET_COLS, SHEET_ROWS = 4, 3

#: Tracklet-level prompt. One JSON line out; ``null`` is an explicitly wanted answer.
PROMPT_TRACKLET = """\
{intro}

Read the jersey number printed on this player's shirt.

Rules:
- Give a number ONLY if you can actually see the printed digits in at least one view.
- Jersey numbers are 1 to 99. Never infer a number from the player's role, position, kit colour,
  body shape or anything other than digits you can see.
- If no digits are legible in any view, answer null. Answering null is correct and expected: in
  this dataset most players' numbers are not visible.

Reply with one line of JSON and nothing else:
{{"number": <integer 1-99, or null>}}"""

#: Per-regime opening sentence. Everything after it is byte-identical across regimes.
INTRO = {
    "sheet": (
        "This image is a 4x3 contact sheet of 12 cropped views of ONE football (soccer) player, "
        "taken at 12 different moments of the same 30-second broadcast clip. The small yellow "
        "digits 1-12 in the top-left corner of each cell are cell indices, NOT jersey numbers."),
    "multi12": (
        "These images are 12 cropped views of ONE football (soccer) player, taken at 12 different "
        "moments of the same 30-second broadcast clip."),
    "zoom1": (
        "This image is the closest available cropped view of ONE football (soccer) player from a "
        "30-second broadcast clip, enlarged."),
}

#: Per-crop prompt for the tier-B visibility read (one view, one verdict).
PROMPT_PERCROP = """\
This image is one cropped view of a football (soccer) player from a broadcast match, enlarged.

Read the jersey number printed on this player's shirt.

Rules:
- Give a number ONLY if you can actually see the printed digits in THIS view.
- Jersey numbers are 1 to 99. Never infer a number from the player's role, position, kit colour,
  body shape or anything other than digits you can see.
- If the digits are not legible here, answer null. Answering null is correct and expected: in this
  dataset most views do not show the number.

Reply with one line of JSON and nothing else:
{"number": <integer 1-99, or null>}"""

_JSON_NUM = re.compile(r'"number"\s*:\s*(?:"?(\d+)"?|null|None)', re.I)
_BARE_NUM = re.compile(r"\b(\d+)\b")


# --------------------------------------------------------------------------- data


def load_queue(manifest: Path, train_list: Path) -> pd.DataFrame:
    """Load the annotation queue and refuse anything outside the train split.

    Args:
        manifest: ``labelling_manifest.csv`` (499 answered rows).
        train_list: One sequence id per line -- the only sequences that may be opened.

    Returns:
        The manifest with ``cells`` exploded to a list of frame ids.

    Raises:
        SystemExit: If any row names a sequence outside ``train_list``.
    """
    df = pd.read_csv(manifest)
    allowed = {s.strip() for s in train_list.read_text(encoding="utf-8").split() if s.strip()}
    bad = sorted(set(df["sequence"]) - allowed)
    if bad:
        raise SystemExit(f"non-train sequences in manifest: {bad}")
    df["cells"] = df["cell_frames"].map(lambda s: str(s).split(";"))
    return df


def cut_cells(df: pd.DataFrame, gsr_root: Path, out_root: Path) -> dict:
    """Cut every contact-sheet cell to its own JPEG (the exact GT box, no resize).

    Args:
        df: Queue from :func:`load_queue`.
        gsr_root: ``gamestate-2024`` root holding ``SNGS-xxx/img1/*.jpg``.
        out_root: Destination root; cells land in ``out_root/cells/<queue_id>/<i>.jpg``.

    Returns:
        ``{"rows": n, "cells": n, "missing": n, "seconds": s}``.
    """
    import cv2  # noqa: PLC0415

    t0 = time.time()
    boxes: dict[str, dict[tuple[str, int], tuple[int, int, int, int]]] = {}
    n_cell = n_missing = 0
    meta: list[dict] = []
    for row in df.itertuples():
        seq = row.sequence
        if seq not in boxes:
            boxes[seq] = _boxes_of(gsr_root / seq / "Labels-GameState.json")
        dst = out_root / "cells" / row.queue_id
        dst.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(row.cells):
            out = dst / f"{i:02d}.jpg"
            box = boxes[seq].get((frame, int(row.tracklet_id)))
            if box is None:
                n_missing += 1
                continue
            if not out.exists():
                img = cv2.imread(str(gsr_root / seq / "img1" / f"{frame}.jpg"))
                if img is None:
                    n_missing += 1
                    continue
                x, y, w, h = box
                sub = img[max(y, 0): y + h, max(x, 0): x + w]
                if sub.size == 0:
                    n_missing += 1
                    continue
                cv2.imwrite(str(out), sub, [int(cv2.IMWRITE_JPEG_QUALITY), 97])
            meta.append({"queue_id": row.queue_id, "cell": i + 1, "sequence": seq,
                         "tracklet_id": int(row.tracklet_id), "frame": frame,
                         "path": str(out), "box_h": box[3], "box_w": box[2]})
            n_cell += 1
    pd.DataFrame(meta).to_parquet(out_root / "cells.parquet", index=False)
    return {"rows": len(df), "cells": n_cell, "missing": n_missing,
            "seconds": round(time.time() - t0, 1)}


def _boxes_of(labels_json: Path) -> dict[tuple[str, int], tuple[int, int, int, int]]:
    """Map ``(frame stem, track id) -> xywh`` for every player/GK GT box of a sequence."""
    labels = json.loads(labels_json.read_text(encoding="utf-8"))
    frame_of = {im["image_id"]: Path(im["file_name"]).stem for im in labels["images"]}
    out: dict[tuple[str, int], tuple[int, int, int, int]] = {}
    for ann in labels["annotations"]:
        if ann.get("category_id") not in (1, 2) or "bbox_image" not in ann:
            continue
        b = ann["bbox_image"]
        out[(frame_of[ann["image_id"]], int(ann["track_id"]))] = (
            int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"]))
    return out


# --------------------------------------------------------------------------- regimes


def _upscale(path: str, target_h: int, max_scale: float):  # noqa: ANN202
    """Open a crop and Lanczos-enlarge it to ``target_h`` px tall, never beyond ``max_scale``."""
    from PIL import Image  # noqa: PLC0415

    img = Image.open(path).convert("RGB")
    scale = min(max(target_h / max(img.height, 1), 1.0), max_scale)
    if scale > 1.0:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                         Image.LANCZOS)
    return img


def build_items(df: pd.DataFrame, cells: pd.DataFrame, regime: str, sheet_root: Path,
                out_root: Path) -> list[dict]:
    """Build ``{"key", "images", "prompt"}`` work items for one regime.

    Args:
        df: Queue from :func:`load_queue`.
        cells: ``cells.parquet`` written by :func:`cut_cells`.
        regime: ``sheet`` (the exact human contact sheet), ``multi12`` (the 12 cells as 12 separate
            images, each Lanczos-enlarged to 256 px tall), ``zoom1`` (the tallest cell alone,
            Lanczos-enlarged 4x capped at 512 px tall), or ``percrop`` (every tier-B cell alone,
            same enlargement as ``zoom1``).
        sheet_root: Folder holding the 499 annotation contact sheets.
        out_root: Root holding ``cells/``.

    Returns:
        One work item per unit of evidence (tracklet, or crop for ``percrop``).
    """
    del out_root
    by_q: dict[str, pd.DataFrame] = dict(tuple(cells.groupby("queue_id")))
    items: list[dict] = []
    for row in df.itertuples():
        got = by_q.get(row.queue_id)
        if got is None or got.empty:
            continue
        got = got.sort_values("cell")
        if regime == "sheet":
            items.append({"key": row.queue_id, "cell": 0,
                          "images": [str(sheet_root / row.sheet)],
                          "prompt": PROMPT_TRACKLET.format(intro=INTRO["sheet"])})
        elif regime == "multi12":
            items.append({"key": row.queue_id, "cell": 0, "images": list(got["path"]),
                          "prompt": PROMPT_TRACKLET.format(intro=INTRO["multi12"])})
        elif regime == "zoom1":
            best = got.loc[got["box_h"].idxmax()]
            items.append({"key": row.queue_id, "cell": int(best["cell"]),
                          "images": [best["path"]],
                          "prompt": PROMPT_TRACKLET.format(intro=INTRO["zoom1"])})
        elif regime == "percrop":
            if row.tier != "B":
                continue
            items.extend({"key": row.queue_id, "cell": int(c.cell), "images": [c.path],
                          "prompt": PROMPT_PERCROP} for c in got.itertuples())
        else:
            raise SystemExit(f"unknown regime {regime!r}")
    return items


def _open_images(paths: list[str], regime: str) -> list:
    """Open a work item's images with the enlargement its regime declares."""
    from PIL import Image  # noqa: PLC0415

    if regime == "sheet":
        return [Image.open(p).convert("RGB") for p in paths]
    if regime == "multi12":
        return [_upscale(p, 256, 4.0) for p in paths]
    return [_upscale(p, 512, 4.0) for p in paths]  # zoom1 / percrop


# --------------------------------------------------------------------------- inference


def parse_read(text: str) -> int | None:
    """Parse the model's reply into a jersey number or ``None`` (abstain).

    Deliberately generous on format and strict on content: a reply that names no 1-99 integer is an
    abstain, and a number outside 1-99 is an abstain (never a silent clamp).
    """
    m = _JSON_NUM.search(text)
    if m:
        if m.group(1) is None:
            return None
        n = int(m.group(1))
        return n if 1 <= n <= 99 else None
    if re.search(r"\bnull\b|\bnone\b|not (visible|legible|readable)", text, re.I):
        return None
    m2 = _BARE_NUM.search(text)
    if m2:
        n = int(m2.group(1))
        return n if 1 <= n <= 99 else None
    return None


def run_reads(items: list[dict], model_id: str, regime: str, out: Path, *, batch: int,
              max_new: int = 24) -> dict:
    """Run the VLM over the work items and write one row per item.

    Args:
        items: From :func:`build_items`.
        model_id: HuggingFace model id (open weights).
        regime: Regime name, recorded in the output and driving image enlargement.
        out: Destination parquet.
        batch: Items per forward pass.
        max_new: Generation cap; the registered reply is one short JSON line.

    Returns:
        ``{"items": n, "seconds": s, "emitted": n}``.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

    proc = AutoProcessor.from_pretrained(model_id, min_pixels=64 * 28 * 28,
                                         max_pixels=1024 * 28 * 28)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda:0").eval()

    rows: list[dict] = []
    t0 = time.time()
    for start in range(0, len(items), batch):
        chunk = items[start: start + batch]
        texts, images = [], []
        for it in chunk:
            imgs = _open_images(it["images"], regime)
            msg = [{"role": "user", "content": [*({"type": "image"} for _ in imgs),
                                                {"type": "text", "text": it["prompt"]}]}]
            texts.append(proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
            images.extend(imgs)
        enc = proc(text=texts, images=images, return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 return_dict_in_generate=True, output_scores=True)
        seqs = gen.sequences[:, enc["input_ids"].shape[1]:]
        lp = _mean_logprob(gen.scores, seqs, proc.tokenizer.pad_token_id)
        for j, it in enumerate(chunk):
            txt = proc.tokenizer.decode(seqs[j], skip_special_tokens=True).strip()
            rows.append({"key": it["key"], "cell": it["cell"], "regime": regime,
                         "model": model_id, "raw": txt, "read": parse_read(txt),
                         "mean_logprob": float(lp[j]), "n_images": len(it["images"])})
        if start % (batch * 20) == 0:
            done = start + len(chunk)
            print(f"  {done}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df["read"] = df["read"].astype("Int64")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return {"items": len(rows), "seconds": round(time.time() - t0, 1),
            "emitted": int(df["read"].notna().sum())}


def _mean_logprob(scores, seqs, pad_id: int | None) -> np.ndarray:
    """Mean log-probability of each generated sequence's non-pad tokens."""
    import torch  # noqa: PLC0415

    tot = torch.zeros(seqs.shape[0], dtype=torch.float32)
    cnt = torch.zeros(seqs.shape[0], dtype=torch.float32)
    for t, step in enumerate(scores):
        tok = seqs[:, t]
        lp = torch.log_softmax(step.float(), dim=-1).gather(1, tok[:, None])[:, 0].cpu()
        live = (tok != pad_id).cpu() if pad_id is not None else torch.ones_like(lp, dtype=torch.bool)
        tot += lp * live
        cnt += live.float()
    return (tot / cnt.clamp(min=1)).numpy()


# --------------------------------------------------------------------------- scoring


def score_tier_a(df: pd.DataFrame, reads: pd.DataFrame, drop_noted: bool = True) -> dict:
    """Confusion of one regime's tracklet reads against the human tier-A verdicts.

    Args:
        df: Queue from :func:`load_queue`.
        reads: One regime's read rows (``key`` = queue id).
        drop_noted: Exclude the two numbered rows whose ``note`` holds a stray ``u`` keypress
            (kb ``v8-w3-012``), giving the registered 13-number confident pool.

    Returns:
        Counts and the two registered rates.
    """
    a = df[df["tier"] == "A"].copy()
    a["gt"] = pd.to_numeric(a["label"], errors="coerce")
    noted = a["note"].astype(str).str.strip().str.lower().eq("u")
    numbered = a[a["gt"].notna() & (~noted if drop_noted else True)]
    nones = a[a["label"] == "none"]
    r = reads.set_index("key")["read"]
    num_read = numbered["queue_id"].map(r)
    hit = int((num_read == numbered["gt"]).sum())
    emitted = int(num_read.notna().sum())
    none_read = nones["queue_id"].map(r)
    fp = int(none_read.notna().sum())
    return {"n_numbered": len(numbered), "emitted_on_numbered": emitted, "correct": hit,
            "precision": round(hit / emitted, 4) if emitted else float("nan"),
            "recall": round(hit / len(numbered), 4) if len(numbered) else float("nan"),
            "n_none": len(nones), "fp_on_none": fp,
            "fp_rate": round(fp / len(nones), 4) if len(nones) else float("nan")}


def score_tier_b_tracklet(df: pd.DataFrame, reads: pd.DataFrame) -> dict:
    """Tracklet-level accuracy on tier B, where SoccerNet's own GT number is known."""
    b = df[df["tier"] == "B"].copy()
    b["gt"] = pd.to_numeric(b["gt_number"], errors="coerce")
    r = reads.set_index("key")["read"]
    got = b["queue_id"].map(r)
    emitted = int(got.notna().sum())
    hit = int((got == b["gt"]).sum())
    return {"n": len(b), "emitted": emitted, "correct": hit,
            "precision": round(hit / emitted, 4) if emitted else float("nan"),
            "recall": round(hit / len(b), 4) if len(b) else float("nan")}


def score_tier_b_percrop(df: pd.DataFrame, reads: pd.DataFrame) -> dict:
    """Per-crop agreement against Sid's cell-level visibility labels (kb ``v8-w3-009``)."""
    b = df[df["tier"] == "B"].copy()
    b["gt"] = pd.to_numeric(b["gt_number"], errors="coerce")
    vis: dict[tuple[str, int], bool] = {}
    gt_of: dict[str, float] = {}
    for row in b.itertuples():
        gt_of[row.queue_id] = row.gt
        lab = str(row.label).strip().lower()
        cells = (set(range(1, 13)) if lab == "all" else set() if lab in ("none", "unsure", "nan")
                 else {int(x) for x in lab.split(",") if x.strip().isdigit()})
        for c in range(1, len(row.cells) + 1):
            vis[(row.queue_id, c)] = c in cells
    out = {"visible": 0, "vis_emitted": 0, "vis_correct": 0,
           "hidden": 0, "hid_emitted": 0, "hid_correct": 0}
    for row in reads.itertuples():
        k = (row.key, int(row.cell))
        if k not in vis:
            continue
        tag = "vis" if vis[k] else "hid"
        out["visible" if vis[k] else "hidden"] += 1
        if pd.notna(row.read):
            out[f"{tag}_emitted"] += 1
            if float(row.read) == gt_of[row.key]:
                out[f"{tag}_correct"] += 1
    out["vis_precision"] = (round(out["vis_correct"] / out["vis_emitted"], 4)
                            if out["vis_emitted"] else float("nan"))
    out["vis_recall"] = (round(out["vis_correct"] / out["visible"], 4)
                         if out["visible"] else float("nan"))
    out["hid_fp_rate"] = (round(out["hid_emitted"] / out["hidden"], 4)
                          if out["hidden"] else float("nan"))
    return out


# --------------------------------------------------------------------------- cli


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=("cells", "read", "score"))
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--train-list", type=Path, required=True)
    ap.add_argument("--gsr-root", type=Path)
    ap.add_argument("--sheet-root", type=Path)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--regime", choices=("sheet", "multi12", "zoom1", "percrop"))
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test cap on work items")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--cells", type=Path, help="override cells.parquet (e.g. the torso RoI set)")
    ap.add_argument("--reads", type=Path, nargs="*", default=())
    args = ap.parse_args()

    df = load_queue(args.manifest, args.train_list)
    if args.stage == "cells":
        print(json.dumps(cut_cells(df, args.gsr_root, args.out_root)))
        return
    if args.stage == "read":
        cells = pd.read_parquet(args.cells or (args.out_root / "cells.parquet"))
        items = build_items(df, cells, args.regime, args.sheet_root, args.out_root)
        if args.limit:
            items = items[: args.limit]
        print(f"regime {args.regime}: {len(items)} items", flush=True)
        out = args.out or (args.out_root / f"reads_{args.regime}.parquet")
        print(json.dumps(run_reads(items, args.model, args.regime, out, batch=args.batch)))
        return
    for p in args.reads:
        reads = pd.read_parquet(p)
        regime = str(reads["regime"].iloc[0])
        print(f"== {p.name} regime={regime} model={reads['model'].iloc[0]}")
        if regime == "percrop":
            print("  tierB percrop:", json.dumps(score_tier_b_percrop(df, reads)))
        else:
            print("  tierA (13):", json.dumps(score_tier_a(df, reads, True)))
            print("  tierA (15):", json.dumps(score_tier_a(df, reads, False)))
            print("  tierB trk:", json.dumps(score_tier_b_tracklet(df, reads)))


if __name__ == "__main__":
    main()
