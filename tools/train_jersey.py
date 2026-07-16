"""Train / evaluate the jersey-number recognizer on SoccerNet jersey-2023 (resumable).

Single 100-way crop classifier (``generator.jersey_id``); per-tracklet decisions by
confidence-weighted voting. Weak crop labels (each crop inherits its tracklet's number even when the
number is not visible) are absorbed by label smoothing at train time and voting at test time.

Resumable by design -- Claude Code restarts orphan detached processes, so training runs in
wall-clock *slices*: each invocation resumes from ``outputs/jersey/ckpt.pt``, trains until
``--max-minutes`` elapses (checkpointing every epoch), then exits. Re-run until it prints
``TARGET REACHED``. State lives on disk (the checkpoint's epoch counter).

Usage::

    python tools/train_jersey.py train --max-minutes 30 --target-epochs 20
    python tools/train_jersey.py eval --split test          # tracklet accuracy on held-out test
    python tools/train_jersey.py tune                       # pick min_conf on the train-val split
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator import jersey_id as J  # noqa: E402

DATA = Path("data/soccernet/jersey-2023")
OUT = Path("outputs/jersey")
CKPT = OUT / "ckpt.pt"
LOG = OUT / "train.log"
VAL_FRAC = 0.10
SEED = 0


def _log(msg: str) -> None:
    """Print (flushed) and append to the persistent training log so slices survive orphaning."""
    print(msg, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def _tracklets(split: str) -> list[tuple[str, int, list[Path]]]:
    """List ``(tracklet_id, class, crop_paths)`` for a split (``train`` or ``test``)."""
    gt = J.load_gt(DATA / split / f"{split}_gt.json")
    root = DATA / split / "images"
    out = []
    for tid, label in gt.items():
        crops = sorted((root / tid).glob("*.jpg"))
        if crops:
            out.append((tid, J.to_class(label), crops))
    return out


def _split_train_val(items: list) -> tuple[list, list]:
    """Deterministic 90/10 tracklet split for monitoring + threshold tuning (test untouched)."""
    rng = random.Random(SEED)
    shuffled = items[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * VAL_FRAC)
    return shuffled[n_val:], shuffled[:n_val]


class _CropSampler(Dataset):
    """One random crop per tracklet-slot per epoch (fresh draw each ``__getitem__``).

    ``len`` is ``n_tracklets * crops_per_tracklet``; slot ``i`` maps to a fixed tracklet and draws a
    random crop from it, so an epoch sees ``crops_per_tracklet`` random crops of every tracklet and
    successive epochs see different crops -- coverage over 733k crops without a giant epoch.
    """

    def __init__(self, items: list[tuple[str, int, list[Path]]], per_tracklet: int) -> None:
        self.items = items
        self.per = per_tracklet
        self.tf = J.build_transform(train=True)

    def __len__(self) -> int:
        return len(self.items) * self.per

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        _, cls, crops = self.items[i // self.per]
        for _ in range(4):  # retry a few unreadable crops before giving up on a zero tensor
            try:
                img = Image.open(random.choice(crops)).convert("RGB")
                return self.tf(img), cls
            except (OSError, ValueError):
                continue
        return torch.zeros(3, J.INPUT_H, J.INPUT_W), cls


def _crop_acc(model: nn.Module, items: list, device: str, per: int = 8) -> float:
    """Quick per-crop top-1 accuracy on a sample of crops (training-progress signal, not the metric)."""
    tf = J.build_transform(train=False)
    xs, ys = [], []
    for _, cls, crops in items:
        for p in random.sample(crops, min(per, len(crops))):
            try:
                xs.append(tf(Image.open(p).convert("RGB")))
                ys.append(cls)
            except (OSError, ValueError):
                continue
    if not xs:
        return 0.0
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(xs), 256):
            xb = torch.stack(xs[i : i + 256]).to(device)
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                pred = model(xb).float().argmax(1).cpu().numpy()
            correct += int((pred == np.array(ys[i : i + 256])).sum())
    model.train()
    return correct / len(xs)


def train(args: argparse.Namespace) -> None:
    """Train in a wall-clock slice, resuming from and checkpointing to ``outputs/jersey/ckpt.pt``."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = J.build_model(pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device) if device == "cuda" else None
    start_epoch = 0
    if CKPT.exists():
        ck = torch.load(CKPT, map_location=device)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["opt_state"])
        start_epoch = ck["epoch"]
        _log(f"resumed from epoch {start_epoch} (ckpt {CKPT})")
    if start_epoch >= args.target_epochs:
        _log(f"TARGET REACHED ({start_epoch}/{args.target_epochs} epochs)")
        return

    tr, val = _split_train_val(_tracklets("train"))
    _log(f"train tracklets {len(tr)}, val {len(val)}, device {device}")
    loader = DataLoader(
        _CropSampler(tr, args.crops_per_tracklet), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device == "cuda"), persistent_workers=args.workers > 0,
        drop_last=True,
    )
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    model.train()
    deadline = time.time() + args.max_minutes * 60
    epoch = start_epoch
    while epoch < args.target_epochs and time.time() < deadline:
        t0, tot, n = time.time(), 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                loss = lossf(model(xb), yb)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        epoch += 1
        acc = _crop_acc(model, val, device)
        torch.save({"model_state": model.state_dict(), "opt_state": opt.state_dict(),
                    "epoch": epoch}, CKPT)
        _log(f"epoch {epoch}/{args.target_epochs}  loss {tot / max(n, 1):.4f}  "
             f"val_crop_acc {acc:.3f}  {time.time() - t0:.0f}s")
    if epoch >= args.target_epochs:
        _log(f"TARGET REACHED ({epoch}/{args.target_epochs} epochs)")
    else:
        _log(f"SLICE DONE (epoch {epoch}/{args.target_epochs}) -- re-run to continue")


def _tracklet_means(rec: J.JerseyRecognizer, items: list, eval_crops: int) -> list[tuple[int, np.ndarray]]:
    """Pooled ``(true_class, weighted_mean_vector)`` per tracklet (cache for threshold sweeps)."""
    means = []
    for k, (_, cls, crops) in enumerate(items):
        paths = random.sample(crops, min(eval_crops, len(crops)))
        means.append((cls, J.tracklet_mean(rec.crop_probs(paths))))
        if (k + 1) % 200 == 0:
            _log(f"  scored {k + 1}/{len(items)} tracklets")
    return means


def _score(means: list[tuple[int, np.ndarray]], min_conf: float) -> dict:
    """Tracklet-level metrics at a given ``min_conf`` (accuracy, legibility P/R, confusions)."""
    correct = legible_tp = legible_fp = legible_fn = 0
    num_total = num_correct = 0
    confusions: Counter = Counter()
    for true_cls, mean_p in means:
        pred_label, _ = J.decide(mean_p, min_conf=min_conf)
        true_label = J.from_class(true_cls)
        if pred_label == true_label:
            correct += 1
        # legibility: predicting a number vs -1
        if true_label != -1 and pred_label != -1:
            legible_tp += 1
        elif true_label == -1 and pred_label != -1:
            legible_fp += 1
        elif true_label != -1 and pred_label == -1:
            legible_fn += 1
        if true_label != -1:
            num_total += 1
            if pred_label == true_label:
                num_correct += 1
            elif pred_label != -1:
                confusions[(true_label, pred_label)] += 1
    n = len(means)
    prec = legible_tp / (legible_tp + legible_fp) if (legible_tp + legible_fp) else 0.0
    rec = legible_tp / (legible_tp + legible_fn) if (legible_tp + legible_fn) else 0.0
    return {
        "n": n, "accuracy": correct / n if n else 0.0,
        "numbered_acc": num_correct / num_total if num_total else 0.0,
        "legibility_precision": prec, "legibility_recall": rec,
        "top_confusions": confusions.most_common(10),
    }


def tune(args: argparse.Namespace) -> None:
    """Sweep ``min_conf`` on the train-val split; print the accuracy-maximizing threshold."""
    rec = J.JerseyRecognizer.from_checkpoint(CKPT)
    _, val = _split_train_val(_tracklets("train"))
    _log(f"tuning min_conf on {len(val)} val tracklets ...")
    means = _tracklet_means(rec, val, args.eval_crops)
    best = max((round(t, 2) for t in np.arange(0.05, 0.81, 0.05)),
               key=lambda t: _score(means, t)["accuracy"])
    for t in np.arange(0.05, 0.81, 0.05):
        s = _score(means, round(t, 2))
        _log(f"  min_conf {t:.2f}  val_acc {s['accuracy']:.3f}  "
             f"legP {s['legibility_precision']:.3f}  legR {s['legibility_recall']:.3f}")
    _log(f"BEST min_conf {best} (val_acc {_score(means, best)['accuracy']:.3f})")


def evaluate(args: argparse.Namespace) -> None:
    """Tracklet-level accuracy on a held-out split (official metric incl. the ``-1`` class)."""
    rec = J.JerseyRecognizer.from_checkpoint(CKPT, min_conf=args.min_conf)
    items = _tracklets(args.split)
    _log(f"evaluating {len(items)} {args.split} tracklets at min_conf {args.min_conf} ...")
    means = _tracklet_means(rec, items, args.eval_crops)
    s = _score(means, args.min_conf)
    _log(f"== {args.split} tracklet accuracy {s['accuracy']:.4f} (n={s['n']}) ==")
    _log(f"   numbered-only accuracy {s['numbered_acc']:.4f}")
    _log(f"   legibility precision {s['legibility_precision']:.4f}  "
         f"recall {s['legibility_recall']:.4f}")
    _log(f"   top confusions (true->pred): {s['top_confusions']}")
    (OUT / f"eval_{args.split}.json").write_text(
        json.dumps({k: v for k, v in s.items() if k != "top_confusions"}
                   | {"top_confusions": [list(k) + [c] for k, c in s["top_confusions"]],
                      "min_conf": args.min_conf}, indent=2), encoding="utf-8")


def main() -> None:
    """CLI: ``train`` (resumable slices), ``tune`` (pick min_conf), ``eval`` (held-out metric)."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("train")
    t.add_argument("--max-minutes", type=float, default=30.0)
    t.add_argument("--target-epochs", type=int, default=20)
    t.add_argument("--crops-per-tracklet", type=int, default=24)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--workers", type=int, default=4)
    tn = sub.add_parser("tune")
    tn.add_argument("--eval-crops", type=int, default=48)
    e = sub.add_parser("eval")
    e.add_argument("--split", default="test", choices=["train", "test"])
    e.add_argument("--min-conf", type=float, default=0.30)
    e.add_argument("--eval-crops", type=int, default=48)
    args = ap.parse_args()
    random.seed(SEED)
    {"train": train, "tune": tune, "eval": evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
