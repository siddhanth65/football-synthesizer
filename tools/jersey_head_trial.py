"""Is the trained CLIP jersey head a usable second OCR voter?

Session 5B. The cluster session-3 checkpoint (``outputs/gsr/clip_ckpt/epoch8.pt``) carries a
42-way jersey head trained on GSR train crops (41 observed numbers + an explicit no-number class;
sn-reid's action-level numbers were ignored and only its "never visible" letters trained the
abstention class -- see ``results/CLUSTER_SESSION3.md`` section 2). This module measures it on GSR
**valid** ground-truth crops, which the head never saw, and reruns the Qwen-trial design
(``results/OCR_DENSIFICATION.md`` section 7): on crops where the shipped PARSeq chain abstains,
does the head read at >= 0.80 precision?

The head's class order is not recorded locally, so it is **recovered on train** (the split it was
fitted on) by optimal assignment against the GT jersey labels and frozen to
``results/jersey_head_vocab.json``. Recovery that lands on the sorted-integer order and a high
train accuracy is self-verifying; anything else would show up as a near-chance train accuracy.

CLI::

    python -m tools.jersey_head_trial --head train        # GPU: head over train GT crops
    python -m tools.jersey_head_trial --recover           # freeze the class order
    python -m tools.jersey_head_trial --head validation   # GPU: head over valid GT crops
    python -m tools.jersey_head_trial --chain validation  # GPU: PARSeq chain over the same crops
    python -m tools.jersey_head_trial --report            # CPU: every table
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tools.gsr_crops import Crop, plan_split

logger = logging.getLogger("jersey_head_trial")

CROP_ROOT = Path("outputs/gsr/gt_crops")
OUT_ROOT = Path("outputs/gsr/jersey_head")
VOCAB_PATH = Path("results/jersey_head_vocab.json")
RESULTS_DIR = Path("results/gsr_benchmark")
#: Per-crop confidence floors of the two shipped aggregation rules (``ocr_density_rule.json``).
CHAIN_FLOORS = {"0.85-rule": 0.99, "0.80-rule": 0.90}


def _truth(df: pd.DataFrame) -> np.ndarray:
    """GT jersey per crop as strings, with the abstention label spelled ``<none>``.

    The GT jersey is constant within a GSR track (verified: 0 of 1,343 train and 0 of 1,340 valid
    tracks carry two values), so this is a track-level label attached to every one of its crops.
    """
    return np.array([j if isinstance(j, str) and j else "<none>" for j in df["jersey"]])


def dev20_sequences() -> list[str]:
    """The declared DEV-20 partition of the valid split (``eval.gsr_identity.split_sequences``)."""
    from tools.gsr_crops import split_sequences  # noqa: PLC0415

    return sorted(split_sequences("validation"))[::3]


def _crop_frame(crops: list[Crop]) -> pd.DataFrame:
    """Planned crops -> a table keyed by the on-disk crop file name."""
    return pd.DataFrame({
        "name": [c.name for c in crops], "seq": [c.seq for c in crops],
        "track_id": [c.track_id for c in crops], "frame": [c.frame for c in crops],
        "role": [c.role for c in crops], "team": [c.team for c in crops],
        "jersey": [c.jersey for c in crops],
    })


def run_head(split: str, batch_size: int = 16, seqs: list[str] | None = None,
             per_label: int | None = None, device: str | None = None) -> Path:
    """Run the CLIP jersey head over a split's GT crops; persist probs + the crop table.

    Args:
        split: ``train`` / ``validation`` / ``test``.
        batch_size: Tower batch size (16 fits alongside anything else on the 4 GB laptop GPU).
        seqs: Restrict to these sequences.
        per_label: Cap crops per GT jersey label (seeded) -- enough for the class-order recovery
            without paying for all 20k crops.
        device: ``cuda`` / ``cpu``; auto when ``None``.

    Returns:
        Path of the persisted ``.npz``.
    """
    import cv2  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from tools.clip_embedder import ClipEmbedder  # noqa: PLC0415

    df = _crop_frame(plan_split(split))
    if seqs:
        df = df[df["seq"].isin(seqs)]
    root = CROP_ROOT / split
    df["path"] = [str(root / n) for n in df["name"]]
    df = df[[Path(p).exists() for p in df["path"]]].reset_index(drop=True)
    if per_label:
        df = (df.groupby(df["jersey"].fillna("<none>"), group_keys=False)
              .apply(lambda g: g.sample(min(len(g), per_label), random_state=0))
              .sort_index().reset_index(drop=True))
    emb = ClipEmbedder(batch_size=batch_size, device=device)
    probs, pre = [], []
    for start in range(0, len(df), 512):
        chunk = df["path"].iloc[start:start + 512]
        imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in chunk]
        feats, raw = emb._forward(imgs, batch_size)  # noqa: SLF001 (one tower pass, two heads)
        with torch.inference_mode():
            probs.append(torch.softmax(emb.jersey(torch.from_numpy(feats)), 1).numpy())
            pre.append(torch.softmax(emb.jersey(torch.from_numpy(raw)), 1).numpy())
        logger.info("head %s: %d / %d", split, start + len(chunk), len(df))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_ROOT / f"head_{split}.npz",
                        probs=np.concatenate(probs), pre=np.concatenate(pre))
    df.drop(columns=["path"]).to_parquet(OUT_ROOT / f"crops_{split}.parquet")
    return OUT_ROOT / f"head_{split}.npz"


def run_chain(split: str, seqs: list[str] | None = None, chunk: int = 1500) -> Path:
    """Run the shipped Koshkina/PARSeq chain over the same GT crops (GPU)."""
    from eval.gsr_jersey import _build_recognizer  # noqa: PLC0415
    from generator.jersey_id import ILLEGIBLE  # noqa: PLC0415

    df = pd.read_parquet(OUT_ROOT / f"crops_{split}.parquet")
    if seqs:
        df = df[df["seq"].isin(seqs)].reset_index(drop=True)
    paths = [str(CROP_ROOT / split / n) for n in df["name"]]
    recog = _build_recognizer()
    leg, torso, num, pnum = [], [], [], []
    for start in range(0, len(paths), chunk):
        probs, detail = recog.crop_reads(paths[start:start + chunk])
        best = probs[:, 1:].argmax(axis=1) + 1
        conf = probs[np.arange(len(probs)), best]
        read = np.where(probs.argmax(axis=1) == ILLEGIBLE, -1, best)
        leg.append(detail["leg"])
        torso.append(detail["torso"])
        num.append(read)
        pnum.append(conf)
        logger.info("chain %s: %d / %d", split, start + len(probs), len(paths))
    df["leg"] = np.concatenate(leg)
    df["torso"] = np.concatenate(torso)
    df["chain_number"] = np.concatenate(num)
    df["chain_conf"] = np.concatenate(pnum)
    out = OUT_ROOT / f"chain_{split}.parquet"
    df.to_parquet(out)
    return out


def recover_vocab(split: str = "train") -> dict:
    """Recover the head's class order by optimal assignment against train GT labels.

    Returns:
        The frozen vocabulary dict, also written to :data:`VOCAB_PATH`.
    """
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    df = pd.read_parquet(OUT_ROOT / f"crops_{split}.parquet")
    blob = np.load(OUT_ROOT / f"head_{split}.npz")
    gt = _truth(df)
    labels = ["<none>"] + sorted({j for j in gt if j != "<none>"}, key=int)
    truth = np.array([labels.index(j) for j in gt])
    #: The natural construction order: observed numbers ascending, abstention appended last.
    natural = labels[1:] + ["<none>"]
    best = None
    for key in ("probs", "pre"):
        pred = blob[key].argmax(axis=1)
        counts = np.zeros((blob[key].shape[1], len(labels)), np.int64)
        np.add.at(counts, (pred, truth), 1)
        rows, cols = linear_sum_assignment(-counts)
        free = float(counts[rows, cols].sum() / len(truth))
        nat = float((np.asarray(natural)[pred] == gt).mean())
        logger.info("%s: free-assignment train accuracy %.4f, natural order %.4f", key, free, nat)
        if best is None or nat > best["train_accuracy"]:
            best = {"input": key, "train_accuracy": nat, "classes": natural,
                    "free_assignment_accuracy": free,
                    "ties_free_assignment": bool(abs(free - nat) < 1e-9),
                    "n_crops": int(len(truth))}
    if not best["ties_free_assignment"]:
        raise RuntimeError(f"natural class order is not an optimal assignment: "
                           f"{best['train_accuracy']:.4f} vs "
                           f"{best['free_assignment_accuracy']:.4f}")
    VOCAB_PATH.write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best


def _load(split: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Crops + head posteriors (in the frozen input space) + the frozen class order."""
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    df = pd.read_parquet(OUT_ROOT / f"crops_{split}.parquet")
    probs = np.load(OUT_ROOT / f"head_{split}.npz")[vocab["input"]]
    return df, probs, vocab["classes"]


def _digit_structure(pairs: list[tuple[str, str]]) -> dict:
    """Confusion structure of wrong reads: length, first/second digit, truncation."""
    out = {"n": len(pairs), "same_length": 0, "length_mismatch": 0, "truncation": 0,
           "first_digit_right": 0, "last_digit_right": 0, "no_digit_shared": 0}
    for true, pred in pairs:
        if len(true) == len(pred):
            out["same_length"] += 1
        else:
            out["length_mismatch"] += 1
            if true.startswith(pred):
                out["truncation"] += 1
        out["first_digit_right"] += int(true[0] == pred[0])
        out["last_digit_right"] += int(true[-1] == pred[-1])
        out["no_digit_shared"] += int(not set(true) & set(pred))
    return out


def head_eval(split: str = "validation", seqs: list[str] | None = None) -> dict:
    """Per-crop accuracy, abstention quality and confusion structure of the head."""
    from collections import Counter  # noqa: PLC0415

    df, probs, classes = _load(split)
    if seqs:
        keep_rows = df["seq"].isin(seqs).to_numpy()
        df, probs = df[keep_rows].reset_index(drop=True), probs[keep_rows]
    pred = np.array([classes[i] for i in probs.argmax(axis=1)])
    conf = probs.max(axis=1)
    truth = _truth(df)
    in_vocab = np.isin(truth, classes)
    numbered = truth != "<none>"
    emits = pred != "<none>"
    correct = pred == truth
    wrong = [(t, p) for t, p, e in zip(truth, pred, emits) if e and t != "<none>" and t != p]
    res = {
        "split": split, "sequences": len(df["seq"].unique()), "crops": int(len(df)),
        "crops_with_gt_number": int(numbered.sum()),
        "gt_number_outside_head_vocab": int((numbered & ~in_vocab).sum()),
        "accuracy_all": float(correct.mean()),
        "accuracy_numbered": float(correct[numbered].mean()),
        "accuracy_numbered_where_emits": float(correct[numbered & emits].mean()),
        "emit_rate_numbered": float(emits[numbered].mean()),
        "nonum_precision": float((~numbered)[~emits].mean()),
        "nonum_recall": float((~emits)[~numbered].mean()),
        "read_precision_all_emits": float(correct[emits].mean()),
        "read_precision_numbered_only": float(correct[emits & numbered].mean()),
        "confusion": _digit_structure(wrong),
        "top_confusions": Counter(f"{t}->{p}" for t, p in wrong).most_common(12),
        "top_predicted": Counter(pred[emits]).most_common(8),
        "conf_floor_sweep": {},
    }
    # Chance references on the same emitted subset: always guessing that subset's most common
    # true number, and drawing the head's own prediction marginal independently of the truth.
    sub_t, sub_p = truth[emits & numbered], pred[emits & numbered]
    if len(sub_t):
        tcount, pcount = Counter(sub_t), Counter(sub_p)
        res["chance_modal_number"] = tcount.most_common(1)[0][1] / len(sub_t)
        res["chance_independent_marginals"] = sum(
            pcount[k] / len(sub_t) * tcount.get(k, 0) / len(sub_t) for k in pcount)
    for floor in (0.0, 0.5, 0.7, 0.9, 0.95, 0.99):
        keep = emits & (conf >= floor)
        res["conf_floor_sweep"][f"{floor:.2f}"] = {
            "emitted": int(keep.sum()),
            "precision_numbered_only": float(correct[keep & numbered].mean())
            if (keep & numbered).any() else None,
            "precision_all": float(correct[keep].mean()) if keep.any() else None,
        }
    return res


def abstain_trial(split: str = "validation", seqs: list[str] | None = None) -> dict:
    """The Qwen-trial question: on crops the shipped chain abstains on, is the head >= 0.80 precise?"""
    df, probs, classes = _load(split)
    chain = pd.read_parquet(OUT_ROOT / f"chain_{split}.parquet")
    if seqs:
        chain = chain[chain["seq"].isin(seqs)]
    idx = pd.Series(range(len(df)), index=df["name"])
    rows = idx.reindex(chain["name"]).to_numpy()
    pred = np.array([classes[i] for i in probs.argmax(axis=1)])[rows]
    conf = probs.max(axis=1)[rows]
    truth = _truth(chain)
    numbered = truth != "<none>"
    emits = pred != "<none>"
    correct = pred == truth
    chain_num = chain["chain_number"].to_numpy()
    chain_conf = chain["chain_conf"].to_numpy()
    chain_pred = np.where(chain_num > 0, chain_num.astype(str), "<none>")
    modes = {"hard-abstain (no read at all)": chain_num <= 0}
    for name, floor in CHAIN_FLOORS.items():
        modes[f"abstain at {name} floor {floor}"] = (chain_num <= 0) | (chain_conf < floor)
    out = {"split": split, "sequences": sorted(chain["seq"].unique().tolist()),
           "crops": int(len(chain)), "modes": {}, "where_chain_reads": {}}
    for name, mask in modes.items():
        sub = {"crops": int(mask.sum()),
               "head_emits": int((mask & emits).sum()),
               "head_emit_rate": float(emits[mask].mean()) if mask.any() else None}
        for tag, extra in (("all_tracks", np.ones(len(mask), bool)),
                           ("numbered_tracks_only", numbered)):
            keep = mask & emits & extra
            sub[f"precision_{tag}"] = float(correct[keep].mean()) if keep.any() else None
            sub[f"n_{tag}"] = int(keep.sum())
        sub["precision_by_head_conf"] = {}
        for floor in (0.5, 0.7, 0.9, 0.95, 0.99):
            keep = mask & emits & numbered & (conf >= floor)
            sub["precision_by_head_conf"][f"{floor:.2f}"] = {
                "n": int(keep.sum()),
                "precision": float(correct[keep].mean()) if keep.any() else None,
                "added_read_rate": float((mask & emits & (conf >= floor)).sum() / mask.sum())
                if mask.any() else None,
            }
        out["modes"][name] = sub
    reads = (chain_num > 0) & numbered
    out["where_chain_reads"] = {
        "crops": int(reads.sum()),
        "chain_precision": float((chain_pred[reads] == truth[reads]).mean()),
        "head_precision": float(correct[reads].mean()),
        "head_emit_rate": float(emits[reads].mean()),
        "agreement": float((chain_pred[reads] == pred[reads]).mean()),
        "chain_only_right": int(((chain_pred[reads] == truth[reads])
                                 & (pred[reads] != truth[reads])).sum()),
        "head_only_right": int(((chain_pred[reads] != truth[reads])
                                & (pred[reads] == truth[reads])).sum()),
    }
    return out


def generous_variants(split: str = "validation", seqs: list[str] | None = None) -> dict:
    """The two most generous readings of the head, so the verdict is not an artefact of ``argmax``.

    1. **Oracle roster mask** -- zero every class that is not a GT jersey of that sequence, which is
       strictly better than the self-roster the pipeline could build.
    2. **Track-level aggregation** -- majority vote over a track's emitted numbers, and the argmax of
       the track's mean posterior, since a per-crop-noisy head can still be track-consistent.
    """
    from collections import Counter  # noqa: PLC0415

    df, probs, classes = _load(split)
    if seqs:
        keep = df["seq"].isin(seqs).to_numpy()
        df, probs = df[keep].reset_index(drop=True), probs[keep]
    cls = np.asarray(classes)
    truth = _truth(df)
    pred = cls[probs.argmax(axis=1)]
    masked = pred.copy()
    for _seq, idx in df.groupby("seq").indices.items():
        present = {n for n in truth[idx] if n != "<none>"} & set(classes)
        sub = probs[idx].copy()
        sub[:, [c not in present and c != "<none>" for c in classes]] = 0.0
        masked[idx] = cls[sub.argmax(axis=1)]

    def score(p: np.ndarray, t: np.ndarray) -> dict:
        emit, num = p != "<none>", t != "<none>"
        return {"emit_rate": float(emit.mean()), "n": int((emit & num).sum()),
                "precision_numbered_only": float((p == t)[emit & num].mean())
                if (emit & num).any() else None}

    out = {"per_crop_argmax": score(pred, truth), "per_crop_oracle_roster": score(masked, truth)}
    rows = []
    for _key, idx in df.groupby(["seq", "track_id"]).indices.items():
        votes = [p for p in pred[idx] if p != "<none>"]
        mvotes = [p for p in masked[idx] if p != "<none>"]
        rows.append((truth[idx][0],
                     Counter(votes).most_common(1)[0][0] if votes else "<none>",
                     Counter(mvotes).most_common(1)[0][0] if mvotes else "<none>",
                     cls[probs[idx].mean(axis=0).argmax()]))
    tr = np.array(rows)
    out["tracks"] = len(tr)
    for col, name in ((1, "track_majority_vote"), (2, "track_majority_vote_oracle_roster"),
                      (3, "track_mean_posterior_argmax")):
        out[name] = score(tr[:, col], tr[:, 0])
    return out


def _demo() -> None:
    """Self-check on the two pieces of non-trivial logic: the grader and the digit structure."""
    st = _digit_structure([("42", "62"), ("11", "1"), ("7", "9")])
    assert st == {"n": 3, "same_length": 2, "length_mismatch": 1, "truncation": 1,
                  "first_digit_right": 1, "last_digit_right": 2, "no_digit_shared": 1}, st
    assert _digit_structure([]) == {"n": 0, "same_length": 0, "length_mismatch": 0,
                                    "truncation": 0, "first_digit_right": 0,
                                    "last_digit_right": 0, "no_digit_shared": 0}
    print("jersey_head_trial demo OK")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--head", default=None, help="split to run the CLIP jersey head over")
    ap.add_argument("--chain", default=None, help="split to run the PARSeq chain over")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--per-label", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dev20", action="store_true", help="restrict --seqs to the DEV-20 partition")
    ap.add_argument("--recover", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.dev20:
        args.seqs = dev20_sequences()
    if args.demo:
        _demo()
    if args.head:
        run_head(args.head, seqs=args.seqs, per_label=args.per_label, device=args.device)
    if args.recover:
        print(json.dumps({k: v for k, v in recover_vocab().items() if k != "classes"}, indent=2))
    if args.chain:
        run_chain(args.chain, args.seqs)
    if args.report:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        dev = dev20_sequences()
        test = [s for s in sorted(set(pd.read_parquet(OUT_ROOT / "crops_validation.parquet")["seq"]))
                if s not in set(dev)]
        out = {"vocab": json.loads(VOCAB_PATH.read_text(encoding="utf-8")),
               "train_fit_set": head_eval("train"), "partitions": {}}
        for tag, part in (("VALID-58", None), ("DEV-20", dev), ("TEST-38", test)):
            out["partitions"][tag] = {"head_eval": head_eval("validation", part),
                                      "abstain_trial": abstain_trial(seqs=part),
                                      "generous_variants": generous_variants(seqs=part)}
        (RESULTS_DIR / "jersey_head_trial.json").write_text(
            json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
