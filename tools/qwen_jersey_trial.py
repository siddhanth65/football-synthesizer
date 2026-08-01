"""Boxed trial: Qwen2-VL-2B as a second jersey-number voter next to the Koshkina PARSeq chain.

Survey lead F of ``docs/ATTRIBUTION_RESEARCH_PLAN.md`` (the one VLM path viable at 4 GB). Graded on
GSR crops with jersey ground truth, stratified so both halves of the question are answered:

* crops where the shipped chain **reads** a number -> agreement and relative precision;
* crops where the shipped chain **abstains** (legibility gate rejected it, so it never reached
  PARSeq) -> added-read rate and the precision of those added reads.

Adoption rule, pre-declared in ``results/OCR_DENSIFICATION.md``: keep it only if it adds reads at
>= 0.80 precision, fits 4 GB, and runs at <= 2 s/crop. GPU: one job at a time -- run this only after
the per-crop OCR pass has finished.

CLI::

    python -m tools.qwen_jersey_trial --seqs SNGS-021 SNGS-024 --n 200
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from eval.gsr_jersey import PERCROP_SUBDIR, extract_track_crops
from eval.gsr_score import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, DEFAULT_RESULTS_DIR
from tools.ocr_density import load_gt_cache

logger = logging.getLogger("qwen_trial")

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
PROMPT = ("What number is printed on this football player's shirt? "
          "Reply with just the number, or NONE if no number is visible.")
MAX_CROPS = 60


def pick_crops(percrop: pd.DataFrame, gt: dict[int, str | None], n: int, seed: int = 0) -> pd.DataFrame:
    """Stratified crop sample: half where the chain read a number, half where it abstained (pure).

    Args:
        percrop: One sequence's per-crop frame.
        gt: ``{track_id: GT jersey string or None}`` for that sequence.
        n: Total crops wanted.
        seed: RNG seed.

    Returns:
        The sampled rows with a ``gt`` column, restricted to tracks whose GT player wears a number.
    """
    df = percrop.copy()
    df["gt"] = df["track_id"].map(lambda t: gt.get(int(t)))
    df = df[df["gt"].notna()]
    rng = np.random.default_rng(seed)
    read = df[df["number"] > 0]
    abstain = df[df["number"] <= 0]
    take = min(n // 2, len(read))
    parts = [read.iloc[rng.choice(len(read), take, replace=False)] if take else read.iloc[:0]]
    take2 = min(n - take, len(abstain))
    parts.append(abstain.iloc[rng.choice(len(abstain), take2, replace=False)] if take2
                 else abstain.iloc[:0])
    return pd.concat(parts)


def parse_number(text: str) -> int:
    """First 1-2 digit run in the model's reply, or ``-1`` when it declines (pure)."""
    m = re.search(r"\d{1,2}", text)
    return int(m.group()) if m else -1


def run(seqs: list[str], n: int, data_dir: Path, out_dir: Path, results_dir: Path) -> dict:
    """Extract crops, run Qwen2-VL-2B in 4-bit, and grade agreement / added reads against GT."""
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2VLForConditionalGeneration,
    )

    from generator.extract import _build_detector  # noqa: PLC0415

    gt_all = load_gt_cache(data_dir, out_dir, seqs)
    work = out_dir / PERCROP_SUBDIR / "_qwen"
    shutil.rmtree(work, ignore_errors=True)
    detector = _build_detector("cuda" if torch.cuda.is_available() else "cpu", "football")
    rows: list[dict] = []
    for seq in seqs:
        percrop = pd.read_parquet(out_dir / PERCROP_SUBDIR / f"{seq}.parquet")
        sample = pick_crops(percrop, gt_all[seq], n // len(seqs))
        df = pd.read_parquet(out_dir / "positions" / f"{seq}.parquet")
        keep = set(zip(sample["track_id"].astype(int), sample["frame"].astype(int)))
        paths = extract_track_crops(data_dir / seq, df[df.apply(
            lambda r: (int(r["track_id"]), int(r["frame"])) in keep, axis=1)],
            detector, work / seq, max_crops=MAX_CROPS)
        by_key = {(t, int(p.stem.split("_")[-1])): p for t, ps in paths.items() for p in ps}
        for r in sample.itertuples():
            p = by_key.get((int(r.track_id), int(r.frame)))
            if p is not None:
                rows.append({"seq": seq, "track_id": int(r.track_id), "frame": int(r.frame),
                             "gt": str(r.gt), "chain": int(r.number),
                             "chain_conf": float(r.p_number), "legibility": float(r.legibility),
                             "path": str(p)})
    del detector
    torch.cuda.empty_cache()
    logger.info("%d crops prepared; loading %s in 4-bit", len(rows), MODEL_ID)

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_quant_type="nf4")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant, dtype=torch.float16, device_map="cuda:0")
    proc = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=64 * 28 * 28,
                                         max_pixels=256 * 28 * 28)
    vram = torch.cuda.max_memory_allocated() / 2**30
    t0 = time.time()
    for i, row in enumerate(rows):
        img = Image.open(row["path"]).convert("RGB")
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        reply = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0].strip()
        row["qwen_raw"] = reply
        row["qwen"] = parse_number(reply)
        if (i + 1) % 25 == 0:
            logger.info("  %d/%d  %.2f s/crop", i + 1, len(rows), (time.time() - t0) / (i + 1))
    sec_per_crop = (time.time() - t0) / max(len(rows), 1)
    shutil.rmtree(work, ignore_errors=True)

    df = pd.DataFrame(rows)
    read = df[df["chain"] > 0]
    abst = df[df["chain"] <= 0]
    added = abst[abst["qwen"] > 0]
    payload = {
        "model": MODEL_ID, "n_crops": len(df), "sequences": seqs,
        "vram_peak_gib": round(float(vram), 3), "sec_per_crop": round(sec_per_crop, 3),
        "on_chain_reads": {
            "n": len(read),
            "chain_precision": float((read["chain"].astype(str) == read["gt"]).mean())
            if len(read) else 0.0,
            "qwen_precision": float((read["qwen"].astype(str) == read["gt"]).mean())
            if len(read) else 0.0,
            "agreement": float((read["chain"] == read["qwen"]).mean()) if len(read) else 0.0,
        },
        "on_chain_abstentions": {
            "n": len(abst), "n_qwen_read": len(added),
            "added_read_rate": len(added) / max(len(abst), 1),
            "added_read_precision": float((added["qwen"].astype(str) == added["gt"]).mean())
            if len(added) else 0.0,
        },
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "qwen_jersey_trial.json").write_text(
        json.dumps({**payload, "rows": df.drop(columns=["path"]).to_dict("records")}, indent=2),
        encoding="utf-8")
    logger.info("VRAM peak %.2f GiB, %.2f s/crop", payload["vram_peak_gib"], sec_per_crop)
    logger.info("chain reads n=%d: chain prec %.3f, qwen prec %.3f, agreement %.3f",
                payload["on_chain_reads"]["n"], payload["on_chain_reads"]["chain_precision"],
                payload["on_chain_reads"]["qwen_precision"],
                payload["on_chain_reads"]["agreement"])
    logger.info("chain abstentions n=%d: qwen read %d (%.3f) at precision %.3f",
                payload["on_chain_abstentions"]["n"], payload["on_chain_abstentions"]["n_qwen_read"],
                payload["on_chain_abstentions"]["added_read_rate"],
                payload["on_chain_abstentions"]["added_read_precision"])
    return payload


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqs", nargs="+", default=["SNGS-021", "SNGS-024"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = ap.parse_args()
    run(args.seqs, args.n, args.data_dir, args.out_dir, args.results_dir)


if __name__ == "__main__":
    main()
