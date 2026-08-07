"""PARSeq STR sidecar (py3.11) for the Koshkina close-up reader: torso crops -> positional softmaxes.

Runs ONLY in the sidecar venv (``~/jersey-str-env``, Python 3.11 + strhub). It is invoked as a
subprocess by :class:`generator.jersey_id.KoshkinaRecognizer`; the py3.14 main env never imports
strhub -- it consumes only this script's JSON output.

For every ``*.jpg`` in ``--crops-dir`` this runs the SoccerNet-fine-tuned PARSeq and writes
``{crop_name: {"p0": [11 floats], "p1": [11 floats]}}`` to ``--out-json``, where ``p0``/``p1`` are the
softmax distributions over tokens ``[E, 0, 1, ..., 9]`` at string positions 0 and 1 (the two jersey
digits). This mirrors ``repro_str.py``'s decode (``logits[:, :3, :11].softmax``) but emits the full
positional distributions instead of the argmax label, so the main-env roster mask can redistribute
off-roster mass rather than hard-gating.

With ``--edl-head`` the same forward pass additionally feeds a frozen-trunk **Dirichlet evidential
head** (:mod:`generator.evidential_jersey`, campaign v7 V3) and each entry gains
``"alpha": [100 floats]`` -- the concentration parameters over ``{no-number} u {1..99}``. The trunk
is untouched either way; the head reads the pooled encoder memory that the decode already computed,
so the evidential arm costs one extra MLP per crop and nothing else. Without the flag the output is
byte-identical to every on-record run.

Resumable by disk state: crops already present in ``--out-json`` are skipped, so a kill is cheap.
"""

from __future__ import annotations

import argparse
import json
import os
import string
import sys
import time
from pathlib import Path

import torch
from PIL import Image


def main() -> None:
    """Read torso crops, run PARSeq, checkpoint ``{crop: {p0, p1}}`` positional softmaxes."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--parseq-repo", required=True, help="jersey-number-pipeline root (holds str/parseq)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ckpt-every", type=int, default=4000)
    ap.add_argument("--edl-head", default=None,
                    help="Dirichlet evidential head checkpoint; adds an 'alpha' field per crop")
    # Every on-record read was produced on CPU (this script never moved the model), so CPU stays
    # the default for the incumbent path; only the evidential arm opts into the GPU.
    ap.add_argument("--device", default=None,
                    help="'cuda'/'cpu' (default: cpu, or cuda when --edl-head is given)")
    args = ap.parse_args()

    sys.path.append(os.path.join(args.parseq_repo, "str", "parseq"))
    from strhub.data.module import SceneTextDataModule  # noqa: PLC0415
    from strhub.models.utils import load_from_checkpoint  # noqa: PLC0415

    crops_dir = Path(args.crops_dir)
    out_json = Path(args.out_json)

    device = args.device or ("cuda" if args.edl_head and torch.cuda.is_available() else "cpu")
    model = load_from_checkpoint(args.ckpt, charset_test=string.digits).eval().to(device)
    tf = SceneTextDataModule.get_transform(model.hparams.img_size)

    head, memory = None, []
    if args.edl_head:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from generator.evidential_jersey import load_head  # noqa: PLC0415

        head = load_head(args.edl_head, device)
        model.encoder.register_forward_hook(lambda _m, _i, o: memory.append(o))
    results = json.loads(out_json.read_text()) if out_json.exists() else {}
    files = sorted(f for f in crops_dir.iterdir() if f.suffix == ".jpg")
    todo = [f for f in files if f.name not in results]
    print(f"STR sidecar: {len(todo)} torso crops to do ({len(results)} cached, {len(files)} total)")

    t0 = time.time()
    buf: list[torch.Tensor] = []
    names: list[str] = []

    def flush() -> None:
        if not buf:
            return
        x = torch.stack(buf).to(device)
        memory.clear()
        with torch.no_grad():
            logits = model(x)
            # positions 0,1,2 over tokens [E,0..9]; positions 0 and 1 are the two jersey digits.
            pos = logits[:, :3, :11].softmax(-1)
            alpha = None
            if head is not None:
                mem = memory[0]
                alpha = head(torch.cat([mem.mean(1), mem.amax(1), pos.flatten(1)], 1))
                alpha = alpha.float().cpu().numpy()
        probs = pos.float().cpu().numpy()
        for i, (nm, row) in enumerate(zip(names, probs)):
            results[nm] = {"p0": row[0].tolist(), "p1": row[1].tolist()}
            if alpha is not None:
                results[nm]["alpha"] = [round(float(v), 5) for v in alpha[i]]
        buf.clear()
        names.clear()

    for i, f in enumerate(todo, 1):
        try:
            buf.append(tf(Image.open(f).convert("RGB")))
            names.append(f.name)
        except (OSError, ValueError):
            continue
        if len(buf) >= args.batch:
            flush()
        if i % args.ckpt_every == 0 or i == len(todo):
            flush()
            out_json.write_text(json.dumps(results))
            rate = i / (time.time() - t0)
            print(f"  {i}/{len(todo)}  {rate:.1f}/s")
    flush()
    out_json.write_text(json.dumps(results))
    print(f"STR sidecar done: {len(results)} predictions -> {out_json.name}")


if __name__ == "__main__":
    main()
