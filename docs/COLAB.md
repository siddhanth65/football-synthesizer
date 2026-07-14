# Colab GPU runner — offload the two GPU-bound jobs

Your RTX 3050 (4 GB) is the bottleneck for exactly two jobs; both belong on Colab's free **T4 (16 GB)**.
Everything else (possession linking, metrics, networks, roles, phases) is pure CPU — keep it local.

| Job | local 3050 | Colab T4 |
|---|---|---|
| Ball fine-tune (`tools/finetune_ball.py`) | the 18-h stall; batch 2, fp16 forced | batch 16, fp16, minutes |
| Player tracking (`tools/batch_match.py`) | slow, GPU-bound | much faster |

Notebook: [`notebooks/football_synthesizer_colab.ipynb`](../notebooks/football_synthesizer_colab.ipynb).

## One-time prep

1. **Get the code to Colab** — either:
   - push this repo to GitHub and set `GITHUB_URL` in the notebook's CONFIG cell, **or**
   - copy the repo folder to Drive and leave `GITHUB_URL = ''` (it reads `REPO_IN_DRIVE`).
2. **Footage on Drive.** Put your `chunk_*.mp4` in a Drive folder; set `DRIVE_FOOTAGE_DIR`. (Copyright is
   your call — the footage never leaves your Drive.)
3. **Pretrained zoo weight.** Drop `tracknetv2_soccer_best.pth.tar` in `DRIVE_OUT_DIR`; the notebook
   copies it into the WASB checkout so the fine-tune has a base checkpoint.

## Run

`Runtime → Change runtime type → T4 GPU`, edit the CONFIG cell, `Runtime → Run all`. Outputs (fine-tuned
weights + `outputs/match/*.parquet`) are saved back to `DRIVE_OUT_DIR`.

## Back home

Copy the weights + parquets from Drive into the repo, then run the CPU half locally — ball possession,
roles, phase split, possession report. No GPU needed for any of it.

> The VS Code Colab extension can open and run this notebook directly; this repo can't drive the
> extension for you, but the notebook is self-contained once the CONFIG cell is set.
