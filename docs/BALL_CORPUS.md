# Ball annotation corpus — annotate once, reuse forever

The ball detector is the one component that needs labelled data per *visual domain* (resolution, broadcast
style, stadium, ball design). Instead of re-annotating every match, we keep a **permanent, accumulating
corpus** and always retrain the detector on the **union** of every match labelled so far. As the corpus
covers more visual variety, a new match increasingly "just works" with no labelling.

## Layout (committed to git — the labels are the valuable asset)

```
data/ball_annotations/
  manifest.csv            # match,chunk,annotations,video,positions  (the full plan across matches)
  mun_mci/chunk_000.csv   # frame,x,y,visible  (hand-clicked)
  fra_sen/chunk_004.csv   # ...
```

The CSVs are tiny and tracked in git (videos are not — the manifest just records where each video lives
locally). `outputs/ball_annotations/` is the old, git-ignored location; the corpus under `data/` is the
permanent one.

## Add a new match (the only manual step)

1. **Annotate** ~150-300 live-play frames per chunk (the tool auto-picks busy frames; left-click the
   ball, Enter if not visible, close to stop — resumable):
   ```bash
   python tools/annotate_ball.py \
     --video <path>/chunk_004.mp4 \
     --positions outputs/<match>/chunk_004_dense.parquet \
     --out data/ball_annotations/<match>/chunk_004.csv --n 200
   ```
2. **Add a manifest row** for each annotated chunk (`match,chunk,annotations,video,positions`).

## Retrain on the whole corpus (one command, no per-match edits)

```bash
python tools/finetune_ball.py --manifest data/ball_annotations/manifest.csv \
  --out outputs/ball_finetuned/tracknetv2_v3.pth --epochs 15 --batch-size 2   # batch 16 on a Colab T4
```

`--manifest` trains on every row whose CSV exists and **skips** rows not yet labelled, so the training set
grows just by annotating more chunks. The result is one detector that covers all labelled domains.

## When can you stop annotating?

Before labelling a new match, **test the current detector on it first**:
```bash
python -m eval.ball_eval --weights outputs/ball_finetuned/tracknetv2_v3.pth \
  --chunks-dir <path> --chunks chunk_004   # needs a small annotation set to score against
```
Once the corpus spans enough variety (several broadcasts / resolutions), new matches tend to score well
out of the box — annotate only if recall is poor. Each match you do add makes the next one less likely to
need it.
```
```
