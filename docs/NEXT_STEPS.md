# Next steps — annotation (#4) and multi-match ingestion (#5)

Everything below is wired and ready. Two workstreams: **fine-tune a ball detector** (so passing/line-
breaks/GK distribution unlock) and **ingest more matches** (so the C5 opponent synthesizer can work).

## #4 — Annotate the ball, then fine-tune (≈45 min of clicking)

The pretrained TrackNetV2/WASB models don't transfer to this 1024×576 footage (domain gap), so we adapt
them on frames labelled from *this* footage.

**Step 1 — annotate (you do this):**
```bash
python tools/annotate_ball.py \
  --video "C:/Users/siddh_ygv5bws/OneDrive/Desktop/cv-football/chunks/chunk_000.mp4" \
  --positions outputs/chunk000_dense.parquet \
  --out outputs/ball_annotations/chunk_000.csv --n 300
```
A window pops up with one live-play frame at a time:
- **left-click the centre of the ball** → records its position,
- **press Enter** (no click) if the ball isn't visible/findable,
- **close the window** to stop — progress is saved, rerun to resume.
Aim for ~300 frames across 1–2 chunks (more chunks = more variety = better). The tool auto-picks
busy live-play frames and skips replays/graphics.

**Step 2 — fine-tune (one command, ~10 min on the GPU):**
```bash
python tools/finetune_ball.py \
  --annotations outputs/ball_annotations/chunk_000.csv \
  --video "C:/Users/.../chunk_000.mp4" --base tracknetv2 --epochs 30 \
  --out outputs/ball_finetuned/tracknetv2_ours.pth
```
**Step 3 — use it:** point the detector at the new weights and re-run the ball pipeline:
`WASBBallDetector(weights="outputs/ball_finetuned/tracknetv2_ours.pth")` → `link_ball` → `assign_possession`
→ then possession %, PPDA, line breaks, passing networks become computable.

## #5 — Ingest more matches

**What footage to provide (this is the gating input):** broadcast video of a full match, **split into
~10-minute `chunk_*.mp4` files** in one folder (same as the existing MUN–MCI footage). Resolution like the
current 1024×576 is fine; a *single wide tactical camera* works best (the calibration assumes a broadcast
main camera). National-team tournament matches are the project's scope. Put them at e.g.
`.../matches/<name>/chunks/chunk_000.mp4 …`.

**Then one command does the whole pipeline:**
```bash
python tools/ingest_match.py --name france_senegal \
  --chunks-dir "<...>/france_senegal/chunks" --enrich
```
→ `outputs/matches/france_senegal/{anchored_dense,facets}.parquet` — that match's style fingerprint.
With `--enrich` it also adds the GAT relational reads + pitch control + synchrony (slower). Once ≥2 matches
exist, `fingerprint.style_metrics.style_distance` compares any two, and the C5 synthesizer has data.

> **Note:** I can't download copyrighted match footage. Provide your own recordings / legally-sourced
> video, chunked as above.

## Player recognition — what's feasible

- **Individual names: not feasible from this footage.** Jersey numbers are ~3–5 px (unreadable); face/kit
  recognition needs resolution we don't have. OCR/number-detection would fail.
- **Positional *roles*: feasible now** (no extra data). From the persistent tracks we can assign each
  player a role (GK / full-back / centre-back / midfielder / winger / striker) by clustering their average
  position + role-template (Hungarian) matching — the standard formation-ID method. That gives "the left
  winger", "the deepest midfielder", etc., and lets metrics be reported per role (e.g. line-breaks by the
  #6). **Not implemented yet — a clean next feature.**
- **Bridge to names:** if you supply the **starting XI + formation** (text), we can map roles→names
  heuristically (GK = the keeper, the two widest defenders = full-backs, etc.). Approximate, but turns
  "left winger" into a name for the report.

## Feature-rich ideas (beyond the above)
- **Per-role metrics** (line height by the back line, build-up by the #6, threat by the front three).
- **Set-piece / transition detection** (corners, throw-ins, counter triggers) — partly positional.
- **Temporal momentum** (xT / control over match time — the "who's on top now" curve).
- **Tracking-network centrality** (key connectors) and **GNN disruption** (per-defender value via the
  counterfactuals we already do in `attacker/instinct`).
- **Style clustering across matches** (once #5 lands) → "this team plays like a high-press 2010s side".
- **LLM-narrated report** (ground every sentence in a computed number) — the supervisor-facing layer.
