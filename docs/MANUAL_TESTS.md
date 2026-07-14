# Manual tests — see the pipeline work with your own eyes

Every command assumes you're in the repo root. Prefix with `PYTHONPATH=.` (already shown). Nothing here
needs a paid API. Grouped from "watch it" to "check the numbers".

## 1. Watch the CV overlay (the money shot)
A rendered clip with player boxes, team colours, ByteTrack ids, the **ball trail**, and **pitch lines
projected from the calibration** — all on real France footage.

```bash
# already rendered for you:
open results/cv_demo_france_senegal.mp4        # or double-click it
```
**What it proves:** detection + tracking + team split + ball (v4) + calibration all line up at once.
Render your own on any chunk:
```bash
PYTHONPATH=. python tools/render_clip.py \
  --video matches/france_norway/h1/chunk_003.mp4 \
  --positions outputs/france_norway/h1/match/chunk_003_dense.parquet \
  --weights outputs/ball_finetuned/tracknetv2_v4.pth \
  --start 5000 --end 6500 --fps 25 --team0 France --team1 Norway \
  --out results/my_demo.mp4
```
(Drop `--weights` to skip the ball overlay. Pick a `--start/--end` window; wide-play windows look best.)

## 2. Eyeball the calibration (top-down)
Projects players to a bird's-eye pitch — if the shapes look like real formations, calibration is sound.
```bash
PYTHONPATH=. python tools/verify_topdown.py     # writes results/verify_topdown.png
```

## 3. Read the grounded pundit report
```bash
cat results/pundit_france_senegal.md            # or _iraq / _norway
```
**What it proves:** every claim is a real CV metric or FIFA number, with player names.

## 4. Natural-language narration (free, no paid key)
```bash
# Option A — local, free (recommended): install Ollama from ollama.com, then:
ollama pull llama3.2
PYTHONPATH=. python -m report.narrate --match france_senegal   # auto-detects Ollama
# Option B — Groq free key (groq.com): 
LLM_API_KEY=<your_groq_key> LLM_MODEL=llama-3.1-8b-instant \
  PYTHONPATH=. python -m report.narrate --match france_senegal \
  --backend openai --base-url https://api.groq.com/openai/v1
# Option C — no LLM: paste results/narrate_<match>_prompt.txt into any chatbot.
```
See `results/narrate_france_iraq.md` for a hand-verified example of the target output.

## 5. Ball detector accuracy (numbers)
```bash
PYTHONPATH=. python -c "
from eval.ball_eval import evaluate_chunk
r=evaluate_chunk('matches/france_senegal/h1/chunk_002.mp4',
  'data/ball_annotations/france_iraq/h1_chunk_002.csv',   # (any labelled chunk)
  'outputs/france_senegal/h1/match/chunk_002_dense.parquet',
  'outputs/ball_finetuned/tracknetv2_v4.pth')
print(r)"
```
**Expect:** precision ~0.99, recall ~0.6–0.8, localisation ~0.2 m.

## 6. France's structural fingerprint + Tier-A forecast
```bash
PYTHONPATH=. python -m synthesizer.predict --positions outputs/france_senegal/final/match_aligned.parquet --team 0 --name France
```
**Expect:** high line (~47–55 m), attacking-third share, wing share, each with an 80% interval.

## 7. Validation vs FIFA (C6)
```bash
PYTHONPATH=. python tools/phase_pct_c6.py      # results/phase_pct_c6.png  (mean abs err ~16 pp)
```
**What it proves:** our phase distribution tracks FIFA's; the residual is the diagnosed partial-view bias.

## 8. The synthesizer's opponent signal + Tier-B backtest
```bash
PYTHONPATH=. python tools/france_profile.py        # results/france_opponent_signal.png
PYTHONPATH=. python -m synthesizer.backtest        # Tier-A vs Tier-B (honest: needs more matches)
```

## 9. Cross-match style DNA
```bash
PYTHONPATH=. python tools/style_matrix.py          # results/style_matrix.png  (France≈Man City)
```

## 10. The test suite (all layers)
```bash
PYTHONPATH=. python -m pytest -q                   # ~155 pass (run when the GPU is idle)
```

---
**Fastest sanity loop:** open `results/cv_demo_france_senegal.mp4` (does it look right?) → `cat
results/pundit_france_senegal.md` (do the numbers read true?) → `tools/phase_pct_c6.py` (do we match FIFA?).
