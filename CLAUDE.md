# football-synthesizer — working rules for Claude Code

Broadcast match video → CV tracking → validated team metrics → opponent-conditioned synthesizer →
grounded pundit reports (France focus). Roadmap anchor: `docs/PROJECT_AUDIT_2026-07.md`. Living log:
`STATUS.md` (update it at every milestone).

## Multi-agent orchestration (MANDATORY)

- **Main session = orchestrator/tech-lead ONLY.** It plans, decomposes, reviews results,
  makes calls, talks to the user, and updates docs/STATUS/memory. It does **NOT** write or edit code.
  **Current orchestrator model: Opus 5 (Sid, 2026-07-24)** — "opus 5 will now be working instead of
  fable 5 to plan; use opus 4.8 and sonnet 5 for coding and lower level tasks; for clarity and
  confirmation use fable 5 to confirm." (History: Fable 5 held the seat 2026-07-11 → 07-24, with
  Opus 4.8 standing in during quota gaps. The orchestrator/worker split held through every swap.)
- **ALL code work → `deep-worker` subagent (Opus 4.8):** pipeline modules, metric implementations,
  debugging, validation harnesses, GPU scripts, architecture review.
- **Mechanical, fully-specified work → `fast-worker` subagent (Sonnet 5):** parsers, boilerplate,
  spec'd tests, batch runs, cleanups, doc formatting.
- The ONLY files the main session may edit directly: `CLAUDE.md`, `STATUS.md`, `docs/*`,
  `.claude/**`, and auto-memory. Everything under `core/ generator/ fingerprint/ synthesizer/
  attacker/ report/ eval/ tools/ tests/` goes through a subagent. If a change feels "too small to
  delegate", delegate it to fast-worker anyway.
- **Model honesty rule (hard):** never state that a specific model performed work unless verified
  (user's usage dashboard or explicit `/model` output). Earlier in this project, "Fable 5 consults"
  were claimed while the dashboard showed 0% Fable usage. Label delegated work by the *requested*
  route ("deep-worker (requested: opus)") until the user confirms routing on the dashboard once.

## Standing constraints (from the user — do not relax)

- **Git:** commit ONLY when explicitly asked. Never push or force-push. Author is "Sid" — no Claude
  co-author line, no "Generated with Claude" footer.
- Sibling repos (e.g. `football-state-of-play`) are READ-ONLY.
- Scope (2026-07-17 pivot, per Sid): **Manchester United, EPL 2024-25 season** — demo = opposition
  scouting pack for ManU. France/WC material is REFERENCE ONLY (PMSR PDFs stay as method-calibration
  ground truth). Footage source: PL website 24-25 replays, supplied by Sid.
- **Scope amendment (2026-07-31, per Sid): primary near-term goal = score well on the SoccerNet
  GSR benchmark (GS-HOTA), using the SoccerNet GSR dataset.** ManU-corpus adaptation deferred.
  College GPU cluster is approved (Sid books slots on request); the 4 GB constraint applies to the
  LAPTOP only. Campaign docs: `docs/GSR_CAMPAIGN_BRIEF.md`, `docs/GSR_CLUSTER_ROADMAP.md`.
- Never download copyrighted footage; the user supplies match video. **Narrow exception (Sid,
  2026-07-28, explicit):** research-dataset video distributed under an NDA that Sid has personally
  signed (e.g. SoccerNet / FOOTPASS) may be downloaded, each fetch with a stated size plan and
  Sid's per-fetch approval on record.
- **Hardware:** 4 GB GPU laptop — one heavy GPU/data job at a time; never run the full pytest suite
  during GPU training (OOM killed a fine-tune once); targeted tests only while GPU is busy.
- Free-tier LLM narration only (Ollama / OpenAI-compatible); no paid keys assumed.

## Engineering discipline

- Python ≥3.11, Google docstrings, type hints, ruff-clean at 100 cols, pytest green.
- Files written with `encoding="utf-8"`; the Windows console is cp1252 — keep `print()` ASCII-safe.
- Match paths ALWAYS via `core.registry` (`data/matches.yaml`). Never hardcode `outputs/...` paths.
- Metric changes bump `METRICS_VERSION` in `core/pitch.py`; artifacts are version-stamped.
- **Validated-or-nothing:** no metric ships without a validator (FIFA PMSR ground truth, LOMO
  backtest, or annotated frames). Report honest error bars and negative results; retract loudly when
  a claim fails end-to-end verification.
- **Ball-metric lesson (cost us a retraction):** the only ball-coverage number that counts is the
  post-`link_ball` usable track. Never quote pre-link detection/projection counts as "coverage".
- End-to-end before claiming: run the real pipeline (the regeneration, the full report), not just the
  intermediate probe, before updating STATUS/memory with a win.
