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

- **Git:** commit ONLY when explicitly asked. Push ONLY to origin (siddhanth65/football-synthesizer,
  private) under Sid's standing authorization (2026-07, in memory) — never force-push, never any
  other remote. Author is "Sid" — no Claude co-author line, no "Generated with Claude" footer.
- Sibling repos (e.g. `football-state-of-play`) are READ-ONLY.
- Scope (2026-07-17 pivot, per Sid): **Manchester United, EPL 2024-25 season** — demo = opposition
  scouting pack for ManU. France/WC material is REFERENCE ONLY (PMSR PDFs stay as method-calibration
  ground truth). Footage source: PL website 24-25 replays, supplied by Sid.
- **Scope amendment (2026-08-14, per Sid, OVERRIDING): SOLE goal = beat 61.48 GS-HOTA on the
  live GSR test board. The vehicle is the v9 association campaign (learned association in
  metric pitch coordinates — TWiX/MOTIP/SUSHI lineage on our cue basis, targeting the measured
  +11.9 connector-achievable ceiling). Do NOT propose or mention challenge pivots
  (PCBAS/SynLoc/2027 etc.) unless Sid explicitly raises them. Spend priority: the jugular,
  not cheap probes.** (Prior 2026-07-31 amendment — GSR focus over ManU corpus — still holds
  underneath.)
- **Scope amendment (2026-07-31, per Sid): primary near-term goal = score well on the SoccerNet
  GSR benchmark (GS-HOTA), using the SoccerNet GSR dataset.** ManU-corpus adaptation deferred.
  College GPU cluster is approved (Sid books slots on request); the 4 GB constraint applies to the
  LAPTOP only. Campaign docs: `docs/GSR_CAMPAIGN_BRIEF.md`, `docs/GSR_CLUSTER_ROADMAP.md`.
- Never download copyrighted footage; the user supplies match video. **Narrow exception (Sid,
  2026-07-28, explicit):** research-dataset video distributed under an NDA that Sid has personally
  signed (e.g. SoccerNet / FOOTPASS) may be downloaded, each fetch with a stated size plan and
  Sid's per-fetch approval on record.
- **Hardware (amended 2026-08-14, per Sid: "do all gpu sessions on the cluster"):** ALL GPU work
  runs on the cluster (a100server1 = 192.168.3.19, GPU 1 only, nvidia-smi occupancy check
  immediately before every launch; GPU 0 belongs to another user; supervised processes only, no
  bare nohup). The laptop GPU belongs to Sid — workers must not use it. Cluster is reachable
  ONLY on campus network/IIITDVPN; if unreachable, GPU work queues rather than falling back to
  the laptop. Laptop CPU work is unrestricted. Legacy rule stays for any laptop run Sid himself
  approves: one heavy job at a time; never the full pytest suite during GPU training; targeted
  tests only while a GPU is busy.
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
- **Never `git checkout`/`git restore` shared mutable files** (`knowledge/claims.json`, `STATUS.md`,
  any results ledger): they routinely carry other sessions' uncommitted work. Undo your own edit
  surgically (Edit tool, or re-apply from your diff) — a checkout destroyed 25 uncommitted claims
  on 2026-08-05 (22 recovered by script re-run, 3 lost). Check `git diff` scope before ANY revert.
