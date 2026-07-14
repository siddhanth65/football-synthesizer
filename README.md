# Football Synthesizer

Broadcast match video -> CV tracking and calibration -> validated team tactical metrics ->
opponent-conditioned model -> grounded, guardrailed pundit reports. The system slices a match into
chunks, detects and tracks players, projects them onto a 105x68 m pitch via camera calibration,
tracks the ball, computes tactical metrics only on frames whose geometry survives explicit quality
gates, and generates a report whose every number is checked against the underlying fact store before
it reaches the reader.

Academic framing: this is a **Game State Reconstruction (GSR)**-style pipeline (SoccerNet's
formalized task — single moving broadcast camera to 2D pitch positions + role + team for every
player) extended with validated tactical metrics, opponent conditioning, and evidence-gated
reporting. It is a 2-semester IIIT Delhi BTP (Review 1: Dec 2026, Review 2: May 2027). Corpus:
WC-France (validated methodology chapter, closed) plus Premier League Manchester United 24/25
(current focus).

## Pipeline

```
match.mp4
  -> chunk           tools/chunk_video.py            (ffmpeg stream-copy, ~10-min chunks)
  -> detect           football-trained YOLOv8         (players, every 5th frame)
  -> track             ByteTrack                       (per-chunk track_id)
  -> team              CIELAB KMeans + cross-chunk anchoring (jersey colour, not number)
  -> calibrate         PnLCalib homography + fit gate + plausibility gate
  -> project            105 x 68 m pitch coordinates
  -> ball                TrackNetV2 (v5 WC / v6 PL) + link_ball (physics gate) + carry-over
  -> possession            Viterbi-smoothed nearest-carrier proxy
  -> fingerprint metrics    position-only (spine) + ball-dependent (proxy, coverage-gated)
  -> fact store               outputs/facts/<match>.json (version-stamped)
  -> report v2 + numeric guardrail   results/report_v2_<match>.{md,html}
```

Every stage is described in detail, with honest failure modes, in `docs/CV_EXPLAINER.md`.

## Key validated results (as of 2026-07-14; see `docs/CAPABILITY_LEDGER.md` for the full ledger)

| Metric | Result |
|---|---|
| Defensive line height vs FIFA (raw -> de-biased) | 16.4 m -> ~7.2 m held-out (5.5 m in-sample) |
| Post-link ball coverage (best full match) | 51.9% (Brighton, carry-over on) |
| Pass-recall proxy (PL) | ~48%, team-symmetric (relative claims usable, absolute not) |
| Guardrail precision on shipped report bodies | 100% |
| v6 ball detector, PL held-out recall | 87% |

Honest absences: no event layer (shots, tackles, duels are not detected — nothing distinguishes a
tackle from two players near the ball); no player identity from video yet (team only, via jersey
colour; jersey-number recognition is Layer 2, in progress). Possession is reported as a caveated
"trackable-frame possession share", not event possession — it is measurably biased toward settled
build-up and cannot be corrected without event data (see the ledger for the negative result).

## Quickstart

```bash
# 1. Split a full match video into chunks
python tools/chunk_video.py <match.mp4> <output_dir> ...

# 2. Register the match (one entry in data/matches.yaml)

# 3. Run the pipeline: extract -> align -> ball -> facts -> gate (resumable)
python -m tools.pl_pilot_run

# 4. Generate the gated, guardrailed report
python -m report.report_v2 --match <match_id>
```

Match paths are always resolved via `core.registry` / `data/matches.yaml` — never hardcoded.

Model weights: PnLCalib HRNets in `~/PnLCalib/weights` (env `FOOTBALL_PNLCALIB_PATH`); ball weights
in `outputs/ball_finetuned/` (`tracknetv2_v5.pth` = WC production, `tracknetv2_v6.pth` = PL
production); player YOLO auto-downloads from HuggingFace on first run.

## Documentation

- [`docs/CV_EXPLAINER.md`](docs/CV_EXPLAINER.md) — how the pipeline works, stage by stage
- [`docs/CAPABILITY_LEDGER.md`](docs/CAPABILITY_LEDGER.md) — what's validated, unvalidated, or absent
- [`docs/BTP_DECEMBER_PLAN.md`](docs/BTP_DECEMBER_PLAN.md) — roadmap to the December 2026 review
- [`STATUS.md`](STATUS.md) — living log, updated at every milestone

## Layout

| Dir | Role |
|---|---|
| `core/` | match registry, pitch constants, `METRICS_VERSION` |
| `generator/` | video -> tracks: detect/track, calibrate, team assignment, ball, projection |
| `attacker/` | per-player run + receiver heads (novelty track, cross-chunk track fix in progress) |
| `fingerprint/` | aggregate tracks into team tactical metrics |
| `synthesizer/` | opponent-conditioned tendency model |
| `report/` | fact store, gated report v2, numeric guardrail |
| `eval/` | validation harnesses (FIFA PMSR, GS-HOTA, attacker eval) |
| `tools/` | chunking, batch runs, oracle fetch, calibration/fitting scripts |
| `ingest/` | legacy data-source/EFI-fetch helpers (WC-era; not part of the pl_pilot_run flow) |

## Data and licensing

Match footage is user-supplied and used only under lawful access for local academic processing; it
is never redistributed. All published numbers are derived metrics (tracking, tactical statistics),
not the source video itself. FIFA PMSR PDFs and raw broadcast footage are excluded from this
repository.

License: MIT (code only; see above for data terms).
