# Football Synthesizer

**Broadcast video → freeze-frames → relational reads → an auto-generated, FIFA-style team report
that also *predicts* how a team will play.**

This is a **new, standalone project**. It depends on, but does not modify, two earlier projects:
- `football-state-of-play` — the trained relational (GAT) model + the freeze-frame contract.
- `cv-football` — the Phase-1 broadcast CV pipeline (YOLO + ByteTrack + homography).

## What it does (one pipeline, end to end)

```
match video ──▶ freeze-frame generator ──▶ unified freeze-frames ──▶ relational model
 (WC22, Euros,     (detect + track +          (substrate-aware,        (success / xT /
  Copa, friendlies)  calibrate + complete)     masked contract)         run / receiver / press)
                                                                              │
   FIFA EFI reports ──▶ validation + features ──────────────────────────────▶│
                                                                              ▼
                                                          team identity fingerprint
                                                                              │
                                                                              ▼
                                          ┌──────────────────────────────────────────────┐
                                          │  SYNTHESIZER → FIFA-style team report (.pdf)   │
                                          │  • how they play now (phases, lines, channels) │
                                          │  • how they will play vs opponent O (predicted)│
                                          └──────────────────────────────────────────────┘
```

## Status

Scaffolding. **First focus: the freeze-frame generator** (`generator/`). See
[`docs/PLAN.md`](docs/PLAN.md) for the full technical plan and [`STATUS.md`](STATUS.md) for progress.

## Quickstart (once deps are installed)

```bash
pip install -e .                       # or: pip install -r requirements.txt
pytest                                 # the freeze-frame contract is tested first
python -m ingest.sources --list        # show data sources (WC22, Euro, Copa, friendlies)
```

## Layout

| Dir | Role |
|---|---|
| `ingest/` | fetch footage (yt-dlp), parse FIFA EFI PDFs, data-source registry |
| `generator/` | **video → freeze-frames**: detect/track, calibrate, team/role, off-screen completion |
| `attacker/` | per-player run + receiver heads retrained on dense video tracks (novelty **C3**) |
| `fingerprint/` | aggregate model reads → team identity vector + style/formation axes |
| `synthesizer/` | opponent-conditioned tendency prediction (novelty **C5**) |
| `report/` | render the FIFA-style team report (PDF/HTML) |
| `eval/` | GS-HOTA, FIFA-EFI validation, attacker-vs-360 baselines |

License: MIT.
