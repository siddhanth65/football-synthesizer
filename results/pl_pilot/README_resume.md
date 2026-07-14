# Brighton v Manchester United - full-match pilot (Phase B first brick)

Fixture: **Brighton 2-1 Manchester Utd**, ENG Premier League 2024-25 **Matchweek 2, 2024-08-24**
(Amex, 12:30 KO, ref Craig Pawson, att 31,537; both 4-2-3-1). Ten Hag era; Man Utd in **red home**
kit, Brighton **blue/white stripes**.

Source: `Brighton v Manchester United 2024-25 Full Match Replay.mp4` (repo root), 1920x1080 @ 25 fps,
150,024 frames, 6000.9 s.

## Halftime / chunking
The replay is **continuous** (halftime break trimmed - no long non-play gap; green-content scan found
no >45s dead stretch). Football runs ~0:15 to ~99:15. The half boundary is a trimmed cut inside the
45:00-48:15 transition zone (whistle + graphics + H2 KO; brightness steps up after 48:15). **Split at
file t = 2895 s (48:15).** Verify post-extraction via the attack-direction flip in `pl_pilot_align`.

Chunks (10-min, ffmpeg stream-copy):
- h1: `matches/brighton_manutd/h1/chunk_000..004.mp4`  (004 = 495 s)
- h2: `matches/brighton_manutd/h2/chunk_000..005.mp4`  (005 = 108 s)  -> 11 chunks total
Kickoff offsets: H1 KO ~= 0:15 into h1/chunk_000; H2 KO ~= start of h2/chunk_000 (file 48:15).

## Stage status
- [x] CHUNK
- [x] REGISTRY: data/matches.yaml `brighton_manutd` entry (schema extended w/ optional club fields)
- [~] EXTRACT (GPU, ~50 min/chunk, ~9 h total): `outputs/brighton_manutd/{h1,h2}/match/chunk_NNN_dense.parquet`
      Resumable via `--skip-existing`. Logs: `results/pl_pilot/extract_{h1,h2}.log`.
- [ ] ALIGN: `outputs/brighton_manutd/final/match_aligned.parquet`  (tools/pl_pilot_align.py)
- [ ] BALL (GPU, v6 + carry-over): `outputs/brighton_manutd/final/ball/ball_<half>_chunkNNN.parquet`
- [ ] FACTS: `outputs/facts/brighton_manutd.json`  (report.facts, degrades w/o PMSR)
- [ ] GATE: `results/pl_pilot/fbref_gate.md`  (FBref ref cached in `fbref_ref.json`)

## Resume (one command, turnkey, each stage resumable)
    python -m tools.pl_pilot_run                 # from extraction
    python -m tools.pl_pilot_run --from align    # once extraction is done

## FBref reference (gate = possession within ~5 pp; ours is a proxy)
Possession: Brighton 48% / Man Utd 52%. Shots + passes in `fbref_ref.json`. No CV shot detector
(shots = honest gap). teams[0]=Man Utd (red, dark anchor), teams[1]=Brighton - confirm via the
align colour diagnostic.

## GPU discipline
ONE GPU stage at a time. Extraction and ball are both GPU - never run concurrently. No full pytest
while either is running (targeted tests only).
