# Jersey-number legibility probe -- brighton_manutd (2026-07-12)

Decides Layer 2 (player identity) of `docs/CAPABILITY_LEDGER.md` sec.3. PROBE ONLY -- nothing
written into the pipeline. Tool: `tools/jersey_feasibility.py`. Artifacts: this file,
`jersey_probe_stats.json`, `montage_{wide,medium,close}.png`, `closeup_diag/*.jpg`.

## Question
At 1920x1080 wide-camera distance, how tall (px) is a player torso / back number, and is that
number human-legible -- either (path 1) directly in wide play, or (path 2) in close-ups/replays that
we discard for geometry but which could anchor identity onto a whole track?

## Method
- Content split from the dense aligned parquet's players-per-frame (the ledger's live-wide-play
  proxy): `>=8` outfield people = WIDE, `4-7` = MEDIUM, `<=3` = CLOSE-UP/replay. Match mix by
  frame-instance: **wide 45.4% / medium 28.5% / close 26.1%**.
- 160 frames sampled per bucket (seed 0), detector re-run to recover person boxes (the parquet has
  only foot points). Torso = upper 40% of the box; number height estimated at ~0.35 x torso.

### Detector caveat that changed the answer (important)
The football-trained YOLO `extract.py` uses (`uisikdag/yolo-v8`) is **blind to close-up players**: on
close-bucket frames it caps box height at ~150-160 px while **COCO yolov8s finds boxes up to ~1050
px on the identical frames**. A first pass with the football detector wrongly showed close-ups
yielding the same ~40 px torsos as wide play. All numbers below use **COCO yolov8s person
detection**, which spans both scales (it agrees with the football detector on wide-play boxes).

## Results

### Per-crop torso height (every detected person, incl. small background players)
| content | crops | torso p10/p50/p90 (px) | >=32px | >=64px | est. number >=12px |
|---|---|---|---|---|---|
| wide   | 2839 | 28.0 / 38.8 / 51.6 | 80% | 1%  | 71% |
| medium | 2614 | 27.2 / 38.0 / 51.2 | 77% | 1%  | 68% |
| close  | 2026 | 27.6 / 39.2 / 68.8 | 79% | 11% | 71% |

Median torso in tactical play ~38-39 px -> estimated back-number ~**13-14 px**: right on the
human-legibility floor. Confidently-legible crops (torso >=64 px -> number ~22 px) are only **1%** in
wide/medium play.

### Per-frame TALLEST torso (the path-2 metric: is a legibly-large subject present?)
| content | frame-max p50/p90 (px) | frames with torso >=64px | frames with torso >=100px |
|---|---|---|---|
| wide   | 56.4 / 65.2  | 14% | 2%  |
| medium | 55.6 / 64.0  | 11% | 2%  |
| close  | 61.0 / **360.4** | **38%** | **30%** |

Close-up frames carry a huge tail: **p90 frame-max torso = 360 px**; **30% of close-up frames
contain a torso >=100 px** (number ~35 px+ -- trivially legible).

## Legibility read (honest, from the montages)
- **Wide/medium montages** (`montage_wide.png`, `montage_medium.png`): the *tallest* crops
  (~36-56 px torso) show readable back numbers -- "41", "7", "6", "9", "3" -- but the median/small
  crops are blurred, folded, or angled and are **not confidently readable**. The ~12 px number at
  the median is at the ragged edge for a human and would need a trained OCR model.
- **Close montage + `closeup_diag/`**: genuine broadcast close-ups are unambiguous. Sampled
  examples show **"GILMOUR 11"** with the **surname nameplate** fully legible (number ~200 px), a red
  "#6" back with nameplate, and a Man Utd player face-legible (Garnacho). Numbers "34/22/29" clear at
  ~50-64 px. Close-ups deliver number AND surname -- two independent identity keys.
- **Backs-vs-fronts caveat:** numbers are on backs only; fronts show the sponsor. At any instant only
  players moving away from camera present a number, and side-on players show it foreshortened. So the
  *effective* legible-number fraction in wide play is materially below the raw torso-size fraction.

## Two-path viability
**Path 1 -- DIRECT wide-play OCR.** Passes the *literal* pre-committed bar (71% of wide crops clear
the ~12 px number / ~32 px torso floor, well over the 20% threshold), **but marginally**: median
numbers are ~13 px, confidently-legible (>=64 px torso) crops are ~1%, and only back-facing players
qualify. Verdict: **viable only with a model trained for small/blurred numbers** (SoccerNet jersey
scale), not human-trivial; expect partial per-appearance recall, best treated as a per-track vote
aggregated over many frames.

**Path 2 -- CLOSE-UP-ANCHORED.** The strong path. ~26% of the match is close-up/replay content, and
**30-38% of those frames contain a legibly-large subject** (number + often surname). Legibility is a
solved problem here; the binding constraint is **linking the close-up identity back to a tactical
track across the camera cut** (close-up frames carry no geometry). Once linked, one read propagates
along the entire track id -- exactly the leverage the task flagged.

## Verdict
**Layer 2 is VIABLE.** Both pre-committed conditions are met: (a) wide-play crops clear the >=20%
legible-number bar numerically (though marginally, model-dependent, backs-only), and (b) close-ups
provide plainly legible number+surname crops. The recommended architecture is **both paths together**:
use close-ups/replays as high-confidence identity anchors propagated along track ids, and a
SoccerNet-trained jersey-number model as a per-track vote over wide-play back-facing frames to
fill/confirm. The hard remaining engineering is cross-cut track linking (path 2), not legibility.

## SoccerNet comparison
`data/soccernet/JERSEY_DATASET_NOTES.md` was still **absent** at probe completion (parallel download
not finished) -- comparison skipped. Note for when it lands: our wide-play crops (~38 px torso, ~13 px
number) sit at the small end; the SoccerNet jersey benchmark crops are the right training scale to
target, and our close-up crops (100-360 px torso) are well above it.
