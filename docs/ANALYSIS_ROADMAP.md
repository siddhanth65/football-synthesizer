# Analysis Roadmap — getting to a more comprehensive team-style read from footage

Synthesised from the **xG Football Club** corpus (91 articles, summarised in
`football-state-of-play/docs/research_notes.md`) + `football-state-of-play/eval/team_metrics.py`. Goal:
understand a team's footballing style more completely **from broadcast freeze-frames**, beyond what we
already compute.

## The two biggest under-used assets

1. **Our own GAT has heads we don't read.** `eval/team_metrics.py` builds attacking + defensive
   fingerprints from the *same checkpoint* using **receiver-distribution entropy** (passing-option
   richness), **presser_probs** (press decisiveness) and **xpass_dense** (lane suppression / opponent
   lane openness) — none of which our `sop_bridge` currently extracts. **Free enrichment, same model.**
2. **We have no spatial-control layer.** Almost every tactical method below sits on a **pitch-control
   surface** (Spearman 2018 / Fernández) — who controls each pitch zone — which needs only **positions +
   velocities** (we have both; velocity from `build_tracks`). **No ball required.** This is the single
   highest-leverage addition for team-style analysis.

## Method → our pipeline (what to build, what it needs, what it adds)

| Method (corpus) | Needs | Status | What it adds to team style |
|---|---|---|---|
| **Full GAT readout** (recv entropy, presser, xPass) | trained GAT (have) | **free** | option richness, press decisiveness, lane suppression — attacking+defending depth |
| **Pitch control surface** (Spearman/Fernández) | positions + velocity | **missing** | space dominance, territory control, control in each third — the spatial backbone |
| **OBSO** (off-ball scoring opportunity) | pitch control + goal geom | missing | where a team manufactures *off-ball* danger (no ball needed for the control term) |
| **Velocity synchrony** ("Team Mind") | velocity | missing | attacking coordination — strongest chance-creation predictor in the corpus |
| **Tactical DNA / style distance** (EMD over positional distributions) | positions | missing | one interpretable **style-distance** number between teams → matchup/clustering |
| **Tracking networks** (influence-distance graph + centrality) | positions | partial (passing-proxy) | connectivity structure, key-connector identification |
| **Formation / shape-ambiguity** transitions | positions over time | missing | *when* a team breaks shape (63% of play; where attacks happen) |
| **GNN disruption** (counterfactual defender ablation) | our GAT | missing | per-defender defensive value (we already do counterfactuals in `instinct`) |
| **Markov defensive signature** (high-press vs deep-block states) | phase/block metrics | partial (have phase split) | Klopp-vs-Simeone style steady-state signature |
| PPDA, line breaks, penetrative passes, passing networks, space utilisation, pass quality, chemistry | **ball events** | parked (ball) | the FIFA "right column"; gated on ball tracking |
| Opponent-conditioned tendencies (C5) | **many matches** | blocked (1 match) | "how A plays vs B" — the end goal |

## What we currently have (today)
Generator (calibration-fixed, detection, ByteTrack, tactical frames) → deterministic `z_T` (shape /
lanes / direction / rough physical) → **4-facet fingerprint** (attacking/defending/passing/GK, positional
+ GAT xT/success/recovery) → **possession phase-split** (in/out) → **C3 run head** (beats 360) → **C4
instinct v2** (receiver-value off-ball runs) → **GAT relational reads** (per-team, ball-carrier-attributed)
→ **report v1** (HTML). Colour-anchored full-match teams.

## What to improve, in priority order
1. **Full GAT heads** → option richness / press decisiveness / lane suppression in the facet fingerprint
   (mirror `eval/team_metrics.py`). *Free, high value, do first.*
2. **Pitch control surface** (positions+velocity) → space-dominance metrics + OBSO. *No ball; the spatial
   backbone the corpus is built on.*
3. **Style-distance (Tactical DNA) + velocity synchrony** → cheap, interpretable team-style axes; tee up
   matchup comparison for C5.
4. **Ball-tracking fine-tuning** (parked) → unlocks PPDA / line breaks / passing networks / pass quality.
5. **More matches** → the C5 opponent-conditioned synthesizer (the end goal).

## Honest "are we on track?"
The **machinery is deep and working** on one match (generator → fingerprint → relational reads →
report). The **headline goal (C5: predict how a team plays vs an opponent) is gated on data scale** —
it needs many team-matches, and we have one. So: on track on *depth*; the *end goal* needs the
wide pass (more matches) plus the ball layer for full EFI parity. Items 1–3 above make the **single-match
style read much more comprehensive without solving ball detection**, which is the best use of effort now.
