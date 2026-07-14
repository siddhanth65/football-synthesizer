# Man Utd v Brighton -- CV-primary tactical read

*Body written entirely from our computer-vision metrics (the [CV] sections). Sofascore appears only in the validation appendix [ORACLE], as an oracle we check ourselves against -- never as content.*

**Ball-evidence gate:** post-link coverage 52% (>= 40%), pass-recall proxy 48.2% (>= 50%) -> **ABSTAIN on ball families**.  Body guardrail precision: 100% (38/38 numbers CV-backed).

*Gate inputs read from `outputs/eval/brighton_manutd_ball_eval.json` (oracle: Sofascore).*

*Gate readout: coverage clears the 40% bar by 11.9 pp, but the pass-recall proxy falls 1.8 pp short of the 50% bar -- a narrow miss. Ball families are withheld; the position-only structural sections render in full.*

## [CV] How Man Utd set up

Our shape classifier reads Man Utd in a **3-5-2**. The structural read below is measured directly from tracked positions and does not depend on the ball, so it renders for every match regardless of ball-track quality.

In shape terms, their visibility-corrected defensive line sits **33 m** up the pitch (the raw broadcast line is censoring-inflated; the visibility de-biasing was validated on the World Cup set, where FIFA per-phase lines exist); they build from a base line around **56 m**; the block is **31 m** wide and **19 m** deep; nearest-team-mate compactness holds near **11.4** (a spread index, lower is tighter).

Where the shape lives (lane occupation, whole match):
- half-spaces **49%** -- the dominant channel
- centre **33%**
- wings **18%** -- comparatively thin, a structural handle for opponents

## [CV] In possession

Man Utd move as a unit -- velocity synchrony **72%** on a 0-1 scale (this is a position-only measure and always renders).

> **Withheld.** Ball-dependent possession detail (tempo, PPDA, ball-xT, phase split, verticality): insufficient ball-track evidence (coverage 52%, recall proxy 48.2%) -- withheld.

## [CV] Out of possession

Man Utd defend from a high starting point -- the visibility-corrected line at **33 m** is a front-foot posture (position-only, always rendered).

> **Withheld.** Ball-dependent pressing detail (pressing intensity, counterpress decay curve, regains): insufficient ball-track evidence (coverage 52%, recall proxy 48.2%) -- withheld.

## [CV] Counter-structure seams -- how to play against Man Utd

Each seam pairs a concrete idea with the exact CV number behind it and a one-line condition that would close it. Seams resting on ball-tracked families abstain when the match fails the evidence gate.

> **Withheld.** Seam withheld (counterpress-timing window, ball-derived): insufficient ball-track evidence (coverage 52%, recall proxy 48.2%) -- withheld.

- **Seam.** Attack and switch into the wide lanes. Man Utd concentrate centrally and in the half-spaces (**49%** half-space, **33%** centre) with only **18%** in the wings -- isolating the full-backs stretches a compact block.
  - *Backing:* lane occupation: half-space 49%, centre 33%, wing 18% (Man Utd, whole match, position-only).
  - *Closes if:* Man Utd's wing occupation climbs well above 18% toward their central load.

- **Seam.** Target the space behind the defensive line. Visibility-corrected, Man Utd's line sits **33 m** up the pitch behind a base line near 56 m; early, direct balls in behind exploit that depth before the block resets.
  - *Backing:* de-biased defensive line 33 m (visibility de-biasing validated on the World Cup set; position-only).
  - *Closes if:* Man Utd drop the line well below 33 m toward their own half.

> **Withheld.** Seam withheld (transition after beating the press, ball-derived): insufficient ball-track evidence (coverage 52%, recall proxy 48.2%) -- withheld.

- **Seam.** Stretch the block before entering it. Man Utd hold a tight, narrow shape -- **31 m** wide by **19 m** deep at a nearest-team-mate spread of **11.4** -- so pinning both flanks and switching fast forces the gaps a compact block does not want to open.
  - *Backing:* block size 31 m wide x 19 m deep, compactness 11.4 (Man Utd, whole match, position-only).
  - *Closes if:* Man Utd's block width climbs well above 31 m -- a shape already spread enough to defend the flanks.

## [ORACLE] Validation against Sofascore -- oracle only, not used above

None of the numbers below feed the analysis above; the Sofascore oracle only checks our CV output where the two measure the same thing. Two honesty notes carry over from the pilot gate: (1) our possession figures are measured on trackable ball frames only, so they are biased by ball coverage and here INVERT the oracle -- our space-control hands the territory to the side the event oracle has with less of the ball; the separately diagnosed trackable ball-possession share inverts identically (Man Utd ~44%; results/pl_pilot/possession_diagnosis.md). (2) We have no CV shot detector, so shots are an open gap shown for context only, never claimed as a CV output.

*Possession -- CV proxy vs Sofascore (biased-by-construction, reported caveated).*

| team | CV (ours) | Sofascore | note |
| --- | --- | --- | --- |
| Man Utd | 43% (space-control proxy) | 52% | trackable-frame bias; inverts the oracle |
| Brighton | 57% (space-control proxy) | 48% | trackable-frame bias; inverts the oracle |

*Pass volume in the usable ball track vs Sofascore completed passes.*

| team | CV (ours) | Sofascore | recall proxy |
| --- | --- | --- | --- |
| Man Utd | 213 | 446 | 47.8% |
| Brighton | 198 | 407 | 48.6% |

*Shots -- shown for context; we do not detect shots from broadcast video.*

| team | CV (ours) | Sofascore | note |
| --- | --- | --- | --- |
| Man Utd | -- (no CV shot detector) | 11 | honest gap |
| Brighton | -- (no CV shot detector) | 14 | honest gap |

---
*football-synthesizer report v2. Body = CV metrics (team-level; no player roster for this club fixture). Appendix = Sofascore oracle. No C5 opponent-model claim (it is World-Cup-pooled).*