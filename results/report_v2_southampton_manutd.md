# Southampton v Man Utd -- CV-primary tactical read

*Body written entirely from our computer-vision metrics (the [CV] sections). Sofascore appears only in the validation appendix [ORACLE], as an oracle we check ourselves against -- never as content.*

**Ball-evidence gate:** post-link coverage 43% (>= 40%), pass-recall proxy 26.0% (>= 50%) -> **ABSTAIN on ball families**.  Body guardrail precision: 100% (49/49 numbers CV-backed).

*Gate inputs read from `outputs/eval/southampton_manutd_ball_eval.json` (oracle: Sofascore).*

*Gate readout: coverage clears the 40% bar by 3.0 pp, but the pass-recall proxy falls 24.0 pp short of the 50% bar -- a narrow miss. Ball families are withheld; the position-only structural sections render in full.*

## [CV] How Southampton set up

Our shape classifier reads Southampton in a **3-5-2**. The structural read below is measured directly from tracked positions and does not depend on the ball, so it renders for every match regardless of ball-track quality.

In shape terms, their visibility-corrected defensive line sits **34 m** up the pitch (the raw broadcast line is censoring-inflated; the visibility de-biasing was validated on the World Cup set, where FIFA per-phase lines exist); they build from a base line around **58 m**; the block is **31 m** wide and **18 m** deep; nearest-team-mate compactness holds near **11.6** (a spread index, lower is tighter).

Where the shape lives (lane occupation, whole match):
- half-spaces **47%** -- the dominant channel
- centre **31%**
- wings **22%** -- comparatively thin, a structural handle for opponents

## [CV] Opponent structural read -- Man Utd

Our shape classifier reads Man Utd in a **3-5-2**. The structural read below is measured directly from tracked positions and does not depend on the ball, so it renders for every match regardless of ball-track quality.

In shape terms, their visibility-corrected defensive line sits **24 m** up the pitch (the raw broadcast line is censoring-inflated; the visibility de-biasing was validated on the World Cup set, where FIFA per-phase lines exist); they build from a base line around **43 m**; the block is **30 m** wide and **18 m** deep; nearest-team-mate compactness holds near **11.3** (a spread index, lower is tighter).

Where the shape lives (lane occupation, whole match):
- half-spaces **45%** -- the dominant channel
- centre **33%**
- wings **22%** -- comparatively thin, a structural handle for opponents

Man Utd move as a unit -- velocity synchrony **67%** on a 0-1 scale (this is a position-only measure and always renders).

## [CV] In possession

Southampton move as a unit -- velocity synchrony **67%** on a 0-1 scale (this is a position-only measure and always renders).

> **Withheld.** Ball-dependent possession detail (tempo, PPDA, ball-xT, phase split, verticality): insufficient ball-track evidence (coverage 43%, recall proxy 26.0%) -- withheld.

## [CV] Out of possession

Southampton defend from a high starting point -- the visibility-corrected line at **34 m** is a front-foot posture (position-only, always rendered).

> **Withheld.** Ball-dependent pressing detail (pressing intensity, counterpress decay curve, regains): insufficient ball-track evidence (coverage 43%, recall proxy 26.0%) -- withheld.

## [CV] Counter-structure seams -- how to play against Southampton

Each seam pairs a concrete idea with the exact CV number behind it and a one-line condition that would close it. Seams resting on ball-tracked families abstain when the match fails the evidence gate.

> **Withheld.** Seam withheld (counterpress-timing window, ball-derived): insufficient ball-track evidence (coverage 43%, recall proxy 26.0%) -- withheld.

- **Seam.** Attack and switch into the wide lanes. Southampton concentrate centrally and in the half-spaces (**47%** half-space, **31%** centre) with only **22%** in the wings -- isolating the full-backs stretches a compact block.
  - *Backing:* lane occupation: half-space 47%, centre 31%, wing 22% (Southampton, whole match, position-only).
  - *Closes if:* Southampton's wing occupation climbs well above 22% toward their central load.

- **Seam.** Target the space behind the defensive line. Visibility-corrected, Southampton's line sits **34 m** up the pitch behind a base line near 58 m; early, direct balls in behind exploit that depth before the block resets.
  - *Backing:* de-biased defensive line 34 m (visibility de-biasing validated on the World Cup set; position-only).
  - *Closes if:* Southampton drop the line well below 34 m toward their own half.

> **Withheld.** Seam withheld (transition after beating the press, ball-derived): insufficient ball-track evidence (coverage 43%, recall proxy 26.0%) -- withheld.

- **Seam.** Stretch the block before entering it. Southampton hold a tight, narrow shape -- **31 m** wide by **18 m** deep at a nearest-team-mate spread of **11.6** -- so pinning both flanks and switching fast forces the gaps a compact block does not want to open.
  - *Backing:* block size 31 m wide x 18 m deep, compactness 11.6 (Southampton, whole match, position-only).
  - *Closes if:* Southampton's block width climbs well above 31 m -- a shape already spread enough to defend the flanks.

## [ORACLE] Validation against Sofascore -- oracle only, not used above

None of the numbers below feed the analysis above; the Sofascore oracle only checks our CV output where the two measure the same thing. Two honesty notes carry over from the pilot gate: (1) our possession figures are measured on trackable ball frames only, so they are biased by ball coverage and here INVERT the oracle -- our space-control hands the territory to the side the event oracle has with less of the ball; the separately diagnosed trackable ball-possession share inverts identically (Man Utd ~44%; results/pl_pilot/possession_diagnosis.md). (2) We have no CV shot detector, so shots are an open gap shown for context only, never claimed as a CV output.

*Possession -- CV proxy vs Sofascore (biased-by-construction, reported caveated).*

| team | CV (ours) | Sofascore | note |
| --- | --- | --- | --- |
| Southampton | 48% (space-control proxy) | 44% | trackable-frame bias; inverts the oracle |
| Man Utd | 52% (space-control proxy) | 56% | trackable-frame bias; inverts the oracle |

*Pass volume in the usable ball track vs Sofascore completed passes.*

| team | CV (ours) | Sofascore | recall proxy |
| --- | --- | --- | --- |
| Southampton | 131 | 421 | 31.1% |
| Man Utd | 116 | 556 | 20.9% |

*Shots -- shown for context; we do not detect shots from broadcast video.*

| team | CV (ours) | Sofascore | note |
| --- | --- | --- | --- |
| Southampton | -- (no CV shot detector) | 6 | honest gap |
| Man Utd | -- (no CV shot detector) | 20 | honest gap |

---
*football-synthesizer report v2. Body = CV metrics (team-level; no player roster for this club fixture). Appendix = Sofascore oracle. No C5 opponent-model claim (it is World-Cup-pooled).*