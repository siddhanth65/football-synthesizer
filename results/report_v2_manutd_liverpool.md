# Man Utd v Liverpool -- CV-primary tactical read

*Body written entirely from our computer-vision metrics (the [CV] sections). Sofascore appears only in the validation appendix [ORACLE], as an oracle we check ourselves against -- never as content.*

**Ball-evidence gate:** post-link coverage 46% (>= 40%), pass-recall proxy 29.0% (>= 50%) -> **COMPARATIVE (relative claims only)**.  Body guardrail precision: 100% (100/100 numbers CV-backed).

*Gate inputs read from `outputs/eval/manutd_liverpool_ball_eval.json` (oracle: Sofascore).*

*Gate readout: coverage clears the 40% bar by 5.7 pp; the pass-recall proxy is 29.0% -- below the 50% absolute bar, but team-symmetric (spread 0.028 <= 0.05). Ball families therefore render in COMPARATIVE form only -- team shares, ratios and team-vs-team differences; absolute ball volumes are withheld. Position-only structural sections render in full.*

## [CV] How Man Utd set up

Our shape classifier reads Man Utd in a **4-2-3-1**. The structural read below is measured directly from tracked positions and does not depend on the ball, so it renders for every match regardless of ball-track quality.

In shape terms, their visibility-corrected defensive line sits **29 m** up the pitch (the raw broadcast line is censoring-inflated; the visibility de-biasing was validated on the World Cup set, where FIFA per-phase lines exist); they build from a base line around **52 m**; the block is **28 m** wide and **21 m** deep; nearest-team-mate compactness holds near **11.3** (a spread index, lower is tighter).

Where the shape lives (lane occupation, whole match):
- half-spaces **46%** -- the dominant channel
- centre **35%**
- wings **19%** -- comparatively thin, a structural handle for opponents

## [CV] Opponent structural read -- Liverpool

Our shape classifier reads Liverpool in a **3-5-2**. The structural read below is measured directly from tracked positions and does not depend on the ball, so it renders for every match regardless of ball-track quality.

In shape terms, their visibility-corrected defensive line sits **24 m** up the pitch (the raw broadcast line is censoring-inflated; the visibility de-biasing was validated on the World Cup set, where FIFA per-phase lines exist); they build from a base line around **46 m**; the block is **30 m** wide and **20 m** deep; nearest-team-mate compactness holds near **12.1** (a spread index, lower is tighter).

Where the shape lives (lane occupation, whole match):
- half-spaces **47%** -- the dominant channel
- centre **33%**
- wings **20%** -- comparatively thin, a structural handle for opponents

Liverpool move as a unit -- velocity synchrony **71%** on a 0-1 scale (this is a position-only measure and always renders).

## [CV] In possession

Man Utd move as a unit -- velocity synchrony **71%** on a 0-1 scale (this is a position-only measure and always renders).

> **Relative claims only** -- capture ~29%, team-symmetric (spread 0.028); absolute volumes withheld.

By our phase classifier the two sides split their in-possession time differently: Man Utd spend **17%** of it in the final third against Liverpool's **17%**, and **65%** in build-up against **60%** -- a team-vs-team tilt, not an absolute volume.

With the ball, Man Utd account for **54%** of the two sides' tracked passing volume (**46%** Liverpool) at a similar mean length (**8.5 m** vs **8.4 m**); ball-progression threat (xT) splits **52%**/**48%** in Man Utd's favour; goalward directness (verticality) runs **7%** for Man Utd vs **9%** for Liverpool; Man Utd produced **57%** of the defensive-line breaks between the sides.

Relative pressing tempo (PPDA proxy, lower is more aggressive): Man Utd **0.8** vs Liverpool **0.9** -- a ratio of Man Utd build-up passes allowed per defensive action, reported team-vs-team.

## [CV] Out of possession

Man Utd defend from a high starting point -- the visibility-corrected line at **29 m** is a front-foot posture (position-only, always rendered).

> **Relative claims only** -- capture ~29%, team-symmetric (spread 0.028); absolute volumes withheld.

Pressing intensity (a 0-1 time-to-intercept index) runs **36%** for Man Utd against **40%** for Liverpool -- an intensity comparison, not a count of actions.

Counterpress decay, side by side: within 5 s of a loss Man Utd regain the ball **49%** of the time against Liverpool's **46%**, and by 8 s **58%** vs **52%** -- both curves are per-team regain shares.

Man Utd commit to the counterpress on **71%** of losses vs Liverpool's **72%**; but convert **26%** of their losses into high regains against Liverpool's **37%** -- relative conversion, not absolute regains.

## [CV] Counter-structure seams -- how to play against Man Utd

Each seam pairs a concrete idea with the exact CV number behind it and a one-line condition that would close it. Seams resting on ball-tracked families abstain when the match fails the evidence gate.

> **Relative claims only** -- capture ~29%, team-symmetric (spread 0.028); absolute volumes withheld.

- **Seam.** Retain through the first press and play forward within about 5 s. Side by side, Man Utd recover **49%** of losses inside 5 s vs Liverpool's **46%**, and by 8 s **58%** vs **52%** -- the earlier-flattening side is the one to play through.
  - *Backing:* regain-curve comparison, Man Utd vs Liverpool: 5 s 49% vs 46%, 8 s 58% vs 52% (per-team regain shares; relative claim only).
  - *Closes if:* Man Utd's 5 s regain rate pulls clear of Liverpool's 46% -- the relative edge closes.

- **Seam.** Attack and switch into the wide lanes. Man Utd concentrate centrally and in the half-spaces (**46%** half-space, **35%** centre) with only **19%** in the wings -- isolating the full-backs stretches a compact block.
  - *Backing:* lane occupation: half-space 46%, centre 35%, wing 19% (Man Utd, whole match, position-only).
  - *Closes if:* Man Utd's wing occupation climbs well above 19% toward their central load.

- **Seam.** Target the space behind the defensive line. Visibility-corrected, Man Utd's line sits **29 m** up the pitch behind a base line near 52 m; early, direct balls in behind exploit that depth before the block resets.
  - *Backing:* de-biased defensive line 29 m (visibility de-biasing validated on the World Cup set; position-only).
  - *Closes if:* Man Utd drop the line well below 29 m toward their own half.

- **Seam.** Commit to the counter the instant you win it back. Man Utd pour into the counterpress on **71%** of losses vs Liverpool's **72%**, yet convert only **26%** of their losses into high regains against Liverpool's **37%** -- survive first contact and the pitch opens.
  - *Backing:* counterpress rate 71% vs Liverpool 72%; high-regain conversion 26% vs 37% (relative rates, no absolute regain counts).
  - *Closes if:* Man Utd's high-regain conversion pulls clear of Liverpool's 37% -- a counterpress that wins the ball high rather than just delaying.

- **Seam.** Stretch the block before entering it. Man Utd hold a tight, narrow shape -- **28 m** wide by **21 m** deep at a nearest-team-mate spread of **11.3** -- so pinning both flanks and switching fast forces the gaps a compact block does not want to open.
  - *Backing:* block size 28 m wide x 21 m deep, compactness 11.3 (Man Utd, whole match, position-only).
  - *Closes if:* Man Utd's block width climbs well above 28 m -- a shape already spread enough to defend the flanks.

## [ORACLE] Validation against Sofascore -- oracle only, not used above

None of the numbers below feed the analysis above; the Sofascore oracle only checks our CV output where the two measure the same thing. Two honesty notes carry over from the pilot gate: (1) our possession figures are measured on trackable ball frames only, so they are biased by ball coverage and here INVERT the oracle -- our space-control hands the territory to the side the event oracle has with less of the ball; the separately diagnosed trackable ball-possession share inverts identically (Man Utd ~44%; results/pl_pilot/possession_diagnosis.md). (2) We have no CV shot detector, so shots are an open gap shown for context only, never claimed as a CV output.

*Possession -- CV proxy vs Sofascore (biased-by-construction, reported caveated).*

| team | CV (ours) | Sofascore | note |
| --- | --- | --- | --- |
| Man Utd | 47% (space-control proxy) | 53% | trackable-frame bias; inverts the oracle |
| Liverpool | 53% (space-control proxy) | 47% | trackable-frame bias; inverts the oracle |

*Pass volume in the usable ball track vs Sofascore completed passes.*

| team | CV (ours) | Sofascore | recall proxy |
| --- | --- | --- | --- |
| Man Utd | 128 | 421 | 30.4% |
| Liverpool | 108 | 391 | 27.6% |

*Shots -- shown for context; we do not detect shots from broadcast video.*

| team | CV (ours) | Sofascore | note |
| --- | --- | --- | --- |
| Man Utd | -- (no CV shot detector) | 8 | honest gap |
| Liverpool | -- (no CV shot detector) | 11 | honest gap |

---
*football-synthesizer report v2. Body = CV metrics (team-level; no player roster for this club fixture). Appendix = Sofascore oracle. No C5 opponent-model claim (it is World-Cup-pooled).*