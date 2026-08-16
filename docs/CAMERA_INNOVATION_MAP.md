# Camera-geometry innovation map

*Literature deep-dive, 2026-08-16, for the v10 campaign (goal: 70 GS-HOTA). Full agent
report with all URLs lives in the session record; this is the decision version.*

## The one insight that reframes the problem

GS-HOTA's position matching uses a **5-metre tolerance kernel**. Our solved-frame median
error is 0.43 m — already far inside it. **Median calibration accuracy is past the
bottleneck; what bleeds score is the TAIL:** dead frames whose interpolated homographies
drift metres off, and the identity errors that ride on jittery geometry. This explains why
BroadTrack's headline number on GSR clips is **100% completeness** (5.02 px mean error,
33 re-inits over 49 sequences) rather than accuracy — and why the leader's edge is
plausibly completeness + tail behaviour, not medians. *(Derived from the 5 m kernel, not a
published result — flagged as our inference; cheap to verify, see "sensitivity curve".)*

## What the world knows (six families, compressed)

1. **Temporal sports-camera models.** BroadTrack (WACV'25): 8-param tripod model
   {f, k1, pan, tilt, roll, C}, tripod centre fitted post-hoc from unconstrained per-frame
   estimates, then re-tracked with keypoints + line segmentation + player-masked optical
   flow under robust LM. Public precedent for GSR use: Playbox & MIXI extended BroadTrack
   with optical-flow parameter propagation → 61.64 in the 2025 challenge. BHITK (2024):
   two-stage Kalman over homographies — the principled version of our dead-frame fill.
   Two-point PTZ (Chen & Little): under a tripod prior, TWO correspondences solve a frame
   (our per-frame DLT needs 4+) — why sparse-marking frames stop being dead.
2. **SLAM/SfM framing.** Classic SLAM is degenerate here (no translation → no parallax);
   the correct geometry is rotation-only tracking against a known map. **The unpublished
   formulation: offline full-clip bundle adjustment** — one optimization per 30 s clip,
   shared {tripod, distortion}, smooth per-frame {pan, tilt, roll, f}, every correspondence
   + flow link jointly, covariance out the back. We are non-causal (broadcast AR systems
   must be real-time; we don't) — that asymmetry is OUR structural advantage.
3. **Broadcast AR industry.** The industry SOTA *is* BroadTrack (EVS published their AR
   tracker). Vizrt/Supponor: tripod prior + line lock-on + temporal tracking — same family,
   nothing beyond it published.
4. **Learned regression.** Our PnLCalib is per-frame SOTA lineage. Falaleev & Chen
   (MMSports'24 challenge winners): denser correspondence harvest — conic tangent points,
   line-conic intersections. Central-view rescue (arXiv:2504.20052): circle→line
   conversion makes centre-circle-only frames solvable. Foundation-feature registration:
   open gap.
5. **Robustness.** Confidence-gated re-init (BroadTrack), clamped smoothing (±2°/±2 m,
   the 2024 GSR winner's guard), uncertainty-weighted fusion (BHITK). Lines survive blur
   better than points → weight lines/flow during fast pans.
6. **Evaluation.** JaC@5px + completeness (SN-Calib), or GS-HOTA directly. Nobody has
   published the **calibration-error → GS-HOTA sensitivity curve** — perturb GT calibration
   on valid, replot GS-HOTA. Cheap, novel, and it tells us exactly when to stop investing
   in geometry.

## The build ladder (each rung gates the next)

**Rung 0 — the sensitivity curve + quick wins (days, CPU).** Produce the unpublished
calibration-vs-GS-HOTA curve; adopt conic-tangent keypoints + central-view circle→line to
shrink the dead-frame set at the source.

**Rung 1 — BHITK-style RTS smoother over our existing solves (2-4 days, CPU). THE PROBE.**
Forward-backward smoothing of [pan, tilt, roll, f] time series with per-frame measurement
covariance from solver residuals; dead frames get model-based prediction + clamps instead
of linear interpolation. No re-solving, no GPU. **Decision rule: if this moves DEV GS-HOTA
< +0.5, temporal camera modelling is confirmed non-bottleneck for us and the budget goes
back to identity/tracking; that negative is itself decision-grade.**

**Rung 2 — tripod-constrained clip solver on our stack (1-2 weeks).** Fit tripod centre +
distortion per clip from our existing camera solves, re-solve frames with only
[pan, tilt, roll, f] free + player-masked optical-flow links; dead frames solvable from
2 correspondences. Claim: 100% completeness with tripod-consistent geometry.

**Rung 3 — offline full-clip bundle adjustment (2-3 weeks). THE DIFFERENTIATOR.** The
formulation nobody has published for GSR: batch-MAP camera solve per clip, uncertainty-
stamped positions, optional rolling-shutter parameter (also untouched in sports
literature). This is the thesis-grade novelty claim of the geometry axis.

## Verified context worth knowing

- Published GS-HOTA landscape: 2025 challenge podium 63.90/63.81/62.76; SoccerMaster
  (foundation model, arXiv:2512.11016) reports 64.1; Broadcast2Pitch (WACV'26) is
  per-frame + VLM identity. The 68.3 board entry exists on the eval server only — no
  paper anywhere; "leader uses BroadTrack" remains inference (EVS lineage + the Playbox
  precedent), currently being tested directly by our W3 instrument run.
- BroadTrack code: github.com/evs-broadcast/BroadTrack — license restrictive (EVS,
  noncommercial internal research; instrument-only for us). The PAPER is the clean-room
  source for anything that ships.
