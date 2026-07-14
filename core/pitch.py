"""Single source of truth for pitch geometry + metric versioning (P0 audit item #6/#5).

Every module that previously redefined ``PITCH_LEN, PITCH_WID = 105.0, 68.0`` (or the 120x80
StatsBomb contract dims) imports from here instead. ``METRICS_VERSION`` is stamped into every
metric artifact (facet parquets, fact-store JSON) so a definition change never silently
invalidates historical outputs — bump it whenever a metric definition or threshold changes.
"""
from __future__ import annotations

# Real-world pitch (metres) — the coordinate system of all dense/aligned position tables.
PITCH_LEN = 105.0
PITCH_WID = 68.0

# StatsBomb-style contract grid used by generator.contract FreezeFrames.
CONTRACT_LEN = 120.0
CONTRACT_WID = 80.0

ATT_THIRD_X = 2 * PITCH_LEN / 3  # the 70 m line; presence beyond it = attacking third

# Bump on any metric-definition/threshold change (see docs/PROJECT_AUDIT_2026-07.md section 3.5).
# 2026.07.1: transitions counterpress window 1 s -> 5 s (StatsBomb spec alignment).
# 2026.07.2: pass max-gap now seconds-based (0.9 s) via true per-chunk fps, replacing 50 native frames
#            (which meant 2.0 s @25fps vs 0.83 s @59.94fps) -- fps-consistent pass physics.
# 2026.07.3: ALL time-based fact-store metrics now use the chunk's TRUE native fps (was a hardcoded
#            FPS=50 in report/facts.py + WINDOW_FRAMES=250 in transitions): set-piece settle speed,
#            pressing_intensity/synchrony/space velocity, counterpress_curve windows, transition
#            counter-press window (now 5 s * native fps) and mean_recovery_s. At 25 fps native the old
#            constants ran ~2x fast (recovery 2x too small, regain curve shifted early); this makes the
#            counter-press / recovery / set-piece physics fps-consistent across the 25 vs 59.94 corpus.
METRICS_VERSION = "2026.07.3"
