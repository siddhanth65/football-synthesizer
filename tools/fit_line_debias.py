"""Fit the global defensive-line de-bias slope over all processed matches (P2 calibration).

Regenerates ``generator.impute.LINE_DEBIAS_SLOPE``. The slope removes the visibility-censoring inflation
of the deepest-N line estimate: as fewer back-line defenders are visible the line reads higher, and this
single coefficient corrects each frame to a full-back-line view. Print the fitted value and paste it into
``generator/impute.py`` if it drifts.

CONTAMINATION NOTE (2026-07-14): ``main`` fits over ``matches(processed_only=True)`` = ALL 5 processed
matches, which INCLUDES the 3 FIFA France matches that ``tools/line_c6`` then validates against — so the
shipped slope's FIFA validation is in-sample, not held-out. Measured cost: refitting on only the 2
non-FIFA matches (mun_mci + brighton_manutd) gives -6.31 (vs the shipped -5.62), and that clean slope
raises the France per-phase mean |line - FIFA| from ~5.5 m to ~7.2 m. Leave-one-match-out slopes span
-5.95..-5.01 (median -5.62), so the scalar itself is stable; the clean club-only slope is steeper because
PL broadcasts censor defenders differently. The shipped constant is unchanged pending a decision (would
bump METRICS_VERSION and regenerate fact stores).

Run: ``python -m tools.fit_line_debias``.
"""
from __future__ import annotations

from core.registry import matches
from generator.impute import BACK_REF, LINE_DEBIAS_SLOPE, fit_line_debias


def main() -> None:
    ms = matches(processed_only=True)
    aligned = [m.load_aligned() for m in ms]
    b_back = fit_line_debias(aligned, predictor="n_back")
    b_vis = fit_line_debias(aligned, predictor="n_vis")
    print(f"pooled over {len(ms)} matches")
    print(f"  n_back slope = {b_back:+.3f} m per back-line defender (BACK_REF={BACK_REF})  <- shipped")
    print(f"  n_vis  slope = {b_vis:+.3f} m per outfielder (reference)")
    print(f"\ncurrent generator.impute.LINE_DEBIAS_SLOPE = {LINE_DEBIAS_SLOPE:+.3f}")
    if abs(b_back - LINE_DEBIAS_SLOPE) > 0.2:
        print(f"  -> drift; update LINE_DEBIAS_SLOPE to {b_back:.2f}")


if __name__ == "__main__":
    main()
