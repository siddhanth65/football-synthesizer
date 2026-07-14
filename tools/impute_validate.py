"""P2 validation: does off-screen imputation actually recover the true structure — and un-bias the line?

Two self-supervised tests (no FIFA needed) plus the headline effect:

1. **Leave-one-visible-out accuracy.** In each frame, hide one *visible* player whose role we know, impute
   it from its role offset, and measure the metre error against where it really was. Baseline = predict the
   team's visible centroid (offset 0). If the role offset beats the centroid, the imputer carries real
   structural signal, not noise.

2. **Masking line-height recovery.** Take near-complete frames (>=9 visible, where ``def_line_height`` is
   trustworthy ~= FIFA), record the true line, then drop the deepest defenders down to a broadcast-like ~6
   visible (which inflates the line — the bias). Impute the dropped roles and measure how much of the true
   line the imputer restores vs the naive masked estimate. DoD: ``|imputed - true| < |naive - true|``.

3. **Real-frame effect.** Match-level ``def_line_height`` before vs after imputation on the *actual* (mostly
   low-visibility) frames — the deep shift imputation applies, which should be ~the +11 m the hand offset
   was faking, now earned from the data instead of hard-coded.

Run: ``python -m tools.impute_validate`` (all processed matches).
"""
from __future__ import annotations

import numpy as np

from core.registry import matches
from fingerprint.roles import assign_roles
from fingerprint.structural_metrics import DEF_LINE_QUANTILE, attacking_coord, resolve_attack_directions
from fingerprint.theory_metrics import complete_directions
from generator.impute import (
    MIN_VIS_FOR_CENTROID,
    _chunk_meta,
    _formation_roles,
    impute_frame,
    learn_role_offsets,
)

PLAYERS = ("player", "goalkeeper")
NEAR_COMPLETE = 9   # visible count treated as pseudo-ground-truth
BROADCAST_VIS = 6   # simulate a broadcast view by masking down to this many


def _line(ax: np.ndarray) -> float:
    return float(np.percentile(ax, DEF_LINE_QUANTILE))


def loo_accuracy(aligned, roles, offsets) -> tuple[list[float], list[float]]:
    """Leave-one-visible-out: (imputer errors, centroid-baseline errors) in metres."""
    role_map, formation = _chunk_meta(roles)
    imp_err, base_err = [], []
    for ck, g in (aligned.groupby("chunk") if "chunk" in aligned.columns else [("_", aligned)]):
        dirs = complete_directions(resolve_attack_directions(g))
        pl = g[g["role"].isin(PLAYERS)].dropna(subset=["pitch_x", "pitch_y"])
        for (fr, team), fg in pl.groupby(["frame", "team"]):
            team = int(team)
            if team not in dirs or len(fg) <= MIN_VIS_FOR_CENTROID:
                continue
            adir = dirs[team]
            role_of = {int(t): role_map.get((ck, int(t)), (None, None))[1] for t in fg["track_id"]}
            for tid in fg["track_id"].astype(int):
                role = role_of.get(tid)
                if role is None or (team, role) not in offsets:
                    continue
                held = fg[fg["track_id"] == tid]
                rest = fg[fg["track_id"] != tid]
                if len(rest) < MIN_VIS_FOR_CENTROID:
                    continue
                true_x = float(held["pitch_x"].iloc[0])
                true_y = float(held["pitch_y"].iloc[0])
                rest_roles = {k: v for k, v in role_of.items() if k != tid}
                imp = impute_frame(rest, team=team, adir=adir, role_of=rest_roles,
                                   froles=[role], offsets=offsets)
                if not imp:
                    continue
                imp_err.append(float(np.hypot(imp[0]["pitch_x"] - true_x, imp[0]["pitch_y"] - true_y)))
                cx, cy = float(rest["pitch_x"].mean()), float(rest["pitch_y"].mean())
                base_err.append(float(np.hypot(cx - true_x, cy - true_y)))
    return imp_err, base_err


def masking_recovery(aligned, roles, offsets) -> tuple[list, list]:
    """On >=9-visible frames: (|imputed - true|, |naive - true|) line-height errors in metres."""
    role_map, formation = _chunk_meta(roles)
    imp_e, naive_e = [], []
    for ck, g in (aligned.groupby("chunk") if "chunk" in aligned.columns else [("_", aligned)]):
        dirs = complete_directions(resolve_attack_directions(g))
        pl = g[g["role"].isin(PLAYERS)].dropna(subset=["pitch_x", "pitch_y"])
        for (fr, team), fg in pl.groupby(["frame", "team"]):
            team = int(team)
            if team not in dirs or not (NEAR_COMPLETE <= len(fg) <= 11):
                continue
            adir = dirs[team]
            ax_all = attacking_coord(fg["pitch_x"].to_numpy(), adir)
            true_line = _line(ax_all)
            # mask: keep the BROADCAST_VIS highest-up (largest attacking-x) players -> drops deep defenders
            order = np.argsort(ax_all)  # ascending: deepest first
            keep_idx = order[-BROADCAST_VIS:]
            masked = fg.iloc[keep_idx]
            naive_line = _line(attacking_coord(masked["pitch_x"].to_numpy(), adir))
            role_of = {int(t): role_map.get((ck, int(t)), (None, None))[1] for t in masked["track_id"]}
            froles = _formation_roles(formation.get((ck, team), ""))
            imp = impute_frame(masked, team=team, adir=adir, role_of=role_of, froles=froles, offsets=offsets)
            imp_ax = attacking_coord(
                np.concatenate([masked["pitch_x"].to_numpy(), [r["pitch_x"] for r in imp]]) if imp
                else masked["pitch_x"].to_numpy(), adir)
            imp_line = _line(imp_ax)
            imp_e.append(abs(imp_line - true_line))
            naive_e.append(abs(naive_line - true_line))
    return imp_e, naive_e


def real_frame_shift(aligned, roles, offsets) -> tuple[float, float, int]:
    """Match-level def_line_height on real frames: (raw, imputed, n_frames)."""
    role_map, formation = _chunk_meta(roles)
    raw, imp = [], []
    for ck, g in (aligned.groupby("chunk") if "chunk" in aligned.columns else [("_", aligned)]):
        dirs = complete_directions(resolve_attack_directions(g))
        pl = g[g["role"].isin(PLAYERS)].dropna(subset=["pitch_x", "pitch_y"])
        for (fr, team), fg in pl.groupby(["frame", "team"]):
            team = int(team)
            if team not in dirs or len(fg) < MIN_VIS_FOR_CENTROID:
                continue
            adir = dirs[team]
            raw.append(_line(attacking_coord(fg["pitch_x"].to_numpy(), adir)))
            role_of = {int(t): role_map.get((ck, int(t)), (None, None))[1] for t in fg["track_id"]}
            froles = _formation_roles(formation.get((ck, team), ""))
            rows = impute_frame(fg, team=team, adir=adir, role_of=role_of, froles=froles, offsets=offsets)
            allx = np.concatenate([fg["pitch_x"].to_numpy(), [r["pitch_x"] for r in rows]]) if rows \
                else fg["pitch_x"].to_numpy()
            imp.append(_line(attacking_coord(allx, adir)))
    return float(np.mean(raw)), float(np.mean(imp)), len(raw)


def main() -> None:
    print("=== P2 validation: off-screen imputation ===\n")
    agg = {"loo_i": [], "loo_b": [], "mask_i": [], "mask_n": []}
    for m in matches(processed_only=True):
        aligned = m.load_aligned()
        roles = assign_roles(aligned)
        offsets = learn_role_offsets(aligned, roles)
        ie, be = loo_accuracy(aligned, roles, offsets)
        mi, mn = masking_recovery(aligned, roles, offsets)
        raw, imp, n = real_frame_shift(aligned, roles, offsets)
        agg["loo_i"] += ie
        agg["loo_b"] += be
        agg["mask_i"] += mi
        agg["mask_n"] += mn
        print(f"[{m.id}]")
        if ie:
            print(f"  LOO position error   imputer {np.mean(ie):5.2f} m  vs centroid {np.mean(be):5.2f} m "
                  f"({'imputer better' if np.mean(ie) < np.mean(be) else 'baseline better'}, n={len(ie)})")
        if mi:
            print(f"  masking line recovery |imputed-true| {np.mean(mi):4.1f} m  vs |naive-true| "
                  f"{np.mean(mn):4.1f} m  ({'imputer better' if np.mean(mi) < np.mean(mn) else 'no gain'}, "
                  f"n={len(mi)})")
        print(f"  real-frame line height  raw {raw:4.1f} m -> imputed {imp:4.1f} m  (shift {imp-raw:+.1f} m)")
    print("\n=== pooled ===")
    if agg["loo_i"]:
        print(f"  LOO   imputer {np.mean(agg['loo_i']):.2f} m  vs centroid {np.mean(agg['loo_b']):.2f} m")
    if agg["mask_i"]:
        print(f"  mask  |imputed-true| {np.mean(agg['mask_i']):.1f} m  vs |naive-true| "
              f"{np.mean(agg['mask_n']):.1f} m  (n={len(agg['mask_i'])})")
        gain = 100 * (np.mean(agg["mask_n"]) - np.mean(agg["mask_i"])) / np.mean(agg["mask_n"])
        print(f"  bias reduction from imputation: {gain:+.0f}%")


if __name__ == "__main__":
    main()
