"""Does the broadcast calibration failure of the frozen B4 v1 regions replicate across matches?

``results/FULL_MATCH_RECONSTRUCTION.md`` measured the FROZEN 50%/90% conformal regions against
real re-appearances on one match (``tottenham_manutd``) and found 20.5% / 55.3% coverage where
they promise 50 / 90, after passing 6/6 on Metrica simulation. One match is not a finding. This
aggregates the same harness (``tools.full_match_reconstruction``, run per match) over several
matches and answers one question: does it replicate?

It **measures only**. Nothing is re-fitted and nothing is re-conformalised -- the re-appearance
sample is a biased, near-stationary subset of occlusions (the tracker only re-associates a player
who came back near where it left), so multipliers fitted on it would be tight exactly where the
model is used. The displacement columns are printed so that bias stays visible.

Inputs are the per-match artifacts the harness already writes::

    python -m tools.full_match_reconstruction --match manutd_liverpool
    python -m tools.calibration_replication --matches tottenham_manutd,manutd_liverpool

The observation-noise sigma is recomputed here with the harness's own
``full_match_reconstruction.noise_floor`` rather than parsed out of a log.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core import registry
from synthesizer.imputation import BIN_LABELS
from tools.full_match_reconstruction import OUT_ROOT, noise_floor
from tools.tactical_clip import CALIB_MAX_M

DEFAULT_MATCHES = ("tottenham_manutd", "manutd_liverpool", "manutd_brighton", "liverpool_manutd")
MIN_BUCKET_N = 30  # events a bucket needs before its PICP is quoted as testable


def load_events(match_id: str, root: Path = OUT_ROOT) -> pd.DataFrame:
    """Within-track (primary) scored re-appearances for one match.

    Args:
        match_id: Registry match id.
        root: Directory the reconstruction harness wrote its artifacts to.

    Returns:
        The primary-link event rows, with a ``match`` column added.

    Raises:
        SystemExit: If the match has no event parquet (run the harness first).
    """
    path = root / f"{match_id}_reappearance_events.parquet"
    if not path.exists():
        raise SystemExit(f"{path} missing -- run tools.full_match_reconstruction --match {match_id}")
    ev = pd.read_parquet(path)
    ev = ev[ev["link"] == "within-track"].copy()
    ev["match"] = match_id
    return ev


def sigma_of(match_id: str) -> tuple[float, int]:
    """Per-axis observation-noise sigma of our projected positions, and its triple count."""
    m = registry.get(match_id)
    df = m.load_aligned()
    gated = df[(df["calib_error_m"] <= CALIB_MAX_M) & df["pitch_x"].notna() & df["pitch_y"].notna()
               & df["team"].isin([0, 1]) & df["role"].isin(["player", "goalkeeper"])]
    return noise_floor(gated)


def pooled_sigma(sigmas: list[tuple[float, int]]) -> float:
    """Triple-count-weighted pooling of per-match noise sigmas (variances add, not sigmas)."""
    n = np.array([k for _, k in sigmas], float)
    s = np.array([v for v, _ in sigmas], float)
    ok = np.isfinite(s) & (n > 0)
    return float(np.sqrt(np.sum(n[ok] * s[ok] ** 2) / np.sum(n[ok]))) if ok.any() else float("nan")


def _rmse(v) -> float:
    """Root mean square of a series."""
    a = np.asarray(v, float)
    return float(np.sqrt(np.mean(a**2))) if a.size else float("nan")


def bucket_rows(ev: pd.DataFrame, widen: float = 0.0) -> pd.DataFrame:
    """Census, error, coverage and last-seen displacement per horizon bucket, plus an ALL row.

    Args:
        ev: Scored within-track events.
        widen: Per-axis observation-noise sigma. When non-zero, two extra columns report the
            coverage of the SAME frozen regions inflated in quadrature by it -- a diagnostic of
            why they miss, never a re-calibration.

    Returns:
        One row per bucket.
    """
    rows = []
    for b, lab in enumerate([*BIN_LABELS, "ALL"]):
        e = ev if lab == "ALL" else ev[ev["bucket"] == b]
        if e.empty:
            rows.append({"bucket": lab, "n": 0})
            continue
        wide = {}
        if widen:
            for lvl in (50, 90):
                r = np.sqrt(e[f"r{lvl}_m"].to_numpy() ** 2 + 2.0 * widen**2)
                wide[f"picp{lvl}_wide"] = 100 * float((e["err_m"].to_numpy() <= r).mean())
        rows.append({
            **wide,
            "bucket": lab, "n": len(e), "rmse": _rmse(e["err_m"]),
            "rmse_hold": _rmse(e["err_hold_m"]),
            "picp50": 100 * float(e["inside50"].mean()),
            "picp90": 100 * float(e["inside90"].mean()),
            "r50": float(e["r50_m"].mean()), "r90": float(e["r90_m"].mean()),
            # hold-last error IS |truth(t1) - last seen(t0)|: the selection-bias check.
            "disp_p50": float(e["err_hold_m"].median()),
            "disp_mean": float(e["err_hold_m"].mean()),
            "testable": len(e) >= MIN_BUCKET_N and lab != "ALL",
        })
    return pd.DataFrame(rows)


def print_match(match_id: str, ev: pd.DataFrame, sig: tuple[float, int],
                widen: float = 0.0) -> pd.DataFrame:
    """Print one match's bucket table and return it."""
    tab = bucket_rows(ev, widen)
    print(f"\n{match_id}: {len(ev)} within-track re-appearances, "
          f"sigma = {sig[0]:.2f} m per axis ({sig[1]} midpoint triples)")
    print("  horizon   |    n | emitted |   hold | PICP50 | PICP90 |  r50 |  r90 | disp p50 | "
          "disp mean | testable" + ("| 50+sig | 90+sig" if widen else ""))
    for r in tab.itertuples():
        if r.n == 0:
            print(f"  {r.bucket:9s} |    0 |       - |      - |      - |      - |    - |    - |"
                  "        - |         - | -")
            continue
        flag = "yes" if r.testable else ("(ALL)" if r.bucket == "ALL" else f"n<{MIN_BUCKET_N}")
        extra = f"| {r.picp50_wide:6.1f} | {r.picp90_wide:6.1f}" if widen else ""
        print(f"  {r.bucket:9s} | {r.n:4d} | {r.rmse:7.2f} | {r.rmse_hold:6.2f} | {r.picp50:6.1f} | "
              f"{r.picp90:6.1f} | {r.r50:4.1f} | {r.r90:4.1f} | {r.disp_p50:8.2f} | "
              f"{r.disp_mean:9.2f} | {flag:8s}" + extra)
    return tab.assign(match=match_id)


def verdict(per_match: pd.DataFrame) -> str:
    """One-line replication verdict from the per-match ALL rows and testable buckets."""
    a = per_match[per_match["bucket"] == "ALL"]
    t = per_match[per_match["testable"].fillna(False)]
    fails = int(((t["picp50"] < 50) & (t["picp90"] < 90)).sum())
    return (f"ALL-row PICP50 spread {a['picp50'].min():.1f}-{a['picp50'].max():.1f} "
            f"(nominal 50), PICP90 {a['picp90'].min():.1f}-{a['picp90'].max():.1f} (nominal 90); "
            f"{fails}/{len(t)} testable buckets under-cover at BOTH levels")


def _self_check() -> None:
    """Assert the pooling algebra and the bucket table on synthetic inputs."""
    # variances pool by count, not sigmas: two equal counts of 1 and 3 -> sqrt(5), not 2
    assert abs(pooled_sigma([(1.0, 10), (3.0, 10)]) - np.sqrt(5.0)) < 1e-12
    assert abs(pooled_sigma([(0.5, 100), (0.5, 7)]) - 0.5) < 1e-12
    assert np.isnan(pooled_sigma([(float("nan"), 0)]))
    ev = pd.DataFrame({
        "bucket": [0, 0, 1], "err_m": [3.0, 4.0, 0.0], "err_hold_m": [1.0, 3.0, 0.0],
        "inside50": [True, False, True], "inside90": [True, True, True],
        "r50_m": [1.0, 1.0, 1.0], "r90_m": [2.0, 2.0, 2.0],
    })
    t = bucket_rows(ev).set_index("bucket")
    assert t.loc["0-1s", "n"] == 2 and abs(t.loc["0-1s", "rmse"] - 3.5355) < 1e-3
    assert t.loc["0-1s", "picp50"] == 50.0 and t.loc["ALL", "picp90"] == 100.0
    assert t.loc["0-1s", "disp_p50"] == 2.0 and t.loc["ALL", "n"] == 3
    assert not t.loc["0-1s", "testable"] and t.loc["3-5s", "n"] == 0
    print("calibration_replication self-check OK (sigma pooling, bucket table)")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matches", default=",".join(DEFAULT_MATCHES))
    ap.add_argument("--root", default=str(OUT_ROOT))
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        _self_check()
        return
    ids = [s.strip() for s in args.matches.split(",") if s.strip()]
    root = Path(args.root)
    print("=" * 100)
    print("CALIBRATION REPLICATION -- frozen B4 v1 regions vs real re-appearances (measure only)")
    print("=" * 100)
    tabs, sigmas, evs = [], [], []
    for mid in ids:
        ev = load_events(mid, root)
        sig = sigma_of(mid)
        tabs.append(print_match(mid, ev, sig))
        sigmas.append(sig)
        evs.append(ev)
    per_match = pd.concat(tabs, ignore_index=True)

    print("\n" + "=" * 100)
    print("POOLED (all matches, within-track events)")
    print("=" * 100)
    pool = pd.concat(evs, ignore_index=True)
    ps = pooled_sigma(sigmas)
    print_match("POOLED", pool, (ps, sum(k for _, k in sigmas)), widen=ps)
    print("  (50+sig / 90+sig = the SAME frozen regions widened in quadrature by the POOLED "
          "observation-noise sigma -- a diagnostic, not a re-calibration.)")
    print("\nper-match sigma: " + ", ".join(f"{m} {s:.3f} m (n={k})"
                                            for m, (s, k) in zip(ids, sigmas, strict=True)))
    print(f"pooled sigma: {pooled_sigma(sigmas):.3f} m per axis "
          f"({np.sqrt(2) * pooled_sigma(sigmas):.2f} m RMS radially)")
    print("\nVERDICT: " + verdict(per_match))
    print("NOTE: no re-conformalisation is performed here; the displacement columns show the "
          "sample is easier than a real occlusion.")


if __name__ == "__main__":
    main()
