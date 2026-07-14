"""C6 validation of the P1 theory metrics against their FIFA EFI counterparts (France, 3 matches).

For each theory metric we pull France's CV value and the FIFA number it should track across the three
processed France matches, then report **direction** (does the CV metric move the same way as FIFA?) and
**rank agreement** (Spearman over the 3 matches). Honest scope: n=3 is coarse (like the phase-% C6) — this
tells us whether each metric is *directionally trustworthy*, not a calibrated count. Absolute gaps are
expected where the partial broadcast view undercounts (line breaks especially).

Reads the fact stores (``outputs/facts/<match>.json``); run ``python -m report.facts --match all`` first.
"""
from __future__ import annotations

from report.facts import load_facts

FRANCE_MATCHES = ("france_iraq", "france_senegal", "france_norway")

# metric (in cv.theory.France)  ->  (FIFA source, key, higher-CV-implies-higher-FIFA, mapping note)
#   fifa source: "ks" = key_stats, "ph" = phases (both are [home, away] lists)
#   "clean" = a like-for-like mapping (the real test); "weak" = a known dimensional/semantic mismatch
#   flagged for honesty (no better FIFA field exists), so a miss there is about the validator, not the metric.
VALIDATORS = {
    "pressing_intensity": ("ks", "defensive_pressures", True, "clean"),
    "line_breaks": ("ks", "completed_line_breaks", True, "clean (partial view undercounts absolute)"),
    "mean_overload": ("ks", "forced_turnovers", True, "clean"),
    "verticality": ("ph", "long_ball", True, "weak: France ~1-2% long every match, no variance to rank"),
    "final_third_lane_share": ("ks", "receptions_final_third", True, "weak: ratio vs volume count"),
    "counterpress_5s": ("ph", "counter-press", True, "weak: our success-rate vs FIFA time-share (freq)"),
}


def _france_idx(match: str) -> int:
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415
    r = json.loads(Path("data/france_roster.json").read_text())["matches"][match]
    return 0 if r["france_is_home"] else 1


def _spearman(xs: list[float], ys: list[float]) -> float:
    import numpy as np  # noqa: PLC0415
    if len(xs) < 2 or len({*xs}) < 2 or len({*ys}) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(xs))
    ry = np.argsort(np.argsort(ys))
    return float(np.corrcoef(rx, ry)[0, 1])


def collect() -> dict:
    """Per metric: the France CV series, the FIFA series, direction-correct flag, and rank agreement."""
    out: dict = {}
    for metric, (src, key, higher, note) in VALIDATORS.items():
        cv_series, fifa_series, matches = [], [], []
        for m in FRANCE_MATCHES:
            f = load_facts(m)
            if not f or not f.get("fifa"):
                continue
            th = f["cv"]["theory"].get("France")
            if not th:
                continue
            cvv = (th["counterpress_regain_curve"].get("5s") if metric == "counterpress_5s"
                   else th.get(metric))
            fifa_block = f["fifa"]["key_stats" if src == "ks" else "phases"]
            fv = fifa_block.get(key)
            fi = _france_idx(m)
            if cvv is None or not isinstance(fv, list):
                continue
            cv_series.append(float(cvv))
            fifa_series.append(float(fv[fi]))
            matches.append(m)
        if len(cv_series) < 2:
            out[metric] = {"n": len(cv_series), "note": "insufficient data"}
            continue
        rho = _spearman(cv_series, fifa_series)
        direction_ok = (rho >= 0) == higher if rho == rho else None  # noqa: PLR0124
        out[metric] = {"fifa_key": key, "matches": matches, "cv": cv_series, "fifa": fifa_series,
                       "rank_rho": rho, "direction_ok": direction_ok, "n": len(cv_series),
                       "mapping": "clean" if note.startswith("clean") else "weak", "note": note}
    return out


def main() -> None:
    res = collect()
    print("=== P1 theory-metric C6: CV vs FIFA EFI (France, per match) ===\n")
    print(f"  {'metric':<24}{'FIFA key':<24}{'map':>6}{'rho':>7}  dir  per-match (CV | FIFA)")
    for metric, r in res.items():
        if r.get("n", 0) < 2:
            print(f"  {metric:<24}{'—':<24}{'':>6}{'n<2':>7}")
            continue
        dirflag = "OK " if r["direction_ok"] else "off"
        pairs = "  ".join(f"{c:.2f}|{fi:.0f}" for c, fi in zip(r["cv"], r["fifa"]))
        print(f"  {metric:<24}{r['fifa_key']:<24}{r['mapping']:>6}{r['rank_rho']:>7.2f}  {dirflag}  {pairs}")
    clean = [r for r in res.values() if r.get("mapping") == "clean"]
    ok = sum(1 for r in clean if r.get("direction_ok"))
    print(f"\n  Clean-mapping metrics direction-correct: {ok}/{len(clean)} "
          f"(pressing, line-breaks, overloads — the real test).")
    print("  'weak'-mapped rows have a known validator mismatch (see notes), not a metric fault:")
    for m, r in res.items():
        if r.get("mapping") == "weak":
            print(f"    - {m}: {r['note']}")
    print("  n=3 matches — directional, not calibrated; absolute line-break count undercounts (partial view).")


if __name__ == "__main__":
    main()
