"""Automated team-mapping screen: BAS possession-link majority vs the Sofascore oracle.

The kit-anchor (``generator.team_anchor.align_teams_by_color``) can assign the wrong cluster to
``teams[0]`` for degenerate near-identical kits (red-vs-red / striped), which silently flips every
per-team ``score_state``-derived number (root cause: ``fingerprint.score_state.manu_index`` trusts the
registry order). This is the cheap permanent guard the 2026-07-23 pair-analysis audit used to catch two
such flips (``southampton_manutd``, ``tottenham_manutd``):

* **Man Utd poss-link share** -- ManU's share of BAS carrier possession links
  (``fingerprint.pass_network``); the dominant-link cluster must be the dominant-possession team.
* **Sofascore possession** -- ground-truth majority (``tools.oracle.parse_sofascore_team_stats``).

If the two disagree on *who had the ball* by more than BAS noise, the mapping is almost certainly
flipped: a loud WARNING is printed and a flag file is written to ``outputs/<id>/facts/``. Distinct-kit
matches sit within noise and pass. Run after align (or standalone) per match::

    python tools/verify_team_mapping.py                 # all registered matches with both oracles
    python tools/verify_team_mapping.py --match tottenham_manutd

Poss-links are read from the cached ``outputs/<id>/facts/pass_network.json`` (indexed by cluster id, so
a stale name label cannot fool the screen); if absent they are computed live.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import registry  # noqa: E402
from core.registry import Match  # noqa: E402
from fingerprint.score_state import manu_index  # noqa: E402
from tools import bas_validate  # noqa: E402
from tools.oracle import parse_sofascore_team_stats  # noqa: E402

MANU = "Man Utd"
# Opposite-majority disagreement wider than this (share vs oracle, both as fractions) => flip.
# Distinct-kit corpus max opposite-side gap is ~0.045 (brighton legs); the two known flips are ~0.20.
NOISE = 0.08
FLAG_NAME = "team_mapping_flag.json"


def manu_link_share(match: Match) -> tuple[float, int, int] | None:
    """Man Utd's share of BAS possession links (by cluster id, not stale name label).

    Reads ``outputs/<id>/facts/pass_network.json`` when present, else computes it live. Returns
    ``None`` when the match has no BAS pass stream (no poss links on either cluster).
    """
    cache = match.aligned.parent.parent / "facts" / "pass_network.json"
    if cache.exists():
        teams = json.loads(cache.read_text(encoding="utf-8"))["teams"]
        links = {int(v["team_int"]): int(v["n_poss_links"]) for v in teams.values()}
    else:
        from fingerprint import pass_network  # noqa: PLC0415
        rep = pass_network.match_report(match)
        links = {int(v["team_int"]): int(v["n_poss_links"]) for v in rep["teams"].values()}
    n0, n1 = links.get(0, 0), links.get(1, 0)
    total = n0 + n1
    if total == 0:
        return None
    mi = manu_index(match)
    return (n0, n1)[mi] / total, n0, n1


def oracle_manu_possession(match: Match) -> float | None:
    """Sofascore possession fraction for the Man Utd side (home/away from the registry)."""
    path = bas_validate.TRUTH.get(match.id)
    if path is None or not path.exists():
        return None
    parsed = parse_sofascore_team_stats(pd.read_parquet(path))
    side = "home" if match.home_team == MANU else "away"
    poss = parsed[side].get("possession_pct")
    return None if poss is None else float(poss) / 100.0


def verdict(share: float, oracle: float, noise: float = NOISE) -> str:
    """``"FLIP"`` if share and oracle name opposite majorities by more than ``noise``, else ``"OK"``.

    Same-majority matches pass at any gap (they agree on who had the ball); only an opposite-side
    disagreement wider than the noise band is a flip.
    """
    opposite = (share - 0.5) * (oracle - 0.5) < 0
    return "FLIP" if opposite and abs(share - oracle) > noise else "OK"


def screen_match(match: Match) -> dict | None:
    """Run the screen for one match; write a flag file on FLIP. ``None`` if an oracle is missing."""
    ls = manu_link_share(match)
    oracle = oracle_manu_possession(match)
    if ls is None or oracle is None:
        return None
    share, n0, n1 = ls
    v = verdict(share, oracle)
    row = {"match": match.id, "manu_index": manu_index(match), "poss_links_team0": n0,
           "poss_links_team1": n1, "manu_link_share": round(share, 3),
           "oracle_manu_possession": round(oracle, 3), "gap": round(abs(share - oracle), 3),
           "verdict": v}
    if v == "FLIP":
        flag = match.aligned.parent.parent / "facts" / FLAG_NAME
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(f"WARNING team-mapping FLIP suspected for {match.id}: ManU poss-link share "
              f"{share:.3f} vs Sofascore possession {oracle:.3f} (gap {abs(share - oracle):.3f} > "
              f"{NOISE}). Registry teams order for '{match.id}' is likely wrong. Flag -> {flag}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", default="all", help="registered match id, or 'all'")
    args = ap.parse_args()
    ids = [m.id for m in registry.matches()] if args.match == "all" else [args.match]
    rows: list[dict] = []
    for mid in ids:
        m = registry.get(mid)
        r = screen_match(m)
        if r is not None:
            rows.append(r)
    if not rows:
        print("no matches had both a poss-link stream and a Sofascore oracle")
        return
    df = pd.DataFrame(rows)
    print("\nteam-mapping screen (ManU poss-link share vs Sofascore possession):")
    print(df.to_string(index=False))
    n_flip = int((df["verdict"] == "FLIP").sum())
    print(f"\n{len(df)} screened | {n_flip} FLIP | {len(df) - n_flip} OK")


def _demo() -> None:
    """Self-check on the pure verdict seam (the known cases + the noise band)."""
    assert verdict(0.361, 0.56) == "FLIP"      # southampton pre-fix: opp sides, 0.20 gap
    assert verdict(0.647, 0.44) == "FLIP"      # tottenham pre-fix: opp sides, 0.21 gap
    assert verdict(0.639, 0.56) == "OK"        # southampton corrected: same majority
    assert verdict(0.475, 0.52) == "OK"        # brighton: opp sides but only 0.045 gap
    assert verdict(0.844, 0.67) == "OK"        # palace: same majority, wide gap
    assert verdict(0.517, 0.49) == "OK"        # fulham away: opp sides, 0.027 gap
    print("verify_team_mapping self-check OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        _demo()
    else:
        main()
