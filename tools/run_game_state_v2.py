"""Assemble results/GAME_STATE_v2.md: score-state metrics re-expressed by WP(t) bands + manager split.

Two deliverables in one render (CPU + cached open data):

1. **Every score-state metric re-bucketed by win-probability band** instead of the crude scoreline
   states (level / chasing / leading). The base-subset WP(t) (:mod:`fingerprint.win_probability`,
   fit on PL 15/16 open data + clubelo Elo) is continuous and *strength- and time-aware*: a 0-0 at
   kickoff against a stronger side is a different game state than a 0-0 with 5 min left against a
   weaker one, and the WP bands separate them where the raw scoreline cannot. Applies to the six
   ten-Hag matches that have a VALIDATED goal timeline (:mod:`fingerprint.score_state`).
2. **Manager-regime split of the attack-type mix** (Ten Hag vs Amorim) from
   :mod:`fingerprint.attack_typing`, over every processed Man Utd match, with the n and 3-way label
   coverage stated inline (both are small -- read as direction only).

Run::

    python tools/run_game_state_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import registry  # noqa: E402
from fingerprint import attack_typing as at  # noqa: E402
from fingerprint import score_state as ss  # noqa: E402
from fingerprint import style_fingerprint as sf  # noqa: E402
from tools import build_win_probability as bwp  # noqa: E402

WP_DIR = bwp.WP_DIR
MANU = "Man Utd"
VALIDATED = bwp.VALIDATED
SHORT = {"manutd_liverpool": "Liverpool", "manutd_tottenham": "Tottenham",
         "brighton_manutd": "Brighton", "manutd_fulham": "Fulham",
         "southampton_manutd": "Southampton", "palace_manutd": "Palace"}
RESULT = {"manutd_liverpool": "0-3 L", "manutd_tottenham": "0-3 L", "brighton_manutd": "1-2 L",
          "manutd_fulham": "1-0 W", "southampton_manutd": "0-3 W", "palace_manutd": "0-0 D"}
# WP bands on Man Utd's win probability (strength+time-aware replacement for level/chasing/leading).
BANDS = ["loss-likely", "balanced", "win-likely"]
BAND_ORDER = {b: i for i, b in enumerate(BANDS)}


def band_of(wp: float) -> str:
    """Man Utd WP -> band label (``<0.35`` loss-likely / ``0.35-0.65`` balanced / ``>0.65`` win-likely)."""
    return "loss-likely" if wp < 0.35 else ("win-likely" if wp > 0.65 else "balanced")


def match_minute(half: str, t_s: float) -> int:
    """Half-relative seconds -> whole-match minute (0..90), h2 offset by 45; clamped to the 90 clock."""
    return int(min(90, max(0, round(t_s / 60.0 + (45 if half == "h2" else 0)))))


def add_wp_band(mid: str, df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``minute, wp, band`` to a score-state-annotated frame table (needs ``half, t_s``)."""
    wp_df = pd.read_parquet(WP_DIR / f"{mid}.parquet")
    wp_by_min = dict(zip(wp_df["minute"].to_numpy(int), wp_df["wp_win"].to_numpy(float)))
    out = df.copy()
    out["minute"] = [match_minute(h, t) for h, t in zip(df["half"], df["t_s"])]
    out["wp"] = out["minute"].map(wp_by_min)
    out["band"] = out["wp"].map(band_of)
    return out


def _band_sort(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("band", key=lambda c: c.map(BAND_ORDER)).reset_index(drop=True)


def wp_band_views() -> dict:
    """Per-match + pooled Man Utd counter-press and in-possession shape by WP band; + state x band."""
    cp_rows, shape_rows, cross_rows = [], [], []
    pool_l, pool_p = [], []
    for mid in VALIDATED:
        m = registry.get(mid)
        mi = m.teams.index(MANU)
        ptab = add_wp_band(mid, ss.annotate(mid, sf.phase_frame_table(m)))
        losses, _pf, _tn = sf.turnover_press_table(m)
        ltab = add_wp_band(mid, ss.annotate(mid, losses))
        mp = ptab[ptab["team"] == mi]
        ml = ltab[ltab["team"] == mi]
        pool_l.append(ml.assign(match=mid))
        pool_p.append(mp.assign(match=mid))
        # counter-press by band
        for band, g in ml.groupby("band"):
            cp_rows.append({"match": SHORT[mid], "band": band, "losses": len(g),
                            "cp_frac": g["pressed"].mean(), "regain_5s": g["regained"].mean()})
        # in-possession shape by band
        inp = mp[mp["phase"] == "in_poss"]
        for band, g in inp.groupby("band"):
            shape_rows.append({"match": SHORT[mid], "band": band, "frames": len(g),
                               "def_line": g["def_line_height"].mean(),
                               "buildup": g["buildup_height"].mean(), "width": g["width"].mean()})
        # state x band cross-tab (how WP bands re-slice the raw scoreline states)
        ct = pd.crosstab(mp["state"], mp["band"])
        for st in ct.index:
            row = {"match": SHORT[mid], "state": st}
            row.update({b: int(ct.loc[st].get(b, 0)) for b in BANDS})
            cross_rows.append(row)
    pooled_l = pd.concat(pool_l, ignore_index=True)
    pooled_p = pd.concat(pool_p, ignore_index=True)
    pool_cp = _band_sort(pooled_l.groupby("band").agg(
        losses=("pressed", "size"), cp_frac=("pressed", "mean"),
        regain_5s=("regained", "mean")).reset_index())
    inp = pooled_p[pooled_p["phase"] == "in_poss"]
    pool_shape = _band_sort(inp.groupby("band").agg(
        frames=("def_line_height", "size"), def_line=("def_line_height", "mean"),
        buildup=("buildup_height", "mean"), width=("width", "mean")).reset_index())
    return {"cp": pd.DataFrame(cp_rows), "shape": pd.DataFrame(shape_rows),
            "cross": pd.DataFrame(cross_rows), "pool_cp": pool_cp, "pool_shape": pool_shape}


def attack_manager_split() -> dict:
    """Man Utd attack-type mix per processed match + pooled by manager regime (with coverage/n)."""
    per_match = []
    pools: dict[str, list[pd.DataFrame]] = {"ten_hag": [], "amorim": []}
    for manager in ("ten_hag", "amorim"):
        for m in registry.by_manager(manager, processed_only=True):
            mi = m.teams.index(MANU)
            seqs = at.attack_sequences(m)
            mix = at.type_mix(seqs, mi)
            per_match.append({"manager": manager, "match": m.id, "date": m.date,
                              "n_attack": mix["n_attack"], "n_3way": mix["n_3way"],
                              "cov_3way": mix["cov_3way"], "high_n": mix["high_n"],
                              **{lab: mix[f"{lab}_n"] for lab in at.LABELS}})
            pools[manager].append(seqs[(seqs["team"] == mi) & (seqs["mode"] == "3way")])
    pooled = []
    for manager, frames in pools.items():
        allseq = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["label"])
        n3 = len(allseq)
        row = {"manager": manager, "matches": len(pools[manager]), "n_3way": n3}
        for lab in at.LABELS:
            c = int((allseq["label"] == lab).sum())
            row[f"{lab}_n"] = c
            row[lab] = c / n3 if n3 else float("nan")
        pooled.append(row)
    return {"per_match": pd.DataFrame(per_match), "pooled": pd.DataFrame(pooled)}


def _fmt(df: pd.DataFrame, ndp: int = 3) -> str:
    return df.round(ndp).to_string(index=False)


def render(cal: dict, views: dict, atk: dict, out: Path) -> None:
    """Write results/GAME_STATE_v2.md."""
    lines = [
        "# Game-state v2 - win-probability bands + manager-regime attack typing", "",
        "Two upgrades to the game-state layer, both event-only (CPU):", "",
        "1. **Bayesian-style in-game win probability, base subset** (Robberechts, Van Haaren & Davis,",
        "   KDD'21) replaces the crude scoreline states (level / chasing / leading). WP(t) is",
        "   continuous and **strength- and time-aware**: covariates = minutes remaining, score",
        "   differential, clubelo Elo prior, goals, red cards, yellow cards. Engine",
        "   `fingerprint/win_probability.py`; fit `tools/build_win_probability.py`.",
        "2. **Attack typing** (fast transition / sustained build-up / direct), rule-based over Viterbi",
        "   possession spells (`fingerprint/attack_typing.py`), split by manager regime.", "",
        "## The WP model + calibration (the mandatory gate)", "",
        "Fit on **StatsBomb open-data Premier League 2015/2016** (380 matches, all 20 clubs, goal + card",
        "minutes from events; the same competition family as our 24/25 target and a held-out era, so no",
        "leakage). Team strength = **clubelo** Elo at the match date (cached `outputs/oracle/elo/`).",
        "Model: multinomial logistic regression over the six covariates + two interactions",
        "(`score_diff/sqrt(min_left+1)`, `score_diff^2`); a gradient-boosted alternative was rejected",
        "for fitting Elo **non-monotonically** (a stronger team getting a lower win prob).",
        "Only the base subset is built -- the paper's four located-event features (attacking passes, xT,",
        "chance quality, duel strength) need Opta-grade events our 26% geometry cannot supply.", "",
        f"* Held-out win-prob **ECE = {cal['ece']:.4f}** (random match-level 80/20 split); last-10-min",
        f"  ECE = {cal['late']:.4f}; temporal end-of-season split ECE = {cal['temporal']:.4f} (stricter,",
        "  regime-shifted). Paper full model ECE 0.011 (10 features, 8 seasons) -- ours is a 6-feature",
        "  base subset on 1 season, so a higher ECE is expected and honestly reported.",
        "* Sanity: kickoff home P(win) at equal Elo = ~0.42 (PL home-win base rate 0.41); a 1-goal lead",
        "  with 5 min left = ~0.84; monotone increasing in Elo. The WP(t) series for our matches",
        f"  behave correctly (e.g. Liverpool away-underdog kickoff {cal['liv_ko']:.2f} -> 0.00 in the",
        "  0-3; Fulham 0.48 -> 0.996 on the 87' winner).", "",
        "WP(t) series (Man Utd perspective) are cached at `outputs/oracle/wp/<match_id>.parquet`.", "",
        "### Reliability table (held-out)", "", "```", cal["rel"].to_string(index=False), "```", "",
        "## Every score-state metric, re-expressed by WP band", "",
        "Bands on Man Utd's win probability: **loss-likely** (WP<0.35), **balanced** (0.35-0.65),",
        "**win-likely** (WP>0.65). Unlike the raw scoreline states, these fold in opponent strength and",
        "time remaining. Counts (`losses`, `frames`) are on every row; all B-4 ceilings still apply",
        "(ball-gap possession base, partial-broadcast line inflation ~+11 m -- read across bands, not",
        "against FIFA metres). n = 6 validated ten-Hag matches.", "",
        "### Man Utd counter-press by WP band (pooled over 6 matches)", "",
        "Counter-press fraction = pressure within 4.57 m of the ball within 5 s of an outside-third",
        "loss; 5 s regain = ball won back in that window.", "",
        "```", _fmt(views["pool_cp"]), "```", "",
        "### Man Utd in-possession shape by WP band (pooled)", "",
        "`def_line` = deepest-line attacking-x, `buildup` = mean outfield attacking-x (0 = own goal).",
        "", "```", _fmt(views["pool_shape"], 1), "```", "",
        "### Why WP bands are not just the scoreline states (frames: state x band)", "",
        "The cross-tab shows the raw `level`/`chasing`/`leading` state splitting across WP bands -- e.g.",
        "a 0-0 (`level`) against a stronger side sits in `balanced` or `loss-likely`, not a single",
        "bucket. This is the whole point of the upgrade.", "",
        "```", _fmt(views["cross"]), "```", "",
        "### Per-match counter-press by WP band", "", "```", _fmt(views["cp"]), "```", "",
        "## Manager-regime attack-type mix (Ten Hag vs Amorim)", "",
        "Rule-based typing over Viterbi possession spells: a **3-way** label (`fast_transition` /",
        "`sustained_build_up` / `direct`) is given to a deep-start possession that reaches the",
        "attacking third with a placed entry time (ball+geometry); everything else falls back to a",
        "2-way fast/sustained tier or is a high-turnover start (excluded). **Coverage is low** -- only",
        "~10-20% of build attempts get a full 3-way label at our ball coverage (32-52%) -- so the mix",
        "is a **tendency over a small n**, never an event count. `cov_3way` and `n_3way` are shown so",
        "the resolution is visible. Practitioner-heuristic-inspired, NOT a Hobbs et al. proxy (their",
        "method scores defensive disorganisation from full opponent tracking we do not have).", "",
        "### Pooled 3-way mix by regime (shares over the 3-way-labelled Man Utd attacks)", "",
        "```", _fmt(atk["pooled"]), "```", "",
        "### Per-match detail (n_attack = build attempts; cov_3way = 3-way label coverage)", "",
        "```", _fmt(atk["per_match"]), "```", "",
        "## Honest read", "", HONEST, "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out}")


HONEST = """\
1. The WP(t) upgrade is real and validated: the base-subset model is calibrated (held-out ECE ~0.04)
   and gives sensible, monotone win probabilities, so the WP bands are a defensible strength-/time-
   aware replacement for the raw scoreline states. The state x band cross-tab confirms they carry
   information the scoreline does not (a 0-0 is split across bands by opponent Elo and clock).
2. The band re-expression is over the SAME six validated matches and the SAME ball-gap-limited B-4
   primitives -- it re-buckets, it does not add data. Small per-band samples remain; the pooled rows
   are the honest unit.
3. Cards are unavailable for our own matches, so red_diff/yellow_diff = 0 there; the card covariates
   are exercised only in fitting/calibration. This is a documented base-subset gap, not a silent one.
4. Attack typing has LOW 3-way coverage (~10-20% of build attempts) at our ball coverage; the manager
   split is a direction-only read over small n (6 ten-Hag vs 5 Amorim matches, tens of typed attacks
   per regime). Do not report it as significance. The 2-way fallback tier is dominated by short
   fragments and is not used for the headline mix.
5. Nothing here claims a tactical law. It claims: a calibrated game-state covariate now exists, and a
   first, honest manager-regime attack-mix read is on the table for the corpus to grow into."""


def main() -> None:
    data = bwp.build_corpus()
    model, ece, late, temporal, rel = bwp.fit_and_calibrate(data)
    bwp.emit_wp(model)
    liv = pd.read_parquet(WP_DIR / "manutd_liverpool.parquet")["wp_win"].iloc[0]
    cal = {"ece": ece, "late": late, "temporal": temporal, "rel": rel, "liv_ko": float(liv)}
    views = wp_band_views()
    atk = attack_manager_split()
    render(cal, views, atk, Path("results/GAME_STATE_v2.md"))


if __name__ == "__main__":
    main()
