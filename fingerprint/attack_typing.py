"""Attack-type typing: counter / sustained-possession / direct, rule-based over possession spells.

Practitioner-heuristic-inspired attack labelling (NOT a Hobbs et al. proxy -- their innovation is
defensive-disorganisation scoring from full opponent positions, which our event/geometry stack cannot
supply; see the STYLE_RESEARCH_METHODS Axis-2 honesty trail). We only read, per possession spell of a
team, two things practitioners use to eyeball a fast break vs a worked build-up: **how long from the
regain to the attacking-third entry**, and **how many passes** the spell contained.

Pipeline per match (reuses validated primitives only):
  * possession = Viterbi-smoothed nearest-carrier track (:func:`generator.ball.assign_possession`),
  * spells = contiguous same-team runs of that track,
  * passes = :func:`fingerprint.possession_metrics.extract_passes` (carrier hand-offs),
  * attacking-third entry = the ball crossing ``ATT_THIRD_X`` in the team's attack direction.

Labels (mutually exclusive):
  * ``sustained_build_up`` -- slow OR many-pass sequence (possession domination),
  * ``direct``             -- reached the third fast with <=2 passes (long ball / vertical bypass),
  * ``fast_transition``    -- reached the third fast with a few (3-5) passes (a counter).

A **3-way** label needs the ball+geometry to place an attacking-third entry time; where that is
missing but the spell still has a duration and pass count we fall back to a **2-way**
``fast`` / ``sustained`` label. Sequences with neither a pass nor an entry are untyped. Every consumer
gets the label-coverage split (3-way / 2-way / untyped) so the mix is never read as fully resolved.

Ceilings (ponytail): ball links only 32-52% of frames and homography ~26% (STYLE_RESEARCH_METHODS),
so spells are fragments of the true possessions and the mix is a *tendency*, not an event count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.pitch import ATT_THIRD_X
from fingerprint.possession_metrics import extract_passes
from fingerprint.structural_metrics import attacking_coord, resolve_attack_directions
from generator.ball import assign_possession

# Rule thresholds (seconds / pass counts). Deliberately round practitioner numbers, not fitted --
# ponytail: naive fixed thresholds; expose + retune if a labelled clip set ever lands.
FAST_ENTRY_S = 12.0     # regain -> attacking-third entry at or under this = a "fast" attack
SUSTAINED_PASSES = 6    # this many passes in a spell = a worked build-up regardless of speed
DIRECT_PASSES = 2       # <= this many passes but still reached the third = a direct / long-ball attack
COUNTER_PASSES = 5      # fast attacks with 3..this passes are counters (else fold into sustained)
TWO_WAY_SUSTAINED = 3   # 2-way fallback: this many passes (no confirmed third-entry) = "sustained"

LABELS = ("fast_transition", "sustained_build_up", "direct")


def possession_spells(possession: pd.DataFrame, *, max_gap_frames: int | None = None) -> list[dict]:
    """Contiguous same-team runs of a possession track: ``team, f0, f1, n_samples``.

    A spell breaks on a team change OR (when ``max_gap_frames`` is set) a sampling gap larger than
    that many frames -- the ball links only intermittently, so without the gap break two genuinely
    separate possessions with an unlinked stretch between them fuse into one implausible mega-spell.

    Args:
        possession: ``frame, carrier, team`` (one chunk), any order.
        max_gap_frames: max frame gap between consecutive samples inside one spell (``None`` = no
            gap break, contiguous by team only).

    Returns:
        One dict per spell with the winning ``team``, first/last frame ``f0``/``f1`` and the number
        of possession samples in the run, in frame order.
    """
    if possession.empty:
        return []
    p = possession.sort_values("frame").reset_index(drop=True)
    team = p["team"].to_numpy(int)
    frame = p["frame"].to_numpy(int)
    out: list[dict] = []
    start = 0
    for i in range(1, len(p) + 1):
        gap = (max_gap_frames is not None and i < len(p)
               and frame[i] - frame[i - 1] > max_gap_frames)
        if i == len(p) or team[i] != team[start] or gap:
            out.append({"team": int(team[start]), "f0": int(frame[start]),
                        "f1": int(frame[i - 1]), "n_samples": i - start})
            start = i
    return out


def _progression(spell: dict, ball_x: dict[int, float], adir: int | None,
                 fps: float) -> tuple[float, float]:
    """``(start_ac, entry_s)``: the regain-zone attacking-x and seconds to the attacking-third entry.

    ``start_ac`` is the team's attacking coordinate of the ball at the spell's first linked sample
    (where the possession began); ``entry_s`` is the seconds from the spell start to the first frame
    the ball crosses :data:`core.pitch.ATT_THIRD_X`. Both are ``nan`` without geometry or a ball
    sample in the spell.
    """
    if adir is None:
        return float("nan"), float("nan")
    frames = sorted(f for f in ball_x if spell["f0"] <= f <= spell["f1"])
    if not frames:
        return float("nan"), float("nan")
    start_ac = float(attacking_coord(np.array([ball_x[frames[0]]]), adir)[0])
    entry_s = float("nan")
    for f in frames:
        if attacking_coord(np.array([ball_x[f]]), adir)[0] >= ATT_THIRD_X:
            entry_s = (f - spell["f0"]) / fps
            break
    return start_ac, entry_s


def label_sequence(n_passes: int, start_ac: float, entry_s: float,
                   duration_s: float) -> tuple[str | None, str | None]:
    """Label one attacking sequence and say whether it is a 3-way / 2-way / high-start decision.

    Gating: a spell whose regain was **already in the attacking third** (``start_ac >= ATT_THIRD_X``)
    is a high turnover, not a build-up/counter/direct attack -- it is labelled ``high_start`` (mode
    ``"high"``) and kept out of the attack-type mix. A **3-way** label is given to a genuine deep-start
    progression with a known attacking-third ``entry_s`` (ball+geometry present): ``direct`` (fast,
    <=2 passes = long-ball bypass), ``fast_transition`` (fast, 3-5 passes = counter),
    ``sustained_build_up`` (slow entry OR many passes). Otherwise a **2-way** ``fast`` / ``sustained``
    fallback keys off the **pass count** only (fused-spell duration is unreliable at our ball
    coverage). Nothing-happened spells are ``(None, None)``.

    Args:
        n_passes: passes completed in the spell.
        start_ac: attacking-x of the ball at the regain, or ``nan`` if no ball sample / no geometry.
        entry_s: seconds regain->attacking-third entry, or ``nan`` if no entry was placed.
        duration_s: spell duration in seconds (reported, not used in the rule -- see docstring).

    Returns:
        ``(label, mode)`` with ``mode`` in ``{"3way", "2way", "high"}`` (or ``(None, None)``).
    """
    del duration_s  # reported alongside, but fused-spell duration is too noisy to threshold on
    if not np.isnan(start_ac) and start_ac >= ATT_THIRD_X:
        return "high_start", "high"
    if not np.isnan(entry_s):
        if n_passes >= SUSTAINED_PASSES or entry_s > FAST_ENTRY_S:
            return "sustained_build_up", "3way"
        if n_passes <= DIRECT_PASSES:
            return "direct", "3way"
        if n_passes <= COUNTER_PASSES:
            return "fast_transition", "3way"
        return "sustained_build_up", "3way"
    if n_passes >= 1 or not np.isnan(start_ac):
        return ("sustained" if n_passes >= TWO_WAY_SUSTAINED else "fast"), "2way"
    return None, None


def attack_sequences(match) -> pd.DataFrame:
    """Per-spell attack-type table for one match: ``chunk, team, f0, f1, n_passes, entry_s, label, mode``.

    Iterates the match's linked-ball chunks, builds the Viterbi possession track, splits it into
    spells and labels each. Only spells that either completed a pass or reached the attacking third
    are rows; the caller reads label coverage from the ``mode`` column.

    Args:
        match: a :class:`core.registry.Match`.

    Returns:
        One row per typed spell (empty frame with the schema if nothing links).
    """
    cols = ["chunk", "team", "f0", "f1", "duration_s", "n_passes", "start_ac", "entry_s",
            "label", "mode"]
    df = match.load_aligned()
    players = df[df["role"].isin(["player", "goalkeeper"])]
    rows: list[dict] = []
    for ck, path in match.ball_chunks():
        g = players[players["chunk"] == ck]
        if g.empty:
            continue
        adir = resolve_attack_directions(g)
        ball = pd.read_parquet(path)
        poss = assign_possession(ball, g.dropna(subset=["pitch_x", "pitch_y"]), smooth=True)
        if poss.empty:
            continue
        fps = match.chunk_fps(ck)
        passes = extract_passes(poss, g, max_gap_s=None)
        ball_x = {int(r.frame): float(r.x) for r in ball.dropna(subset=["x"]).itertuples(index=False)}
        # No gap-split: splitting fragments genuine deep->third progressions and destroys 3-way
        # coverage; the 3-way rule already filters mega-spells via the deep-start + real-entry gate.
        for sp in possession_spells(poss):
            t = sp["team"]
            if t < 0:
                continue
            n_passes = int(((passes["team"] == t) & (passes["frame"] >= sp["f0"])
                            & (passes["frame"] <= sp["f1"])).sum())
            duration_s = (sp["f1"] - sp["f0"]) / fps
            start_ac, entry_s = _progression(sp, ball_x, adir.get(t), fps)
            label, mode = label_sequence(n_passes, start_ac, entry_s, duration_s)
            if label is None:
                continue
            rows.append({"chunk": ck, "team": t, "f0": sp["f0"], "f1": sp["f1"],
                         "duration_s": round(duration_s, 2), "n_passes": n_passes,
                         "start_ac": round(start_ac, 1) if not np.isnan(start_ac) else float("nan"),
                         "entry_s": round(entry_s, 2) if not np.isnan(entry_s) else float("nan"),
                         "label": label, "mode": mode})
    return pd.DataFrame(rows, columns=cols)


def type_mix(seqs: pd.DataFrame, team: int) -> dict:
    """Attack-type mix + label coverage for one team from an :func:`attack_sequences` table.

    Returns the share of that team's typed spells in each 3-way label (over the 3-way-labelled spells
    only), the raw counts, and coverage = the 3-way / 2-way / untyped split so the mix's resolution is
    always visible. 2-way ``fast``/``sustained`` counts ride along separately.

    Args:
        seqs: an :func:`attack_sequences` table.
        team: team index to summarise.

    Returns:
        Dict with ``n_seq``, ``n_3way``, ``n_2way``, ``cov_3way`` and, per 3-way label, ``<label>``
        (share) and ``<label>_n`` (count); plus ``fast_n`` / ``sustained_n`` for the fallback tier.
    """
    t = seqs[seqs["team"] == team]
    three = t[t["mode"] == "3way"]
    two = t[t["mode"] == "2way"]
    high = int((t["mode"] == "high").sum())
    n_attack = len(three) + len(two)   # build/progression attempts (high starts excluded)
    out: dict = {"n_attack": n_attack, "n_3way": len(three), "n_2way": len(two), "high_n": high,
                 "cov_3way": len(three) / n_attack if n_attack else float("nan")}
    for lab in LABELS:
        c = int((three["label"] == lab).sum())
        out[f"{lab}_n"] = c
        out[lab] = c / len(three) if len(three) else float("nan")
    out["fast_n"] = int((two["label"] == "fast").sum())
    out["sustained_n"] = int((two["label"] == "sustained").sum())
    return out


def _demo() -> None:
    """Self-check on the pure seams: spell splitting + the label rules."""
    poss = pd.DataFrame({"frame": [0, 10, 20, 30, 40, 100, 110],
                         "carrier": [1, 2, 1, 3, 4, 9, 8],
                         "team": [0, 0, 0, 0, 0, 1, 1]})
    sp = possession_spells(poss)
    assert len(sp) == 2 and sp[0]["team"] == 0 and sp[0]["f0"] == 0 and sp[0]["f1"] == 40, sp
    assert sp[1]["team"] == 1 and sp[1]["n_samples"] == 2, sp
    # gap split: a > max_gap jump inside a same-team run breaks the spell
    gappy = pd.DataFrame({"frame": [0, 10, 20, 90, 100], "carrier": [1, 2, 1, 3, 4],
                          "team": [0, 0, 0, 0, 0]})
    spg = possession_spells(gappy, max_gap_frames=15)
    assert len(spg) == 2 and spg[0]["f1"] == 20 and spg[1]["f0"] == 90, spg
    # high start (regain already in the attacking third) is excluded from the mix
    assert label_sequence(2, 90.0, 0.0, 1.0) == ("high_start", "high")
    # 3-way rules (deep start = start_ac below the 70 m third)
    assert label_sequence(1, 30.0, 5.0, 6.0) == ("direct", "3way")            # fast, <=2 passes
    assert label_sequence(4, 30.0, 5.0, 6.0) == ("fast_transition", "3way")   # fast, 3-5 passes
    assert label_sequence(8, 30.0, 4.0, 9.0) == ("sustained_build_up", "3way")  # many passes
    assert label_sequence(2, 30.0, 20.0, 25.0) == ("sustained_build_up", "3way")  # slow entry
    # 2-way fallback keys off pass count only (no confirmed third-entry)
    assert label_sequence(3, float("nan"), float("nan"), 4.0) == ("sustained", "2way")
    assert label_sequence(1, float("nan"), float("nan"), 15.0) == ("fast", "2way")
    assert label_sequence(0, 30.0, float("nan"), 5.0) == ("fast", "2way")  # deep start, no entry
    assert label_sequence(0, float("nan"), float("nan"), 0.0) == (None, None)  # nothing happened
    print("attack_typing self-check OK")


if __name__ == "__main__":
    _demo()
