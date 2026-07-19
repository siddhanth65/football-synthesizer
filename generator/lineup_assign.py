"""B-1: lineup-prior identity assignment (bipartite assignment, not open-set recognition).

Team sheets are public pre-match, so identity is an *assignment* problem: the candidates are the
known rostered players (11 starters + subs keyed by entry/exit minute), the targets are our track
evidence (jersey-vote groups from the close-up anchors), and we solve a per-team, per-window
Hungarian match that fuses four costs:

* **jersey posterior** -- the dominant term. Sofascore ``shirtNumber`` is a *unique key within a
  team*, so a group that votes number ``N`` can only be the one rostered player wearing ``N``. A
  number mismatch is therefore infeasible, not merely expensive: the jersey term never trades off
  against geometry. The two teams *do* reuse numbers, which is why the solve is per-team.
* **formation-position prior** -- a soft corroborator. The oracle position is coarse (G/D/M/F), so
  this maps to an expected longitudinal band and only nudges confidence; it can never override a
  jersey match (weight :data:`W_POS` caps its contribution well below a mismatch).
* **GK role flag** -- a hard veto: a keeper number on an outfield track (or vice-versa) is rejected.
* **team / kit** -- handled structurally by solving each team separately (per-team candidate set).

Abstention is first-class: every player and every group gets a dummy option at :data:`ABSTAIN_FLOOR`,
so nothing is forced below confidence. A group whose voted number belongs to no in-window candidate
(e.g. a sub's number read before their entry minute) simply abstains -- that is how the
substitution-window constraint is enforced.

All functions here are CPU-pure and unit-tested; artifact loading and the pitch-geometry join live in
the runner :mod:`tools.run_lineup_assign`.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# --- FROZEN cost weights (reasoned defaults; no per-assignment ground truth exists) ----------------
# The calibration knob: EXPECTED_U are rough longitudinal bands (0 = own goal, 1 = opponent goal) for
# the four coarse oracle positions. They tune the soft position term only -- jersey dominates.
HALF_MIN: float = 45.0
EXPECTED_U: dict[str, float] = {"G": 0.05, "D": 0.30, "M": 0.55, "F": 0.82}
W_POS: float = 0.25          # soft position weight (max contribution; < any real trade-off)
GK_VETO: float = 10.0        # keeper<->outfield mismatch: effectively infeasible
INFEASIBLE: float = 1.0e6    # jersey-number mismatch: never assignable
ABSTAIN_FLOOR: float = 0.90  # never assign a real pair whose cost exceeds this
NEUTRAL_U_COST: float = 0.5  # position cost when the group has no reliable pitch position


@dataclass(frozen=True)
class PlayerCand:
    """A rostered candidate for assignment in one match window.

    Attributes:
        team: Kit-anchor team id (0/1), matching the aligned positions table.
        name: Player display name (NFC-normalised).
        shirt: Back-of-shirt number (oracle ``shirtNumber``) -- the jersey key.
        position: Coarse oracle position, one of ``G``/``D``/``M``/``F``.
        is_sub: True if the player started on the bench.
        minutes: Minutes played (NaN for an unused sub).
        on_h1: Whether the player was on the pitch during the first half.
        on_h2: Whether the player was on the pitch during the second half.
    """

    team: int
    name: str
    shirt: int
    position: str
    is_sub: bool
    minutes: float
    on_h1: bool
    on_h2: bool

    def on(self, half: str) -> bool:
        """Whether this player is a candidate in ``half`` (``h1``/``h2``)."""
        return self.on_h1 if half == "h1" else self.on_h2


@dataclass(frozen=True)
class TrackGroup:
    """A jersey-vote group: the track evidence for one voted number in one team-half window.

    Attributes:
        team: Kit-anchor team id (0/1).
        half: ``h1`` or ``h2``.
        number: The voted back number (``None`` = a number-less group, position drives assignment).
        n_fragments: Distinct ByteTrack fragments carrying this number in the window.
        vote_mass: Total independent close-up anchors backing the number (sum of ``n_anchors``).
        mean_u: Mean longitudinal pitch position in ``[0, 1]`` (own goal -> opponent), or ``None`` if
            no reliable orientation could be inferred for the window.
        is_keeper_track: True if the group's tracks are detector goalkeeper tracks.
        track_ids: The fragment ids in this group (for the output ledger).
    """

    team: int
    half: str
    number: int | None
    n_fragments: int
    vote_mass: int
    mean_u: float | None
    is_keeper_track: bool
    track_ids: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Assign:
    """One resolved assignment (or abstention) for the output ledger.

    Attributes:
        team: Team id (0/1).
        half: Window (``h1``/``h2``).
        player: Assigned player name, or ``None`` on abstention.
        shirt: Assigned player's shirt number, or ``None``.
        position: Assigned player's coarse position, or ``None``.
        number: The group's voted number.
        vote_mass: Anchors backing the group.
        n_fragments: Fragments in the group.
        mean_u: Group longitudinal position (or ``None``).
        confidence: Assignment confidence in ``[0, 1]`` (0 on abstention).
        method: Evidence label (e.g. ``jersey+position``, ``jersey``, ``abstain:<reason>``).
        track_ids: The group's fragment ids.
    """

    team: int
    half: str
    player: str | None
    shirt: int | None
    position: str | None
    number: int | None
    vote_mass: int
    n_fragments: int
    mean_u: float | None
    confidence: float
    method: str
    track_ids: tuple[int, ...]


def _norm_name(name: object) -> str:
    """NFC-normalise a possibly-mojibake oracle name (accents kept as proper unicode)."""
    return unicodedata.normalize("NFC", str(name)).strip()


def _window_flags(is_sub: bool, minutes: float) -> tuple[bool, bool]:
    """Derive ``(on_h1, on_h2)`` from the substitute flag and minutes played (pure).

    Entry/exit minutes are inferred without explicit sub events: a starter runs from minute 0 to
    ``minutes`` (subbed off there if < 90); a sub enters at ``90 - minutes``. A window is active if
    the player's interval overlaps it. An unused sub (NaN minutes) is on neither.

    Args:
        is_sub: Whether the player started on the bench.
        minutes: Minutes played (NaN for an unused sub).

    Returns:
        ``(on_h1, on_h2)`` booleans.
    """
    if minutes is None or (isinstance(minutes, float) and np.isnan(minutes)) or minutes <= 0:
        return False, False
    if is_sub:
        entry, exit_ = max(0.0, 90.0 - minutes), 90.0
    else:
        entry, exit_ = 0.0, minutes
    on_h1 = entry < HALF_MIN and exit_ > 0.0
    on_h2 = entry < 90.0 and exit_ > HALF_MIN
    return on_h1, on_h2


def roster_candidates(oracle_df: pd.DataFrame, team_id_fn) -> list[PlayerCand]:
    """Build the per-player candidate list from the Sofascore player-stats oracle (pure).

    Args:
        oracle_df: Cached per-player oracle parquet (one row per rostered player).
        team_id_fn: Maps an oracle ``teamName`` to a kit-anchor team id (0/1), or ``None`` to drop.

    Returns:
        One :class:`PlayerCand` per player with a valid ``shirtNumber`` and resolvable team.
    """
    cands: list[PlayerCand] = []
    for row in oracle_df.itertuples(index=False):
        team = team_id_fn(getattr(row, "teamName"))
        if team is None:
            continue
        try:
            shirt = int(str(getattr(row, "shirtNumber")))
        except (TypeError, ValueError):
            continue
        minutes = getattr(row, "minutesPlayed", float("nan"))
        minutes = float(minutes) if minutes is not None else float("nan")
        is_sub = bool(getattr(row, "substitute"))
        on_h1, on_h2 = _window_flags(is_sub, minutes)
        pos = str(getattr(row, "position") or "").strip().upper()[:1] or "M"
        cands.append(PlayerCand(int(team), _norm_name(getattr(row, "name")), shirt, pos, is_sub,
                                minutes, on_h1, on_h2))
    return cands


def position_cost(position: str, mean_u: float | None) -> float:
    """Soft longitudinal-band cost of a group at ``mean_u`` for a player of coarse ``position``.

    Args:
        position: Coarse oracle position (``G``/``D``/``M``/``F``); unknown -> midfield band.
        mean_u: Group longitudinal position in ``[0, 1]``, or ``None`` (no reliable orientation).

    Returns:
        Cost in ``[0, 1]`` (0 = perfect band match); :data:`NEUTRAL_U_COST` when ``mean_u`` is None.
    """
    if mean_u is None:
        return NEUTRAL_U_COST
    return abs(float(mean_u) - EXPECTED_U.get(position, EXPECTED_U["M"]))


def _pair_cost(p: PlayerCand, g: TrackGroup) -> float:
    """Fused assignment cost for one (player, group) pair (pure)."""
    if g.number is not None and g.number != p.shirt:
        return INFEASIBLE
    jersey = 0.0 if g.number is not None else 0.5  # number-less group: neutral, let position drive
    if (p.position == "G") != g.is_keeper_track:
        return GK_VETO + jersey  # keeper<->outfield mismatch: reject
    return jersey + W_POS * position_cost(p.position, g.mean_u)


def assignment_confidence(g: TrackGroup, position: str) -> tuple[float, str]:
    """Confidence and evidence label for assigning group ``g`` to a player of ``position`` (pure).

    Jersey vote mass is the dominant driver (``1 - 2^-mass``: 1 anchor -> 0.50, 3 -> 0.88, 5 -> 0.97);
    the position band contributes a small corroboration only when a pitch position is available.

    Args:
        g: The assigned track group.
        position: The assigned player's coarse position.

    Returns:
        ``(confidence in [0, 1], method label)``.
    """
    vote_conf = 1.0 - 0.5 ** min(max(g.vote_mass, 0), 12)
    if g.mean_u is None:
        return round(vote_conf, 3), "jersey" if g.number is not None else "position"
    pos_agree = 1.0 - position_cost(position, g.mean_u)
    conf = round(0.85 * vote_conf + 0.15 * pos_agree, 3)
    label = "jersey+position" if pos_agree >= 0.5 else "jersey(position-weak)"
    return conf, label


def build_cost_matrix(players: list[PlayerCand], groups: list[TrackGroup]) -> np.ndarray:
    """Dense ``len(players) x len(groups)`` fused-cost matrix (pure).

    Args:
        players: Candidate players (rows).
        groups: Track groups to assign (columns).

    Returns:
        Cost matrix; :data:`INFEASIBLE` entries are never assignable (jersey mismatch).
    """
    m = np.full((len(players), len(groups)), INFEASIBLE, dtype=float)
    for i, p in enumerate(players):
        for j, g in enumerate(groups):
            m[i, j] = _pair_cost(p, g)
    return m


def solve_assignment(players: list[PlayerCand], groups: list[TrackGroup]) -> list[Assign]:
    """Solve one team-half Hungarian assignment with first-class abstention (pure).

    Every player and every group is given a dummy option at :data:`ABSTAIN_FLOOR`, so a real pair is
    committed only when its fused cost falls below the floor; otherwise the group abstains. Groups
    whose number matches no in-window player abstain (``no_candidate``); groups the solver leaves
    unassigned abstain (``low_confidence``).

    Args:
        players: In-window candidate players for this team-half.
        groups: Track groups (voted numbers) observed in this team-half.

    Returns:
        One :class:`Assign` per group (assigned or abstained), sorted by descending confidence.
    """
    if not groups:
        return []
    n_p, n_g = len(players), len(groups)
    cost = build_cost_matrix(players, groups)
    # Square block matrix: real P x G, plus per-player and per-group abstain dummies.
    big = np.full((n_p + n_g, n_g + n_p), INFEASIBLE, dtype=float)
    big[:n_p, :n_g] = cost
    for i in range(n_p):          # player i may go unassigned via its own dummy column
        big[i, n_g + i] = ABSTAIN_FLOOR
    for j in range(n_g):          # group j may abstain via its own dummy row
        big[n_p + j, j] = ABSTAIN_FLOOR
    big[n_p:, n_g:] = 0.0         # dummy-dummy corner is free
    rows, cols = linear_sum_assignment(big)
    picked: dict[int, int] = {c: r for r, c in zip(rows, cols) if r < n_p and c < n_g}

    out: list[Assign] = []
    for j, g in enumerate(groups):
        i = picked.get(j)
        if i is not None and cost[i, j] < ABSTAIN_FLOOR:
            p = players[i]
            conf, method = assignment_confidence(g, p.position)
            out.append(Assign(g.team, g.half, p.name, p.shirt, p.position, g.number, g.vote_mass,
                              g.n_fragments, g.mean_u, conf, method, g.track_ids))
        else:
            has_cand = g.number is None or any(
                pl.shirt == g.number for pl in players)
            reason = "low_confidence" if has_cand else "no_candidate"
            out.append(Assign(g.team, g.half, None, None, None, g.number, g.vote_mass,
                              g.n_fragments, g.mean_u, 0.0, f"abstain:{reason}", g.track_ids))
    out.sort(key=lambda a: -a.confidence)
    return out


def _demo() -> None:
    """Self-check of window flags, cost fusion, abstention, and sub re-keying (asserts; runnable)."""
    # Substitution-window derivation.
    assert _window_flags(False, 45.0) == (True, False)   # starter off at HT -> h1 only
    assert _window_flags(False, 65.0) == (True, True)     # starter subbed at 65' -> both
    assert _window_flags(True, 11.0) == (False, True)     # late sub -> h2 only
    assert _window_flags(True, float("nan")) == (False, False)  # unused sub -> neither

    # Position band: a forward group sits high, a defender group low.
    assert position_cost("F", 0.82) == 0.0
    assert position_cost("D", 0.82) > position_cost("F", 0.82)
    assert position_cost("F", None) == NEUTRAL_U_COST

    P = PlayerCand
    mu = P(0, "Bruno", 8, "M", False, 79.0, True, True)   # #8 midfielder, both halves
    df = P(0, "Dalot", 20, "D", False, 90.0, True, True)  # #20 defender
    sub = P(0, "Antony", 21, "M", True, 8.0, False, True)  # #21 sub, h2 only

    # Jersey mismatch is infeasible; the right number wins.
    g8 = TrackGroup(0, "h1", 8, 3, 5, 0.55, False, (100,))
    assert _pair_cost(mu, g8) < ABSTAIN_FLOOR
    assert _pair_cost(df, g8) >= INFEASIBLE           # #20 cannot be group #8

    res = solve_assignment([mu, df], [g8])
    assert len(res) == 1 and res[0].player == "Bruno" and res[0].confidence > 0.9

    # Sub re-keying: #21 read in h1 (Antony not yet on) -> no h1 candidate -> abstain(no_candidate).
    g21_h1 = TrackGroup(0, "h1", 21, 1, 1, None, False, (200,))
    res = solve_assignment([mu, df], [g21_h1])        # h1 window: sub excluded
    assert res[0].player is None and res[0].method == "abstain:no_candidate"
    # Same read in h2, sub present -> assigned.
    g21_h2 = TrackGroup(0, "h2", 21, 1, 1, None, False, (200,))
    res = solve_assignment([mu, df, sub], [g21_h2])
    assert res[0].player == "Antony"

    # GK veto: a keeper track cannot take an outfield player.
    gk_grp = TrackGroup(0, "h1", 8, 1, 1, 0.05, True, (300,))
    assert _pair_cost(mu, gk_grp) >= GK_VETO
    res = solve_assignment([mu, df], [gk_grp])
    assert res[0].player is None                       # abstains rather than force

    # Two groups, two numbers -> both assigned to the right players (Hungarian, no conflict).
    g20 = TrackGroup(0, "h1", 20, 2, 3, 0.30, False, (400,))
    res = solve_assignment([mu, df], [g8, g20])
    by_num = {a.number: a.player for a in res}
    assert by_num == {8: "Bruno", 20: "Dalot"}

    print("lineup_assign demo OK: window flags + cost fusion + abstention + sub re-keying + GK veto")


if __name__ == "__main__":
    _demo()
