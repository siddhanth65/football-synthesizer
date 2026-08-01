"""Global identity solve: tracklets -> roster identities under mutual exclusion (Lu-style).

Stage 2 of ``docs/ATTRIBUTION_RESEARCH_PLAN.md``. The per-tracklet identity decision is currently
made greedily and independently (an OCR read is spread inside its merge group and nothing else can
contradict it). Lu et al., *Learning to Track and Identify Players from Broadcast Sports Videos*
(TPAMI 2013) showed that the same unary features rise from 50-55% to 85-89% once the decision is a
single constrained solve over a whole half. This module is that solve, with modern unaries:

* **OCR soft votes** -- a tracklet's Koshkina reads enter as a likelihood
  ``P(read | identity number)`` built from a *measured* digit-confusion prior
  (:func:`digit_confusion_prior`), never as a hard roster mask. A wrong read therefore costs
  evidence instead of silently deciding the tracklet (``results/MANUTD_IDENTITY_PROFILE.md``).
* **Appearance** -- mean-of-top-k cosine similarity between the tracklet's per-detection PRTreID
  embeddings and each identity's OCR-anchored gallery. Mean-of-top-k rather than max, because
  ``results/GTA_LINK_STAGE1.md`` A3 measured max-over-crops to be the scoring rule most exposed to
  a contaminated merge group.
* **Team / role compatibility** -- multiplicative gates in ``[eps, 1]``; a role gate is what lets a
  goalkeeper identity be filled by a goalkeeper-looking tracklet with no number read at all.
* **(unknown)** -- a real class with a calibrated prior, so the solver abstains instead of forcing a
  name onto a tracklet whose player was never read.

The evidence is combined into a per-tracklet posterior over ``identities + {unknown}``
(:func:`posterior`), and the objective is the **expected number of correctly identified rows**,
``sum_t n_rows(t) * P(t -> i)``, which is exactly the per-row metric the arm is graded on. Hard
constraints: one label per tracklet; two tracklets sharing a frame cannot share an identity (0-frame
slack -- the SoccerNet evaluator rejects a repeated id in one timestep); at most ``max_concurrent``
named tracklets per team in any frame; identities may carry an availability window (substitutions).

Solved as a small binary program with ``scipy.optimize.milp`` (HiGHS -- already a dependency; no
OR-tools). CPU only, milliseconds per sequence. Self-check: ``python -m generator.identity_solve``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

logger = logging.getLogger("identity_solve")

#: Version stamp for frozen configs and result payloads (bump when the model changes).
SOLVER_VERSION = "identity-solve-1.0"

#: Label used for the abstain class in reported assignments.
UNKNOWN = None


@dataclass(frozen=True)
class Identity:
    """One roster slot the solver may assign.

    Attributes:
        key: Caller-side identity handle (``(team_side, number)`` on GSR, a player name on a real
            match). Only used to report the assignment back.
        team: Team index in the caller's own labelling, matched against ``Tracklet.team``.
        number: Shirt number, or ``None`` for a roster slot with no number evidence.
        role: ``"player"`` or ``"goalkeeper"``.
        window: ``(first_frame, last_frame)`` this identity may be on the pitch, or ``None`` for the
            whole sequence (substitution gating on real matches).
    """

    key: object
    team: int
    number: int | None
    role: str = "player"
    window: tuple[int, int] | None = None


@dataclass
class Tracklet:
    """One (merged) tracklet with all its unary evidence already summarised.

    Attributes:
        track_id: Tracklet id in the caller's partition.
        n_rows: Number of prediction rows the tracklet owns (the metric weight).
        team: Majority team index, or ``None`` when unknown.
        team_frac: Fraction of rows carrying ``team``.
        role_frac: ``{role: fraction of rows}``.
        reads: OCR votes contributed by the tracklet's members as ``(number, confidence)``.
        emb: ``(n, D)`` L2-normalised per-detection embeddings (may be empty).
        span: ``(first_frame, last_frame)`` envelope, used only for substitution windows.
    """

    track_id: int
    n_rows: int
    team: int | None
    team_frac: float
    role_frac: dict[str, float]
    reads: tuple[tuple[int, float], ...]
    emb: np.ndarray
    span: tuple[int, int]


@dataclass
class SolverConfig:
    """Frozen solver calibration. Fit on dev sequences only, written to disk, then never touched.

    Attributes:
        p_correct: Probability an OCR read equals the true number (measured on dev).
        confusion: ``{"digit": 10x10 row-stochastic list, "len": P(length error)}`` -- the digit
            confusion prior from :func:`digit_confusion_prior`.
        app_gain: Inverse temperature on the appearance similarity (Gibbs likelihood).
        sim_none: Similarity the ``unknown`` class is credited with (same cosine units).
        topk: Number of best crop pairs averaged for the gallery similarity.
        pi_none: Prior mass on ``unknown`` before any evidence.
        r_abstain: Reward per row for abstaining -- the calibrated floor. It is deliberately NOT the
            posterior mass on ``unknown``: that mass covers both "this player carries no number at
            all" (abstaining is right) and "this player IS on the roster but left no evidence"
            (abstaining is just as wrong as a wrong name). Only the first case earns anything, so
            the floor is a separate fitted constant and the solver names whenever some identity's
            posterior beats it.
        team_eps: Likelihood floor for a team mismatch (0 would make it a hard constraint).
        role_eps: Likelihood floor for a role mismatch.
        max_concurrent: Maximum simultaneously-named tracklets per team.
        use_confusion: When False the digit prior is replaced by a uniform one (ablation).
        use_ocr: When False the OCR term is dropped entirely (ablation).
        use_app: When False the appearance term is dropped entirely (ablation).
        use_mutex: When False every constraint is dropped and each tracklet takes its own argmax
            (the ablation that isolates the joint-inference contribution).
    """

    p_correct: float = 0.875
    confusion: dict = field(default_factory=dict)
    app_gain: float = 20.0
    sim_none: float = 0.90
    topk: int = 3
    pi_none: float = 0.5
    r_abstain: float = 0.15
    team_eps: float = 0.02
    role_eps: float = 0.05
    max_concurrent: int = 11
    use_confusion: bool = True
    use_ocr: bool = True
    use_app: bool = True
    use_mutex: bool = True

    def save(self, path: Path) -> None:
        """Write the frozen config as JSON (UTF-8), with the version stamp."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": SOLVER_VERSION, **asdict(self)}, indent=2),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SolverConfig:
        """Read a frozen config written by :meth:`save`."""
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        blob.pop("version", None)
        return cls(**blob)


# === OCR confusion prior =========================================================================
def digit_confusion_prior(
    pairs: Iterable[tuple[int, int]], *, alpha: float = 1.0,
) -> dict:
    """Estimate ``P(read digit | true digit)`` from observed ``(true_number, read_number)`` pairs.

    Same-length pairs contribute one substitution count per digit position; different-length pairs
    contribute only to the length-error rate (a 2-digit number read as 1 digit carries no aligned
    substitution). Add-``alpha`` smoothing keeps every digit reachable -- with the tens of wrong
    reads a jersey OCR actually produces, the smoothing is expected to dominate, which is precisely
    why the prior must be *reported* alongside its own sample size.

    Args:
        pairs: ``(true_number, read_number)`` observations, correct reads included.
        alpha: Add-alpha smoothing mass per cell.

    Returns:
        ``{"digit": 10x10 row-stochastic matrix, "len": P(length differs), "n": n_pairs,
        "n_wrong": n_wrong, "n_aligned": substitution observations}``.
    """
    counts = np.full((10, 10), float(alpha))
    n = n_wrong = n_len = n_aligned = 0
    for true, read in pairs:
        n += 1
        st, sr = str(int(true)), str(int(read))
        if st != sr:
            n_wrong += 1
        if len(st) != len(sr):
            n_len += 1
            continue
        for a, b in zip(st, sr):
            counts[int(a), int(b)] += 1.0
            n_aligned += 1
    return {"digit": (counts / counts.sum(axis=1, keepdims=True)).tolist(),
            "len": (n_len + alpha) / (n + 2 * alpha) if n else 0.5,
            "n": n, "n_wrong": n_wrong, "n_aligned": n_aligned}


def confusion_weight(true: int, read: int, prior: dict) -> float:
    """Unnormalised ``P(read | true, read != true)`` under the digit prior (pure).

    Equal-length numbers multiply their per-position digit confusion probabilities; a length
    mismatch is charged the measured length-error rate instead.
    """
    st, sr = str(int(true)), str(int(read))
    if len(st) != len(sr):
        return float(prior.get("len", 0.5)) * 0.1
    digit = np.asarray(prior["digit"], float)
    w = 1.0 - float(prior.get("len", 0.5))
    for a, b in zip(st, sr):
        w *= float(digit[int(a), int(b)])
    return w


def read_likelihoods(
    reads: Sequence[tuple[int, float]], numbers: Sequence[int | None], cfg: SolverConfig,
) -> np.ndarray:
    """Likelihood of a tracklet's OCR reads under each candidate number (pure).

    Args:
        reads: ``(number, confidence)`` votes carried by the tracklet.
        numbers: Candidate numbers, one per identity (``None`` = identity has no number).
        cfg: Frozen calibration.

    Returns:
        ``(len(numbers) + 1,)`` likelihoods; the last entry is the ``unknown`` background, under
        which any read is equally likely to be any admissible number.
    """
    out = np.ones(len(numbers) + 1)
    if not reads or not cfg.use_ocr:
        return out
    known = [n for n in numbers if n is not None]
    k = max(len(set(known)), 2)
    prior = cfg.confusion if cfg.use_confusion else {"digit": np.full((10, 10), 0.1).tolist(),
                                                     "len": 0.5}
    for num, conf in reads:
        pc = cfg.p_correct * float(np.clip(conf, 0.0, 1.0)) + (1.0 - cfg.p_correct) * 0.5
        pc = float(np.clip(pc, 0.05, 0.99))
        # The error mass is distributed over the OTHER numbers only, so the read's own slot never
        # dilutes the share it leaves to its confusable neighbours.
        weights = np.array([0.0 if (j is None or j == num) else confusion_weight(j, num, prior)
                            for j in numbers])
        tot = weights.sum()
        share = weights / tot if tot > 0 else np.zeros_like(weights)
        lik = np.where(np.array([j == num for j in numbers]), pc, (1.0 - pc) * share)
        # A read is only informative about numbered slots; an unnumbered slot (e.g. a GK roster
        # entry with no number) is treated like the background so it is never favoured by a read.
        lik = np.where(np.array([j is None for j in numbers]), 1.0 / k, lik)
        out[:-1] *= np.clip(lik, 1e-6, None)
        out[-1] *= 1.0 / k
    return out


# === Appearance ==================================================================================
def gallery_similarity(query: np.ndarray, gallery: np.ndarray, topk: int) -> float:
    """Mean of the ``topk`` largest cosine similarities between two embedding sets (pure).

    Both inputs must be L2-normalised. Returns ``nan`` when either side is empty.
    """
    if query.size == 0 or gallery.size == 0:
        return float("nan")
    sims = (query @ gallery.T).ravel()
    k = min(int(topk), sims.size)
    return float(np.mean(np.sort(sims)[-k:]))


def similarity_matrix(
    tracklets: Sequence[Tracklet], galleries: Sequence[np.ndarray], topk: int,
) -> np.ndarray:
    """``(n_tracklets, n_identities)`` gallery similarities, ``nan`` where evidence is missing."""
    out = np.full((len(tracklets), len(galleries)), np.nan)
    for t, trk in enumerate(tracklets):
        for i, gal in enumerate(galleries):
            out[t, i] = gallery_similarity(trk.emb, gal, topk)
    return out


# === Posterior ===================================================================================
def posterior(
    trk: Tracklet, identities: Sequence[Identity], sims: np.ndarray, cfg: SolverConfig,
) -> np.ndarray:
    """Per-tracklet posterior over ``identities + [unknown]`` (pure).

    ``likelihood(i) = P_ocr * P_app * gate_team * gate_role``; the ``unknown`` class carries the
    background OCR likelihood, ``exp(app_gain * sim_none)`` for appearance and no gates.

    Args:
        trk: The tracklet and its summarised evidence.
        identities: Candidate identities (same order as ``sims``).
        sims: ``(n_identities,)`` gallery similarities, ``nan`` where the identity has no gallery.
        cfg: Frozen calibration.

    Returns:
        ``(n_identities + 1,)`` probabilities summing to 1; the last entry is ``unknown``.
    """
    n = len(identities)
    lik = read_likelihoods(trk.reads, [i.number for i in identities], cfg)
    if cfg.use_app:
        s = np.where(np.isfinite(sims), sims, cfg.sim_none)
        app = np.exp(cfg.app_gain * (np.append(s, cfg.sim_none) - cfg.sim_none))
    else:
        app = np.ones(n + 1)
    gate = np.ones(n + 1)
    for i, ident in enumerate(identities):
        if trk.team is not None and ident.team != trk.team:
            gate[i] *= max(cfg.team_eps, 1.0 - trk.team_frac)
        elif trk.team is not None:
            gate[i] *= max(trk.team_frac, cfg.team_eps)
        frac = float(trk.role_frac.get(ident.role, 0.0))
        gate[i] *= max(frac, cfg.role_eps)
        if ident.window is not None and (trk.span[1] < ident.window[0]
                                         or trk.span[0] > ident.window[1]):
            gate[i] = 0.0
    prior = np.append(np.full(n, (1.0 - cfg.pi_none) / max(n, 1)), cfg.pi_none)
    post = prior * lik * app * gate
    total = post.sum()
    return post / total if total > 0 else np.append(np.zeros(n), 1.0)


# === Constraints =================================================================================
def exclusion_groups(alive_by_frame: Iterable[Sequence[int]]) -> list[tuple[int, ...]]:
    """Maximal sets of tracklets that co-occur in some frame (pure).

    Two tracklets in the same group share at least one frame, so they may not share an identity.
    Frames whose alive-set is a subset of another frame's are dropped, which is exact (their
    constraint is implied) and keeps the program small.

    Args:
        alive_by_frame: One sequence of tracklet indices per frame.

    Returns:
        Maximal co-occurrence groups of size >= 2, as sorted tuples.
    """
    seen = {frozenset(g) for g in alive_by_frame if len(set(g)) >= 2}
    out: list[frozenset[int]] = []
    # ponytail: O(k^2) subset filter over distinct alive-sets (k ~ 1 per frame). Fine at a 750-frame
    # sequence and at a match chunk; index by member if a much longer clip ever needs it.
    for g in sorted(seen, key=len, reverse=True):
        if not any(g <= k for k in out):
            out.append(g)
    return [tuple(sorted(g)) for g in out]


def solve_assignment(
    probs: np.ndarray, weights: np.ndarray, groups: Sequence[Sequence[int]],
    identity_team: Sequence[int], *, max_concurrent: int = 11,
    max_candidates: int | None = None, time_limit: float | None = None,
) -> np.ndarray:
    """Maximise expected correct rows under mutual exclusion -> identity index per tracklet.

    Args:
        probs: ``(n_tracklets, n_identities + 1)`` payoffs; column ``-1`` is the abstain reward
            (see :attr:`SolverConfig.r_abstain`), the rest are identity posteriors.
        weights: ``(n_tracklets,)`` row counts (the metric weight of each tracklet).
        groups: Co-occurrence groups from :func:`exclusion_groups`.
        identity_team: Team index of each identity.
        max_concurrent: Maximum named tracklets per team inside one group.
        max_candidates: Keep only each tracklet's ``max_candidates`` best-paying identities (plus
            abstain) as admissible. ``None`` (default, used for every GSR number) considers the
            whole roster. On a match chunk with ~700 tracklets the full program does not solve in
            reasonable time; pruning is a documented approximation, not an exact reduction.
        time_limit: Seconds given to HiGHS; the incumbent is used if the limit is hit.

    Returns:
        ``(n_tracklets,)`` int array of identity indices; ``-1`` means ``unknown``.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp  # noqa: PLC0415
    from scipy.sparse import csr_array  # noqa: PLC0415

    n_t, n_cols = probs.shape
    n_i = n_cols - 1
    if n_t == 0:
        return np.zeros(0, int)
    var = np.arange(n_t * n_cols).reshape(n_t, n_cols)
    obj = -(weights[:, None] * probs).ravel()
    upper = np.ones((n_t, n_cols))
    if max_candidates is not None and n_i > max_candidates:
        keep = np.argsort(-probs[:, :n_i], axis=1)[:, :max_candidates]
        upper[:, :n_i] = 0.0
        np.put_along_axis(upper[:, :n_i], keep, 1.0, axis=1)

    rows, cols = [], []
    for t in range(n_t):  # exactly one label per tracklet
        rows.extend([t] * n_cols)
        cols.extend(var[t].tolist())
    eq = LinearConstraint(csr_array((np.ones(len(rows)), (rows, cols)), shape=(n_t, obj.size)),
                          1.0, 1.0)

    rows, cols, lo, hi = [], [], [], []
    r = 0
    for g in groups:
        for i in range(n_i):  # one identity per group
            if len(g) < 2:
                continue
            rows.extend([r] * len(g))
            cols.extend(var[list(g), i].tolist())
            lo.append(0.0)
            hi.append(1.0)
            r += 1
        for team in sorted(set(identity_team)):  # squad size on the pitch
            cand = [i for i in range(n_i) if identity_team[i] == team]
            if len(g) <= max_concurrent or not cand:
                continue
            for t in g:
                rows.extend([r] * len(cand))
                cols.extend(var[t, cand].tolist())
            lo.append(0.0)
            hi.append(float(max_concurrent))
            r += 1
    cons = [eq]
    if r:
        cons.append(LinearConstraint(
            csr_array((np.ones(len(rows)), (rows, cols)), shape=(r, obj.size)),
            np.array(lo), np.array(hi)))
    options = {"time_limit": time_limit} if time_limit else {}
    res = milp(c=obj, constraints=cons, integrality=np.ones(obj.size),
               bounds=Bounds(np.zeros(obj.size), upper.ravel()), options=options)
    if res.x is None:
        logger.warning("MILP produced no solution (%s); falling back to independent argmax",
                       res.message)
        return np.where(probs.argmax(axis=1) == n_i, -1, probs.argmax(axis=1))
    if not res.success:
        logger.warning("MILP stopped early (%s); using the incumbent", res.message)
    x = np.asarray(res.x).reshape(n_t, n_cols).round().astype(int)
    pick = x.argmax(axis=1)
    return np.where(pick == n_i, -1, pick)


def solve_sequence(
    tracklets: Sequence[Tracklet], identities: Sequence[Identity], galleries: Sequence[np.ndarray],
    alive_by_frame: Iterable[Sequence[int]], cfg: SolverConfig,
    sims: np.ndarray | None = None,
) -> tuple[list[Identity | None], np.ndarray]:
    """End-to-end solve for one sequence/half -> ``(identity per tracklet, posterior matrix)``.

    Args:
        tracklets: Tracklets to label.
        identities: Roster slots.
        galleries: One ``(n, D)`` embedding gallery per identity (may be empty arrays).
        alive_by_frame: Per-frame sequences of tracklet indices (mutual-exclusion evidence).
        cfg: Frozen calibration.
        sims: Optional precomputed similarity matrix (skips the gallery pass when grid-searching).

    Returns:
        ``(assignment, probs)`` with ``assignment[t]`` the :class:`Identity` or ``None``.
    """
    if sims is None:
        sims = similarity_matrix(tracklets, galleries, cfg.topk)
    probs = np.stack([posterior(t, identities, sims[k], cfg) for k, t in enumerate(tracklets)]) \
        if tracklets else np.zeros((0, len(identities) + 1))
    weights = np.array([t.n_rows for t in tracklets], float)
    groups = exclusion_groups(alive_by_frame) if cfg.use_mutex else []
    payoff = probs.copy()
    payoff[:, -1] = cfg.r_abstain
    pick = solve_assignment(payoff, weights, groups, [i.team for i in identities],
                            max_concurrent=cfg.max_concurrent)
    return [None if p < 0 else identities[p] for p in pick], probs


def _demo() -> None:
    """Self-check: mutual exclusion overrides a stronger unary score (asserts; runnable)."""
    d = 8
    a, b = np.zeros(d), np.zeros(d)
    a[0], b[1] = 1.0, 1.0
    cfg = SolverConfig(confusion=digit_confusion_prior([(7, 7), (1, 7), (4, 9)]),
                       app_gain=30.0, sim_none=0.80, topk=1, pi_none=0.50)

    ids = [Identity(("left", 7), 0, 7), Identity(("left", 9), 0, 9)]
    gal = [a[None, :], b[None, :]]

    def trk(tid: int, lo: int, hi: int, emb: np.ndarray, reads=()) -> Tracklet:
        return Tracklet(tid, hi - lo + 1, 0, 1.0, {"player": 1.0}, tuple(reads), emb, (lo, hi))

    # Three tracklets, two identities. T0 and T1 overlap in time and both look like #7; T2 is
    # disjoint from both and looks like #9. Independently, T0 and T1 would both take #7.
    t0 = trk(0, 0, 10, a[None, :])
    t1 = trk(1, 5, 15, (0.99 * a + 0.14 * b)[None, :])
    t2 = trk(2, 20, 30, b[None, :])
    tracklets = [t0, t1, t2]
    alive = [[0]] * 5 + [[0, 1]] * 6 + [[1]] * 5 + [[2]] * 11
    assert np.argmax(posterior(t0, ids, np.array([1.0, 0.14]), cfg)) == 0
    assert np.argmax(posterior(t1, ids, np.array([0.99, 0.28]), cfg)) == 0  # unary: both want #7
    assign, _p = solve_sequence(tracklets, ids, gal, alive, cfg)
    assert assign[0] is ids[0], assign          # the better match keeps #7
    assert assign[1] is not ids[0], assign      # mutex forces the overlapping twin off #7
    assert assign[2] is ids[1], assign          # the disjoint #9-looking tracklet is named

    # Same three tracklets with the overlap removed: nothing forces T1 off #7 any more.
    t1b = trk(1, 40, 50, (0.99 * a + 0.14 * b)[None, :])
    assign2, _ = solve_sequence([t0, t1b, t2], ids, gal,
                                [[0]] * 11 + [[1]] * 11 + [[2]] * 11, cfg)
    assert assign2[1] is ids[0], assign2

    # OCR soft vote: a read of 9 flips a mildly #7-looking tracklet, but it is NOT a hard mask --
    # strong enough appearance evidence outvotes the read instead of being masked out by it.
    t3 = trk(3, 60, 70, a[None, :], reads=[(9, 0.9)])
    soft = SolverConfig(**{**vars(cfg), "app_gain": 1.0})
    p_soft = posterior(t3, ids, np.array([1.0, 0.14]), soft)
    assert p_soft[1] > p_soft[0], p_soft            # the read outweighs a mild appearance lean
    hard = SolverConfig(**{**vars(cfg), "app_gain": 10.0})
    p_hard = posterior(t3, ids, np.array([1.0, 0.14]), hard)
    assert p_hard[0] > p_hard[1], p_hard            # ...and is outvoted, not obeyed, when it is not
    # (unknown) wins when there is no evidence at all: no reads, no gallery similarity.
    t4 = Tracklet(4, 10, 0, 1.0, {"player": 1.0}, (), np.zeros((0, d)), (80, 90))
    assert np.argmax(posterior(t4, ids, np.array([np.nan, np.nan]), cfg)) == len(ids)
    # Role gate: a goalkeeper-looking tracklet fills a GK slot that carries no number at all.
    gk = [Identity(("left", None), 0, None, "goalkeeper")]
    tgk = Tracklet(5, 10, 0, 1.0, {"goalkeeper": 1.0}, (), np.zeros((0, d)), (0, 10))
    assert np.argmax(posterior(tgk, gk, np.array([np.nan]), cfg)) == 0
    tpl = Tracklet(6, 10, 0, 1.0, {"player": 1.0}, (), np.zeros((0, d)), (0, 10))
    assert np.argmax(posterior(tpl, gk, np.array([np.nan]), cfg)) == 1  # -> unknown
    # Substitution window: an identity that is not on the pitch cannot be assigned.
    late = [Identity(("left", 7), 0, 7, window=(100, 200))]
    assert np.argmax(posterior(t0, late, np.array([1.0]), cfg)) == 1
    print("identity_solve demo OK: mutex overrides unaries, OCR votes stay soft, GK slot opens")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _demo()
