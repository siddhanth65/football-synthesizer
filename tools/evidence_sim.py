"""Evidence-density simulator: how much identity evidence does joint attribution actually need?

``results/IDENTITY_SOLVER_STAGE2.md`` measured that the Lu-style joint solve does not beat the
appearance-only arm *because the evidence is starved*: 8.7% of tracklets carry an OCR read, 24.1% of
rows belong to a player read anywhere in the sequence, and the evidence-restricted oracle therefore
caps at 0.365. That is a statement about the inputs, not about the solver, and it can only be turned
into a design rule by varying the inputs -- which real footage does not let us do.

This module varies them, on top of FOOTPASS ground truth (``data/footpass/README.md``): dense
per-frame per-player tracking for 54 matches, with ROI boxes that are ``NaN`` exactly when a player
is off-screen. The truth is fragmented into tracklets that reproduce our **measured** post-connector
fragmentation, given a synthetic appearance channel calibrated to our **measured** 0.62-0.64 top-1,
and then fed evidence at densities and precisions we choose. Everything the simulator invents is
pinned to a number this project measured:

===========================  =========================================  =========================
simulator component          calibrated against                         source
===========================  =========================================  =========================
tracklet length distribution merged-tracklet spans, ``tau = 0.040``      Stage 1 / Stage 2 partition
fragments per 30 s visible   4.72 (connector) / 5.87 (GSR baseline)      ``GTA_LINK_STAGE1.md``
appearance top-1             0.6403 at 4.55 candidates                   ``IDENTITY_SOLVER_STAGE2.md``
OCR read density / precision 8.7% of tracklets, 0.875 precision          ``IDENTITY_SOLVER_STAGE2.md``
solve unit                   the 750-frame GSR sequence (gate arm)       ``eval/gsr_identity.py``
===========================  =========================================  =========================

The solver itself is **not** forked: :func:`solve_half` builds
:class:`generator.identity_solve.Tracklet` / :class:`~generator.identity_solve.Identity` objects and
calls the shipped :func:`~generator.identity_solve.solve_assignment`.

Self-check: ``python -m tools.evidence_sim`` (calibration assertions, no I/O).
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from generator.identity_solve import (
    Identity,
    SolverConfig,
    Tracklet,
    exclusion_groups,
    posterior,
    solve_assignment,
)

logger = logging.getLogger("evidence_sim")

#: Broadcast frame rate of FOOTPASS (and of our own match video).
FPS = 25.0
#: Digest directory written by ``tools/footpass_digest.py``.
DIGEST_DIR = Path("data/footpass/digest")

#: Percentiles 0,5,...,100 of merged-tracklet **span** (frames) in the Stage-2 input partition
#: (GTA connector, ``tau = 0.040``, 3,790 tracklets over the 58 GSR sequences). The fragmenter draws
#: fragment lengths from this empirical shape rather than from a parametric law, so the short-blip
#: mass (13.6% of tracklets span <= 5 frames) survives into the simulation.
GSR_SPAN_Q = (1.0, 1.0, 3.0, 7.0, 13.0, 20.0, 32.0, 44.0, 60.0, 78.0, 105.0,
              135.0, 168.0, 213.0, 270.0, 343.0, 431.2, 518.0, 635.1, 745.0, 747.0)

#: OCR reads are **not** uniform over tracklets: a legible number needs a large, sustained crop.
#: Measured on the Stage-2 partition, a tracklet carrying a Koshkina read spans 1.63x the mean
#: (321 vs 196 frames; median 240 vs 95). The simulator draws reads with probability proportional to
#: ``span ** READ_LENGTH_ALPHA``, with alpha fitted to reproduce that ratio -- without it the
#: anchored fraction comes out ~40% below the measured 24.1%.
READ_LENGTH_RATIO = 1.63

#: Measured appearance characteristic to reproduce: top-1 rate and the candidate count it holds at.
APP_TOP1_TARGET = 0.6366
APP_CANDIDATES_TARGET = 4.55
#: Cosine scale of the synthetic similarity channel (GSR: within-identity distance median 0.044,
#: cross-identity median 0.11-0.15 -> similarities 0.956 and 0.85-0.89).
APP_MU_CORRECT = 0.956
APP_SIGMA = 0.05

#: Fragmentation arms, in fragments per 30 s of on-screen time (0 = one tracklet per player-chunk).
FRAG_ARMS = {"none": 0.0, "connector": 4.72, "gsr_baseline": 5.87, "fulham_baseline": 6.74}


# === Configuration ===============================================================================
@dataclass(frozen=True)
class SimConfig:
    """One experimental condition.

    Attributes:
        frag_rate: Target fragments per 30 s of on-screen time; ``0`` = one tracklet per
            player-chunk (the no-fragmentation arm).
        chunk_frames: Solve unit in frames. 750 reproduces the GSR sequence the Stage-2 numbers
            were measured on; 3,000 (2 min) is the match-scale default.
        ocr_density: Fraction of tracklets carrying an OCR-like read.
        ocr_precision: Probability such a read names the tracklet's true player.
        mention_rate: Commentary mentions per minute of the half.
        mention_precision: Probability a mention names the true actor of the event it was drawn from.
        mention_lag_iqr: Interquartile range, in seconds, of the mention's timestamp lag.
        mention_lag_compensate: Subtract the lag distribution's *median* before binding. Any real
            implementation would calibrate this out, so the uncompensated arm is the pessimistic
            bound and this one isolates the residual spread.
        mention_bind_oracle: Bind each mention to the acting player's own tracklet, i.e. assume the
            alignment problem is solved. Upper bound on what the commentary channel can ever buy.
        team_error: Probability a tracklet's majority team is wrong.
        role_error: Probability a goalkeeper's role is not recognised (and, at a fifth of the rate,
            that an outfielder is called a keeper).
        seed: Base RNG seed; the per-half stream is derived from it.
    """

    frag_rate: float = 4.72
    chunk_frames: int = 3000
    ocr_density: float = 0.087
    ocr_precision: float = 0.86
    mention_rate: float = 0.0
    mention_precision: float = 0.6
    mention_lag_iqr: float = 4.0
    mention_lag_compensate: bool = False
    mention_bind_oracle: bool = False
    team_error: float = 0.05
    role_error: float = 0.10
    seed: int = 0


def solver_config(cfg: SimConfig) -> SolverConfig:
    """Stage-2's frozen calibration, with the read model told each channel's true precision.

    ``p_correct = 1`` makes the per-read confidence *be* ``P(read correct)``, so OCR-like and
    commentary-like reads with different precisions can coexist in one solve. The digit-confusion
    prior is off: Stage 2 measured it inert (p = 0.52) and recommended deleting it.
    """
    return SolverConfig(p_correct=1.0, use_confusion=False, app_gain=10.0, sim_none=0.92,
                        topk=3, pi_none=0.70, r_abstain=0.0, team_eps=0.02, role_eps=0.05,
                        max_concurrent=11)


# === Fragmentation ===============================================================================
def draw_spans(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    """Sample ``n`` fragment lengths from the measured GSR span shape, scaled by ``scale`` (pure)."""
    u = rng.random(n) * 100.0
    return np.maximum(1.0, np.interp(u, np.arange(0, 101, 5), GSR_SPAN_Q) * scale)


def fragment(runs: np.ndarray, chunk: tuple[int, int], rate: float, scale: float,
             rng: np.random.Generator) -> np.ndarray:
    """Cut on-screen runs into tracklets inside one chunk (pure given ``rng``).

    Args:
        runs: ``(n, 3)`` array of ``(player index, first frame, last frame)`` on-screen runs.
        chunk: Inclusive ``(first, last)`` frame bounds of the solve unit.
        rate: Target fragments per 30 s of on-screen time; ``0`` merges every run of a player in the
            chunk into a single tracklet (the no-fragmentation arm).
        scale: Multiplier on the drawn span lengths (set by :func:`calibrate_fragmentation`).
        rng: Random source.

    Returns:
        ``(m, 4)`` array of ``(player index, start, end, visible frames)``.
    """
    lo, hi = chunk
    sel = runs[(runs[:, 2] >= lo) & (runs[:, 1] <= hi)]
    if sel.size == 0:
        return np.zeros((0, 4), np.int64)
    a = np.maximum(sel[:, 1], lo)
    b = np.minimum(sel[:, 2], hi)
    if rate <= 0.0:
        out = []
        for p in np.unique(sel[:, 0]):
            m = sel[:, 0] == p
            out.append((p, a[m].min(), b[m].max(), int((b[m] - a[m] + 1).sum())))
        return np.array(out, np.int64)
    out = []
    for p, s, e in zip(sel[:, 0], a, b):
        pos = int(s)
        while pos <= e:
            span = int(draw_spans(rng, 1, scale)[0])
            end = min(int(e), pos + span - 1)
            out.append((p, pos, end, end - pos + 1))
            pos = end + 1
    return np.array(out, np.int64)


def calibrate_fragmentation(halves: list[dict], rate: float, chunk_frames: int,
                            seed: int = 7) -> float:
    """Bisect the span scale until the achieved fragments per 30 s of on-screen time hits ``rate``.

    Chunk boundaries fragment tracks on their own, so the scale is fitted *after* chunking -- the
    achieved rate is what the arm is named for.

    Args:
        halves: Digests to calibrate on.
        rate: Target fragments per 30 s on-screen.
        chunk_frames: Solve unit length.
        seed: RNG seed for the calibration draws.

    Returns:
        The span scale to pass to :func:`fragment`.
    """
    if rate <= 0.0:
        return 1.0

    def achieved(scale: float) -> float:
        n_frag = vis = 0
        for h in halves:
            rng = np.random.default_rng(seed)
            for chunk in chunks_of(h, chunk_frames):
                t = fragment(h["runs"], chunk, rate, scale, rng)
                n_frag += t.shape[0]
                vis += int(t[:, 3].sum()) if t.size else 0
        return n_frag / max(vis / FPS / 30.0, 1e-9)

    lo, hi = 1e-3, 40.0
    if achieved(hi) > rate:
        # Visibility breaks plus chunk boundaries already fragment more than the target asks for;
        # no amount of lengthening the drawn spans can reach it. Report and use the floor.
        logger.warning("fragmentation floor: rate %.2f unreachable at chunk %d (floor %.2f)",
                       rate, chunk_frames, achieved(hi))
        return hi
    for _ in range(24):
        mid = (lo + hi) / 2
        if achieved(mid) > rate:
            lo = mid          # too many fragments -> longer spans
        else:
            hi = mid
    return (lo + hi) / 2


def chunks_of(half: dict, chunk_frames: int) -> list[tuple[int, int]]:
    """Inclusive frame bounds of every solve unit in a half (pure)."""
    f0 = int(half["runs"][:, 1].min()) if half["runs"].size else 0
    f1 = int(half["runs"][:, 2].max()) if half["runs"].size else 0
    return [(s, min(s + chunk_frames - 1, f1)) for s in range(f0, f1 + 1, chunk_frames)]


# === Appearance channel ==========================================================================
def _top1_rate(d: float, k: float) -> float:
    """P(correct wins) when the true score is ``N(d, 1)`` and ``k - 1`` distractors are ``N(0, 1)``."""
    x = np.linspace(-8.0, 12.0, 4001)
    from scipy.stats import norm  # noqa: PLC0415
    return float(np.trapezoid(norm.pdf(x - d) * norm.cdf(x) ** (k - 1.0), x))


def appearance_separation(top1: float = APP_TOP1_TARGET,
                          candidates: float = APP_CANDIDATES_TARGET) -> float:
    """Signal-to-noise ``d`` reproducing a measured top-1 rate at a measured candidate count."""
    lo, hi = 0.0, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _top1_rate(mid, candidates) < top1:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


#: Fitted once at import: the separation the synthetic similarity channel runs at.
APP_D = appearance_separation()


def read_length_alpha(ratio: float = READ_LENGTH_RATIO) -> float:
    """Exponent making read probability ``prop. span**alpha`` reproduce a measured length bias.

    Args:
        ratio: Mean span of a tracklet carrying a read, over the mean span of all tracklets.

    Returns:
        The exponent ``alpha``, fitted against the measured GSR span distribution.
    """
    lens = np.interp(np.linspace(0, 100, 2001), np.arange(0, 101, 5), GSR_SPAN_Q)
    lo, hi = 0.0, 4.0
    for _ in range(50):
        mid = (lo + hi) / 2
        w = lens ** mid
        got = float((w * lens).sum() / w.sum() / lens.mean())
        if got < ratio:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


#: Fitted once at import.
READ_ALPHA = read_length_alpha()


def similarity(true_player: np.ndarray, gallery_owner: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
    """``(n_tracklets, n_identities)`` synthetic cosine similarities (pure given ``rng``).

    An identity with no gallery (``gallery_owner < 0``) scores ``nan``, exactly as an identity with
    no OCR-anchored crops does in ``generator.identity_solve``. An identity whose gallery was
    anchored by a *wrong* read carries the crops of whoever was actually in those tracklets, so its
    high-similarity partner is the gallery's owner, not the roster slot -- read errors propagate into
    the appearance channel the way they do in the real pipeline.
    """
    n_t, n_i = true_player.size, gallery_owner.size
    sim = APP_MU_CORRECT - APP_D * APP_SIGMA + APP_SIGMA * rng.standard_normal((n_t, n_i))
    hit = true_player[:, None] == gallery_owner[None, :]
    sim = np.where(hit, sim + APP_D * APP_SIGMA, sim)
    return np.where(gallery_owner[None, :] >= 0, sim, np.nan)


# === Evidence channels ===========================================================================
def ball_track(half: dict) -> tuple[np.ndarray, np.ndarray]:
    """Ball proxy: the acting player's position at each action event (pure).

    FOOTPASS ships no ball track. Events are ~3 s apart, so the chain of actors' positions is a
    usable stand-in: the ball is at the actor when an action is annotated, and is interpolated in
    between. Returns ``(frames, (n, 2) positions)``.
    """
    ev = half["events"]
    f0, stride = half["grid"]
    step = np.clip((ev[:, 0] - f0) // stride, 0, half["pos"].shape[0] - 1)
    return ev[:, 0].astype(np.int64), half["pos"][step, ev[:, 1]]


def positions_at(half: dict, frame: int) -> np.ndarray:
    """``(n_players, 2)`` positions at the grid step nearest ``frame`` (pure)."""
    f0, stride = half["grid"]
    step = int(np.clip((frame - f0) // stride, 0, half["pos"].shape[0] - 1))
    return half["pos"][step]


def bind_mentions(half: dict, tracklets: np.ndarray, chunk: tuple[int, int], cfg: SimConfig,
                  rng: np.random.Generator) -> list[tuple[int, int, float]]:
    """Generate commentary mentions in a chunk and bind each to the tracklet nearest the ball.

    A mention is drawn from an action event, names the true actor with probability
    ``cfg.mention_precision`` (otherwise another player on the pitch), and is *timestamped late* by a
    lag drawn from a Gamma with the requested IQR. It then attaches to whichever tracklet is alive at
    the lagged time and closest to the ball proxy there -- which is frequently the wrong player, and
    that is the failure mode being priced.

    Args:
        half: Digest of the half.
        tracklets: ``(m, 4)`` tracklets of this chunk.
        chunk: Inclusive frame bounds.
        cfg: Condition.
        rng: Random source.

    Returns:
        ``(tracklet index, mentioned shirt-number index, precision)`` triples.
    """
    lo, hi = chunk
    ev = half["events"]
    ev = ev[(ev[:, 0] >= lo) & (ev[:, 0] <= hi)]
    minutes = (hi - lo + 1) / FPS / 60.0
    n = rng.poisson(cfg.mention_rate * minutes)
    if n == 0 or ev.shape[0] == 0 or tracklets.size == 0:
        return []
    pick = rng.choice(ev.shape[0], size=int(n), replace=ev.shape[0] < n)
    # Gamma(k=2) is right-skewed like ASR/commentary delay; theta set so the IQR matches the arm.
    theta = cfg.mention_lag_iqr / 1.7311
    lag = rng.gamma(2.0, theta, size=int(n))
    if cfg.mention_lag_compensate:
        lag -= 1.6783 * theta          # median of Gamma(2, theta): the calibratable part of the lag
    lag *= FPS
    ball_f, ball_p = ball_track(half)
    out: list[tuple[int, int, float]] = []
    n_players = half["shirt"].size
    for k, lg in zip(pick, lag):
        actor = int(ev[k, 1])
        named = actor
        if rng.random() >= cfg.mention_precision:
            named = int(rng.integers(n_players - 1))
            named += int(named >= actor)
        t_lag = int(np.clip(ev[k, 0] + lg, lo, hi))
        if cfg.mention_bind_oracle:
            own = np.flatnonzero((tracklets[:, 0] == actor) & (tracklets[:, 1] <= ev[k, 0])
                                 & (tracklets[:, 2] >= ev[k, 0]))
            if own.size:
                out.append((int(own[0]), named, cfg.mention_precision))
            continue
        alive = np.flatnonzero((tracklets[:, 1] <= t_lag) & (tracklets[:, 2] >= t_lag))
        if alive.size == 0:
            continue
        bx = np.array([np.interp(t_lag, ball_f, ball_p[:, 0]),
                       np.interp(t_lag, ball_f, ball_p[:, 1])])
        pos = positions_at(half, t_lag)[tracklets[alive, 0]]
        d = np.linalg.norm(np.nan_to_num(pos, nan=9.0) - bx, axis=1)
        out.append((int(alive[int(np.argmin(d))]), named, cfg.mention_precision))
    return out


def ocr_reads(tracklets: np.ndarray, cfg: SimConfig,
              rng: np.random.Generator, n_players: int) -> list[tuple[int, int, float]]:
    """OCR-like reads at mean density ``cfg.ocr_density``, biased towards long tracklets (pure).

    The per-tracklet probability is ``prop. span ** READ_ALPHA``, rescaled (with clipping) so the
    mean over tracklets equals the requested density -- so ``ocr_density`` keeps meaning "fraction of
    tracklets carrying a read", the quantity Stage 2 measured at 8.7%.
    """
    if tracklets.size == 0 or cfg.ocr_density <= 0:
        return []
    w = tracklets[:, 3].astype(float) ** READ_ALPHA
    p = w / w.mean() * cfg.ocr_density
    for _ in range(4):  # fixed point: clipping at 1 removes mass, put it back on the rest
        p = np.clip(p, 0.0, 1.0)
        deficit = cfg.ocr_density * p.size - p.sum()
        room = p < 1.0
        if deficit <= 1e-9 or not room.any():
            break
        p[room] += deficit * w[room] / w[room].sum()
    take = np.flatnonzero(rng.random(tracklets.shape[0]) < np.clip(p, 0.0, 1.0))
    out = []
    for t in take:
        true = int(tracklets[t, 0])
        named = true
        if rng.random() >= cfg.ocr_precision:
            named = int(rng.integers(n_players - 1))
            named += int(named >= true)
        out.append((int(t), named, cfg.ocr_precision))
    return out


# === One solve unit ==============================================================================
def _alive_by_frame(tracklets: np.ndarray, chunk: tuple[int, int], stride: int = 5) -> list[list]:
    """Per-frame alive tracklet indices, sampled every ``stride`` frames.

    ponytail: stride-5 sampling can miss an overlap shorter than 5 frames; that costs at most a
    handful of mutex constraints and keeps the exclusion-group count linear in chunk length. Drop to
    stride 1 if a future arm cares about sub-200 ms overlaps.
    """
    lo, hi = chunk
    if tracklets.size == 0:
        return []
    return [np.flatnonzero((tracklets[:, 1] <= f) & (tracklets[:, 2] >= f)).tolist()
            for f in range(lo, hi + 1, stride)]


def solve_chunk(half: dict, chunk: tuple[int, int], cfg: SimConfig, scale: float,
                rng: np.random.Generator, scfg: SolverConfig) -> dict:
    """Simulate evidence in one chunk, solve it, and return per-tracklet assignments.

    Returns:
        ``{"tracklets", "assigned", "conf", "anchored", "read_direct", "app_argmax"}`` -- the
        tracklet table, the solver's identity index per tracklet (``-1`` = abstain), the posterior of
        that identity, whether the tracklet's true identity had any read in the chunk, the
        direct-read baseline's answer, and the unconstrained appearance argmax.
    """
    trk = fragment(half["runs"], chunk, cfg.frag_rate, scale, rng)
    n_players = int(half["shirt"].size)
    if trk.size == 0:
        return {"tracklets": trk, "assigned": np.zeros(0, int), "conf": np.zeros(0),
                "anchored": np.zeros(0, bool), "read_direct": np.zeros(0, int),
                "app_argmax": np.zeros(0, int)}

    reads = ocr_reads(trk, cfg, rng, n_players)
    reads += bind_mentions(half, trk, chunk, cfg, rng)

    # Galleries: an identity is anchored by the tracklets read as it; its appearance owner is the
    # player who actually dominates those tracklets.
    owner = np.full(n_players, -1)
    for i in range(n_players):
        mine = [t for t, named, _ in reads if named == i]
        if mine:
            w = np.zeros(n_players)
            np.add.at(w, trk[mine, 0], trk[mine, 3])
            owner[i] = int(w.argmax())
    sims = similarity(trk[:, 0], owner, rng)

    lo, hi = chunk
    ident = [Identity(key=i, team=int(half["team"][i]), number=int(half["shirt"][i]),
                      role="goalkeeper" if half["role"][i] == 1 else "player",
                      window=(int(half["window"][i, 0]), int(half["window"][i, 1])))
             for i in range(n_players)]
    numbers = np.array([id_.number for id_ in ident])
    by_trk: dict[int, list[tuple[int, float]]] = {}
    for t, named, prec in reads:
        by_trk.setdefault(t, []).append((int(numbers[named]), float(prec)))

    teams = half["team"][trk[:, 0]]
    flip = rng.random(trk.shape[0]) < cfg.team_error
    teams = np.where(flip, 3 - teams, teams)
    is_gk = half["role"][trk[:, 0]] == 1
    seen_gk = np.where(is_gk, rng.random(trk.shape[0]) >= cfg.role_error,
                       rng.random(trk.shape[0]) < cfg.role_error / 5.0)

    objs = [Tracklet(track_id=int(k), n_rows=int(trk[k, 3]), team=int(teams[k]), team_frac=0.9,
                     role_frac={"goalkeeper" if seen_gk[k] else "player": 1.0},
                     reads=tuple(by_trk.get(k, ())), emb=np.zeros((0, 1)),
                     span=(int(trk[k, 1]), int(trk[k, 2])))
            for k in range(trk.shape[0])]
    probs = np.stack([posterior(o, ident, sims[k], scfg) for k, o in enumerate(objs)])
    payoff = probs.copy()
    payoff[:, -1] = scfg.r_abstain
    groups = exclusion_groups(_alive_by_frame(trk, (lo, hi)))
    pick = solve_assignment(payoff, trk[:, 3].astype(float), groups,
                            [id_.team for id_ in ident], max_concurrent=scfg.max_concurrent,
                            max_candidates=6, time_limit=20.0)
    conf = np.array([probs[k, p] if p >= 0 else 0.0 for k, p in enumerate(pick)])

    # Reference arm 1: the greedy rule -- a tracklet takes its own majority read, resolved against
    # its observed team (a jersey number alone is ambiguous when both squads field it).
    id_team = np.array([id_.team for id_ in ident])
    direct = np.full(trk.shape[0], -1)
    for t, lst in by_trk.items():
        nums = [n for n, _ in lst]
        cand = np.flatnonzero(numbers == max(set(nums), key=nums.count))
        same = cand[id_team[cand] == teams[t]]
        direct[t] = int(same[0]) if same.size else int(cand[0])
    # Reference arm 2: unconstrained appearance argmax within the observed team (the 4.5-candidate
    # decision Stage 2 measured at 0.64 top-1).
    masked = np.where(id_team[None, :] == teams[:, None], np.nan_to_num(sims, nan=-9.0), -9.0)
    app = np.where((masked > -9.0).any(1), masked.argmax(1), -1)
    return {"tracklets": trk, "assigned": pick, "conf": conf,
            "anchored": np.isin(trk[:, 0], np.flatnonzero(owner >= 0)),
            "read_direct": direct, "app_argmax": app,
            "n_reads": len(reads),
            "n_reads_correct": sum(1 for t, named, _ in reads if trk[t, 0] == named)}


# === Scoring =====================================================================================
def score_half(half: dict, cfg: SimConfig, scale: float, scfg: SolverConfig) -> dict:
    """Run every chunk of a half and grade the action events, PCBAS-style.

    An event is *attributable* only when its actor is on-screen at the annotated frame (a tracklet
    exists to carry the name); off-screen events count against coverage, never against precision.

    Returns:
        Per-event arrays ``correct`` / ``answered`` / ``conf`` for the solver, plus the two reference
        arms and the per-condition diagnostics used by the calibration gate.
    """
    # zlib.crc32, not hash(): Python's str hash is salted per process and would break determinism
    # across the sweep's worker pool.
    rng = np.random.default_rng(zlib.crc32(str(half["key"]).encode()) ^ (cfg.seed * 2654435761))
    ev = half["events"]
    runs = half["runs"]
    n_frag = n_vis = n_reads = n_reads_ok = 0
    rec = {k: [] for k in ("correct", "answered", "conf", "base_correct", "base_answered",
                           "app_correct", "app_answered", "anchored", "attributable")}
    for chunk in chunks_of(half, cfg.chunk_frames):
        r = solve_chunk(half, chunk, cfg, scale, rng, scfg)
        trk = r["tracklets"]
        if trk.size == 0:
            continue
        n_frag += trk.shape[0]
        n_vis += int(trk[:, 3].sum())
        n_reads += r["n_reads"]
        n_reads_ok += r["n_reads_correct"]
        sel = ev[(ev[:, 0] >= chunk[0]) & (ev[:, 0] <= chunk[1])]
        for f, p, _cls in sel:
            on = np.any((runs[:, 0] == p) & (runs[:, 1] <= f) & (runs[:, 2] >= f))
            hit = np.flatnonzero((trk[:, 0] == p) & (trk[:, 1] <= f) & (trk[:, 2] >= f)) if on \
                else np.zeros(0, int)
            rec["attributable"].append(bool(on and hit.size))
            if not (on and hit.size):
                for k in ("correct", "answered", "base_correct", "base_answered", "app_correct",
                          "app_answered", "anchored"):
                    rec[k].append(False)
                rec["conf"].append(0.0)
                continue
            k = int(hit[0])
            rec["anchored"].append(bool(r["anchored"][k]))
            rec["answered"].append(r["assigned"][k] >= 0)
            rec["correct"].append(r["assigned"][k] == p)
            rec["conf"].append(float(r["conf"][k]))
            rec["base_answered"].append(r["read_direct"][k] >= 0)
            rec["base_correct"].append(r["read_direct"][k] == p)
            rec["app_answered"].append(r["app_argmax"][k] >= 0)
            rec["app_correct"].append(r["app_argmax"][k] == p)
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["n_fragments"] = n_frag
    out["visible_seconds"] = n_vis / FPS
    out["n_reads"] = n_reads
    out["n_reads_correct"] = n_reads_ok
    return out


def load_halves(split: str, limit: int | None = None) -> list[dict]:
    """Load digests for a split (``"train"`` / ``"val"``), sorted by key."""
    fs = sorted(DIGEST_DIR.glob(f"{split}_*.npz"))[:limit]
    return [{k: d[k] for k in d.files} for d in (np.load(f) for f in fs)]


def frontier(correct: np.ndarray, answered: np.ndarray, conf: np.ndarray,
             n_events: int, n_points: int = 60) -> list[tuple[float, float, float]]:
    """Precision-coverage frontier from the posterior-confidence dial (pure).

    Answers are ranked by the posterior of the identity the solver assigned and taken most-confident
    first, so the frontier is sampled evenly **in coverage** rather than on a threshold grid (a
    threshold grid collapses: 80% of the answers sit below posterior 0.08 when 30 roster slots share
    the mass). Coverage is over *all* events in scope, including those whose actor was off-screen;
    precision is over answered events only.

    Returns:
        ``(threshold, coverage, precision)`` triples, most confident first.
    """
    m = np.flatnonzero(answered)
    if m.size == 0:
        return [(1.0, 0.0, float("nan"))]
    order = m[np.argsort(-conf[m], kind="stable")]
    cum = np.cumsum(correct[order].astype(float))
    ranks = np.unique(np.linspace(1, order.size, n_points).astype(int))
    return [(float(conf[order][r - 1]), float(r / max(n_events, 1)), float(cum[r - 1] / r))
            for r in ranks]


def coverage_at(front: list[tuple[float, float, float]], floor: float) -> float:
    """Maximum coverage whose precision clears ``floor`` (pure); 0.0 if the floor is never met."""
    ok = [c for _t, c, p in front if np.isfinite(p) and p >= floor]
    return max(ok) if ok else 0.0


def _demo() -> None:
    """Self-check: the appearance channel and the fragmenter reproduce their measured targets."""
    rng = np.random.default_rng(0)
    # 1. appearance calibration -- top-1 among 4.55 anchored candidates must land at 0.62-0.64.
    hits = tot = 0
    for k in (4, 5):
        owner = np.arange(k)
        true = rng.integers(0, k, 20000)
        s = similarity(true, owner, rng)
        w = 0.45 if k == 4 else 0.55
        hits += w * (s.argmax(1) == true).sum()
        tot += w * true.size
    top1 = hits / tot
    assert 0.62 <= top1 <= 0.65, top1
    # 2. fragmenter: the drawn span shape matches the measured GSR quantiles.
    sp = draw_spans(rng, 200000, 1.0)
    for q, want in ((25, 20.0), (50, 105.0), (75, 343.0)):
        got = np.percentile(sp, q)
        assert abs(got - want) / want < 0.15, (q, got, want)
    print(f"evidence_sim demo OK: appearance top-1 {top1:.4f} at ~4.55 candidates (target "
          f"{APP_TOP1_TARGET:.4f}); span p50 {np.percentile(sp, 50):.0f} frames (GSR 105)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _demo()
