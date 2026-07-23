"""B2 Stage-2c: wire gated close-up jersey anchors into named tactical tracks.

The step-3 gate (``tools/closeup_anchor_probe.py --kit-gate --ocr-gate``) produced 226
high-precision (98.6% verified) close-up anchors, each a ``(chunk, frame, back-number)`` read sitting
inside a close-up shot. This module carries those reads onto the *tactical* ByteTrack fragments so a
track can be named, under two hard rules from the project's Stage-2a/2b findings:

* **Appearance-only ReID cannot separate same-kit players** (Stage-2a: ImageNet OSNet median cosine
  0.81 on constraint-valid same-team pairs; relink merge precision 35%). So attachment commits a name
  to a track only when one candidate is clearly the most similar -- a frozen margin gate
  (:func:`choose_track`), otherwise the anchor stays unattached.
* **Names never cross an untrusted relink merge for free** (Stage-2a production guard): a name
  propagates freely *within* one ByteTrack fragment ``(chunk, track_id)``, but across a relink merge
  only when ``>= 2`` independent anchors agree on the merged identity (:func:`resolve_identities`).
  Two anchors disagreeing on one fragment flag it and name neither.

All functions here are CPU-pure and unit-tested; the GPU work (OSNet embeddings, video crop reads)
lives in the runner :mod:`tools.wire_anchors`.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Survivor crop filename, e.g. ``h1_chunk_000_f10645_n08_c0.799.jpg`` (chunk holds underscores).
_SURVIVOR_RE = re.compile(r"^(h\d_chunk_\d+)_f(\d+)_n(\d+)_c([\d.]+)\.jpg$")

# --- FROZEN attachment-confidence rule (pre-committed before the run) ------------------------------
# No per-attachment ground truth exists (there is no annotated anchor->track map for this match), so
# these are reasoned defaults, not data-tuned thresholds -- stated as such in the report. Rationale:
# OSNet same-team cosines cluster ~0.8 with a small spread (Stage-2a), so a 0.05 gap is a meaningful
# separation between the winning candidate and its nearest same-kit rival; MIN_SIM only rejects a
# clearly-wrong best match (close-up vs wide crop, so kept loose).
REID_MIN_MARGIN = 0.05
REID_MIN_SIM = 0.50

# PRTreID lives on a different cosine scale than OSNet, so it needs its own frozen gate. GT-audited on
# 58 SoccerNet-GSR sequences (cached embeddings, tools/prtreid_probe.py): the ABSOLUTE similarity is
# the precision lever (same-player p50 0.918 vs different-player-same-kit p50 0.820 -- a real signal
# OSNet lacks), so min_sim carries the gate and rises from OSNet's toothless 0.50 (which never fires
# on PRTreID's ~0.82+ floor) to 0.92; a wider margin is precision-NEGATIVE on this embedder (the
# margin-gate simulation shows precision FALLING as min_margin rises once min_sim is meaningful), so
# the margin stays at OSNet's value. See results/gsr_benchmark/prtreid_probe_soccernet_sweep.json.
PRTREID_MIN_SIM = 0.92
PRTREID_MIN_MARGIN = 0.05


@dataclass(frozen=True)
class Anchor:
    """One gated close-up anchor read (a survivor crop of the step-3 extractor).

    Attributes:
        chunk: Chunk key (e.g. ``h1_chunk_000``).
        frame: Sampled frame index the anchor crop came from.
        number: Back-of-shirt number the recognizer read (== oracle ``shirtNumber``).
        conf: Recognizer peak-class confidence.
        crop_path: Absolute path to the survivor crop JPG.
    """

    chunk: str
    frame: int
    number: int
    conf: float
    crop_path: str


@dataclass(frozen=True)
class Attachment:
    """An anchor committed to a tactical fragment by the ReID margin gate.

    Attributes:
        chunk: Chunk key.
        track_id: The ByteTrack fragment id the anchor attached to.
        number: Back number carried onto the fragment.
        team: The fragment's team label (0/1) from the positions table.
        conf: The anchor's recognizer confidence.
        basis: How the track was chosen (``reid_margin`` / ``sole_candidate``).
    """

    chunk: str
    track_id: int
    number: int
    team: int
    conf: float
    basis: str


@dataclass(frozen=True)
class ResolvedFragment:
    """A named fragment after the propagation guard.

    Attributes:
        chunk: Chunk key.
        track_id: Fragment id.
        number: Back number assigned to the fragment.
        team: Team label (0/1).
        n_anchors: Independent anchors supporting the name.
        basis: ``fragment`` (own anchors) or ``merge_2anchor`` (crossed a relink merge on >=2 agree).
    """

    chunk: str
    track_id: int
    number: int
    team: int
    n_anchors: int
    basis: str


def parse_survivor_name(fname: str) -> tuple[str, int, int, float] | None:
    """Parse a step-3 survivor crop filename into ``(chunk, frame, number, conf)`` (pure).

    Args:
        fname: The crop file name (``<chunk>_f<frame>_n<NN>_c<conf>.jpg``).

    Returns:
        ``(chunk, frame, number, conf)`` or ``None`` if the name does not match the pattern.
    """
    m = _SURVIVOR_RE.match(fname)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))


def nearest_wide_frame(frame: int, wide_frames: np.ndarray, win: int) -> int | None:
    """Nearest ``live_wide`` frame within ``+/- win`` of ``frame`` across the cut (pure).

    Args:
        frame: The anchor's close-up frame index.
        wide_frames: Sorted array of the chunk's ``live_wide`` frame indices.
        win: Window half-width in frames (``round(NEAR_WINDOW_S * fps)``).

    Returns:
        The closest wide frame index within the window, or ``None`` if none is in range.
    """
    if wide_frames.size == 0:
        return None
    near = wide_frames[np.abs(wide_frames - frame) <= win]
    if near.size == 0:
        return None
    return int(near[int(np.abs(near - frame).argmin())])


def choose_track(
    sims: dict[int, float], *, min_margin: float = REID_MIN_MARGIN, min_sim: float = REID_MIN_SIM
) -> tuple[int | None, str]:
    """Pick a track for one anchor by the frozen ReID margin rule (pure).

    Args:
        sims: ``track_id -> cosine similarity`` between the anchor crop and each candidate track crop.
        min_margin: Minimum best-minus-second-best cosine gap to commit.
        min_sim: Minimum best cosine to commit at all.

    Returns:
        ``(track_id, basis)`` on a confident attach, else ``(None, reason)`` where reason is one of
        ``no_candidate`` / ``low_sim`` / ``ambiguous``.
    """
    if not sims:
        return None, "no_candidate"
    ranked = sorted(sims.items(), key=lambda kv: -kv[1])
    best_tid, best = ranked[0]
    if best < min_sim:
        return None, "low_sim"
    if len(ranked) == 1:
        return best_tid, "sole_candidate"
    if best - ranked[1][1] < min_margin:
        return None, "ambiguous"
    return best_tid, "reid_margin"


def resolve_identities(
    attachments: list[Attachment], remap: dict[tuple[str, int], object] | None = None
) -> tuple[list[ResolvedFragment], list[dict]]:
    """Apply the propagation guard to a set of attachments (pure).

    Within one fragment ``(chunk, track_id)`` a name propagates freely, but a fragment carrying two
    anchors that read different numbers is flagged and named by neither. When a relink ``remap`` is
    supplied, a name crosses a merge (onto sibling fragments in the same relink group, including
    un-anchored ones) only if ``>= 2`` independent anchors in the group agree on a single number;
    otherwise each fragment keeps only its own directly-attached name.

    Args:
        attachments: Committed anchor->fragment attachments.
        remap: Optional ``(chunk, track_id) -> group_key`` relink map (``None`` = no merges applied,
            the brighton_manutd case: every fragment stands alone).

    Returns:
        ``(resolved, flags)`` -- named fragments and a list of flag dicts (disagreements).
    """
    by_frag: dict[tuple[str, int], list[Attachment]] = defaultdict(list)
    for a in attachments:
        by_frag[(a.chunk, a.track_id)].append(a)

    flags: list[dict] = []
    frag_name: dict[tuple[str, int], tuple[int, int, int]] = {}  # frag -> (number, team, n_anchors)
    for frag, atts in by_frag.items():
        nums = Counter(a.number for a in atts)
        if len(nums) == 1:
            frag_name[frag] = (atts[0].number, atts[0].team, len(atts))
        else:
            flags.append({"type": "fragment_disagreement", "chunk": frag[0],
                          "track_id": frag[1], "numbers": dict(nums)})

    if remap is None:
        resolved = [ResolvedFragment(f[0], f[1], n, t, k, "fragment")
                    for f, (n, t, k) in frag_name.items()]
        return resolved, flags

    groups: dict[object, list[tuple[str, int]]] = defaultdict(list)
    for frag in set(by_frag) | set(remap):  # include un-anchored siblings named only in the remap
        groups[remap.get(frag, frag)].append(frag)

    resolved = []
    for frags in groups.values():
        named = [f for f in frags if f in frag_name]
        if len(frags) > 1 and len(named) >= 2:
            counts: Counter = Counter()
            for f in named:
                num, _team, k = frag_name[f]
                counts[num] += k
            top_num, top_cnt = counts.most_common(1)[0]
            unique_top = list(counts.values()).count(top_cnt) == 1
            if top_cnt >= 2 and unique_top:
                team = frag_name[named[0]][1]
                for f in frags:  # cross the merge: name every sibling, anchored or not
                    resolved.append(ResolvedFragment(f[0], f[1], top_num, team, top_cnt,
                                                     "merge_2anchor"))
                continue
        for f in named:  # no trusted merge: each fragment keeps only its own name
            num, team, k = frag_name[f]
            resolved.append(ResolvedFragment(f[0], f[1], num, team, k, "fragment"))
    return resolved, flags


def _norm_name(name: str) -> str:
    """NFC-normalise a possibly-mojibake oracle name (accents kept as proper unicode)."""
    return unicodedata.normalize("NFC", str(name)).strip()


def build_roster_maps(
    oracle_df: pd.DataFrame, team_id_fn, *, number_col: str = "shirtNumber"
) -> tuple[dict[tuple[int, int], str], dict[int, set[int]]]:
    """Build ``(team, number) -> name`` and ``number -> {team}`` maps from the oracle (pure).

    The recognizer reads the physical back-of-shirt number, which Sofascore stores in
    ``shirtNumber`` -- NOT ``jerseyNumber`` (the two differ for several players; see the report). A
    ``(team, number)`` key is unique because shirt numbers only repeat across the two teams.

    Args:
        oracle_df: The cached per-player oracle parquet.
        team_id_fn: Maps an oracle ``teamName`` to a positions-table team id (0/1).
        number_col: Which oracle column holds the back number (``shirtNumber``).

    Returns:
        ``(name_by_team_number, teams_by_number)``.
    """
    name_by: dict[tuple[int, int], str] = {}
    teams_by: dict[int, set[int]] = defaultdict(set)
    for row in oracle_df.itertuples(index=False):
        raw = getattr(row, number_col)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        try:
            num = int(str(raw))
        except (TypeError, ValueError):
            continue
        team = team_id_fn(getattr(row, "teamName"))
        if team is None:
            continue
        name_by[(team, num)] = _norm_name(getattr(row, "name"))
        teams_by[num].add(team)
    return name_by, dict(teams_by)


def merge_intervals(spans: list[tuple[int, int]]) -> int:
    """Total covered length of a set of ``(start, end)`` frame spans, unioning overlaps (pure)."""
    if not spans:
        return 0
    spans = sorted(spans)
    total = 0
    cs, ce = spans[0]
    for s, e in spans[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            total += ce - cs
            cs, ce = s, e
    total += ce - cs
    return total


def _demo() -> None:
    """Self-check of the margin gate and propagation guard on synthetic inputs (asserts; runnable)."""
    # Margin gate.
    assert choose_track({}) == (None, "no_candidate")
    assert choose_track({7: 0.9, 3: 0.6}) == (7, "reid_margin")          # clear 0.30 margin
    assert choose_track({7: 0.82, 3: 0.80})[0] is None                   # 0.02 < 0.05 -> ambiguous
    assert choose_track({7: 0.4}) == (None, "low_sim")                   # below MIN_SIM
    assert choose_track({7: 0.7}) == (7, "sole_candidate")               # one candidate, above sim

    A = Attachment
    # Within-fragment: two agreeing anchors -> named with n=2; disagreement -> flagged, unnamed.
    res, flags = resolve_identities([A("h1", 5, 8, 0, 0.9, "reid_margin"),
                                     A("h1", 5, 8, 0, 0.8, "reid_margin"),
                                     A("h1", 9, 8, 0, 0.9, "reid_margin"),
                                     A("h1", 9, 10, 0, 0.9, "reid_margin")])
    named = {(r.track_id): r for r in res}
    assert named[5].number == 8 and named[5].n_anchors == 2 and named[5].basis == "fragment"
    assert 9 not in named and any(f["track_id"] == 9 for f in flags)

    # Merge guard: two anchors agree across a merge -> name crosses onto the un-anchored sibling.
    remap = {("h1", 1): "g", ("h1", 2): "g", ("h1", 3): "g"}
    res, _ = resolve_identities([A("h1", 1, 8, 0, 0.9, "reid_margin"),
                                 A("h1", 2, 8, 0, 0.9, "reid_margin")], remap)
    got = {r.track_id: r for r in res}
    assert got[1].number == got[2].number == got[3].number == 8  # sibling 3 (no anchor) inherits
    assert all(r.basis == "merge_2anchor" for r in res)

    # Merge guard: a single anchor does NOT cross the merge (sibling stays un-named).
    res, _ = resolve_identities([A("h1", 1, 8, 0, 0.9, "reid_margin")], remap)
    got = {r.track_id: r for r in res}
    assert set(got) == {1} and got[1].basis == "fragment"  # only the anchored frag is named

    # Merge guard: two anchors that disagree do not cross (each keeps its own).
    res, _ = resolve_identities([A("h1", 1, 8, 0, 0.9, "reid_margin"),
                                 A("h1", 2, 10, 0, 0.9, "reid_margin")], remap)
    got = {r.track_id: (r.number, r.basis) for r in res}
    assert got == {1: (8, "fragment"), 2: (10, "fragment")}

    assert merge_intervals([(0, 10), (5, 20), (30, 35)]) == 25
    assert parse_survivor_name("h1_chunk_000_f10645_n08_c0.799.jpg") == ("h1_chunk_000", 10645, 8,
                                                                         0.799)
    print("anchor_wire demo OK: margin gate + propagation guard + roster helpers")


if __name__ == "__main__":
    _demo()
