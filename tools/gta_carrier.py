"""Re-run the carrier-attribution factorisation with the GTA-Link connector applied.

``results/CARRIER_CONSTRAINED_v2.md`` factorises end-to-end carrier precision into

    gate-hit 0.846  x  team/candidate 0.719  x  naming 0.609

and names the middle and right terms as gallery/appearance problems. The GTA connector
(:mod:`generator.gta_link`, splitter OFF -- see ``results/GTA_LINK_STAGE1.md``) attacks both by
merging tracklets: a merge group inherits its members' OCR name, so one named fragment names every
fragment merged with it. That grows the per-player gallery (more crops, more angles) and can cover
players who had no named fragment at all.

**Leakage control (the thing that would fake a win).** ``carrier_attribution_probe`` scores a query
against the gallery leave-one-track-out, dropping gallery crops from the query's own
``(chunk, track_id)``. After merging, "own track" means the whole merge group, so this module writes
both the events table and the gallery with **merged** track ids. Without that, a sibling fragment of
the same player in the same chunk re-enters the gallery and the naming factor rises for free.

No new GPU work: gallery crops are drawn from the per-detection PRTreID cache built by
``tools/gta_match.py --build-cache`` (identical crop geometry -- ``estimate_player_box`` -- identical
embedder -- ``PrtreidEmbedder`` -- and the same 24 px height floor), and the carrier *query*
embeddings are unchanged by a re-partition, so they are reused verbatim.

Nothing is tuned here. The connector settings are frozen from the SoccerNet-GSR split and the
operating point is the v1 frozen threshold, so this is a pure test pass.

CLI::

    python -m tools.gta_carrier --taus 0.04,0.05,0.06
    python -m tools.gta_carrier --taus 0.04 --json-out results/gta_carrier.json
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import tools.carrier_attribution_probe as probe
import tools.carrier_constrained as cc
from core import registry
from eval.gsr_prtreid_relink import propagation_fill
from generator.gta_link import GTA_VERSION, GtaParams, connect, mean_embeddings
from generator.track_relink import summarize_fragments
from tools.carrier_attribution_probe import GALLERY_PER_TRACK, PROBE_MATCHES
from tools.gta_match import NAMED_TRACKS, cache_dir, load_chunk_cache
from tools.score_carrier_labels import DEFAULT_LABELS, load_labels, norm_name, score_labels

logger = logging.getLogger("gta_carrier")

#: v1 frozen operating point -- the point at which the on-record 0.846/0.719/0.609 was measured.
V1_MIN_SIM, V1_MIN_MARGIN = 0.88, 0.01
#: Baseline configuration of the factorisation: every constraint OFF (the v1 reproduction).
BASELINE_CFG = {"subwindow": False, "exclusion": False, "kinematic": False, "w_territory": 0.0,
                "territory_floor": None, "w_hubness": 0.0, "joint": False}


# === connector + name propagation ================================================================
def merge_match(match_id: str, params: GtaParams) -> tuple[dict[tuple[str, int], int], dict]:
    """Connect each chunk's tracklets -> ``({(chunk, track_id): merged_id}, stats)`` (splitter OFF).

    Merging is within-chunk only: the cross-chunk variant measured 71.4% named-merge precision
    against 82.8% within-chunk (``results/GTA_LINK_STAGE1.md``), below the 80% bar this analysis
    needs, so it is not used here.
    """
    df = pd.read_parquet(registry.get(match_id).aligned)
    people = df[df["role"] != "ball"]
    out: dict[tuple[str, int], int] = {}
    n_before = n_after = 0
    for chunk in sorted(people["chunk"].unique()):
        cache = cache_dir(match_id) / f"detemb_{chunk}.npz"
        if not cache.exists():
            raise FileNotFoundError(f"no per-detection cache for {match_id} {chunk}: {cache}")
        sub = people[people["chunk"] == chunk]
        det = load_chunk_cache(match_id, chunk)
        embs, counts = mean_embeddings(det)
        frags = summarize_fragments(sub)
        remap = connect(frags, embs, counts, params)
        n_before += len(frags)
        n_after += len({remap.get(f.track_id, f.track_id) for f in frags})
        for tid in sub["track_id"].astype(int).unique():
            out[(chunk, int(tid))] = int(remap.get(int(tid), int(tid)))
    return out, {"n_fragments_before": n_before, "n_fragments_after": n_after,
                 "n_merges": n_before - n_after}


def propagate_names(
    match_id: str, merged: dict[tuple[str, int], int],
    base: dict[tuple[str, int], str] | None = None,
) -> tuple[dict[tuple[str, int], str], dict]:
    """Spread each merge group's OCR name to its unnamed members -> ``{(chunk, tid): player}``.

    Uses :func:`eval.gsr_prtreid_relink.propagation_fill` verbatim (names int-coded), i.e. the
    pre-committed rule already validated for jersey numbers: fill only previously-unnamed members,
    and on ANY disagreement inside a group leave the whole group untouched.

    Args:
        match_id: Registry match id.
        merged: ``{(chunk, track_id): merged_id}`` from :func:`merge_match`.
        base: Directly-read names to spread. Defaults to the close-up anchor chain; a caller with
            denser evidence (per-crop OCR) passes its own.
    """
    if base is None:
        named = pd.read_parquet(str(NAMED_TRACKS).format(match=match_id))
        base = {(str(c), int(t)): str(p)
                for c, t, p in zip(named["chunk"], named["track_id"], named["player_name"])}
    codes = {p: i + 1 for i, p in enumerate(sorted(set(base.values())))}
    inv = {v: k for k, v in codes.items()}

    out: dict[tuple[str, int], str] = dict(base)
    n_filled = n_disagree = 0
    by_chunk: dict[str, list[tuple[int, int]]] = {}
    for (chunk, tid), mid in merged.items():
        by_chunk.setdefault(chunk, []).append((tid, mid))
    for chunk, pairs in by_chunk.items():
        remap = {tid: mid for tid, mid in pairs}
        votes = {tid: (codes.get(base.get((chunk, tid), ""), -1), 1.0) for tid in remap}
        fill, st = propagation_fill(remap, votes)
        n_disagree += st["n_groups_disagree"]
        for tid, code in fill.items():
            out[(chunk, int(tid))] = inv[code]
            n_filled += 1
    return out, {"n_named_before": len(base), "n_named_after": len(out),
                 "n_names_propagated": n_filled, "n_groups_name_disagree": n_disagree}


# === arm inputs (gallery from the cached per-detection embeddings; no new GPU) ====================
def build_gallery(
    match_id: str, merged: dict[tuple[str, int], int], names: dict[tuple[str, int], str],
    club: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Gallery rows + embeddings for the merged, name-propagated tracklets (pure w.r.t. the cache).

    ``track_id`` in the returned rows is the **merged** id, so leave-one-track-out in
    :func:`tools.carrier_attribution_probe.identify` excludes the query's whole merge group.
    Up to :data:`tools.carrier_attribution_probe.GALLERY_PER_TRACK` crops are taken per merged
    tracklet, spread evenly over the frames the cache holds for it.
    """
    match = registry.get(match_id)
    roster_team = {n: i for i, n in enumerate(match.teams)}
    df = pd.read_parquet(match.aligned)
    df = df[df["role"].isin(["player", "goalkeeper"])].dropna(subset=["image_x", "image_y"])
    pos = {(str(c), int(t), int(f)): (float(x), float(y)) for c, t, f, x, y in zip(
        df["chunk"], df["track_id"], df["frame"], df["image_x"], df["image_y"])}
    named_df = pd.read_parquet(str(NAMED_TRACKS).format(match=match_id))
    # ``club`` maps player -> club name; the default is the OCR-named set. A caller whose name set
    # is wider (the Stage-2 solver names the whole lineup) must pass its own map, or players absent
    # from the OCR set would silently fall into team 0.
    club = club or dict(zip(named_df["player_name"], named_df["team"]))

    rows: list[dict] = []
    vecs: list[np.ndarray] = []
    n_conflict = 0
    for chunk in sorted({c for c, _ in names}):
        cache = load_chunk_cache(match_id, chunk)
        # group the chunk's cached detections by MERGED id
        grouped: dict[int, list[tuple[int, np.ndarray, int]]] = {}
        for tid, (frames, embs) in cache.items():
            mid = merged.get((chunk, int(tid)))
            player = names.get((chunk, int(tid)))
            if mid is None or player is None:
                continue
            for f, e in zip(frames, embs):
                grouped.setdefault(mid, []).append((int(f), e, int(tid)))
        for mid, items in grouped.items():
            # propagation_fill leaves a disagreeing group untouched, so its members can still carry
            # two different names. Such a group is a proven bad merge -- drop it rather than pick.
            group_names = {names[(chunk, t)] for _f, _e, t in items}
            if len(group_names) != 1:
                n_conflict += 1
                continue
            player = group_names.pop()
            items.sort()
            # Budget scales with the number of ORIGINAL tracklets in the group. Holding it at a flat
            # GALLERY_PER_TRACK would shrink the gallery purely because merging means fewer tracks
            # (fulham: 902 -> 626 crops), confounding the comparison with a sampling artifact.
            budget = GALLERY_PER_TRACK * len({t for _f, _e, t in items})
            pick = np.unique(np.linspace(0, len(items) - 1,
                                         min(budget, len(items))).astype(int))
            for i in pick:
                frame, emb, src = items[int(i)]
                xy = pos.get((chunk, src, frame))
                if xy is None:
                    continue
                rows.append({"chunk": chunk, "frame": frame, "track_id": int(mid),
                             "src_track": int(src),
                             "team": roster_team.get(club.get(player), 0), "player": player,
                             "image_x": xy[0], "image_y": xy[1]})
                vecs.append(emb)
    gal = pd.DataFrame(rows)
    return gal, (np.stack(vecs) if vecs else np.zeros((0, 256), np.float32)), n_conflict


def candidate_table_merged(match_id: str, arm_dir: Path,
                           merged: dict[tuple[str, int], int]) -> pd.DataFrame:
    """:func:`tools.carrier_constrained.candidate_table` with merge-group leave-one-track-out.

    Identical to the shipped function except that the query's excluded track is its **merged** id,
    so every fragment of the query's own player is dropped from its gallery.
    """
    ev = pd.read_parquet(arm_dir / f"{match_id}_events.parquet")
    q = ev[ev["resolvable"]].reset_index(drop=True)
    gal = pd.read_parquet(arm_dir / f"{match_id}_gallery.parquet").reset_index(drop=True)
    qe = np.load(arm_dir / f"{match_id}_query_emb.npy")
    ge = np.load(arm_dir / f"{match_id}_gallery_emb.npy")
    ok_g = np.linalg.norm(ge, axis=1) > 0.5
    players = gal["player"].to_numpy()
    g_team, g_track, g_chunk = (gal["team"].to_numpy(), gal["track_id"].to_numpy(),
                                gal["chunk"].to_numpy())
    rows: list[dict] = []
    for i, r in enumerate(q.itertuples(index=False)):
        if np.linalg.norm(qe[i]) <= 0.5:
            continue
        own = merged.get((str(r.chunk), int(r.track_id)), int(r.track_id))
        mask = ok_g & (g_team == r.team) & ~((g_chunk == r.chunk) & (g_track == own))
        if not mask.any():
            continue
        sims = ge[mask] @ qe[i]
        best: dict[str, float] = {}
        n_crops: dict[str, int] = {}
        for p, s in zip(players[mask], sims):
            n_crops[p] = n_crops.get(p, 0) + 1
            if s > best.get(p, -2.0):
                best[p] = float(s)
        for p, s in best.items():
            rows.append({"qi": i, "player": p, "sim": s, "n_crops": n_crops[p]})
    return pd.DataFrame(rows)


def merged_anchors(match_id: str, names: dict[tuple[str, int], str]) -> pd.DataFrame:
    """:func:`tools.carrier_constrained.named_positions` over the propagated name set."""
    df = pd.read_parquet(registry.get(match_id).aligned)
    df = df[df["role"].isin(["player", "goalkeeper"])].dropna(subset=["pitch_x", "pitch_y"])
    df = df.copy()
    df["player"] = [names.get((str(c), int(t))) for c, t in zip(df["chunk"], df["track_id"])]
    df = df[df["player"].notna()].copy()
    df["half"] = df["chunk"].str[:2]
    return df[["chunk", "frame", "track_id", "player", "half", "pitch_x", "pitch_y"]].reset_index(
        drop=True)


def prepare_arm(match_id: str, arm_dir: Path, params: GtaParams,
                namer=None, club: dict[str, str] | None = None) -> dict:
    """Build every per-match input for one arm into ``arm_dir`` -> repair/propagation stats.

    ``namer`` replaces :func:`propagate_names` (same signature and return shape), which is how the
    Stage-2 solver arm reuses this whole pipeline unchanged; ``club`` is its player -> club map.
    """
    src = Path("results/carrier_attr")
    arm_dir.mkdir(parents=True, exist_ok=True)
    merged, mstat = merge_match(match_id, params)
    names, nstat = (namer or propagate_names)(match_id, merged)
    gal, ge, n_conflict = build_gallery(match_id, merged, names, club)
    gal.to_parquet(arm_dir / f"{match_id}_gallery.parquet", index=False)
    np.save(arm_dir / f"{match_id}_gallery_emb.npy", ge)
    for suffix in ("_events.parquet", "_query_emb.npy"):  # unchanged by a re-partition
        shutil.copyfile(src / f"{match_id}{suffix}", arm_dir / f"{match_id}{suffix}")
    cand = candidate_table_merged(match_id, arm_dir, merged)
    cand.to_parquet(arm_dir / f"{match_id}_candidates.parquet", index=False)
    cc.query_context(match_id).to_parquet(arm_dir / f"{match_id}_qctx.parquet", index=False)
    merged_anchors(match_id, names).to_parquet(arm_dir / f"{match_id}_anchors.parquet", index=False)
    base_gal = pd.read_parquet(src / f"{match_id}_gallery.parquet")
    return {**mstat, **nstat, "n_groups_name_conflict_dropped": n_conflict,
            "n_gallery_crops_before": int(len(base_gal)),
            "n_gallery_crops_after": int(len(gal)),
            "n_gallery_players_before": int(base_gal["player"].nunique()),
            "n_gallery_players_after": int(gal["player"].nunique()) if len(gal) else 0}


# === factorisation ===============================================================================
def factorise(preds: pd.DataFrame, preps: dict[str, dict]) -> dict:
    """The three factors + composite, reproducing ``results/CARRIER_CONSTRAINED_v2.md`` exactly.

    ``team`` is per-query candidate membership (is the true carrier in *this* query's surviving,
    team-restricted, LOTO-filtered candidate set?), which is what the on-record 0.719 measured --
    not the looser "name appears anywhere in the match gallery".
    """
    res = score_labels(load_labels(DEFAULT_LABELS),
                       cc.gate(preds, V1_MIN_SIM, V1_MIN_MARGIN), cc.gallery_sets())
    r = res["rows"]
    cand_sets: dict[tuple[str, str, int], set[str]] = {}
    for mid, p in preps.items():
        key = p["ctx"].set_index("qi")[["chunk", "frame"]]
        for qi, g in p["cand"].groupby("qi"):
            k = key.loc[qi]
            cand_sets[(mid, str(k["chunk"]), int(k["frame"]))] = {
                norm_name(x) for x in g["player"]}
    good = r[~r["wrong_box"]]
    a = good[good["assigned"]].copy()
    a["in_cand"] = [t in cand_sets.get((m, str(c), int(f)), set())
                    for t, m, c, f in zip(a["truth"], a["match"], a["chunk"], a["frame"])]
    cov = a[a["in_cand"]]
    per_match = {}
    for mid in PROBE_MATCHES:
        rm, am = r[r["match"] == mid], a[a["match"] == mid]
        cm = am[am["in_cand"]]
        per_match[mid] = {
            "gate_hit": _frac((~rm["wrong_box"]).sum(), len(rm)),
            "team": _frac(am["in_cand"].sum(), len(am)),
            "naming": _frac(cm["correct"].sum(), len(cm)),
            "n_judged": int(len(rm)), "n_assigned_good_box": int(len(am)),
        }
    return {
        "gate_hit": _frac((~r["wrong_box"]).sum(), len(r)),
        "team": _frac(a["in_cand"].sum(), len(a)),
        "naming": _frac(cov["correct"].sum(), len(cov)),
        "n_judged": int(len(r)), "n_assigned_good_box": int(len(a)), "n_in_cand": int(len(cov)),
        "identification_precision": res["headline"]["identification_precision"],
        "end_to_end_precision": res["headline"]["end_to_end_precision"],
        "n_assigned_all": res["headline"]["n_assigned_all"],
        "abstain_rate": res["abstention"]["abstain_rate"],
        "per_match": per_match,
    }


def _frac(num, den) -> dict:
    """``{"n": num, "d": den, "p": num/den}`` (nan when empty) -- keeps every rate auditable."""
    return {"n": int(num), "d": int(den), "p": (float(num) / den if den else float("nan"))}


def run_arm(taus: list[float], out_root: Path) -> dict:
    """Score the baseline and every tau arm; returns the full payload."""
    src = Path("results/carrier_attr")
    arms: dict[str, dict] = {}

    preps = {m: cc.prepare(m) for m in PROBE_MATCHES}
    preds = pd.concat([cc.run_config(preps[m], BASELINE_CFG) for m in PROBE_MATCHES],
                      ignore_index=True)
    arms["baseline"] = {"factors": factorise(preds, preps), "repair": {}}
    logger.info("baseline: gate %.3f team %.3f naming %.3f",
                arms["baseline"]["factors"]["gate_hit"]["p"],
                arms["baseline"]["factors"]["team"]["p"],
                arms["baseline"]["factors"]["naming"]["p"])

    for tau in taus:
        params = GtaParams(tau=tau)
        arm_dir = out_root / f"tau{tau:.3f}"
        repair = {m: prepare_arm(m, arm_dir, params) for m in PROBE_MATCHES}
        probe.OUT_DIR, cc.OUT_DIR = arm_dir, arm_dir     # test-harness redirection only
        try:
            preps = {m: cc.prepare(m) for m in PROBE_MATCHES}
            preds = pd.concat([cc.run_config(preps[m], BASELINE_CFG) for m in PROBE_MATCHES],
                              ignore_index=True)
            arms[f"tau{tau:.3f}"] = {"factors": factorise(preds, preps), "repair": repair}
        finally:
            probe.OUT_DIR, cc.OUT_DIR = src, src
        f = arms[f"tau{tau:.3f}"]["factors"]
        logger.info("tau=%.3f: gate %.3f team %.3f naming %.3f (e2e %s)",
                    tau, f["gate_hit"]["p"], f["team"]["p"], f["naming"]["p"],
                    f["end_to_end_precision"])
    return {"version": GTA_VERSION, "operating_point": [V1_MIN_SIM, V1_MIN_MARGIN],
            "config": BASELINE_CFG, "arms": arms}


def _print(payload: dict) -> None:
    """ASCII table of every arm (cp1252-safe)."""
    print(f"\n{'arm':<12} {'gate-hit':>14} {'team':>14} {'naming':>14} {'ident':>7} "
          f"{'e2e':>7} {'nAsg':>5}")
    for name, arm in payload["arms"].items():
        f = arm["factors"]
        def cell(k: str) -> str:
            v = f[k]
            return f"{v['n']}/{v['d']}={v['p']:.3f}"
        print(f"{name:<12} {cell('gate_hit'):>14} {cell('team'):>14} {cell('naming'):>14} "
              f"{f['identification_precision']:>7} {f['end_to_end_precision']:>7} "
              f"{f['n_assigned_all']:>5}")
    print("\nrepair (per arm, summed over the 3 matches):")
    for name, arm in payload["arms"].items():
        if not arm["repair"]:
            continue
        r = arm["repair"]
        def tot(k: str) -> int:
            return sum(v[k] for v in r.values())
        print(f"  {name}: frags {tot('n_fragments_before')} -> {tot('n_fragments_after')} "
              f"({tot('n_merges')} merges); named tracks {tot('n_named_before')} -> "
              f"{tot('n_named_after')} (+{tot('n_names_propagated')} propagated, "
              f"{tot('n_groups_name_disagree')} groups vetoed); gallery crops "
              f"{tot('n_gallery_crops_before')} -> {tot('n_gallery_crops_after')}, players "
              f"{tot('n_gallery_players_before')} -> {tot('n_gallery_players_after')}")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taus", default="0.04,0.05,0.06")
    ap.add_argument("--out-root", type=Path, default=Path("outputs/gta_carrier"))
    ap.add_argument("--json-out", type=Path, default=Path("results/gta_carrier.json"))
    ap.add_argument("--loto", action="store_true",
                    help="high-n gallery leave-one-group-out proxy instead of the label scoring")
    ap.add_argument("--extra", action="append", default=[], metavar="NAME=DIR",
                    help="with --loto: extra arm gallery directory to score (repeatable)")
    ap.add_argument("--controls", default="tau0.000",
                    help="with --loto: comma-separated arm names to pair every arm against")
    args = ap.parse_args()
    if args.loto:
        extra = {s.split("=", 1)[0]: Path(s.split("=", 1)[1]) for s in args.extra}
        payload = run_loto([float(t) for t in args.taus.split(",")], args.out_root,
                           extra_dirs=extra, controls=tuple(args.controls.split(",")))
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
        return
    payload = run_arm([float(t) for t in args.taus.split(",")], args.out_root)
    _print(payload)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json_out}")



# === high-power proxy: gallery leave-one-group-out top-1 =========================================
def loto_query_hits(gallery_dir: Path, match_id: str,
                    ocr_named: dict[tuple[str, int], str]) -> dict[tuple, int]:
    """Per-query leave-one-group-out top-1 correctness, keyed ``(chunk, src_track, frame)`` (pure).

    The 23-moment label set cannot resolve a change in the naming factor. This proxy uses the same
    appearance decision on ~1000x more samples: every gallery crop is scored against the rest of the
    gallery of its own team, with its whole merge group removed, and counted correct when the
    nearest player is the crop's true one.

    **Non-circularity:** queries are restricted to crops whose SOURCE tracklet carried a direct OCR
    read, and the label they are graded against is that *anchor* read's player -- never the name the
    arm itself assigned. Grading against the arm's own gallery label would score self-consistency,
    which an arm that relabels a whole match (the Stage-2 solver) can win for free.

    Keyed on the SOURCE crop, so two arms can be compared on the identical set of query crops --
    the unpaired rates move partly because each arm's gallery sampling selects different crops.

    Args:
        gallery_dir: Directory holding ``<match>_gallery.parquet`` and ``<match>_gallery_emb.npy``.
        match_id: Registry match id.
        ocr_named: ``{(chunk, track_id): player}`` of tracklets named directly by OCR.

    Returns:
        ``{(chunk, src_track, frame): 0|1}`` over the eligible query crops.
    """
    gal = pd.read_parquet(gallery_dir / f"{match_id}_gallery.parquet").reset_index(drop=True)
    ge = np.load(gallery_dir / f"{match_id}_gallery_emb.npy")
    if not len(gal):
        return {}
    ok = np.linalg.norm(ge, axis=1) > 0.5
    players, team, grp = gal["player"].to_numpy(), gal["team"].to_numpy(), gal["track_id"].to_numpy()
    chunk, frame = gal["chunk"].to_numpy(), gal["frame"].to_numpy()
    src = gal["src_track"].to_numpy() if "src_track" in gal.columns else grp
    out: dict[tuple, int] = {}
    for i in range(len(gal)):
        truth = ocr_named.get((str(chunk[i]), int(src[i])))
        if not ok[i] or truth is None:
            continue
        mask = ok & (team == team[i]) & ~((chunk == chunk[i]) & (grp == grp[i]))
        if not mask.any():
            continue
        sims = ge[mask] @ ge[i]
        best: dict[str, float] = {}
        for p, s in zip(players[mask], sims):
            if s > best.get(p, -2.0):
                best[p] = float(s)
        out[(str(chunk[i]), int(src[i]), int(frame[i]))] = int(max(best, key=best.get) == truth)
    return out


def loto_top1(gallery_dir: Path, match_id: str, ocr_named: dict[tuple[str, int], str]) -> dict:
    """Unpaired aggregate of :func:`loto_query_hits` -> ``{"n", "correct", "p"}`` (pure)."""
    hits = loto_query_hits(gallery_dir, match_id, ocr_named)
    c = sum(hits.values())
    return {"n": len(hits), "correct": c, "p": c / len(hits) if hits else float("nan")}


def anchor_truth(match_id: str) -> dict[tuple[str, int], str]:
    """``{(chunk, track_id): player}`` for tracklets named directly by the OCR anchor chain."""
    nm = pd.read_parquet(str(NAMED_TRACKS).format(match=match_id))
    return {(str(c), int(t)): str(p)
            for c, t, p in zip(nm["chunk"], nm["track_id"], nm["player_name"])}


def mcnemar(ctrl: dict[tuple, int], arm: dict[tuple, int]) -> dict:
    """Exact paired McNemar of ``arm`` against ``ctrl`` on their shared query keys (pure)."""
    from scipy.stats import binomtest  # noqa: PLC0415

    keys = sorted(set(ctrl) & set(arm))
    n01 = sum(1 for k in keys if ctrl[k] == 1 and arm[k] == 0)
    n10 = sum(1 for k in keys if ctrl[k] == 0 and arm[k] == 1)
    return {"n_paired": len(keys),
            "control_p": sum(ctrl[k] for k in keys) / max(len(keys), 1),
            "arm_p": sum(arm[k] for k in keys) / max(len(keys), 1),
            "control_only_right": n01, "arm_only_right": n10,
            "mcnemar_p": binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0}


def run_loto(taus: list[float], out_root: Path, extra_dirs: dict[str, Path] | None = None,
             controls: tuple[str, ...] = ("tau0.000",)) -> dict:
    """Unpaired LOTO top-1 per arm + the paired McNemar of every arm against every control.

    Args:
        taus: Connector thresholds whose arm directories live under ``out_root``.
        out_root: Root holding ``tau<...>`` arm directories.
        extra_dirs: Further ``{arm name: gallery directory}`` to score (e.g. the solver arm).
        controls: Arm names to pair every other arm against.
    """
    ocr = {m: anchor_truth(m) for m in PROBE_MATCHES}
    dirs = {"baseline": Path("results/carrier_attr")}
    dirs.update({f"tau{t:.3f}": out_root / f"tau{t:.3f}" for t in taus})
    dirs.update(extra_dirs or {})
    unpaired, hits = {}, {}
    for name, d in dirs.items():
        hits[name] = {}
        for m in PROBE_MATCHES:
            hits[name].update({(m, *k): v for k, v in loto_query_hits(d, m, ocr[m]).items()})
        c, n = sum(hits[name].values()), len(hits[name])
        unpaired[name] = {"correct": c, "n": n, "p": c / n if n else float("nan")}
        logger.info("LOTO %-16s %d/%d = %.4f", name, c, n, unpaired[name]["p"])
    paired = {}
    for cname in controls:
        ctrl = hits.get(cname)
        if ctrl is None:
            continue
        for name, h in hits.items():
            if name == cname:
                continue
            paired[f"{name}_vs_{cname}"] = res = mcnemar(ctrl, h)
            logger.info("LOTO paired %-16s vs %-10s n=%d %.4f -> %.4f (McNemar p=%.3g)", name,
                        cname, res["n_paired"], res["control_p"], res["arm_p"], res["mcnemar_p"])
    return {"version": GTA_VERSION, "unpaired": unpaired, "paired": paired}

if __name__ == "__main__":
    main()
