"""Query the project's claims graph (``knowledge/claims.json``).

A claim is one measured/decided fact about the project, harvested from STATUS.md, docs/ and
results/ -- never invented. This module is a read-only query tool; it does not write claims.
See ``knowledge/README.md`` for how to add or retract one.

Commands::

    python -m tools.kb list [--tag TAG] [--status STATUS]
    python -m tools.kb search TEXT
    python -m tools.kb show ID
    python -m tools.kb check

``check`` reports two things for human review, never auto-resolves anything:
(a) claims whose evidence file no longer exists on disk;
(b) pairs of claims that look contradictory (same tag, overlapping statement text or the same
    unit, both status=confirmed, and either conflicting numeric values or no supersession link
    between them).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CLAIMS_PATH = REPO_ROOT / "knowledge" / "claims.json"

# Fraction of matching words (Jaccard over lowercased word sets) above which two statements are
# considered "overlapping" for the contradiction heuristic below.
STATEMENT_OVERLAP_THRESHOLD = 0.30
# Relative difference above which two same-unit numeric values are flagged as conflicting.
NUMERIC_CONFLICT_RATIO = 0.15


def load_claims() -> list[dict[str, Any]]:
    """Load the claims graph.

    Returns:
        The list of claim dicts from ``knowledge/claims.json``.
    """
    data = json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))
    return data["claims"]


def _evidence_path(entry: str) -> str:
    """Strip a ``#section`` anchor off an evidence string, returning the bare file path."""
    return entry.split("#", 1)[0].strip()


def cmd_list(claims: list[dict[str, Any]], tag: str | None, status: str | None) -> None:
    """Print claim id + one-line statement for every claim matching the given filters."""
    rows = claims
    if tag:
        rows = [c for c in rows if tag in c.get("tags", [])]
    if status:
        rows = [c for c in rows if c.get("status") == status]
    if not rows:
        print("No matching claims.")
        return
    for c in rows:
        print(f"[{c['id']:14s}] ({c['status']:10s}) {c['statement']}")
    print(f"\n{len(rows)} claim(s).")


def cmd_search(claims: list[dict[str, Any]], text: str) -> None:
    """Print every claim whose statement or caveats contain the search text (case-insensitive)."""
    needle = text.lower()
    hits = [
        c
        for c in claims
        if needle in c["statement"].lower() or any(needle in cv.lower() for cv in c.get("caveats", []))
    ]
    if not hits:
        print("No matches.")
        return
    for c in hits:
        print(f"[{c['id']:14s}] ({c['status']:10s}) {c['statement']}")
    print(f"\n{len(hits)} match(es).")


def cmd_show(claims: list[dict[str, Any]], claim_id: str) -> None:
    """Print the full record for one claim id."""
    for c in claims:
        if c["id"] == claim_id:
            print(json.dumps(c, indent=2, ensure_ascii=True))
            return
    print(f"No claim with id {claim_id!r}.")


def _find_dead_evidence(claims: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Return (claim_id, missing_path) pairs whose evidence file does not exist on disk."""
    dead = []
    for c in claims:
        for entry in c.get("evidence", []):
            if entry.startswith("http://") or entry.startswith("https://"):
                continue  # URLs are citable evidence; this checker only validates local paths.
            path = _evidence_path(entry)
            if path and not (REPO_ROOT / path).exists():
                dead.append((c["id"], path))
    return dead


def _word_set(statement: str) -> set[str]:
    return {w.strip(".,()-:") for w in statement.lower().split() if len(w) > 3}


def _find_contradictions(claims: list[dict[str, Any]]) -> list[tuple[dict, dict, str]]:
    """Return (claim_a, claim_b, reason) triples flagged for human review.

    Two independent heuristics, both restricted to pairs sharing a tag and both
    status == "confirmed" (anything already retracted/superseded is not a live contradiction):

    1. Same unit + numeric value that differs by more than NUMERIC_CONFLICT_RATIO, with no
       supersession link between the two ids.
    2. Statement text overlapping above STATEMENT_OVERLAP_THRESHOLD (Jaccard on word sets),
       with no supersession link between the two ids.
    """
    flagged = []
    confirmed = [c for c in claims if c.get("status") == "confirmed"]
    for i, a in enumerate(confirmed):
        for b in confirmed[i + 1 :]:
            shared_tags = set(a.get("tags", [])) & set(b.get("tags", []))
            if not shared_tags:
                continue
            linked = (
                b["id"] in a.get("supersedes", []) + a.get("superseded_by", []) + a.get("depends_on", [])
                or a["id"] in b.get("supersedes", []) + b.get("superseded_by", []) + b.get("depends_on", [])
            )
            if linked:
                continue

            # Heuristic 1: same unit, numeric values far apart.
            va, vb = a.get("value"), b.get("value")
            ua, ub = a.get("unit"), b.get("unit")
            if (
                isinstance(va, (int, float))
                and isinstance(vb, (int, float))
                and ua is not None
                and ua == ub
            ):
                denom = max(abs(va), abs(vb), 1e-9)
                if abs(va - vb) / denom > NUMERIC_CONFLICT_RATIO:
                    flagged.append((a, b, f"same unit ({ua}) values differ: {va} vs {vb}"))
                    continue

            # Heuristic 2: overlapping statement text.
            wa, wb = _word_set(a["statement"]), _word_set(b["statement"])
            if not wa or not wb:
                continue
            jaccard = len(wa & wb) / len(wa | wb)
            if jaccard > STATEMENT_OVERLAP_THRESHOLD:
                flagged.append((a, b, f"statement overlap {jaccard:.2f}"))
    return flagged


def cmd_check(claims: list[dict[str, Any]]) -> None:
    """Report dead evidence links and candidate contradictions. Never auto-resolves anything."""
    dead = _find_dead_evidence(claims)
    print(f"=== Dead evidence links ({len(dead)}) ===")
    for claim_id, path in dead:
        print(f"  {claim_id}: {path}")
    if not dead:
        print("  none")

    contradictions = _find_contradictions(claims)
    print(f"\n=== Candidate contradictions for human review ({len(contradictions)}) ===")
    for a, b, reason in contradictions:
        print(f"  {a['id']}  <->  {b['id']}   ({reason})")
        print(f"    A: {a['statement']}")
        print(f"    B: {b['statement']}")
    if not contradictions:
        print("  none")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="python -m tools.kb", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list claims, optionally filtered")
    p_list.add_argument("--tag")
    p_list.add_argument("--status")

    p_search = sub.add_parser("search", help="search claim statements + caveats")
    p_search.add_argument("text")

    p_show = sub.add_parser("show", help="print the full record for one claim id")
    p_show.add_argument("id")

    sub.add_parser("check", help="report dead evidence links + candidate contradictions")

    args = parser.parse_args(argv)
    claims = load_claims()

    if args.command == "list":
        cmd_list(claims, args.tag, args.status)
    elif args.command == "search":
        cmd_search(claims, args.text)
    elif args.command == "show":
        cmd_show(claims, args.id)
    elif args.command == "check":
        cmd_check(claims)


if __name__ == "__main__":
    main()
