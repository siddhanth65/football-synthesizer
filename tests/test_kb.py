"""Targeted tests for tools.kb -- the claims-graph query CLI."""
from __future__ import annotations

from tools import kb


def test_load_claims_schema() -> None:
    """Every claim has the required fields and a resolvable, non-empty id."""
    claims = kb.load_claims()
    assert len(claims) > 50
    ids = [c["id"] for c in claims]
    assert len(ids) == len(set(ids)), "duplicate claim ids"
    required = {
        "id", "statement", "value", "unit", "status", "evidence", "method", "n",
        "caveats", "date", "supersedes", "superseded_by", "depends_on", "tags",
    }
    for c in claims:
        assert required.issubset(c.keys()), c["id"]
        assert c["status"] in {
            "confirmed", "retracted", "superseded", "pending", "refuted_externally", "refuted",
        }, c["id"]
        assert c["evidence"], f"{c['id']} has no evidence"


def test_retraction_names_its_replacement() -> None:
    """Every retracted/superseded claim points at a replacement, inline or via superseded_by.

    Most retractions are two-node (old claim -> superseded_by -> new claim). A few are
    single-node self-corrections (e.g. "X was flipped; corrected order is Y") where the fix is
    stated in the same claim's `statement`/`caveats` -- those are allowed to have an empty
    `superseded_by` as long as the word "correct" appears, so a bare unexplained retraction still
    fails this test.
    """
    claims = kb.load_claims()
    for c in claims:
        if c["status"] in {"retracted", "superseded"}:
            has_link = bool(c["superseded_by"])
            text = (c["statement"] + " ".join(c["caveats"])).lower()
            inline_fix = "correct" in text
            assert has_link or inline_fix, f"{c['id']} is {c['status']} but names no replacement"


def test_evidence_paths_exist() -> None:
    """No dead evidence links -- this is exactly what `kb check` reports."""
    claims = kb.load_claims()
    dead = kb._find_dead_evidence(claims)
    assert dead == [], dead


def test_search_is_case_insensitive_and_finds_known_claim(capsys) -> None:
    claims = kb.load_claims()
    kb.cmd_search(claims, "COUNTERPRESS")
    out = capsys.readouterr().out
    assert "w1-002" in out


def test_check_runs_without_error(capsys) -> None:
    claims = kb.load_claims()
    kb.cmd_check(claims)
    out = capsys.readouterr().out
    assert "Dead evidence links" in out
    assert "Candidate contradictions" in out


if __name__ == "__main__":
    # ponytail: smallest possible self-check, no pytest fixtures needed for this one.
    test_load_claims_schema()
    test_retraction_names_its_replacement()
    test_evidence_paths_exist()
    print("tests/test_kb.py: self-checks OK")
