"""Targeted checks for the adopted B4 abstention policy (``synthesizer.imputation_v1.emit``).

The contract every downstream consumer relies on (``results/B4_ABSTENTION_POLICY.md``): each
emitted position carries a ``source`` of ``"v1" | "anchor" | "abstained"``, the coordinate always
matches that source, and an abstained row is the last-seen position with ``asserted=False`` -- a
silent fallback is the failure mode this test exists to catch.
"""

from __future__ import annotations

import numpy as np

from synthesizer.imputation_v1 import emit


def _fixture() -> dict[str, np.ndarray]:
    """Six samples: one per horizon bucket, with a deliberately wide region in bucket 5."""
    mu = np.arange(12, dtype=float).reshape(6, 2)
    return {
        "samples": {"hold": np.full((6, 2), -1.0)},
        "mu": mu,
        "anchor": mu + 100.0,
        "r90": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 99.0]),
        "bucket": np.arange(6),
    }


def test_source_labels_and_positions_agree() -> None:
    """Every coordinate comes from the estimator its ``source`` names."""
    f = _fixture()
    skill = [1, 2, 3, 4, 5]  # v1 has no edge in bucket 0 -> defer to the anchor there
    e = emit(f["samples"], f["mu"], f["anchor"], f["r90"], f["bucket"], skill, r_max=10.0)
    assert list(e["source"]) == ["anchor", "v1", "v1", "v1", "v1", "abstained"]
    assert e["asserted"].tolist() == [True, True, True, True, True, False]
    v1 = e["source"] == "v1"
    anc = e["source"] == "anchor"
    off = e["source"] == "abstained"
    assert np.array_equal(e["pos"][v1], f["mu"][v1])
    assert np.array_equal(e["pos"][anc], f["anchor"][anc])
    assert np.array_equal(e["pos"][off], f["samples"]["hold"][off])
    assert not e["asserted"][off].any()


def test_no_silent_fallback_and_degenerate_cases() -> None:
    """Abstained rows never carry an estimate; empty skill set defers, it does not abstain."""
    f = _fixture()
    # Layer B rejects everything -> all abstained, all last-seen, nothing asserted.
    e = emit(f["samples"], f["mu"], f["anchor"], f["r90"], f["bucket"], [0, 1, 2], r_max=0.0)
    assert (e["source"] == "abstained").all()
    assert not e["asserted"].any()
    assert np.array_equal(e["pos"], f["samples"]["hold"])
    # No bucket has skill -> the anchor speaks everywhere layer B accepts; still no v1 label.
    e = emit(f["samples"], f["mu"], f["anchor"], f["r90"], f["bucket"], [], r_max=10.0)
    assert list(e["source"]) == ["anchor"] * 5 + ["abstained"]
    assert (e["source"] != "v1").all()
    # Sources partition the rows exactly; asserted == not abstained.
    assert set(np.unique(e["source"])) <= {"v1", "anchor", "abstained"}
    assert np.array_equal(e["asserted"], e["source"] != "abstained")


def test_emitted_region_belongs_to_the_emitted_point() -> None:
    """``r90`` follows the source: anchor rows carry the anchor's own calibrated region."""
    f = _fixture()
    r90_anchor = f["r90"] * 3.0  # the anchor's conformal multipliers differ from v1's
    e = emit(
        f["samples"], f["mu"], f["anchor"], f["r90"], f["bucket"], [1, 2, 3, 4, 5], r_max=10.0,
        r90_anchor=r90_anchor,
    )
    assert e["r90"][0] == r90_anchor[0]  # bucket 0 defers -> anchor region
    assert np.array_equal(e["r90"][1:], f["r90"][1:])  # v1 buckets keep v1's region
    # Layer B tests the EMITTED region: bucket 0's anchor region (3 m) is fine, but widen it
    # past r_max and that row must abstain even though v1's own region is tight.
    wide = r90_anchor.copy()
    wide[0] = 50.0
    e = emit(
        f["samples"], f["mu"], f["anchor"], f["r90"], f["bucket"], [1, 2, 3, 4, 5], r_max=10.0,
        r90_anchor=wide,
    )
    assert e["source"][0] == "abstained"
    assert np.array_equal(e["pos"][0], f["samples"]["hold"][0])
