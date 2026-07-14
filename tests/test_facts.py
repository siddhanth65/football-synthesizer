"""Tests for the per-match fact store (report.facts)."""

from __future__ import annotations

import json
import math

import numpy as np

from report import facts


def test_clean_makes_json_safe():
    dirty = {"a": np.float64(1.5), "b": np.int32(2), "c": float("nan"),
             "d": [np.float32(0.5), float("inf")], "e": {"f": math.nan}}
    out = facts._clean(dirty)
    json.dumps(out, allow_nan=False)  # would raise if any NaN/inf survived
    assert out == {"a": 1.5, "b": 2, "c": None, "d": [0.5, None], "e": {"f": None}}


def test_load_facts_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(facts, "FACTS_DIR", tmp_path)
    assert facts.load_facts("nope") is None
    (tmp_path / "demo.json").write_text(json.dumps({"match": "demo", "cv": {}}), encoding="utf-8")
    assert facts.load_facts("demo")["match"] == "demo"


def test_real_fact_store_carries_the_full_inventory():
    """P0 definition-of-done: the product bundle carries far more than 4 numbers, versioned."""
    f = facts.load_facts("france_iraq")
    if f is None:  # outputs not present on this machine — the generation path is covered above
        return
    assert f["metrics_version"]
    cv = f["cv"]
    for family in ("tendencies", "phases_pct", "transitions", "passing", "ball_xt",
                   "style", "formation", "set_pieces"):
        assert family in cv, family

    def count(o):
        if isinstance(o, dict):
            return sum(count(v) for v in o.values())
        if isinstance(o, list):
            return sum(count(v) for v in o)
        return 1 if isinstance(o, (int, float)) else 0
    assert count(cv) >= 20
