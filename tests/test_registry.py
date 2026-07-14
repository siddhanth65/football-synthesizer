"""Tests for the central match registry (core.registry) + constants module (core.pitch)."""

from __future__ import annotations

from pathlib import Path

import pytest

from core import registry
from core.pitch import CONTRACT_LEN, METRICS_VERSION, PITCH_LEN, PITCH_WID

YAML = """\
version: 1
matches:
  demo_a:
    teams: [Alpha, Beta]
    aligned: {aligned}
    ball_dir: {ball_dir}
    pmsr: null
    roster: false
  demo_b:
    teams: [Gamma, Delta]
    aligned: does/not/exist.parquet
    ball_dir: null
"""


@pytest.fixture()
def reg_yaml(tmp_path: Path) -> Path:
    aligned = tmp_path / "aligned.parquet"
    aligned.write_bytes(b"stub")
    ball = tmp_path / "ball"
    ball.mkdir()
    (ball / "ball_h1_chunk000.parquet").write_bytes(b"stub")
    (ball / "ball_h2_chunk003.parquet").write_bytes(b"stub")
    (ball / "not_a_ball_file.txt").write_bytes(b"stub")
    y = tmp_path / "matches.yaml"
    y.write_text(YAML.format(aligned=aligned.as_posix(), ball_dir=ball.as_posix()))
    return y


def test_registry_loads_and_flags_processed(reg_yaml: Path):
    ms = registry.matches(path=reg_yaml)
    assert [m.id for m in ms] == ["demo_a", "demo_b"]
    assert registry.get("demo_a", path=reg_yaml).processed
    assert not registry.get("demo_b", path=reg_yaml).processed
    assert [m.id for m in registry.matches(processed_only=True, path=reg_yaml)] == ["demo_a"]


def test_ball_chunk_keys_match_aligned_convention(reg_yaml: Path):
    m = registry.get("demo_a", path=reg_yaml)
    keys = [k for k, _ in m.ball_chunks()]
    assert keys == ["h1_chunk_000", "h2_chunk_003"]   # matches aligned['chunk'] values
    assert registry.get("demo_b", path=reg_yaml).ball_chunks() == []


def test_unknown_match_raises_with_known_ids(reg_yaml: Path):
    with pytest.raises(KeyError, match="demo_a"):
        registry.get("nope", path=reg_yaml)


def test_real_registry_parses_and_covers_all_processed_outputs():
    ms = registry.matches()
    assert {m.id for m in ms} >= {"france_iraq", "france_senegal", "france_norway", "mun_mci"}
    for m in ms:
        assert len(m.teams) == 2


def test_constants_single_source():
    assert (PITCH_LEN, PITCH_WID) == (105.0, 68.0)
    assert CONTRACT_LEN == 120.0
    assert METRICS_VERSION  # non-empty stamp
    # the metric modules must use THE shared constants, not local copies
    from fingerprint import structural_metrics
    from generator import contract
    assert structural_metrics.PITCH_LEN is PITCH_LEN
    assert contract.PITCH_LENGTH == CONTRACT_LEN
