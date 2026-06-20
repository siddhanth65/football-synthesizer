"""Tests for the SoP bridge's interpreter-resolution / auto-re-exec logic (pure; no torch needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from generator import sop_bridge


def test_has_modelling_runtime_is_bool_and_consistent():
    import importlib.util

    assert sop_bridge.has_modelling_runtime() is (
        importlib.util.find_spec("torch_geometric") is not None
    )


def test_sop_python_honours_env_override(monkeypatch, tmp_path):
    fake = tmp_path / "python.exe"
    fake.write_text("")
    monkeypatch.setenv(sop_bridge.SOP_PYTHON_ENV, str(fake))
    assert sop_bridge.sop_python() == fake


def test_sop_python_override_missing_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv(sop_bridge.SOP_PYTHON_ENV, str(tmp_path / "nope.exe"))
    assert sop_bridge.sop_python() is None


def test_sop_python_finds_sibling_venv(monkeypatch):
    """With no override, it should locate the SoP .venv interpreter if the sibling repo is present."""
    monkeypatch.delenv(sop_bridge.SOP_PYTHON_ENV, raising=False)
    try:
        sop_bridge.sop_root()
    except FileNotFoundError:
        pytest.skip("football-state-of-play sibling not present")
    py = sop_bridge.sop_python()
    if py is None:
        pytest.skip("SoP repo present but no .venv interpreter")
    assert py.exists() and py.name.startswith("python")


def test_reexec_is_noop_when_already_reexeced(monkeypatch):
    """The guard flag must short-circuit re-exec so we never loop or spawn during tests."""
    monkeypatch.setenv(sop_bridge._REEXEC_FLAG, "1")
    called = False

    def _boom(*a, **k):  # pragma: no cover - must not run
        nonlocal called
        called = True

    monkeypatch.setattr(sop_bridge.subprocess, "run", _boom)
    assert sop_bridge.reexec_under_sop_if_needed([]) is None
    assert called is False


def test_reexec_is_noop_when_runtime_present(monkeypatch):
    monkeypatch.delenv(sop_bridge._REEXEC_FLAG, raising=False)
    monkeypatch.setattr(sop_bridge, "has_modelling_runtime", lambda: True)
    monkeypatch.setattr(
        sop_bridge.subprocess, "run", lambda *a, **k: pytest.fail("should not re-exec")
    )
    assert sop_bridge.reexec_under_sop_if_needed([]) is None


def test_reexec_raises_without_any_modelling_interpreter(monkeypatch):
    monkeypatch.delenv(sop_bridge._REEXEC_FLAG, raising=False)
    monkeypatch.setattr(sop_bridge, "has_modelling_runtime", lambda: False)
    monkeypatch.setattr(sop_bridge, "sop_python", lambda: None)
    with pytest.raises(RuntimeError, match="torch-geometric"):
        sop_bridge.reexec_under_sop_if_needed([])


def test_checkpoint_path_under_sop_root(monkeypatch, tmp_path):
    monkeypatch.setenv(sop_bridge.SOP_PATH_ENV, str(tmp_path))
    assert sop_bridge.checkpoint_path() == Path(tmp_path) / "results" / "checkpoints" / "gnn.pt"
