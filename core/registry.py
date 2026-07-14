"""Central match registry: one YAML (``data/matches.yaml``) that every consumer reads.

Kills the P0 audit blocker (#1): the match list / paths were previously hand-duplicated across
``synthesizer/opponent_model.py``, ``report/pundit.py`` and three tools with two incompatible path
conventions. Adding a match is now one YAML entry; consumers iterate :func:`matches` or call
:func:`get` and never hardcode a path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

REGISTRY_PATH = Path("data/matches.yaml")
VIDEO_ROOT = Path("matches")   # chunk videos: matches/<match_id>/<half>/chunk_<NNN>.mp4
_BALL_RE = re.compile(r"ball_(h\d)_chunk(\d+)\.parquet$")
_CHUNK_KEY_RE = re.compile(r"(h\d)_chunk_(\d+)$")
_FPS_FALLBACK = 25.0
_fps_cache: dict[str, float] = {}


@dataclass(frozen=True)
class Match:
    """One registered match: identity + where its artifacts live.

    The optional club-football fields (``competition``, ``home_team``, ``away_team``) carry the
    metadata WC national-team matches never needed. They default to ``None`` so the four existing
    France/WC entries (which omit them) load unchanged; club entries (e.g. ``brighton_manutd``) set
    them so a PL fixture is self-describing without a France roster.
    """

    id: str
    teams: tuple[str, str]
    aligned: Path
    ball_dir: Path | None
    pmsr: Path | None
    roster: bool
    competition: str | None = None
    home_team: str | None = None
    away_team: str | None = None

    @property
    def processed(self) -> bool:
        """True once the anchored positions parquet exists."""
        return self.aligned.exists()

    def load_aligned(self):
        """Read the anchored per-frame positions table."""
        import pandas as pd  # noqa: PLC0415

        return pd.read_parquet(self.aligned)

    def ball_chunks(self) -> list[tuple[str, Path]]:
        """``(chunk_key, path)`` per linked-ball parquet; chunk_key matches ``aligned['chunk']``."""
        if self.ball_dir is None or not self.ball_dir.exists():
            return []
        out = []
        for p in sorted(self.ball_dir.glob("ball_*_chunk*.parquet")):
            m = _BALL_RE.search(p.name)
            if m:
                out.append((f"{m.group(1)}_chunk_{m.group(2)}", p))
        return out

    def chunk_fps(self, chunk_key: str, *, default: float = _FPS_FALLBACK) -> float:
        """Native container fps of a chunk's source video (``matches/<id>/<half>/chunk_<NNN>.mp4``).

        Broadcast chunks vary in frame rate (25 vs 59.94 fps), so every *time-based* metric must
        convert its seconds-valued parameters through the chunk's TRUE fps rather than a global
        constant. Result is cached per video path; ``default`` is returned when the video is absent
        or reports an implausible rate.

        Args:
            chunk_key: chunk identifier matching ``aligned['chunk']`` (e.g. ``h1_chunk_003``).
            default: fps to assume when the source video cannot be read.

        Returns:
            Frames per second of the chunk's source video.
        """
        mo = _CHUNK_KEY_RE.fullmatch(chunk_key)
        if not mo:
            return default
        video = VIDEO_ROOT / self.id / mo.group(1) / f"chunk_{mo.group(2)}.mp4"
        key = str(video)
        if key in _fps_cache:
            return _fps_cache[key]
        fps = default
        if video.exists():
            import cv2  # noqa: PLC0415

            cap = cv2.VideoCapture(key)
            v = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            if v and v > 1.0:
                fps = v
        _fps_cache[key] = fps
        return fps

    def load_pmsr(self) -> dict | None:
        """Parsed FIFA PMSR ground truth for this match, if it exists."""
        import json  # noqa: PLC0415

        if self.pmsr is None or not self.pmsr.exists():
            return None
        return json.loads(self.pmsr.read_text())


def _load(path: Path = REGISTRY_PATH) -> dict[str, Match]:
    raw = yaml.safe_load(path.read_text())
    out = {}
    for mid, m in raw["matches"].items():
        out[mid] = Match(
            id=mid, teams=tuple(m["teams"]), aligned=Path(m["aligned"]),
            ball_dir=Path(m["ball_dir"]) if m.get("ball_dir") else None,
            pmsr=Path(m["pmsr"]) if m.get("pmsr") else None,
            roster=bool(m.get("roster", False)),
            competition=m.get("competition"),
            home_team=m.get("home_team"),
            away_team=m.get("away_team"),
        )
    return out


def matches(*, processed_only: bool = False, path: Path = REGISTRY_PATH) -> list[Match]:
    """All registered matches (optionally only those with an aligned parquet on disk)."""
    ms = list(_load(path).values())
    return [m for m in ms if m.processed] if processed_only else ms


def get(match_id: str, *, path: Path = REGISTRY_PATH) -> Match:
    """One match by id (KeyError with the known ids if absent)."""
    reg = _load(path)
    if match_id not in reg:
        raise KeyError(f"unknown match {match_id!r}; registered: {sorted(reg)}")
    return reg[match_id]
