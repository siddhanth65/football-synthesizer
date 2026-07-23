"""Runner: build the Phase-A Man Utd player profiles + ``results/PLAYER_ANALYSIS_v1.md`` (CPU).

Thin wrapper over :func:`fingerprint.player_profiles.main` so the analysis has a stable entry point
under ``tools/`` alongside the other validators.

Run::

    python -m tools.build_player_profiles
"""
from __future__ import annotations

from fingerprint import player_profiles


def main() -> None:
    """Delegate to the module builder."""
    player_profiles.main()


if __name__ == "__main__":
    main()
