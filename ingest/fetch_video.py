"""Fetch broadcast footage for the registered tournaments (yt-dlp wrapper).

Downloads clips/matches to ``data/video/<tournament_key>/``. Respect rights/usage; this is for
research on publicly available footage only.
"""

from __future__ import annotations

from pathlib import Path

VIDEO_DIR = Path("data/video")


def fetch(url: str, tournament_key: str, *, out_dir: Path = VIDEO_DIR) -> Path:
    """Download one video to ``out_dir/tournament_key/``; return the file path. TODO: call yt-dlp."""
    raise NotImplementedError("TODO: wrap yt-dlp download + naming")
