"""Robust jersey-colour team assignment (replaces the SigLIP+UMAP classifier that collapsed).

``sports.common.team.TeamClassifier`` runs SigLIP -> UMAP -> KMeans. UMAP's ``transform`` on unseen
crops (a different camera region than the fit set) is unstable and routinely collapses both clusters
onto one team. For a two-team split the discriminative signal is simply the **jersey colour**, so this
classifier works directly on that:

1. For each player crop, take the **torso** region (avoid head, shorts, legs and surrounding grass).
2. **Mask out the pitch** (green) and very dark/See-through pixels, then summarise the remaining
   jersey pixels by their median **CIELAB chrominance** ``(a*, b*)`` plus lightness ``L*``. Chrominance
   is largely invariant to the broadcast brightness changes that break embedding-space clustering.
3. **KMeans(2)** on those per-crop colours. Deterministic (fixed seed), fast, and interpretable.

Pure: numpy + OpenCV + scikit-learn (all core deps); no torch, no UMAP, no network.
"""

from __future__ import annotations

import numpy as np

# Torso crop window as fractions of the player bounding box (top-left origin).
_TORSO_TOP, _TORSO_BOT = 0.18, 0.55
_TORSO_LEFT, _TORSO_RIGHT = 0.20, 0.80
# Grass mask in HSV (OpenCV H in [0, 179]): greenish, reasonably saturated and bright.
_GRASS_H_LO, _GRASS_H_HI = 35, 90
_GRASS_S_MIN, _GRASS_V_MIN = 30, 30
_MIN_JERSEY_PIXELS = 20  # below this after masking, fall back to the whole torso


def jersey_color(crop_bgr: np.ndarray) -> np.ndarray:
    """Summarise a player crop by the median CIELAB ``(L*, a*, b*)`` of its non-grass torso pixels.

    Args:
        crop_bgr: A player crop in BGR (OpenCV) order.

    Returns:
        Length-3 ``float32`` ``[L*, a*, b*]``. Returns mid-grey if the crop is empty.
    """
    import cv2  # noqa: PLC0415 - OpenCV is part of the [cv] stack

    if crop_bgr is None or crop_bgr.size == 0:
        return np.array([50.0, 0.0, 0.0], dtype=np.float32)
    h, w = crop_bgr.shape[:2]
    torso = crop_bgr[
        int(_TORSO_TOP * h):max(int(_TORSO_BOT * h), int(_TORSO_TOP * h) + 1),
        int(_TORSO_LEFT * w):max(int(_TORSO_RIGHT * w), int(_TORSO_LEFT * w) + 1),
    ]
    if torso.size == 0:
        torso = crop_bgr
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grass = (hh >= _GRASS_H_LO) & (hh <= _GRASS_H_HI) & (ss >= _GRASS_S_MIN) & (vv >= _GRASS_V_MIN)
    keep = ~grass & (vv >= 20)  # also drop near-black shadow pixels
    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    mask = keep.reshape(-1)
    pixels = lab[mask] if int(mask.sum()) >= _MIN_JERSEY_PIXELS else lab
    # OpenCV LAB is 0-255; rescale L to 0-100 and a*,b* to roughly -128..127 for KMeans balance.
    med = np.median(pixels, axis=0)
    return np.array([med[0] * 100.0 / 255.0, med[1] - 128.0, med[2] - 128.0], dtype=np.float32)


class JerseyColorTeamClassifier:
    """Two-team classifier from jersey colour: median LAB per crop -> KMeans(2). Deterministic.

    Drop-in for ``sports``'s ``TeamClassifier`` (``fit(crops)`` / ``predict(crops) -> labels``) but
    crop input is **BGR** (OpenCV native), matching the detector output in :mod:`generator.extract`.
    """

    def __init__(self, n_teams: int = 2, *, random_state: int = 0):
        from sklearn.cluster import KMeans  # noqa: PLC0415

        self.n_teams = n_teams
        self._km = KMeans(n_clusters=n_teams, n_init=10, random_state=random_state)
        self._fitted = False

    def _colors(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack([jersey_color(c) for c in crops])

    def fit(self, crops: list[np.ndarray]) -> JerseyColorTeamClassifier:
        """Fit the two team-colour clusters on a list of BGR player crops."""
        colors = self._colors(crops)
        if len(colors) < self.n_teams:
            raise ValueError(f"need >= {self.n_teams} crops to fit, got {len(colors)}")
        self._km.fit(colors)
        self._fitted = True
        return self

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        """Assign each BGR crop to a team (0..n_teams-1)."""
        if not self._fitted:
            raise RuntimeError("call fit() before predict()")
        colors = self._colors(crops)
        if len(colors) == 0:
            return np.array([], dtype=int)
        return self._km.predict(colors).astype(int)
