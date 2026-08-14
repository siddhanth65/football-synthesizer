"""Metric-space TWiX: a learned tracklet-pair associator over pitch coordinates (campaign v9).

Provenance
----------
The architecture follows **TWiX** -- Miah, Bilodeau, Saunier, *Learning Data Association for
Multi-Object Tracking using Only Coordinates* (arXiv:2403.08018, Pattern Recognition 2024), public
implementation ``Guepardow/TWiX``, **MIT licence, (c) 2024 Mehdi Miah**. This module is a
re-implementation of that architecture (two-branch pair encoder with a temporal positional encoding
and an ``inter_pair`` attention stage over the candidate pairs of one window), not a verbatim copy
of their source; the MIT notice is carried because the design is theirs.

Three deviations, all on the metric-coordinate seam that is v9's novelty claim
(registered in ``results/GSR_V9_W1.md`` S5 and ``results/GSR_V9_W2.md`` S1):

1. **No ``NormCoords``.** TWiX rescales each window's boxes into ``[-1, 1]`` because image scale
   moves with the camera. Our positions are already in a fixed metric frame, where absolute scale is
   the signal (a 9 m/s speed bound, a 105x68 m pitch), so the normalisation is by **constants**
   (:data:`X_SCALE`, :data:`Y_SCALE`, :data:`V_SCALE`) and never by per-window extrema.
2. **Tokens are ``(x, y, vx, vy)`` in metres and m/s**, not ``(x1, y1, x2, y2)`` in pixels.
3. **Side features on the pair token** -- team/role agreement, jersey agreement + confidence mass,
   CLIP tracklet-mean cosine distance, temporal gap -- so the signals the incumbent GTA connector
   uses as *hard gates* enter this model as *features* instead (kb v9-w1-004: those gates forbid
   19.83% of all true merges).

The module is inference-and-training plumbing only: it owns no data loading and no metric. The
driver is :mod:`tools.gsr_v9_train`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

#: Version stamp of the architecture + feature layout (bump on any change either side of the seam).
ASSOC_VERSION = "v9-twixm-1.0"

#: Constant metric normalisers (pitch half-length, half-width, player speed ceiling). Deviation 1.
X_SCALE = 52.5
Y_SCALE = 34.0
V_SCALE = 9.0

#: Observation-token feature width: (x, y, vx, vy) + a past/future side flag.
OBS_DIM = 5
#: Pair-level side-feature width; see :func:`tools.gsr_v9_train.side_features` for the layout.
SIDE_DIM = 9


@dataclass(frozen=True)
class TwixConfig:
    """Registered architecture knobs.

    Attributes:
        d_model: Token width.
        n_head: Attention heads (both stages).
        d_ff: Feed-forward width.
        dropout: Dropout in both encoder layers.
        inter_pair: Keep TWiX's cross-pair attention stage (each pair scored in the context of the
            other candidate pairs of its window).
        use_side: Feed the pair-level side features. ``False`` is the registered ablation arm
            (kinematics only -- what metric motion alone buys).
        pe_min_s: Shortest period of the temporal positional encoding, in seconds.
        pe_max_s: Longest period of the temporal positional encoding, in seconds.
    """

    d_model: int = 64
    n_head: int = 4
    d_ff: int = 128
    dropout: float = 0.1
    inter_pair: bool = True
    use_side: bool = True
    pe_min_s: float = 0.2
    pe_max_s: float = 60.0


def time_encoding(dt_s: Tensor, d_model: int, pe_min_s: float, pe_max_s: float) -> Tensor:
    """Sin/cos encoding of a **signed** time offset in seconds (pure).

    TWiX references time to the start of the future tracklet, so offsets are negative for the past
    branch and positive for the future one; the encoding is therefore signed rather than the usual
    index-based one.

    Args:
        dt_s: ``(..., T)`` offsets in seconds.
        d_model: Output width (must be even).
        pe_min_s: Shortest period in seconds.
        pe_max_s: Longest period in seconds.

    Returns:
        ``(..., T, d_model)`` encoding.
    """
    half = d_model // 2
    k = torch.arange(half, device=dt_s.device, dtype=dt_s.dtype) / max(half - 1, 1)
    periods = pe_min_s * (pe_max_s / pe_min_s) ** k
    ang = dt_s.unsqueeze(-1) * (2.0 * math.pi / periods)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class TwixMetric(nn.Module):
    """TWiX in metric space: score every candidate tracklet pair of a window.

    Stage 1 (intra-pair) runs a transformer encoder over ``[CLS] + past observations + future
    observations`` of one candidate pair. Stage 2 (inter-pair, TWiX's ``inter_pair=True``) attends
    across the CLS tokens of all candidate pairs in the same window, so a pair is scored against its
    competitors. A 2-layer head turns each pair token into a merge logit.
    """

    def __init__(self, cfg: TwixConfig = TwixConfig()) -> None:
        """Build the two encoder stages and the scoring head.

        Args:
            cfg: Registered architecture knobs.
        """
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.obs_proj = nn.Linear(OBS_DIM, d)
        self.side_proj = nn.Linear(SIDE_DIM, d)
        self.cls = nn.Parameter(torch.zeros(d))
        layer_kw = dict(d_model=d, nhead=cfg.n_head, dim_feedforward=cfg.d_ff,
                        dropout=cfg.dropout, batch_first=True, norm_first=True)
        self.intra = nn.TransformerEncoderLayer(**layer_kw)
        self.inter = nn.TransformerEncoderLayer(**layer_kw) if cfg.inter_pair else None
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, obs: Tensor, dt_s: Tensor, obs_mask: Tensor, side: Tensor,
                pair_mask: Tensor) -> Tensor:
        """Merge logits for every candidate pair.

        Args:
            obs: ``(G, P, T, OBS_DIM)`` observation tokens (already constant-normalised).
            dt_s: ``(G, P, T)`` signed time offsets in seconds, referenced to the future tracklet's
                first frame.
            obs_mask: ``(G, P, T)`` bool, True where the observation is real (not padding).
            side: ``(G, P, SIDE_DIM)`` pair-level side features.
            pair_mask: ``(G, P)`` bool, True where the pair slot is a real candidate.

        Returns:
            ``(G, P)`` logits. Padded slots hold arbitrary values; callers mask with ``pair_mask``.
        """
        g, p, t, _ = obs.shape
        d = self.cfg.d_model
        tok = self.obs_proj(obs) + time_encoding(dt_s, d, self.cfg.pe_min_s, self.cfg.pe_max_s)
        cls = self.cls.expand(g, p, 1, d)
        if self.cfg.use_side:
            cls = cls + self.side_proj(side).unsqueeze(2)
        x = torch.cat([cls, tok], dim=2).reshape(g * p, t + 1, d)
        pad = torch.cat([torch.ones(g, p, 1, dtype=torch.bool, device=obs.device), obs_mask], 2)
        # An all-padding row would make softmax NaN; the CLS column is always real, so this is safe.
        x = self.intra(x, src_key_padding_mask=~pad.reshape(g * p, t + 1))
        z = x[:, 0].reshape(g, p, d)
        if self.inter is not None:
            z = self.inter(z, src_key_padding_mask=~pair_mask)
        return self.head(z).squeeze(-1)


def _demo() -> None:
    """Self-check: shapes, masking invariance and the metric-scale property (asserts; runnable)."""
    torch.manual_seed(0)
    g, p, t = 2, 5, 9
    obs = torch.randn(g, p, t, OBS_DIM)
    dt = torch.linspace(-3.0, 1.0, t).expand(g, p, t).contiguous()
    obs_mask = torch.ones(g, p, t, dtype=torch.bool)
    obs_mask[:, :, -2:] = False
    side = torch.randn(g, p, SIDE_DIM)
    pair_mask = torch.ones(g, p, dtype=torch.bool)
    pair_mask[1, 3:] = False

    model = TwixMetric().eval()
    with torch.no_grad():
        out = model(obs, dt, obs_mask, side, pair_mask)
    assert out.shape == (g, p), out.shape

    # Padded observation slots must not change a real pair's score.
    obs2 = obs.clone()
    obs2[:, :, -2:] = 99.0
    with torch.no_grad():
        out2 = model(obs2, dt, obs_mask, side, pair_mask)
    assert torch.allclose(out, out2, atol=1e-5), (out - out2).abs().max()

    # Padded PAIR slots must not change a real pair's score (inter-pair masking).
    obs3 = obs.clone()
    obs3[1, 3:] = 5.0
    with torch.no_grad():
        out3 = model(obs3, dt, obs_mask, side, pair_mask)
    assert torch.allclose(out[pair_mask], out3[pair_mask], atol=1e-5)

    # Deviation 1: no per-window rescaling, so a uniform scale change MUST move the scores.
    with torch.no_grad():
        out4 = model(obs * 2.0, dt, obs_mask, side, pair_mask)
    assert not torch.allclose(out[pair_mask], out4[pair_mask], atol=1e-3), "scale is being ignored"

    # The ablation arm must ignore the side features entirely.
    abl = TwixMetric(TwixConfig(use_side=False)).eval()
    with torch.no_grad():
        a1 = abl(obs, dt, obs_mask, side, pair_mask)
        a2 = abl(obs, dt, obs_mask, side * 7.0, pair_mask)
    assert torch.allclose(a1, a2, atol=1e-6)

    n = sum(x.numel() for x in model.parameters())
    print(f"assoc_twix demo OK: {ASSOC_VERSION}, {n} params, masks clean, metric scale preserved")


if __name__ == "__main__":
    _demo()
