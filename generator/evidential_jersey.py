"""Dirichlet evidential jersey head: learned abstention inside the recognizer (campaign v7, V3).

The shipped chain gates crops with an external ResNet34 legibility classifier (19.1% of DEV-20 crops
pass, ``results/GSR_S3_READER.md`` §7.2) and then reads the survivors with PARSeq. Two consequences
were measured: the gate is the binding emit constraint, and the reader has no calibrated way to say
"there is no number here" -- the folded ``ILLEGIBLE`` mass of
:func:`generator.jersey_id.parseq_positions_to_probs` is whatever the STR decoder's end-token
happens to score.

This module replaces both with one head, in the Grad CVPRW2025 style (reimplemented -- no public
code): a **Dirichlet evidential** classifier over ``{no-number} u {1..99}`` sitting on a **frozen**
reader trunk (the GoMatching decoupling: train the head, not the trunk). Per crop it emits
concentration parameters ``alpha = softplus(logits) + 1``; the Dirichlet mean ``alpha / S`` is a
drop-in replacement for the folded softmax the rest of the chain already consumes, and the total
evidence ``S`` yields an explicit uncertainty ``u = K / S`` that no softmax can express. Abstention
is therefore learned *inside* the model on two orthogonal axes:

* **class 0** -- "no number is visible on this crop" (supervised at scale by the 56.8% of
  digit-labelled SoccerNet-v3 crops that FAIL the 0.7 legibility filter, ``CLUSTER_SESSION_S2.md``
  §3.2: per-crop-sound negative evidence);
* **u** -- "this crop supports no confident statement at all", high wherever total evidence is low.

Design guard (``results/JERSEY_HEAD_VOTER.md``): the 5B jersey head collapsed 0.863 -> 0.125 across a
player boundary because its label was a *property of the identity* (1,343 player-number pairs, so
"recognise the player, recall his number" fit the data). Here the trunk is frozen and already a digit
reader, the supervision is per-crop glyph/abstention only, and :func:`player_disjoint_split` exists
so the memorisation check is run by construction rather than remembered.

CLI::

    python -m generator.evidential_jersey        # the toy unit check (CPU, seconds)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

#: Class layout, identical to :data:`generator.jersey_id.NUM_CLASSES`: index 0 = no number visible,
#: indices 1..99 = the jersey number. Sharing the layout is what makes the head a drop-in for the
#: folded PARSeq distribution everywhere downstream (votes, bundles, solver).
NUM_CLASSES = 100
NO_NUMBER = 0


def evidence_to_alpha(logits: Tensor) -> Tensor:
    """Map head logits to Dirichlet concentration parameters (pure).

    Args:
        logits: ``[..., K]`` raw head outputs.

    Returns:
        ``alpha = softplus(logits) + 1``, elementwise ``>= 1``. Softplus (not ``exp``) keeps the
        evidence finite for large logits, which matters because the KL term below actively pushes
        evidence up on the correct class.
    """
    return torch.nn.functional.softplus(logits) + 1.0


def dirichlet_uncertainty(alpha: Tensor) -> Tensor:
    """Total-evidence uncertainty ``u = K / S`` in ``(0, 1]`` (pure).

    ``S = sum(alpha)`` is the Dirichlet strength; a crop the head has no evidence about has
    ``alpha = 1`` everywhere, ``S = K`` and ``u = 1``. This is the quantity a softmax cannot express:
    a uniform softmax and a confident-but-wrong softmax are indistinguishable, but their ``S`` are
    not.

    Args:
        alpha: ``[..., K]`` concentration parameters.

    Returns:
        ``[...]`` uncertainty.
    """
    k = alpha.shape[-1]
    return k / alpha.sum(-1)


def kl_dirichlet_uniform(alpha: Tensor) -> Tensor:
    """``KL(Dir(alpha) || Dir(1))`` per row (pure).

    Args:
        alpha: ``[..., K]`` concentration parameters.

    Returns:
        ``[...]`` non-negative divergence from the uniform Dirichlet.
    """
    k = alpha.shape[-1]
    s = alpha.sum(-1, keepdim=True)
    ln_b = torch.lgamma(s.squeeze(-1)) - torch.lgamma(alpha).sum(-1)
    ln_b_uni = -torch.lgamma(torch.tensor(float(k), device=alpha.device, dtype=alpha.dtype))
    dg = (alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(s))
    return ln_b + ln_b_uni + dg.sum(-1)


def edl_loss(alpha: Tensor, target: Tensor, lam: float = 0.0) -> Tensor:
    """Bayes-risk MSE evidential loss with the annealed uniform-Dirichlet regulariser (pure).

    The first two terms are the Bayes risk of the sum-of-squares loss under ``Dir(alpha)``: a fit
    term on the Dirichlet mean plus its variance, so a row can lower its loss either by being right
    or by *withholding evidence* -- which is what makes abstention learnable rather than
    post-hoc-thresholded. The KL term is applied to ``alpha`` with the ground-truth class's evidence
    removed, so it only penalises evidence assigned to WRONG classes and never punishes a confident
    correct read.

    Args:
        alpha: ``[N, K]`` concentration parameters.
        target: ``[N]`` int64 class indices.
        lam: Weight on the KL regulariser; anneal ``0 -> 1`` over the first epochs (Sensoy et al.
            2018 -- annealing from 0 is load-bearing, a full-strength KL from step 0 collapses every
            row onto ``alpha = 1``).

    Returns:
        Scalar mean loss.
    """
    y = torch.zeros_like(alpha).scatter_(1, target[:, None], 1.0)
    s = alpha.sum(-1, keepdim=True)
    p = alpha / s
    err = ((y - p) ** 2).sum(-1)
    var = (p * (1.0 - p) / (s + 1.0)).sum(-1)
    kl = kl_dirichlet_uniform(y + (1.0 - y) * alpha)
    return (err + var + lam * kl).mean()


class EvidentialHead(nn.Module):
    """Two-layer MLP emitting Dirichlet evidence over :data:`NUM_CLASSES` from frozen trunk features.

    Deliberately small: the trunk (arm-4t PARSeq) is frozen and already reads digits at 0.8344
    per-crop precision, so the head's job is to *re-express* that read as calibrated evidence and to
    add the no-number decision -- not to relearn OCR. A bigger head is also the exact capacity a
    memorisation shortcut would need.
    """

    def __init__(self, in_dim: int, hidden: int = 512, n_classes: int = NUM_CLASSES,
                 dropout: float = 0.1) -> None:
        """Build the head.

        Args:
            in_dim: Frozen-feature width (pooled encoder memory ++ decoder positional logits).
            hidden: Hidden width.
            n_classes: Output classes (index 0 = no number).
            dropout: Dropout on the hidden activation.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, feats: Tensor) -> Tensor:
        """Return ``alpha`` ``[N, n_classes]`` for a batch of frozen features."""
        return evidence_to_alpha(self.net(feats))


#: Width of the decoder block at the tail of a cached feature: 3 string positions x 11 tokens.
POS_DIMS = 33


def trunk_dist(feats: Tensor) -> Tensor:
    """Fold the frozen reader's OWN read out of the tail of its feature vector (pure).

    The last :data:`POS_DIMS` dims of a cached feature are exactly the positional softmaxes the
    sidecar emits, so the trunk's ``[N, 100]`` distribution is recoverable from the feature with no
    second forward pass. Torch port of
    :func:`generator.jersey_id.parseq_positions_to_probs`, asserted equal to it in :func:`_demo`.

    Args:
        feats: ``[N, D]`` cached features (``D >= POS_DIMS``).

    Returns:
        ``[N, 100]`` distribution in the standard layout (index 0 = illegible / not a 1..99 number).
    """
    pos = feats[:, -POS_DIMS:].reshape(-1, 3, 11)
    a0, a1 = pos[:, 0], pos[:, 1]
    d0 = a0[:, 2:11]
    return torch.cat([(a0[:, 0] + a0[:, 1])[:, None],
                      d0 * a1[:, :1],
                      (d0[:, :, None] * a1[:, None, 1:11]).reshape(len(feats), 90)], dim=1)


class EvidentialGateHead(nn.Module):
    """Evidential head that GATES the frozen reader instead of replacing it.

    Measured motivation (``results/GSR_V7_V3.md`` §2.2): a free 100-way MLP on pooled trunk features
    reads DEV-20 at 0.49 per-crop precision where the trunk it sits on reads 0.97 at the same emit.
    The AR decoder's digit composition is sharp and an MLP smooths it, so the free head can only
    lose on numbers -- which is the opposite of the GoMatching decoupling's point.

    This head therefore emits exactly two quantities per crop, **total evidence** ``e`` and the
    **no-number share** ``w``, and lays them over the trunk's own (renormalised) number ranking:

        ``alpha = 1 + e * [ w , (1 - w) * trunk_numbers ]``

    So the number argmax is the trunk's *by construction* (precision cannot regress), ``p_none = w``
    is a learned, calibrated abstention, and ``u = 100 / (100 + e)`` is the total-evidence
    uncertainty a softmax cannot express. Abstention is still learned inside the model -- it is just
    learned in the two dimensions the trunk does not already supply.
    """

    def __init__(self, in_dim: int, hidden: int = 256, n_classes: int = NUM_CLASSES,
                 dropout: float = 0.1, max_evidence: float = 500.0) -> None:
        """Build the gate.

        Args:
            in_dim: Frozen-feature width.
            hidden: Hidden width (small: two outputs).
            n_classes: Must be :data:`NUM_CLASSES`; kept for checkpoint-shape parity.
            dropout: Dropout on the hidden activation.
            max_evidence: Ceiling on total evidence, so ``u`` cannot be driven to 0 by a
                saturating logit (``u >= 100 / (100 + max_evidence)``).
        """
        super().__init__()
        if n_classes != NUM_CLASSES:
            raise ValueError("EvidentialGateHead is defined on the 100-class jersey layout")
        self.max_evidence = max_evidence
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, feats: Tensor) -> Tensor:
        """Return ``alpha`` ``[N, NUM_CLASSES]``."""
        raw = self.net(feats)
        e = self.max_evidence * torch.sigmoid(raw[:, :1])
        w = torch.sigmoid(raw[:, 1:2])
        num = trunk_dist(feats)[:, 1:]
        num = num / num.sum(1, keepdim=True).clamp_min(1e-9)
        return 1.0 + e * torch.cat([w, (1.0 - w) * num], dim=1)


def fuse_tracklet(alpha: np.ndarray, *, max_u: float = 1.01, max_p_none: float = 1.01,
                  min_conf: float = 0.0, min_crops: int = 1) -> list[tuple[int, float]]:
    """Uncertainty-filtered Bayesian fusion of one tracklet's per-crop Dirichlets.

    Dirichlet evidence is additive by construction -- the posterior after observing several
    independent crops of the same shirt is ``Dir(1 + sum_i e_i)`` -- so fusing a tracklet is a sum
    of the per-crop evidence ``alpha - 1``, not a vote. Crops the head is uncertain about
    (``u >= max_u``) or calls empty (``p_none >= max_p_none``) contribute nothing, which is the
    Grad design's point: the filter is a *model output*, not an external legibility classifier.

    This is the arm the Koshkina voting control (:func:`generator.jersey_id.percrop_votes` run on
    the same evidential rows) has to beat, per ``results/GSR_V7_V3.md``.

    Args:
        alpha: ``[n_crops, K]`` concentration parameters for one tracklet.
        max_u: Drop crops whose total-evidence uncertainty reaches this.
        max_p_none: Drop crops whose no-number mass reaches this.
        min_conf: Floor on the fused winner's posterior mass.
        min_crops: Minimum surviving crops before anything is emitted.

    Returns:
        ``[(number, fused_confidence)]`` -- a single-element list, or empty when the tracklet
        abstains. The shape matches :func:`generator.jersey_id.percrop_votes` so the downstream
        vote/bundle/solver path is unchanged.
    """
    if alpha.size == 0:
        return []
    s = alpha.sum(1)
    keep = (alpha.shape[1] / s < max_u) & (alpha[:, NO_NUMBER] / s < max_p_none)
    if int(keep.sum()) < min_crops:
        return []
    fused = (alpha[keep] - 1.0).sum(0) + 1.0
    p = fused / fused.sum()
    best = int(p[1:].argmax()) + 1
    if p[NO_NUMBER] >= p[best] or float(p[best]) < min_conf:
        return []
    return [(best, float(p[best]))]


def player_disjoint_split(player_keys: np.ndarray, holdout_frac: float = 0.10,
                          seed: int = 0) -> np.ndarray:
    """Boolean holdout mask splitting on PLAYER, not on row (pure).

    ``results/JERSEY_HEAD_VOTER.md`` §1: a row-level split hides an identity shortcut completely
    (0.863 train vs 0.125 across a player boundary, same builder, same dataset). Every train/holdout
    number this session reports is split with this function so a memorisation collapse is visible
    instead of averaged away.

    Args:
        player_keys: ``[N]`` per-row player identity key (e.g. ``"v3|<game>|<id>"``).
        holdout_frac: Fraction of DISTINCT players held out.
        seed: RNG seed.

    Returns:
        ``[N]`` bool, ``True`` where the row's player is in the holdout.
    """
    uniq = np.unique(player_keys)
    rng = np.random.default_rng(seed)
    n_hold = max(1, int(round(len(uniq) * holdout_frac)))
    held = set(rng.choice(uniq, size=n_hold, replace=False).tolist())
    return np.array([k in held for k in player_keys], dtype=bool)


def fit_head(feats: np.ndarray, labels: np.ndarray, *, in_dim: int | None = None,
             n_classes: int = NUM_CLASSES, epochs: int = 20, batch: int = 512, lr: float = 1e-3,
             anneal_epochs: int = 10, kl_max: float = 1.0, hidden: int = 512, device: str = "cpu",
             seed: int = 0, weights: np.ndarray | None = None, gate: bool = False,
             log: bool = False) -> tuple[nn.Module, list]:
    """Train an :class:`EvidentialHead` on cached frozen features.

    Args:
        feats: ``[N, D]`` float32 frozen features.
        labels: ``[N]`` int64 class indices (0 = no number).
        in_dim: Feature width; inferred from ``feats`` when ``None``.
        n_classes: Output classes (the toy check uses a small vocabulary; the chain uses 100).
        epochs: Training epochs.
        batch: Minibatch size.
        lr: AdamW learning rate.
        anneal_epochs: Epochs over which the KL weight goes ``0 -> kl_max``.
        kl_max: Ceiling on the KL weight. Below 1.0 this trades calibration for accuracy; at 1.0 on
            a low-margin corpus the regulariser can crush every row onto ``alpha = 1`` (measured on
            the toy: 0.001 accuracy at ``kl_max = 1``), so it is a swept knob, not a constant.
        hidden: Head hidden width.
        device: ``"cpu"`` / ``"cuda"``.
        seed: Torch seed.
        weights: Optional ``[N]`` per-row sampling weights (class balancing); ``None`` = uniform.
        gate: Train an :class:`EvidentialGateHead` (evidence + abstention over the frozen reader's
            own ranking) instead of a free :class:`EvidentialHead`.
        log: Print per-epoch loss.

    Returns:
        ``(head, curve)`` where ``curve`` is ``[(epoch, mean_loss, lam), ...]``.
    """
    torch.manual_seed(seed)
    x = torch.as_tensor(feats, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.int64)
    cls = EvidentialGateHead if gate else EvidentialHead
    head = cls(in_dim or x.shape[1], hidden=hidden, n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    n = len(x)
    w = None if weights is None else torch.as_tensor(weights, dtype=torch.float64)
    curve: list[tuple[int, float, float]] = []
    for ep in range(epochs):
        head.train()
        lam = kl_max * min(1.0, (ep + 1) / max(anneal_epochs, 1))
        perm = (torch.multinomial(w, n, replacement=True) if w is not None
                else torch.randperm(n))
        tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            xb, yb = x[idx].to(device), y[idx].to(device)
            loss = edl_loss(head(xb), yb, lam)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        curve.append((ep, tot / n, lam))
        if log:
            print(f"  epoch {ep:2d}  loss {tot / n:.5f}  lam {lam:.2f}", flush=True)
    head.eval()
    return head, curve


def save_head(head: nn.Module, path, in_dim: int, hidden: int, extra: dict | None = None) -> None:
    """Persist a head with everything :func:`load_head` needs to rebuild it."""
    torch.save({"state": head.state_dict(), "in_dim": in_dim, "hidden": hidden,
                "n_classes": NUM_CLASSES, "gate": isinstance(head, EvidentialGateHead),
                **(extra or {})}, path)


def load_head(path, device: str = "cpu") -> nn.Module:
    """Rebuild a trained head from its checkpoint (the ONE loader every caller uses).

    The checkpoint records which architecture it is (``gate``), because the sidecar, the training
    script and the analysis scripts all load the same file from three different environments and a
    per-caller guess is how a shape mismatch reaches a 20-sequence GPU pass.
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    cls = EvidentialGateHead if blob.get("gate") else EvidentialHead
    head = cls(blob["in_dim"], hidden=blob["hidden"], n_classes=blob["n_classes"]).to(device)
    head.load_state_dict(blob["state"])
    return head.eval()


@torch.no_grad()
def head_alpha(head: nn.Module, feats: np.ndarray, batch: int = 4096,
               device: str = "cpu") -> np.ndarray:
    """Batched inference: ``[N, D]`` features -> ``[N, K]`` alpha (float32)."""
    head.eval()
    out = []
    for s in range(0, len(feats), batch):
        xb = torch.as_tensor(feats[s : s + batch], dtype=torch.float32).to(device)
        out.append(head(xb).float().cpu().numpy())
    return (np.concatenate(out) if out
            else np.empty((0, NUM_CLASSES), np.float32))


def roc_auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based ROC AUC (pure; no sklearn dependency).

    Args:
        score: ``[N]`` higher = more positive.
        positive: ``[N]`` bool labels.

    Returns:
        AUC, or ``nan`` when either class is empty.
    """
    pos, neg = int(positive.sum()), int((~positive).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    # average ranks over ties so a constant score scores exactly 0.5
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[positive].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _toy(n: int = 3000, k: int = 6, dim: int = 16, seed: int = 0, amp: float = 1.5,
         noise: float = 1.0):  # noqa: ANN202
    """Toy corpus mirroring the real asymmetry: legible digits vs no-number crops (pure).

    ``k - 1`` "numbers" are fixed prototype directions (the glyphs, shared across splits); the
    no-number population carries **no** prototype at all, only noise -- an illegible crop is not a
    different glyph, it is the absence of one. Class 0 is therefore learnable (as "no direction
    dominates") but intrinsically evidence-poor, which is exactly the regime the total evidence
    ``S`` is supposed to separate and a softmax cannot.
    """
    proto = np.random.default_rng(1234).normal(size=(k, dim)).astype(np.float32)  # fixed "glyphs"
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, k, size=n)
    sig = np.where(lab == NO_NUMBER, 0.0, amp).astype(np.float32)[:, None]
    feats = proto[lab] * sig + rng.normal(scale=noise, size=(n, dim)).astype(np.float32)
    return feats.astype(np.float32), lab.astype(np.int64)


def _demo() -> None:
    """Unit check on a toy split: the loss falls, illegible rows carry more uncertainty, ROC is sane.

    Pre-declared as the gate that must pass BEFORE any A100 run (plan v7 session V3).
    """
    a = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    assert abs(float(kl_dirichlet_uniform(a))) < 1e-6, kl_dirichlet_uniform(a)
    assert abs(float(dirichlet_uncertainty(a)) - 1.0) < 1e-6
    assert float(kl_dirichlet_uniform(torch.tensor([[9.0, 1.0, 1.0, 1.0]]))) > 0.5
    assert float(dirichlet_uncertainty(torch.tensor([[9.0, 1.0, 1.0, 1.0]]))) < 0.4  # noqa: PLR2004

    k = 6
    tr_f, tr_y = _toy(seed=0)
    te_f, te_y = _toy(seed=1)
    head, curve = fit_head(tr_f, tr_y, n_classes=k, epochs=40, batch=256, lr=3e-3,
                           anneal_epochs=10, kl_max=1.0, hidden=64)
    assert curve[-1][1] < 0.5 * curve[0][1], curve  # (1) the evidential loss decreases

    alpha = head_alpha(head, te_f)
    u = dirichlet_uncertainty(torch.as_tensor(alpha)).numpy()
    p = alpha / alpha.sum(1, keepdims=True)
    illeg = te_y == NO_NUMBER
    assert u[illeg].mean() > u[~illeg].mean(), (u[illeg].mean(), u[~illeg].mean())  # (2)
    auc_u = roc_auc(u, illeg)
    auc_p = roc_auc(p[:, NO_NUMBER], illeg)
    assert auc_p > 0.80, auc_p  # (3) the abstention ROC is sane  # noqa: PLR2004
    assert auc_u > 0.70, auc_u  # noqa: PLR2004
    acc = float((p[~illeg].argmax(1) == te_y[~illeg]).mean())
    assert acc > 0.90, acc  # the head still reads  # noqa: PLR2004

    # fusion: evidence adds, the uncertainty filter bites, and an empty tracklet abstains
    strong = np.ones((3, k), np.float32)
    strong[:, 2] += 20.0
    assert fuse_tracklet(strong) == [(2, fuse_tracklet(strong)[0][1])]
    assert fuse_tracklet(strong)[0][1] > (strong[0, 2] / strong[0].sum())  # 3 crops beat 1
    assert fuse_tracklet(strong, max_u=k / strong.sum(1)[0] * 0.99) == []  # all crops filtered out
    assert fuse_tracklet(np.ones((2, k), np.float32)) == []  # no evidence -> no read
    assert fuse_tracklet(np.empty((0, k), np.float32)) == []

    # the torch fold of the trunk's own read matches the numpy one bit-for-bit in float32
    from generator.jersey_id import parseq_positions_to_probs  # noqa: PLC0415

    rng = np.random.default_rng(7)
    raw = rng.random((5, 3, 11)).astype(np.float32)
    pos = (raw / raw.sum(-1, keepdims=True)).astype(np.float32)
    f = np.concatenate([rng.random((5, 12)).astype(np.float32), pos.reshape(5, 33)], 1)
    got = trunk_dist(torch.as_tensor(f)).numpy()
    want = np.stack([parseq_positions_to_probs(r[0], r[1]) for r in pos])
    assert np.abs(got - want).max() < 1e-6, np.abs(got - want).max()

    # the gate head cannot regress the trunk's number ranking: its argmax IS the trunk's
    gh = EvidentialGateHead(f.shape[1], hidden=8)
    ga = gh(torch.as_tensor(f)).detach().numpy()
    assert np.array_equal(ga[:, 1:].argmax(1), want[:, 1:].argmax(1)), (ga[:, 1:].argmax(1),
                                                                        want[:, 1:].argmax(1))
    assert (ga >= 1.0).all() and (dirichlet_uncertainty(torch.as_tensor(ga)).numpy() <= 1.0).all()

    keys = np.array([f"p{i % 50}" for i in range(500)])
    hold = player_disjoint_split(keys, 0.10, seed=0)
    assert len(set(keys[hold]) & set(keys[~hold])) == 0
    assert 0.05 < hold.mean() < 0.20, hold.mean()  # noqa: PLR2004
    assert abs(roc_auc(np.zeros(10), np.arange(10) < 5) - 0.5) < 1e-9

    print(f"evidential_jersey self-check OK: loss {curve[0][1]:.4f} -> {curve[-1][1]:.4f}, "
          f"u illegible {u[illeg].mean():.4f} vs legible {u[~illeg].mean():.4f}, "
          f"abstain AUC p_none {auc_p:.4f} / u {auc_u:.4f}, number acc {acc:.4f}")


if __name__ == "__main__":
    _demo()
