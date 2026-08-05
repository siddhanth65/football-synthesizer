"""CLIP ViT-B/16 + projection appearance embedder (cluster session 3's encoder).

Duck-type-compatible with :class:`tools.prtreid_probe.PrtreidEmbedder` -- same ``embed``
signature, same RGB-uint8 crop contract, same 256-d L2-normalised output -- so every
consumer of the per-detection embedding cache (``generator.gta_link.detection_embeddings``,
the GTA connector, the solver gallery) is byte-identical across arms and only the embedding
source changes.

The checkpoint carries the full backbone, so the vision tower is built from a local config
and never contacts the HuggingFace hub.

Trained by ``~/work/train_clip.py`` on the unified GSR + SoccerNet-re-ID corpus; see
``results/CLUSTER_SESSION3.md`` for the recipe and the valid-split numbers.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: Default checkpoint (cluster session 3, epoch 8: valid identity mAP 58.75).
CLIP_WEIGHTS = Path("outputs/gsr/clip_ckpt/epoch8.pt")
EMBED_DIM = 256
_CLIP_HW = (224, 224)
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
#: openai/clip-vit-base-patch16 vision tower, spelled out so no hub fetch is needed.
_VISION_CFG = dict(
    hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12,
    image_size=224, patch_size=16, projection_dim=512, hidden_act="quick_gelu",
    layer_norm_eps=1e-5, num_channels=3,
)


class ClipEmbedder:
    """CLIP ViT-B/16 visual tower + the trained 256-d projection head."""

    def __init__(self, weights: Path = CLIP_WEIGHTS, device: str | None = None,
                 batch_size: int = 32) -> None:
        import torch  # noqa: PLC0415
        from torch import nn  # noqa: PLC0415
        from transformers import CLIPVisionConfig, CLIPVisionModel  # noqa: PLC0415

        if not Path(weights).exists():
            raise FileNotFoundError(f"CLIP checkpoint not found at {weights}")
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        state = ckpt["model"]

        backbone = CLIPVisionModel(CLIPVisionConfig(**_VISION_CFG))
        proj = nn.Linear(_VISION_CFG["hidden_size"], EMBED_DIM)
        bn = nn.BatchNorm1d(EMBED_DIM)
        # transformers >=5.14 exposes the tower directly, <5.14 nests it under .vision_model;
        # the checkpoint was written by the former, this may be running on the latter
        tower = getattr(backbone, "vision_model", backbone)
        for mod, prefix in ((tower, "backbone."), (proj, "proj."), (bn, "bn.")):
            sub = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
            missing, unexpected = mod.load_state_dict(sub, strict=False)
            if missing:
                raise RuntimeError(f"CLIP checkpoint missing {len(missing)} {prefix}keys, "
                                   f"e.g. {missing[:3]}")
            if unexpected:
                logger.debug("%s: %d unexpected keys", prefix, len(unexpected))
        logger.info("CLIP embedder loaded: epoch %s, device=%s", ckpt.get("epoch"), self.device)
        self.backbone = backbone.to(self.device).eval()
        self.proj = proj.to(self.device).eval()
        self.bn = bn.to(self.device).eval()
        self.dims = ckpt.get("dims", {})
        self.jersey = None
        if "jersey.weight" in state:  # the trained attribute head, CPU-side (42 x 256)
            self.jersey = nn.Linear(EMBED_DIM, state["jersey.weight"].shape[0])
            self.jersey.load_state_dict({"weight": state["jersey.weight"],
                                         "bias": state["jersey.bias"]})
            self.jersey.eval()

    def embed(self, crops: list[np.ndarray], batch_size: int | None = None) -> np.ndarray:
        """Embed RGB uint8 crops -> ``(N, 256)`` L2-normalised float32 features."""
        return self._forward(crops, batch_size)[0]

    def jersey_probs(self, crops: list[np.ndarray], batch_size: int | None = None,
                     pre_norm: bool = False) -> np.ndarray:
        """Softmax of the trained jersey head -> ``(N, n_jersey)`` float32.

        Args:
            crops: RGB uint8 crops.
            batch_size: Override the instance batch size.
            pre_norm: Feed the head the pre-L2-norm BN output instead of the shared
                L2-normalised embedding (the two differ only by a per-sample scale, but
                that scale changes the softmax temperature).

        Returns:
            Per-crop class posteriors over the head's jersey vocabulary.
        """
        if self.jersey is None:
            raise RuntimeError("checkpoint carries no jersey head")
        feats, raw = self._forward(crops, batch_size)
        torch = self._torch
        x = torch.from_numpy(raw if pre_norm else feats)
        with torch.inference_mode():
            return torch.softmax(self.jersey(x), dim=1).numpy().astype(np.float32)

    def _forward(self, crops: list[np.ndarray],
                 batch_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Run the tower once -> (L2-normalised embedding, pre-norm BN output)."""
        import cv2  # noqa: PLC0415

        torch = self._torch
        if not crops:
            return np.zeros((0, EMBED_DIM), np.float32), np.zeros((0, EMBED_DIM), np.float32)
        bs = batch_size or self.batch_size
        out, raws = [], []
        for start in range(0, len(crops), bs):
            chunk = crops[start:start + bs]
            batch = np.empty((len(chunk), _CLIP_HW[0], _CLIP_HW[1], 3), np.float32)
            for i, c in enumerate(chunk):
                r = cv2.resize(c, (_CLIP_HW[1], _CLIP_HW[0]), interpolation=cv2.INTER_CUBIC)
                batch[i] = (r.astype(np.float32) / 255.0 - _CLIP_MEAN) / _CLIP_STD
            t = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
            with torch.inference_mode():
                pooled = self.backbone(pixel_values=t).pooler_output
                raw = self.bn(self.proj(pooled))
                feats = torch.nn.functional.normalize(raw)
                raws.append(raw.float().cpu().numpy())
                out.append(feats.float().cpu().numpy())
            del t
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return (np.concatenate(out).astype(np.float32),
                np.concatenate(raws).astype(np.float32))


def _demo() -> None:
    """Contract check: shape, L2 norm, and that identical crops embed identically."""
    emb = ClipEmbedder(device="cpu", batch_size=4)
    assert emb.embed([]).shape == (0, EMBED_DIM)
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 255, (120, 60, 3), dtype=np.uint8)
    feats = emb.embed([crop, crop, rng.integers(0, 255, (90, 40, 3), dtype=np.uint8)])
    assert feats.shape == (3, EMBED_DIM), feats.shape
    assert np.allclose(np.linalg.norm(feats, axis=1), 1.0, atol=1e-4), "not L2-normalised"
    assert np.allclose(feats[0], feats[1], atol=1e-5), "same crop must embed identically"
    print("clip_embedder demo OK:", feats.shape, "norms ~1")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
