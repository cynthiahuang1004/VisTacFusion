"""Frozen image encoders. DINOv3 ViT-L/16: 224x224 -> 196 patch tokens + 1 CLS, dim 1024.

Two interchangeable encoders behind one interface:
  - DINOv3Encoder : real frozen DINOv3 (torch.hub arch + local .pth). Needs the gated weights.
  - MockEncoder   : deterministic patch-embed stand-in with identical shapes, so the pipeline
                    runs on CPU before the weights exist.

Interface (tokens at the encoder's native dim E):
    forward(x)            -> (patch [B, N, E], cls [B, 1, E])
    forward_multiscale(x) -> K patch maps [B, N, E]   (for the DPT v2 tap source)

build_encoder() returns the real encoder when a checkpoint is set, else the mock.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _count_blocks(backbone):
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        return None
    try:
        return len(blocks)
    except TypeError:
        return sum(1 for _ in blocks)


def auto_layer_indices(depth, k=4):
    """k evenly-spaced block indices ending at the last layer (ported from the notebook).

    depth 12 (ViT-B) -> (2, 5, 8, 11); depth 24 (ViT-L) -> (5, 11, 17, 23).
    """
    if not depth or depth < k:
        return tuple(range(max(1, k)))
    step = depth / k
    idx = sorted({min(depth - 1, int(round((i + 1) * step)) - 1) for i in range(k)})
    while len(idx) < k:
        for cand in range(depth - 1, -1, -1):
            if cand not in idx:
                idx.append(cand)
                break
        idx = sorted(set(idx))
    return tuple(idx[-k:])


class DINOv3Encoder(nn.Module):
    """Frozen DINOv3: last-layer patch tokens + CLS, plus intermediate layers for DPT v2.

    The hub entrypoint only accepts hashed weight URLs, so we build the arch with
    pretrained=False and load the local .pth ourselves.
    """

    def __init__(self, model_name="dinov3_vitl16", weights=None, multiscale_layers=None):
        super().__init__()
        if weights is None:
            raise ValueError(
                "DINOv3 weights are gated. Pass a local .pth path, or use MockEncoder "
                "(set encoder.checkpoint: null) for scaffolding/tests."
            )
        self.dinov3 = torch.hub.load(
            "facebookresearch/dinov3", model_name, pretrained=False
        )
        print(f"  [encoder] loading DINOv3 weights from {weights}")
        state_dict = torch.load(weights, map_location="cpu", weights_only=True)
        missing, unexpected = self.dinov3.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [encoder] missing keys:    {len(missing)}")
        if unexpected:
            print(f"  [encoder] unexpected keys: {len(unexpected)}")

        self.embed_dim = getattr(self.dinov3, "embed_dim", None)
        if self.embed_dim is None:
            self.embed_dim = self.dinov3.norm.normalized_shape[0]

        depth = _count_blocks(self.dinov3)
        if multiscale_layers is None:
            multiscale_layers = auto_layer_indices(depth, k=4)
        self.multiscale_layers = sorted(multiscale_layers)

        for p in self.dinov3.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        # Keep the frozen backbone in eval mode regardless of the parent's train/eval.
        super().train(mode)
        self.dinov3.eval()
        return self

    @torch.no_grad()
    def forward(self, x):
        out = self.dinov3.get_intermediate_layers(
            x, n=1, reshape=False, return_class_token=True
        )
        patch, cls = out[-1]                # [B, N, E], [B, E]
        return patch, cls.unsqueeze(1)      # [B, N, E], [B, 1, E]

    @torch.no_grad()
    def forward_multiscale(self, x):
        outs = self.dinov3.get_intermediate_layers(
            x, n=self.multiscale_layers, reshape=False, return_class_token=True
        )
        return [patch for (patch, _cls) in outs]   # list of [B, N, E]


class MockEncoder(nn.Module):
    """Deterministic frozen stand-in for DINOv3 with identical output shapes.

    Patch-embed conv -> N tokens; CLS = linear(mean of tokens). forward_multiscale returns K
    distinct projected copies. All params frozen, like the real encoder.
    """

    def __init__(self, embed_dim=1024, patch_size=16, image_size=224, multiscale_k=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.multiscale_layers = list(range(multiscale_k))

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_proj = nn.Linear(embed_dim, embed_dim)
        # per-scale linear so multiscale taps are distinct but deterministic
        self.scale_proj = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(multiscale_k)]
        )
        for p in self.parameters():
            p.requires_grad = False

    def _tokens(self, x):
        f = self.patch_embed(x)                       # [B, E, g, g]
        B, E, g, _ = f.shape
        return f.flatten(2).transpose(1, 2)           # [B, N, E]

    @torch.no_grad()
    def forward(self, x):
        patch = self._tokens(x)                       # [B, N, E]
        cls = self.cls_proj(patch.mean(dim=1, keepdim=True))   # [B, 1, E]
        return patch, cls

    @torch.no_grad()
    def forward_multiscale(self, x):
        patch = self._tokens(x)
        return [proj(patch) for proj in self.scale_proj]       # K x [B, N, E]


def build_encoder(enc_cfg, image_size):
    """Factory: real DINOv3 if a checkpoint is set, else the MockEncoder for testing."""
    checkpoint = enc_cfg.get("checkpoint", None)
    multiscale = enc_cfg.get("multiscale_layers", None)
    if checkpoint:
        return DINOv3Encoder(
            model_name=enc_cfg.get("name", "dinov3_vitl16"),
            weights=checkpoint,
            multiscale_layers=multiscale,
        )
    return MockEncoder(
        embed_dim=enc_cfg.get("embed_dim", 1024),
        patch_size=enc_cfg.get("patch_size", 16),
        image_size=image_size,
    )
