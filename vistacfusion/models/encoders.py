"""Frozen image encoders. DINOv3 ViT-L/16: 224x224 -> 196 patch tokens + 1 CLS, dim 1024.

Two interchangeable encoders behind one interface:
  - DINOv3Encoder : real frozen DINOv3, loaded via HuggingFace `transformers`. The gated
                    weights download as an HF-format state dict (`embeddings.*`, `layer.N.*`),
                    which this loads directly -- no architecture download needed.
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


def auto_layer_indices(depth, k=4):
    """k evenly-spaced layer indices ending at the last layer.

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


def _infer_dinov3_config(sd):
    """Infer a DINOv3ViTConfig from an HF-format state dict (works for any ViT size)."""
    from transformers import DINOv3ViTConfig

    pe = sd["embeddings.patch_embeddings.weight"]          # [hidden, 3, patch, patch]
    hidden = pe.shape[0]
    patch = pe.shape[-1]
    num_register = sd["embeddings.register_tokens"].shape[1]
    num_layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("layer."))
    intermediate = sd["layer.0.mlp.up_proj.weight"].shape[0]
    gated = any("gate" in k for k in sd if k.startswith("layer.0.mlp"))
    return DINOv3ViTConfig(
        patch_size=patch,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=num_layers,
        num_attention_heads=hidden // 64,                  # DINOv3 head_dim = 64
        num_register_tokens=num_register,
        image_size=224,
        use_gated_mlp=gated,
    )


class DINOv3Encoder(nn.Module):
    """Frozen DINOv3 (HuggingFace `transformers`).

    Loads the gated HF-format checkpoint directly. The token layout is
    [CLS, register x R, patch x N]; we expose patches and CLS, and intermediate layers for v2.
    """

    def __init__(self, model_name="dinov3_vitl16", weights=None, multiscale_layers=None):
        super().__init__()
        if weights is None:
            raise ValueError(
                "DINOv3 weights are gated. Pass a local checkpoint path, or use MockEncoder "
                "(set encoder.checkpoint: null) for scaffolding/tests."
            )
        from transformers import DINOv3ViTModel

        print(f"  [encoder] loading DINOv3 weights from {weights}")
        sd = torch.load(weights, map_location="cpu", weights_only=True)
        cfg = _infer_dinov3_config(sd)
        self.dinov3 = DINOv3ViTModel(cfg)
        # HF nests the transformer layers under `model.`; embeddings/norm stay top-level.
        remap = {(f"model.{k}" if k.startswith("layer.") else k): v for k, v in sd.items()}
        self.dinov3.load_state_dict(remap, strict=True)

        self.embed_dim = cfg.hidden_size
        self.num_register = cfg.num_register_tokens
        self._patch_start = 1 + self.num_register               # skip CLS + registers
        if multiscale_layers is None:
            multiscale_layers = auto_layer_indices(cfg.num_hidden_layers, k=4)
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
        tokens = self.dinov3(x).last_hidden_state          # [B, 1+R+N, E]
        patch = tokens[:, self._patch_start:]              # [B, N, E]
        cls = tokens[:, :1]                                # [B, 1, E]
        return patch, cls

    @torch.no_grad()
    def forward_multiscale(self, x):
        # hidden_states[0] = embeddings, hidden_states[i+1] = output of layer i.
        hs = self.dinov3(x, output_hidden_states=True).hidden_states
        return [hs[i + 1][:, self._patch_start:] for i in self.multiscale_layers]


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
