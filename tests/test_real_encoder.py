"""Integration test for the real DINOv3 encoder. Skips automatically when the checkpoint
or `transformers` is unavailable, so it's a no-op in CI / on machines without the weights."""
import os

import pytest
import torch

from vistacfusion.utils.config import merge_configs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _real_cfg():
    c = merge_configs(
        os.path.join(ROOT, "configs", "model.yaml"),
        os.path.join(ROOT, "configs", "train.yaml"),
        os.path.join(ROOT, "configs", "data.yaml"),
    )
    ckpt = c.encoder.checkpoint
    if not ckpt or not os.path.isfile(os.path.join(ROOT, ckpt)):
        pytest.skip("DINOv3 checkpoint not present")
    pytest.importorskip("transformers")
    c.encoder.checkpoint = os.path.join(ROOT, ckpt)
    return c


def test_real_encoder_loads_and_is_frozen():
    from vistacfusion.models.model import build_model

    model = build_model(_real_cfg()).eval()
    assert model.tactile_encoder.embed_dim == 1024
    enc_trainable = sum(p.numel() for p in model.tactile_encoder.parameters() if p.requires_grad)
    assert enc_trainable == 0


def test_real_encoder_full_forward_all_configs():
    from vistacfusion.models.model import build_model

    model = build_model(_real_cfg()).eval()
    rgb = torch.randn(1, 3, 224, 224)
    tac = torch.randn(1, 3, 224, 224)
    for config in ("both", "tactile", "rgb"):
        with torch.no_grad():
            out = model(rgb, tac, config=config)
        assert out["depth"].shape == (1, 1, 224, 224), config
        assert out["normal"].shape == (1, 3, 224, 224), config
        assert out["se2"].shape == (1, 4), config
