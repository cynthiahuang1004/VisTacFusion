"""Shape-assertion tests (CLAUDE.md 10: get shapes right before training logic).

The key fairness invariant: the decoder input is IDENTICAL across all three modality configs:
    4 x [B, 196, D]  spatial taps   +   [B, 1, D]  pose token.
"""
import torch

from vistacfusion.models.model import build_model

B = 2


def _inputs(cfg):
    s = cfg.image_size
    return torch.randn(B, 3, s, s), torch.randn(B, 3, s, s)


def test_output_shapes_all_configs(cfg):
    model = build_model(cfg).eval()
    rgb, tac = _inputs(cfg)
    s = cfg.image_size
    for config in ("both", "tactile", "rgb"):
        out = model(rgb, tac, config=config)
        assert out["depth"].shape == (B, 1, s, s), config
        assert out["normal"].shape == (B, 3, s, s), config
        assert out["se2"].shape == (B, 4), config
        # (cos, sin) must be unit-norm
        cs = out["se2"][:, :2]
        assert torch.allclose(cs.norm(dim=-1), torch.ones(B), atol=1e-4), config


def test_identical_decoder_input_across_configs(cfg):
    """The whole fairness argument: same decoder input shape in every config."""
    model = build_model(cfg).eval()
    rgb, tac = _inputs(cfg)
    D = cfg.trunk_dim
    n_spatial = cfg.tokens.num_spatial_queries
    shapes = {}
    for config in ("both", "tactile", "rgb"):
        dec = model(rgb, tac, config=config, return_decoder_inputs=True)
        assert len(dec["taps"]) == 4, config
        for t in dec["taps"]:
            assert t.shape == (B, n_spatial, D), (config, t.shape)
        assert dec["pose_token"].shape == (B, 1, D), config
        shapes[config] = [tuple(t.shape) for t in dec["taps"]] + [tuple(dec["pose_token"].shape)]
    assert shapes["both"] == shapes["tactile"] == shapes["rgb"]


def test_v2_encoder_multiscale_shapes(cfg):
    """v2 tap_source must produce the same fixed decoder input as v1."""
    cfg.heads.dpt.tap_source = "encoder_multiscale"
    model = build_model(cfg).eval()
    rgb, tac = _inputs(cfg)
    s = cfg.image_size
    for config in ("both", "tactile", "rgb"):
        out = model(rgb, tac, config=config)
        assert out["depth"].shape == (B, 1, s, s), config
        assert out["normal"].shape == (B, 3, s, s), config


def test_v2_injection_zero_without_rgb(cfg):
    """v2 fairness (gotcha 9): with gate init 0 AND RGB absent, taps are pure encoder taps.
    tactile-only output must be invariant to the RGB input tensor."""
    cfg.heads.dpt.tap_source = "encoder_multiscale"
    model = build_model(cfg).eval()
    _, tac = _inputs(cfg)
    rgb_a = torch.randn(B, 3, cfg.image_size, cfg.image_size)
    rgb_b = torch.randn(B, 3, cfg.image_size, cfg.image_size)
    with torch.no_grad():
        out_a = model(rgb_a, tac, config="tactile")
        out_b = model(rgb_b, tac, config="tactile")
    assert torch.allclose(out_a["depth"], out_b["depth"], atol=1e-5)
    assert torch.allclose(out_a["se2"], out_b["se2"], atol=1e-5)


def test_frozen_encoder_has_no_trainable_params(cfg):
    model = build_model(cfg)
    enc_trainable = sum(p.numel() for p in model.tactile_encoder.parameters() if p.requires_grad)
    assert enc_trainable == 0, "encoder must be frozen (CLAUDE.md 1)"
