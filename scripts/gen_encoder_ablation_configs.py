"""Generate all 24 encoder ablation model configs (6 tactile × 4 RGB)."""
import os, yaml

OUT_DIR = "ablation/encoder"

TACTILE_ENCODERS = {
    "t3": {
        "name": "t3_large",
        "checkpoint": "pretrained_encoders/t3_large",
        "embed_dim": 1024,
        "patch_size": 16,
        "frozen": True,
        "share_encoder_weights": False,
        "keep_cls": True,
    },
    "dinov3": {
        "name": "dinov3_vitl16",
        "checkpoint": "weights/dinov3_vitl16_pretrain_lvd1689m.pth",
        "embed_dim": 1024,
        "patch_size": 16,
        "frozen": True,
        "share_encoder_weights": False,
        "keep_cls": True,
    },
    "sparshv2": {
        "name": "sparsh_dino_base",
        "checkpoint": "pretrained_encoders/sparsh_dinov2_base.ckpt",
        "ssl_name": "dinov2",
        "embed_dim": 768,
        "patch_size": 16,
        "frozen": True,
        "share_encoder_weights": False,
        "keep_cls": False,
    },
    "sparshmae": {
        "name": "sparsh_dino_base",
        "checkpoint": "/media/hdd2/ihsuan/sparsh/weights/sparsh/mae_vitbase.ckpt",
        "ssl_name": "mae",
        "embed_dim": 768,
        "patch_size": 16,
        "frozen": True,
        "share_encoder_weights": False,
        "keep_cls": False,
    },
    "sitr": {
        "name": "sitr_b18",
        "checkpoint": "/media/hdd2/ihsuan/gsrl/datasets/checkpoints/SITR_B18.pth",
        "calibration_dir": "/media/hdd2/ihsuan/gs_blender/renders_v3/pattern_01_2_lines_angle_1_2/session_000/sensor_0000/calibration",
        "num_calibration": 18,
        "embed_dim": 768,
        "patch_size": 16,
        "frozen": True,
        "share_encoder_weights": False,
        "keep_cls": True,
    },
    "dav2": {
        "name": "dav2_vitl14",
        "checkpoint": "pretrained_encoders/dav2_vitl14.pth",
        "embed_dim": 1024,
        "patch_size": 14,
        "frozen": True,
        "share_encoder_weights": False,
        "keep_cls": True,
    },
}

RGB_ENCODERS = {
    "mae": {
        "name": "mae_vitl16",
        "checkpoint": "pretrained_encoders/mae_vitl16.pth",
        "embed_dim": 1024,
        "patch_size": 16,
    },
    "dinov3": {
        "name": "dinov3_vitl16",
        "checkpoint": "weights/dinov3_vitl16_pretrain_lvd1689m.pth",
        "embed_dim": 1024,
        "patch_size": 16,
    },
    "clip": {
        "name": "clip_vitl14",
        "model_id": "openai/clip-vit-large-patch14",
        "embed_dim": 1024,
        "patch_size": 14,
    },
    "siglip": {
        "name": "siglip_vitl16",
        "model_id": "google/siglip-large-patch16-256",
        "embed_dim": 1024,
        "patch_size": 16,
    },
}

TAC_LABELS = {
    "t3": "T3-Large",
    "dinov3": "DINOv3 ViT-L",
    "sparshv2": "Sparsh-DINOv2-Base (768 dim, 6ch)",
    "sparshmae": "Sparsh-MAE-Base (768 dim, 6ch)",
    "sitr": "SITR-B18 (18 calibration imgs)",
    "dav2": "DAv2 ViT-L/14 (256 patches)",
}

RGB_LABELS = {
    "mae": "MAE ViT-L",
    "dinov3": "DINOv3 ViT-L",
    "clip": "CLIP ViT-L/14 (256 patches, has CLS)",
    "siglip": "SigLIP ViT-L/16 (no CLS)",
}

SHARED_TAIL = {
    "projection": {
        "out_dim": 768,
        "modality_embedding": True,
        "rgb_positional_embedding": False,
        "tactile_positional_embedding": True,
    },
    "tokens": {
        "num_spatial_queries": 196,
        "num_pose_queries": 1,
        "object_embedding": True,
        "num_objects": 20,
    },
    "fusion_trunk": {
        "num_layers": 4,
        "num_bottleneck_tokens": 16,
        "num_heads": 8,
        "ffn_mult": 4,
        "dropout": 0.1,
        "bottleneck_continuity": "carry",
        "tap_layers": [0, 1, 2, 3],
        "fusion_variant": "asymmetric",
    },
    "heads": {
        "dpt": {
            "features": 256,
            "dropout": 0.1,
            "out_depth_channels": 1,
            "out_normal_channels": 3,
            "output_frame": "tactile",
            "encoder_tap_layers": [2, 5, 8, 11],
            "inject_gate_init": 0.0,
        },
        "pose": {
            "hidden_dim": 256,
            "dropout": 0.1,
            "pose_mode": "regression",
            "rot_num_bins": 72,
            "use_spatial_pool": True,
        },
    },
}

created = []
skipped = []

for tac_key, tac_enc in TACTILE_ENCODERS.items():
    for rgb_key, rgb_enc in RGB_ENCODERS.items():
        fname = f"tac_{tac_key}_rgb_{rgb_key}.yaml"
        fpath = os.path.join(OUT_DIR, fname)

        comment = f"# Ablation: Tactile={TAC_LABELS[tac_key]}, RGB={RGB_LABELS[rgb_key]}\n\n"

        cfg = {"image_size": 224, "trunk_dim": 768}
        cfg["encoder"] = dict(tac_enc)
        cfg["rgb_encoder"] = dict(rgb_enc)
        cfg.update({k: dict(v) if isinstance(v, dict) else v
                    for k, v in SHARED_TAIL.items()})

        # Deep copy heads to avoid mutation
        import copy
        cfg["heads"] = copy.deepcopy(SHARED_TAIL["heads"])

        # DPT tap layers depend on tactile encoder architecture
        TAP_LAYERS = {
            "t3_large": [2, 5, 8, 11],        # 15 blocks
            "dinov3_vitl16": [5, 11, 17, 23],  # 24 blocks
            "dav2_vitl14": [5, 11, 17, 23],    # 24 blocks
            "sparsh_dino_base": [2, 5, 8, 11], # 12 blocks
            "sitr_b18": [2, 5, 8, 11],         # 12 blocks
        }
        cfg["heads"]["dpt"]["encoder_tap_layers"] = TAP_LAYERS[tac_enc["name"]]

        if os.path.exists(fpath):
            skipped.append(fname)
        else:
            with open(fpath, "w") as f:
                f.write(comment)
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            created.append(fname)

print(f"Created {len(created)} configs, skipped {len(skipped)} existing:")
for c in sorted(created):
    print(f"  + {c}")
for s in sorted(skipped):
    print(f"  . {s} (exists)")
