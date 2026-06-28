"""Datasets.

Every sample is a dict:
    rgb [3,H,W], tactile [3,H,W], depth [1,H,W], normal [3,H,W],
    mask [1,H,W] (contact region), pose [4] = (cos, sin, t_x, t_y),
    object: int (for the object-wise sim split).

- SyntheticVisuoTactileDataset: deterministic random tensors, so the whole pipeline runs
  before real data exists.
- SimVisuoTactileDataset: loader for the gs_blender nested layout:
      <root>/<object>/session_*/sensor_*/
          samples/   (tactile PNGs)
          rgb/       (RGB PNGs)
          raw_data/  (depth .npy + pose .json)
"""
from __future__ import annotations

import glob as _glob
import json
import math
import os
import os.path as osp
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import RGBPhotometricAug, TactileAugment, ToTensorResize


def transform_pose(pose, geo):
    """Apply the same geometric augmentation to the SE(2) pose label.

    pose: tensor [4] = (cos θ, sin θ, tx, ty)
    geo:  dict from TactileAugment with {hflip, vflip, rot_deg}

    Transforms applied in the same order as TactileAugment: hflip → vflip → rotation.
    """
    cos, sin, tx, ty = pose[0].item(), pose[1].item(), pose[2].item(), pose[3].item()

    if geo["hflip"]:
        cos, sin, tx, ty = -cos, sin, -tx, ty

    if geo["vflip"]:
        cos, sin, tx, ty = cos, -sin, tx, -ty

    rot_deg = geo["rot_deg"]
    if abs(rot_deg) > 0.5:
        a = math.radians(rot_deg)
        ca, sa = math.cos(a), math.sin(a)
        # cv2 rotates image CCW by α → object heading decreases by α
        cos_new = cos * ca + sin * sa
        sin_new = sin * ca - cos * sa
        # position follows the same cv2 rotation matrix
        tx_new = tx * ca + ty * sa
        ty_new = -tx * sa + ty * ca
        cos, sin, tx, ty = cos_new, sin_new, tx_new, ty_new

    return torch.tensor([cos, sin, tx, ty], dtype=torch.float32)


def depth_to_normal(depth, pixel_size_x, pixel_size_y):
    """Compute unit surface normals from a depth map via central finite differences.

    depth: (H, W) float32.  Returns (H, W, 3) float32 unit normals.
    """
    dz_dx = np.zeros_like(depth)
    dz_dy = np.zeros_like(depth)
    dz_dx[:, 1:-1] = (depth[:, 2:] - depth[:, :-2]) / (2.0 * pixel_size_x)
    dz_dy[1:-1, :] = (depth[2:, :] - depth[:-2, :]) / (2.0 * pixel_size_y)
    dz_dx[:, 0] = (depth[:, 1] - depth[:, 0]) / pixel_size_x
    dz_dx[:, -1] = (depth[:, -1] - depth[:, -2]) / pixel_size_x
    dz_dy[0, :] = (depth[1, :] - depth[0, :]) / pixel_size_y
    dz_dy[-1, :] = (depth[-1, :] - depth[-2, :]) / pixel_size_y

    normal = np.stack([-dz_dx, -dz_dy, np.ones_like(depth)], axis=-1)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True).clip(min=1e-8)
    return (normal / norm).astype(np.float32)


class SyntheticVisuoTactileDataset(Dataset):
    """Deterministic random-tensor stub. Each index yields a fixed sample (seeded by index),
    so a single batch can be memorized by the overfit test."""

    def __init__(self, num_samples=256, image_size=224, num_objects=8, seed=0):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_objects = num_objects
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(self.seed * 100003 + idx)
        H = W = self.image_size
        rgb = torch.randn(3, H, W, generator=g)
        tactile = torch.randn(3, H, W, generator=g)
        depth = torch.rand(1, H, W, generator=g)
        normal = torch.randn(3, H, W, generator=g)
        normal = normal / normal.norm(dim=0, keepdim=True).clamp_min(1e-6)
        mask = (depth > 0.3).float()
        theta = torch.rand(1, generator=g).item() * 2 * math.pi - math.pi
        tx = torch.rand(1, generator=g).item() * 2 - 1
        ty = torch.rand(1, generator=g).item() * 2 - 1
        pose = torch.tensor([math.cos(theta), math.sin(theta), tx, ty], dtype=torch.float32)
        return {
            "rgb": rgb, "tactile": tactile, "depth": depth, "normal": normal,
            "mask": mask, "pose": pose, "object": idx % self.num_objects,
        }


class SimVisuoTactileDataset(Dataset):
    """Sim loader for the gs_blender nested layout:
        <root>/<object>/session_*/sensor_*/{samples, rgb, raw_data}

    Depth is loaded from raw_data/*.npy; normals are computed from depth via finite
    differences; pose is loaded from raw_data/*_pose.json and converted to (cos,sin,tx,ty)
    with tx,ty normalized to [-1,1] by the sensor extent.
    """

    def __init__(self, cfg_data, image_size, augment=False, include_objects=None, seed=0):
        self.image_size = image_size
        self.augment = augment
        sim = cfg_data.sim
        norm = cfg_data.norm
        self.root = sim.root
        self.rgb_subdir = sim.rgb_subdir
        self.use_gt_depth = sim.get("use_gt_depth", True)

        if self.root is None:
            raise ValueError("configs/data.yaml sim.root is null — set the sim data path.")

        self.img_xform = ToTensorResize((image_size, image_size),
                                        norm.imagenet_mean, norm.imagenet_std)
        self.tactile_aug = TactileAugment() if augment else None
        self.rgb_aug = RGBPhotometricAug() if augment else None

        # Discover sensor units (each = one session × one sensor)
        units = sorted(_glob.glob(osp.join(self.root, "*", "session_*", "sensor_*")))
        if not units:
            units = sorted(_glob.glob(osp.join(self.root, "sensor_*")))
        units = [u for u in units if osp.isdir(osp.join(u, "samples"))]
        if include_objects is not None:
            incl = set(include_objects)
            units = [u for u in units
                     if osp.basename(osp.dirname(osp.dirname(u))) in incl]

        # Build flat sample index and per-unit metadata
        self.samples = []
        self.unit_meta = {}
        for unit in units:
            sample_dir = osp.join(unit, "samples")
            pngs = sorted(f for f in os.listdir(sample_dir) if f.endswith(".png"))
            if not pngs:
                continue

            # Read sensor geometry from session.json
            session_dir = osp.dirname(unit)
            session_json = osp.join(session_dir, "session.json")
            with open(session_json) as f:
                sess = json.load(f)
            x_min, x_max = sess["X_MIN"], sess["X_MAX"]
            y_min, y_max = sess["Y_MIN"], sess["Y_MAX"]
            self.unit_meta[unit] = {
                "pixel_size_x": (x_max - x_min) / image_size,
                "pixel_size_y": (y_max - y_min) / image_size,
                "center_x": (x_min + x_max) / 2.0,
                "center_y": (y_min + y_max) / 2.0,
                "half_x": (x_max - x_min) / 2.0,
                "half_y": (y_max - y_min) / 2.0,
            }

            for png in pngs:
                idx = int(osp.splitext(png)[0])
                suffix = "_gt" if self.use_gt_depth else ""
                rgb_ok = osp.exists(osp.join(unit, self.rgb_subdir, f"{idx:04d}.png"))
                depth_ok = osp.exists(osp.join(unit, "raw_data", f"{idx:04d}{suffix}.npy"))
                pose_ok = osp.exists(osp.join(unit, "raw_data", f"{idx:04d}_pose.json"))
                if rgb_ok and depth_ok and pose_ok:
                    self.samples.append((unit, idx))

        if not self.samples:
            raise RuntimeError(f"No samples found under {self.root} "
                               f"(include_objects={include_objects})")

        valid_units = [u for u in units if u in self.unit_meta]
        self.objects = sorted({osp.basename(osp.dirname(osp.dirname(u)))
                               for u in valid_units})
        self._obj_to_id = {o: i for i, o in enumerate(self.objects)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # Retry with a random fallback if a file disappeared (data still generating)
        for _attempt in range(5):
            try:
                return self._load_sample(index)
            except (FileNotFoundError, OSError):
                index = random.randint(0, len(self.samples) - 1)
        return self._load_sample(index)

    def _load_sample(self, index):
        unit, sample_idx = self.samples[index]
        meta = self.unit_meta[unit]

        # --- Load images ---
        tactile = np.array(
            Image.open(osp.join(unit, "samples", f"{sample_idx:04d}.png")),
            dtype=np.float32,
        )
        rgb = np.array(
            Image.open(osp.join(unit, self.rgb_subdir, f"{sample_idx:04d}.png")),
            dtype=np.float32,
        )

        # --- Load depth (float32, H×W) ---
        suffix = "_gt" if self.use_gt_depth else ""
        depth = np.load(
            osp.join(unit, "raw_data", f"{sample_idx:04d}{suffix}.npy")
        ).astype(np.float32)

        # --- Pose: SE(2) = (cos θ, sin θ, tx_norm, ty_norm) ---
        pose = self._load_pose(unit, sample_idx, meta)

        # --- Augmentation (tactile + depth; normals computed after) ---
        if self.tactile_aug is not None:
            tactile, _, depth, _, geo = self.tactile_aug(tactile, [], depth, None)
            pose = transform_pose(pose, geo)
        if self.rgb_aug is not None:
            rgb = self.rgb_aug(rgb)

        # --- Compute normals from (augmented) depth ---
        normal = depth_to_normal(depth, meta["pixel_size_x"], meta["pixel_size_y"])

        # --- Contact mask ---
        mask = (depth > 0).astype(np.float32)

        # --- Object ID ---
        obj_name = osp.basename(osp.dirname(osp.dirname(unit)))
        obj_id = self._obj_to_id[obj_name]

        return {
            "rgb": self.img_xform(rgb),
            "tactile": self.img_xform(tactile),
            "depth": torch.from_numpy(depth).unsqueeze(0),       # (1, H, W)
            "normal": torch.from_numpy(np.ascontiguousarray(normal))
                      .permute(2, 0, 1),                         # (3, H, W)
            "mask": torch.from_numpy(mask).unsqueeze(0),          # (1, H, W)
            "pose": pose,                                         # (4,)
            "object": obj_id,
        }

    def _load_pose(self, unit, sample_idx, meta):
        pose_path = osp.join(unit, "raw_data", f"{sample_idx:04d}_pose.json")
        with open(pose_path) as f:
            data = json.load(f)
        theta = data["rotation_euler"][2]
        tx = (data["sample_x"] - meta["center_x"]) / max(meta["half_x"], 1e-8)
        ty = (data["sample_y"] - meta["center_y"]) / max(meta["half_y"], 1e-8)
        return torch.tensor(
            [math.cos(theta), math.sin(theta), tx, ty], dtype=torch.float32
        )


def build_datasets(cfg):
    """Return (train_ds, val_ds) from the merged config (uses cfg.dataset switch)."""
    image_size = cfg.image_size
    which = cfg.dataset
    if which == "synthetic":
        s = cfg.synthetic
        n = s.num_samples
        n_val = max(1, n // 8)
        train = SyntheticVisuoTactileDataset(n - n_val, image_size, s.num_objects, seed=0)
        val = SyntheticVisuoTactileDataset(n_val, image_size, s.num_objects, seed=1)
        return train, val
    if which == "sim":
        val_objs = list(cfg.sim.val_objects)
        train_all = SimVisuoTactileDataset(cfg, image_size, augment=True, include_objects=None)
        train_objs = [o for o in train_all.objects if o not in set(val_objs)]
        train = SimVisuoTactileDataset(cfg, image_size, augment=True,
                                       include_objects=train_objs)
        val = SimVisuoTactileDataset(cfg, image_size, augment=False,
                                     include_objects=val_objs)
        return train, val
    raise ValueError(f"Unknown dataset {which!r} (configs/data.yaml dataset:)")
