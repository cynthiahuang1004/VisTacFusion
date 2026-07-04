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

from .transforms import (FIXED_CROP, RGBPhotometricAug, TactileAugment,
                         ToTensorResize, fixed_center_crop, rotate_gel_spin)


def rotate_pose_theta(pose, dtheta_rad):
    """Shift the pose theta by dtheta_rad; (x, y) unchanged (gel spins in place).

    pose: tensor [4] = (cos θ, sin θ, x, y).
    Sign convention: image rotated by +φ (cv2) → θ' = θ − φ (pose is sensor-relative-
    to-object: object appearing rotated +φ ⟺ sensor rotated −φ). Verified on real
    cross-session pairs of 5 asymmetric objects (hex_key/edge/patterns, 8/0 votes,
    100× MSE margins) using the corrected object-frame (x,y) for pairing.
    Caller passes dtheta_rad = −radians(φ_cv2).
    """
    cos_t, sin_t = pose[0].item(), pose[1].item()
    c, s = math.cos(dtheta_rad), math.sin(dtheta_rad)
    cos_new = cos_t * c - sin_t * s
    sin_new = sin_t * c + cos_t * s
    return torch.tensor([cos_new, sin_new, pose[2].item(), pose[3].item()],
                        dtype=torch.float32)


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
    differences; pose is computed per pose_calculation.py: relative rotation delta_rz,
    translation in object frame normalized by target_size.
    """

    def __init__(self, cfg_data, image_size, augment=False, include_objects=None, seed=0):
        self.image_size = image_size
        self.augment = augment
        sim = cfg_data.sim
        norm = cfg_data.norm
        self.root = sim.root
        self.mesh_dir = sim.get("mesh_dir", osp.join(osp.dirname(self.root), "meshes"))
        self.rgb_subdir = sim.rgb_subdir
        self.use_gt_depth = sim.get("use_gt_depth", True)
        # Tactile camera view is fixed & square for ALL objects (fov=60, half-width
        # 0.008751m -> 17.5mm). session.json X_MIN/X_MAX is the press sampling range,
        # NOT the camera view — do not use it for pixel size.
        # Every sample gets the fixed 1/sqrt(2) center crop -> effective view 12.37mm,
        # constant pixel size for train/val/inference alike.
        gel_view_m = sim.get("gel_view_m", 0.017502)
        self.pixel_size = gel_view_m * FIXED_CROP / image_size
        # Gel-spin rotation augmentation (train only): rotate tactile+rgb+depth by a
        # random angle, shift GT theta by the same angle (sign verified), (x,y) unchanged.
        self.rot_aug = augment and sim.get("rot_augment", True)
        self.rot_aug_max_deg = sim.get("rot_augment_max_deg", 180.0)

        if self.root is None:
            raise ValueError("configs/data.yaml sim.root is null — set the sim data path.")

        self.img_xform = ToTensorResize((image_size, image_size),
                                        norm.imagenet_mean, norm.imagenet_std)
        self.tactile_aug = TactileAugment() if augment else None
        self.rgb_aug = RGBPhotometricAug() if augment else None

        # Pre-load per-object mesh info for pose computation
        self._obj_pose_info = {}
        self._load_object_pose_info()

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

            session_dir = osp.dirname(unit)
            session_json = osp.join(session_dir, "session.json")
            with open(session_json) as f:
                sess = json.load(f)

            obj_name = osp.basename(osp.dirname(session_dir))
            info = self._obj_pose_info.get(obj_name)
            if info is None:
                continue

            base_rot = sess["base_rotation"]
            delta_rz = base_rot[2] - info["rz0"]
            session_center = self._get_session_center(
                info["vertices"], info["fixed_scale"], base_rot)

            self.unit_meta[unit] = {
                "delta_rz": delta_rz,
                "session_center": session_center,
                "half": info["half"],
                "valid_cells": {c["gx"] * 1000 + c["gy"]: c for c in sess.get("valid_cells", [])},
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

    def _load_object_pose_info(self):
        """Pre-load mesh + session_000 info for each object."""
        try:
            import trimesh
            from scipy.spatial.transform import Rotation
        except ImportError:
            print("[WARN] trimesh/scipy not available, pose labels will be zeros")
            return

        obj_dirs = sorted(d for d in os.listdir(self.root)
                          if osp.isdir(osp.join(self.root, d)))
        for obj_name in obj_dirs:
            mesh_path = osp.join(self.mesh_dir, f"{obj_name}.obj")
            s0_path = osp.join(self.root, obj_name, "session_000", "session.json")
            if not osp.exists(mesh_path) or not osp.exists(s0_path):
                continue
            mesh = __import__("trimesh").load(mesh_path, force="mesh")
            with open(s0_path) as f:
                d0 = json.load(f)
            fixed_scale = d0["fixed_scale"]
            target_size = d0.get("_target_size_mm", 82.0)
            half = target_size / 2.0 / 1000.0
            rz0 = d0["base_rotation"][2]
            self._obj_pose_info[obj_name] = {
                "vertices": mesh.vertices,
                "fixed_scale": fixed_scale,
                "half": half,
                "rz0": rz0,
            }

    @staticmethod
    def _get_session_center(vertices, fixed_scale, base_rotation):
        from scipy.spatial.transform import Rotation
        R_3d = Rotation.from_euler("xyz", base_rotation)
        v = R_3d.apply(vertices) / fixed_scale
        cx = (v[:, 0].min() + v[:, 0].max()) / 2.0
        cy = (v[:, 1].min() + v[:, 1].max()) / 2.0
        return np.array([cx, cy])

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

        # --- Gel-spin rotation aug: tactile+rgb+depth rotate together, θ -= φ ---
        if self.rot_aug:
            phi_deg = random.uniform(-self.rot_aug_max_deg, self.rot_aug_max_deg)
            tactile, rgb, depth = rotate_gel_spin(tactile, rgb, depth, phi_deg)
            pose = rotate_pose_theta(pose, -math.radians(phi_deg))

        # --- Fixed 1/sqrt(2) center crop: EVERY sample (train/val, rotated or not) ---
        tactile = fixed_center_crop(tactile)
        rgb = fixed_center_crop(rgb)
        depth = fixed_center_crop(depth)

        # --- Photometric augmentation ---
        if self.tactile_aug is not None:
            tactile, _, depth, _, _ = self.tactile_aug(tactile, [], depth, None)
        if self.rgb_aug is not None:
            rgb = self.rgb_aug(rgb)

        # --- Compute normals (constant pixel size — fixed crop for all samples) ---
        normal = depth_to_normal(depth, self.pixel_size, self.pixel_size)

        # --- Contact mask (before scaling) ---
        mask = (depth > 0).astype(np.float32)

        # --- Scale depth ×1000 for numerical stability in fp16 SSI loss ---
        depth = depth * 1000.0

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

        delta_rz = meta["delta_rz"]
        half = meta["half"]

        # pose json sample_x/sample_y = press offset from the session object center in
        # WORLD axes with x negated: (sx, sy) = (-(cx - scx), +(cy - scy)) — verified
        # against session.json valid_cells at 0/30/90 deg sessions.
        # Reference label (pose_calculation.py): [x, y] = diag(-1,1) @ R(th).T @ off_cell.
        # Substituting off_cell = diag(-1,1) @ (sx, sy) and simplifying with
        # M R(th).T M = R(th):  label = R(+th) @ (sx, sy) / half.
        cos_rz = math.cos(delta_rz)
        sin_rz = math.sin(delta_rz)
        sx, sy = data["sample_x"], data["sample_y"]
        x_norm = (cos_rz * sx - sin_rz * sy) / max(half, 1e-8)
        y_norm = (sin_rz * sx + cos_rz * sy) / max(half, 1e-8)

        return torch.tensor(
            [cos_rz, sin_rz, x_norm, y_norm],
            dtype=torch.float32,
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
