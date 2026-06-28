"""Augmentation + tensorization.

- TactileAugment   : sim2real domain randomization -- gain/bias/gradient/brightness/residual/
  noise + flips & rotation, with matching normal-map corrections.
- RGBPhotometricAug: photometric jitter on the RGB context (geometry kept stable).
- ToTensorResize   : HWC float32 -> normalized CHW tensor, resized to (H, W).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_AUGMENT_PARAMS = {
    "gain": 0.5, "bias": 45.0, "grad": 0.7, "bright": 25.0,
    "resid": 20.0, "noise": 6.0, "rot_deg": 15.0,
    "hflip": True, "vflip": True,
}


class TactileAugment:
    """Heavy tactile domain randomization. Operates on float32 HWC arrays in-place-ish.

    __call__(sample_diff, calib_diffs, depth, normal) -> same tuple, augmented.
    Geometric ops are applied to ALL images; normal vectors are corrected on flip/rotate.
    """

    def __init__(self, params=None):
        self.p = {**DEFAULT_AUGMENT_PARAMS, **(params or {})}

    def __call__(self, sample_diff, calib_diffs, depth, normal):
        p = self.p
        H, W = sample_diff.shape[:2]

        # ---- Photometric (sample only) ----
        if p["gain"] > 0:
            g = np.random.uniform(1 - p["gain"], 1 + p["gain"], size=(1, 1, 3)).astype(np.float32)
            sample_diff = sample_diff * g
        if p["bias"] > 0:
            b = np.random.uniform(-p["bias"], p["bias"], size=(1, 1, 3)).astype(np.float32)
            sample_diff = sample_diff + b
        if p["bright"] > 0:
            sample_diff = sample_diff + np.float32(np.random.uniform(-p["bright"], p["bright"]))
        if p["grad"] > 0:
            angle = np.random.uniform(0, 2 * np.pi)
            ys = np.linspace(-1, 1, H, dtype=np.float32).reshape(-1, 1)
            xs = np.linspace(-1, 1, W, dtype=np.float32).reshape(1, -1)
            grad_map = np.float32(np.cos(angle)) * xs + np.float32(np.sin(angle)) * ys
            amp = np.random.uniform(0, p["grad"], size=(1, 1, 3)).astype(np.float32)
            sample_diff = sample_diff + grad_map[..., None] * amp * np.float32(50.0)
        if p["resid"] > 0:
            raw = np.random.randn(16, 16, 3).astype(np.float32)
            smooth = cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR)
            smooth = cv2.GaussianBlur(smooth, (0, 0), sigmaX=H / 8.0)
            std = np.float32(smooth.std())
            if std > 1e-6:
                smooth = smooth / std * np.float32(p["resid"])
            sample_diff = sample_diff + smooth
        if p["noise"] > 0:
            noise = np.random.normal(0, p["noise"], sample_diff.shape).astype(np.float32)
            sample_diff = sample_diff + noise

        # ---- Geometric (all images) ----
        do_hflip = p["hflip"] and np.random.random() < 0.5
        do_vflip = p["vflip"] and np.random.random() < 0.5
        rot_angle = np.random.uniform(-p["rot_deg"], p["rot_deg"]) if p["rot_deg"] > 0 else 0.0

        if do_hflip:
            sample_diff = np.ascontiguousarray(sample_diff[:, ::-1])
            calib_diffs = [np.ascontiguousarray(c[:, ::-1]) for c in calib_diffs]
            if depth is not None:
                depth = np.ascontiguousarray(depth[:, ::-1])
            if normal is not None:
                normal = np.ascontiguousarray(normal[:, ::-1])
                normal[:, :, 0] = 255.0 - normal[:, :, 0]

        if do_vflip:
            sample_diff = np.ascontiguousarray(sample_diff[::-1])
            calib_diffs = [np.ascontiguousarray(c[::-1]) for c in calib_diffs]
            if depth is not None:
                depth = np.ascontiguousarray(depth[::-1])
            if normal is not None:
                normal = np.ascontiguousarray(normal[::-1])
                normal[:, :, 1] = 255.0 - normal[:, :, 1]

        if abs(rot_angle) > 0.5:
            M = cv2.getRotationMatrix2D((W / 2, H / 2), rot_angle, 1.0)
            flags, border = cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101
            sample_diff = cv2.warpAffine(sample_diff, M, (W, H), flags=flags, borderMode=border)
            calib_diffs = [cv2.warpAffine(c, M, (W, H), flags=flags, borderMode=border)
                           for c in calib_diffs]
            if depth is not None:
                depth = cv2.warpAffine(depth, M, (W, H), flags=flags, borderMode=border)
            if normal is not None:
                normal = cv2.warpAffine(normal, M, (W, H), flags=flags, borderMode=border)
                rad = np.radians(rot_angle)
                cos_a, sin_a = np.float32(np.cos(rad)), np.float32(np.sin(rad))
                nx = normal[:, :, 0] / 127.5 - 1.0
                ny = normal[:, :, 1] / 127.5 - 1.0
                normal[:, :, 0] = np.clip((cos_a * nx + sin_a * ny + 1.0) * 127.5, 0, 255)
                normal[:, :, 1] = np.clip((-sin_a * nx + cos_a * ny + 1.0) * 127.5, 0, 255)

        geo = {"hflip": do_hflip, "vflip": do_vflip, "rot_deg": rot_angle}
        return sample_diff, calib_diffs, depth, normal, geo


class RGBPhotometricAug:
    """Light photometric jitter on the RGB context image (float32 HWC). No geometry."""

    def __init__(self, gain=0.2, bias=15.0, noise=4.0):
        self.gain, self.bias, self.noise = gain, bias, noise

    def __call__(self, rgb):
        if self.gain > 0:
            g = np.random.uniform(1 - self.gain, 1 + self.gain, size=(1, 1, 3)).astype(np.float32)
            rgb = rgb * g
        if self.bias > 0:
            rgb = rgb + np.random.uniform(-self.bias, self.bias, size=(1, 1, 3)).astype(np.float32)
        if self.noise > 0:
            rgb = rgb + np.random.normal(0, self.noise, rgb.shape).astype(np.float32)
        return rgb


class ToTensorResize:
    """np.float32 HWC -> normalized CHW tensor, bilinear-resized to out_hw."""

    def __init__(self, out_hw, mean, std):
        self.out_hw = tuple(out_hw)
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, arr):
        t = torch.from_numpy(np.ascontiguousarray(arr)).float()
        if t.ndim == 2:
            t = t.unsqueeze(-1)
        t = t.permute(2, 0, 1)                          # HWC -> CHW
        if (t.shape[1], t.shape[2]) != self.out_hw:
            t = F.interpolate(t.unsqueeze(0), size=self.out_hw,
                              mode="bilinear", align_corners=False).squeeze(0)
        return (t - self.mean) / self.std
