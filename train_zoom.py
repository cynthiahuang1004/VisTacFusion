"""
train_zoom.py
=============
Co-training wrapper that ZOOMS sim RGB images on the fly (center crop 1/ZOOM + resize
back to the original size) so the sim camera's mm->pixel scale matches the real camera.
No files are written, no original code is modified: PIL.Image.open used inside
vistacfusion.data.dataset is wrapped; only paths under the SIM root and inside an
'/<rgb_subdir>/' directory are zoomed (real images and tactile/depth are untouched).
Labels are unchanged (physical pose is the same; only the pixel scale changes).

Usage (same args as vistacfusion.engine.train, run from VisTacFusion/):
    GELSIGHT_RGB_ZOOM=1.4 CUDA_VISIBLE_DEVICES=1 python -u train_zoom.py \
      --model ablation/encoder/tac_t3_rgb_mae.yaml --train configs/train_bs32.yaml \
      --data ablation/simqty_gtac/data_ratio_g3s_sim190_transfilt.yaml \
      --output-dir outputs/ratio_g3s_sim190_transfilt_zoom14
"""
import os, sys
import numpy as np
from PIL import Image as _PILImage

ZOOM = float(os.environ.get("GELSIGHT_RGB_ZOOM", "1.4"))
SIM_ROOT = os.environ.get("GELSIGHT_SIM_ROOT", "/media/hdd2/ihsuan/gs_blender/renders_v3")
RGB_SUBDIR = os.environ.get("GELSIGHT_RGB_SUBDIR", "rgb")


def _zoom_center(img, z):
    w, h = img.size
    c = int(round(w / z))
    o = (w - c) // 2
    return img.crop((o, o, o + c, o + c)).resize((w, h), _PILImage.BILINEAR)


class _ImageProxy:
    """Drop-in for the `Image` module inside dataset.py: same attributes, zoomed open()."""

    def __getattr__(self, name):
        return getattr(_PILImage, name)

    def open(self, fp, *a, **k):
        im = _PILImage.open(fp, *a, **k)
        p = str(fp)
        if p.startswith(SIM_ROOT) and f"/{RGB_SUBDIR}/" in p and ZOOM != 1.0:
            im = _zoom_center(im.convert("RGB"), ZOOM)
        return im


if __name__ == "__main__":
    import vistacfusion.data.dataset as ds_mod
    ds_mod.Image = _ImageProxy()
    print(f"[zoom] sim RGB under {SIM_ROOT}/**/{RGB_SUBDIR}/ zoomed x{ZOOM} on the fly "
          f"(center crop {int(round(224/ZOOM))}px -> 224px); real untouched", flush=True)
    import vistacfusion.engine.train as train_mod
    train_mod.main()
