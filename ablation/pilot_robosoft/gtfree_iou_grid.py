"""Review grid for the reference-free contact IoU: tactile | black-hat ridge mask | predicted contact | overlay (IoU)."""
import contextlib, io, json, math, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "scripts"); sys.path.insert(0, "ablation/pilot_robosoft")
from eval_cross_session_gtfree import new_session_frames, OUT, MODEL, TRAIN  # noqa
from calibrate_new_real_poses import blackhat_mask  # noqa
from vistacfusion.data.dataset import SimVisuoTactileDataset  # noqa
from vistacfusion.data.transforms import ToTensorResize, fixed_center_crop  # noqa
from vistacfusion.engine.train import load_checkpoint  # noqa
from vistacfusion.models.model import build_model  # noqa
from vistacfusion.utils.config import merge_configs  # noqa
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
dev = "cuda:0"; run, prefix = "A_blender", "pilot150_"
frames = new_session_frames({"pattern_04_3_lines_angle_2"})
# 2 frames per object, spread over the run
sel = {}
for fr in frames:
    sel.setdefault(fr["obj"], []).append(fr)
picks = []
for obj, lst in sorted(sel.items()):
    picks += [lst[len(lst) // 4], lst[3 * len(lst) // 4]]
cfg = merge_configs(MODEL, TRAIN, f"ablation/pilot_robosoft/data_{run}.yaml")
with contextlib.redirect_stdout(io.StringIO()):
    old = SimVisuoTactileDataset(cfg, cfg.image_size, augment=False, split="val", data_section="real")
    model = build_model(cfg).to(dev).eval(); load_checkpoint(f"{OUT}/{prefix}{run}/best_depth.pt", model, device=dev)
xf = ToTensorResize((224, 224), cfg.norm.imagenet_mean, cfg.norm.imagenet_std)
fig, axes = plt.subplots(len(picks), 4, figsize=(9.5, 2.3 * len(picks)))
with torch.no_grad():
    for i, fr in enumerate(picks):
        im = np.asarray(Image.open(fr["path"]).convert("RGB")); im_c = np.asarray(Image.fromarray(fixed_center_crop(im, crop=0.816)).resize((224, 224)))
        gm = blackhat_mask(im_c)
        x = xf(fixed_center_crop(im, crop=0.816))[None].to(dev)
        pm = (model(torch.zeros_like(x), x, config="tactile", object_ids=torch.tensor([old._obj_to_id.get(fr["obj"], 0)], device=dev))["depth"][0, 0] > 0.05).cpu().numpy()
        iou = (pm & gm).sum() / max(1, (pm | gm).sum())
        ov = im_c.copy(); ov[gm] = (0.5 * ov[gm] + [127, 127, 0]).astype(np.uint8); ov[pm] = (0.5 * ov[pm] + [0, 0, 127]).astype(np.uint8)
        for ax, a, t in zip(axes[i], [im_c, gm, pm, ov], [f"{fr['obj'][8:]} r{fr['run']} f{fr['frame']}", "black-hat ridge mask", f"pred contact ({run})", f"overlay  IoU={iou:.2f}"]):
            ax.imshow(a, cmap="gray" if a.ndim == 2 else None); ax.set_title(t, fontsize=7); ax.axis("off")
fig.tight_layout(h_pad=0.3, w_pad=0.2); fig.savefig("new_real_test/validation/gtfree_iou_grid.png", dpi=90); print("saved new_real_test/validation/gtfree_iou_grid.png", len(picks))
