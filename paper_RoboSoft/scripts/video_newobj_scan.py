"""Scan clips over the four additional objects: consecutive captures (raw index order = consecutive robot moves) of
real_filtered_plus4/<obj>_v2 with the 16-object K=25 real-only and AlignTac students and the GT depth.
usage (cwd = VisTacFusion): python paper_RoboSoft/scripts/video_newobj_scan.py [--n 40] [--device cuda:0]
writes paper_RoboSoft/video/streams/scan_<obj>.{mp4,gif}"""
import argparse, contextlib, io, os, os.path as osp, sys
import numpy as np, torch
from PIL import Image

ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))); sys.path.insert(0, ROOT); sys.path.insert(0, osp.join(ROOT, "paper_RoboSoft", "scripts"))
from video_stream_clips import cmap, up, titled, hstack, band, write_clip, MEAN, STD  # noqa: E402
from vistacfusion.data.dataset import build_datasets  # noqa: E402
from vistacfusion.data.transforms import fixed_center_crop  # noqa: E402
from vistacfusion.engine.train import load_checkpoint  # noqa: E402
from vistacfusion.models.model import build_model  # noqa: E402
from vistacfusion.utils.config import merge_configs  # noqa: E402

OUT = "/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2"; G = "/media/hdd2/ihsuan/gs_blender/real_filtered_plus4"
OBJS = {"button_v2": "button", "hex_key_v2": "hex key", "edge_v2": "block edge", "marble_v2": "marble"}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40); ap.add_argument("--start", type=int, default=200); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); dev = a.device; os.chdir(ROOT)
    cfg = merge_configs("ablation/encoder/tac_sitr_single.yaml", "ablation/pilot_robosoft/train_pilot_e50.yaml", "ablation/pilot_robosoft/data_P4_k25_realonly.yaml")
    with contextlib.redirect_stdout(io.StringIO()):
        omap = build_datasets(cfg)[1]._obj_to_id
        M = {}
        for run in ("P4_k25_realonly", "P4_k25_rf"):
            m = build_model(cfg).to(dev).eval(); load_checkpoint(osp.join(OUT, "pilot150_" + run, "best_depth.pt"), m, device=dev); M[run] = m
    for obj, nice in OBJS.items():
        u = f"{G}/{obj}/session_000/sensor_0000"; oid = torch.tensor([omap[obj]], device=dev); frames = []
        for k in range(a.start, a.start + a.n):
            rgb = np.array(Image.open(f"{u}/samples/{k:04d}.png").convert("RGB")); gt = fixed_center_crop(np.load(f"{u}/raw_data/{k:04d}_gt.npy") * 1e3, crop=0.816)
            x = torch.from_numpy(((fixed_center_crop(rgb.astype(np.float32), crop=0.816) - MEAN) / STD).transpose(2, 0, 1)).float()[None].to(dev)
            d = {r: m(x, x, config="tactile", object_ids=oid)["depth"][0, 0].float().clamp(min=0).cpu().numpy() for r, m in M.items()}
            mae = {r: float(np.abs(d[r] - gt)[gt > 0].mean()) if (gt > 0).any() else 0.0 for r in d}
            tiles = [titled(up(fixed_center_crop(rgb, crop=0.816)), nice), titled(cmap(d["P4_k25_realonly"]), f"real only   MAE {mae['P4_k25_realonly']:.2f} mm"),
                     titled(cmap(d["P4_k25_rf"]), f"AlignTac   MAE {mae['P4_k25_rf']:.2f} mm"), titled(cmap(gt), "ground truth")]
            frames.append(band(hstack(tiles), f"consecutive captures {k - a.start + 1}/{a.n}   (K = 25, models trained on all 16 objects)", bottom=True))
        write_clip(frames, f"scan_{obj}", fps=3); print(obj, len(frames))


if __name__ == "__main__":
    main()
