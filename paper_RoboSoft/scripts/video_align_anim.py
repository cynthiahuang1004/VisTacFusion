"""Aligner-convergence animation for the supplementary video (sensor A, K=25, clean-protocol teacher T_k25_realonly and
aligner outputs/aligner_k25_realonly): on held-out test frames, the mesh outline at the teacher state q0 and after each
of the six aligner updates, drawn over the fixed real image; the GT outline is shown in white.
Writes paper_RoboSoft/video/align_anim/<obj>_<frame>/it0..6.png, a 4-frame strip per iteration, and align_anim.{gif,mp4}.
usage (cwd = VisTacFusion): python paper_RoboSoft/scripts/video_align_anim.py [--device cuda:0]"""
import argparse, contextlib, copy, io, json, math, os, os.path as osp, subprocess, sys
import cv2, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, osp.join(ROOT, "scripts")); sys.path.insert(0, osp.join(ROOT, "scripts", "tactile_render"))
from pseudo_label_rc import B  # noqa: E402
from train_aligner import Aligner, refine  # noqa: E402
from vistacfusion.data.dataset import SimVisuoTactileDataset  # noqa: E402
from vistacfusion.engine.train import load_checkpoint  # noqa: E402
from vistacfusion.models.model import build_model  # noqa: E402
from vistacfusion.utils.config import merge_configs  # noqa: E402

OUT = "/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2"; TEST = "/media/hdd2/ihsuan/gs_blender/real_filtered_testA"
VID = osp.join(ROOT, "paper_RoboSoft", "video", "align_anim"); T = 6; UP = 3          # 224 px -> 672 px tiles
OBJS = ["pattern_06_5_lines_angle_1", "pattern_32", "pattern_36", "pattern_04_3_lines_angle_1"]


def outline(im, depth_mm, color, width=2):
    """Draw the contact outline (depth > 0, 224-px map) on `im`, an UP-times upscaled copy of the frame."""
    m = (depth_mm > 0).astype(np.uint8); cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cs = [(c * UP).astype(np.int32) for c in cs]
    cv2.drawContours(im, cs, -1, (0, 0, 0), width + 4); cv2.drawContours(im, cs, -1, color, width)
    return im


def label(im, text, sub=""):
    pil = Image.fromarray(im); d = ImageDraw.Draw(pil)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30); f2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except OSError:
        f = f2 = ImageFont.load_default()
    d.rectangle([0, 0, pil.width, 44], fill=(0, 0, 0)); d.text((12, 6), text, fill=(255, 255, 255), font=f)
    if sub:
        d.rectangle([0, pil.height - 36, pil.width, pil.height], fill=(0, 0, 0)); d.text((12, pil.height - 32), sub, fill=(255, 255, 255), font=f2)
    return np.array(pil)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--run", default="T_k25_realonly")
    ap.add_argument("--aligner", default="outputs/aligner_k25_realonly/aligner.pt"); a = ap.parse_args(); dev = a.device; os.chdir(ROOT)
    cfg = merge_configs("ablation/encoder/tac_sitr_single.yaml", "ablation/pilot_robosoft/train_pilot_e50.yaml", f"ablation/pilot_robosoft/data_{a.run}.yaml")
    with contextlib.redirect_stdout(io.StringIO()):
        old = SimVisuoTactileDataset(cfg, cfg.image_size, augment=False, split="val", data_section="real")
        c1 = copy.deepcopy(cfg); c1["real"]["root"] = TEST; c1["real"]["val_every"] = 1; c1["real"].pop("exclude_frames", None)
        ds = SimVisuoTactileDataset(c1, cfg.image_size, augment=False, split="all", data_section="real", shared_obj_map=old._obj_to_id)
        mp = build_model(cfg).to(dev).eval(); load_checkpoint(osp.join(OUT, "pilot150_" + a.run, "best_pose.pt"), mp, device=dev)
        md = build_model(cfg).to(dev).eval(); load_checkpoint(osp.join(OUT, "pilot150_" + a.run, "best_depth.pt"), md, device=dev)
    A = Aligner().to(dev).eval(); A.load_state_dict(torch.load(a.aligner, map_location=dev))
    B.VIEW = 0.017502                       # full gel view of the stored frames (importing train_aligner can reset it)
    for o in OBJS: B.get_zmap(o)
    recs = []; i0 = 0
    for b in DataLoader(ds, batch_size=32, num_workers=4):
        n = len(b["tactile"]); units = [ds.samples[i0 + j] for j in range(n)]; i0 += n
        keep = [j for j, (u, _) in enumerate(units) if u.split("/")[-3] in OBJS]
        if not keep: continue
        x = b["tactile"].to(dev)[keep]; oid = b["object"].to(dev)[keep]; units = [units[j] for j in keep]
        se2 = mp(x, x, config="tactile", object_ids=oid)["se2"].float().cpu()
        dep = md(x, x, config="tactile", object_ids=oid)["depth"][:, 0].float().clamp(min=0).amax((1, 2)).cpu()
        objs = [u.split("/")[-3] for u, _ in units]; half = torch.tensor([ds.unit_meta[u]["half"] * 1e3 for u, _ in units])
        img = torch.from_numpy(np.stack([np.array(Image.open(osp.join(u, "samples", f"{s:04d}.png")).convert("RGB"), np.float32).transpose(2, 0, 1) / 127.5 - 1 for u, s in units])).to(dev)
        p = torch.stack([torch.atan2(se2[:, 1], se2[:, 0]), se2[:, 2] * half, se2[:, 3] * half, dep.clamp(0.3, 1.6)], 1).to(dev)
        traj = [p.cpu().numpy().copy()]
        for _ in range(T):
            p = refine(A, objs, img, p, dev, iters=1); traj.append(p.cpu().numpy().copy())
        for j, (u, s) in enumerate(units):
            P = json.load(open(osp.join(u, "raw_data", f"{s:04d}_pose.json"))); g = (P["rotation_euler"][2], P["sample_x"] * 1e3, P["sample_y"] * 1e3)
            gt_mm = np.load(osp.join(u, "raw_data", f"{s:04d}_gt.npy")) * 1000; pg = float(np.clip(gt_mm.max(), 0.3, 1.6))
            m1, m2 = gt_mm > 0, B.gt_depth_mm(objs[j], *g, pg) > 0; iou = float((m1 & m2).sum() / max(1, (m1 | m2).sum()))
            rot = [abs((math.degrees(t[j][0] - g[0]) + 180) % 360 - 180) for t in traj]
            trans = [(abs(t[j][1] - g[1]) + abs(t[j][2] - g[2])) / 2 for t in traj]
            recs.append(dict(obj=objs[j], unit=u, frame=int(s), g=[float(v) for v in g], pg=pg, iou_gt=iou, traj=[[float(v) for v in t[j]] for t in traj], rot=[float(v) for v in rot], trans=[float(v) for v in trans]))
    # one frame per object: a visible but in-range teacher error (rotation 3-8 deg, translation <= 1.5 mm) that the
    # aligner removes (final rotation < 0.7 deg, translation < 0.4 mm); the largest initial rotation error among those
    chosen = []
    for o in OBJS:
        c = [r for r in recs if r["obj"] == o and 3 <= r["rot"][0] <= 8 and r["trans"][0] <= 1.5 and r["rot"][-1] < 0.7 and r["trans"][-1] < 0.4]
        chosen.append(max(c, key=lambda r: r["rot"][0] + r["trans"][0]))
    json.dump(recs, open(osp.join(VID, "all_frames.json"), "w"))
    os.makedirs(VID, exist_ok=True)
    json.dump(chosen, open(osp.join(VID, "chosen.json"), "w"), indent=1)
    strips = []
    for it in range(T + 1):
        tiles = []
        for r in chosen:
            rgb = np.array(Image.open(osp.join(r["unit"], "samples", f"{r['frame']:04d}.png")).convert("RGB"))
            q = r["traj"][it]; d = B.gt_depth_mm(r["obj"], q[0], q[1], q[2], q[3]); dg = B.gt_depth_mm(r["obj"], *r["g"], r["pg"])
            im = cv2.resize(rgb, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
            im = outline(im, dg, (255, 255, 255), 5)                                   # GT outline, white
            t = it / T; col = (int(255 * (1 - t)), int(220 * t), 60)                    # red -> green
            im = outline(im, d, col, 7)
            im = label(im, f"rotation error {r['rot'][it]:.1f} deg,  translation error {r['trans'][it]:.2f} mm", "")
            d_ = osp.join(VID, f"{r['obj']}_{r['frame']:04d}"); os.makedirs(d_, exist_ok=True); Image.fromarray(im).save(osp.join(d_, f"it{it}.png"))
            tiles.append(im)
        strip = np.concatenate(tiles, 1)
        strip = label(strip, f"teacher initial state  ->  aligner update {it} of {T}" if it else "teacher initial state q0 (red); GT outline (white)", "")
        Image.fromarray(strip).save(osp.join(VID, f"strip_it{it}.png")); strips.append(strip)
    # gif / mp4: 0.8 s per update, hold first and last for 1.5 s
    seq = [strips[0]] * 2 + strips[1:] + [strips[-1]] * 2
    Image.fromarray(seq[0]).save(osp.join(VID, "align_anim.gif"), save_all=True, append_images=[Image.fromarray(s) for s in seq[1:]], duration=800, loop=0)
    lst = osp.join(VID, "list.txt")
    with open(lst, "w") as f:
        for it, s in [(0, 1.5)] + [(i, 0.8) for i in range(1, T + 1)] + [(T, 1.5)]:
            f.write(f"file 'strip_it{it}.png'\nduration {s}\n")
        f.write(f"file 'strip_it{T}.png'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p", "-r", "30", osp.join(VID, "align_anim.mp4")], check=True)
    print("GT re-render vs stored GT, mean IoU over frames:", round(float(np.mean([r["iou_gt"] for r in recs])), 3))
    for r in chosen: print(r["obj"], r["frame"], "rot:", [round(v, 2) for v in r["rot"]], "trans:", [round(v, 2) for v in r["trans"]])


if __name__ == "__main__":
    main()
