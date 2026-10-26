"""Clips from the session-2 tactile streams (new_real_test/<obj>/<run>/tactile_images, ~2.3 fps, presses with approach,
contact and lift) for the supplementary video. Models: clean-protocol K=25 real-only and AlignTac students
(pilot150_T_k25_{realonly,rf}), the real-only teacher as aligner initializer and outputs/aligner_k25_realonly.
  pred_<obj>_run<r>   frame | real-only depth | AlignTac depth, with the AlignTac pose glyph on the frame
  label_<obj>_run<r>  frame + teacher outline (red) | frame + aligned outline (green) | aligned mesh depth (pseudo-label)
usage (cwd = VisTacFusion): python paper_RoboSoft/scripts/video_stream_clips.py --obj pattern_32 --run 1 --iters 2 3 [--device cuda:0]
writes paper_RoboSoft/video/streams/<name>.{mp4,gif} and the per-frame PNGs."""
import argparse, contextlib, io, json, math, os, os.path as osp, subprocess, sys
import cv2, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont

ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, osp.join(ROOT, "scripts")); sys.path.insert(0, osp.join(ROOT, "scripts", "tactile_render"))
from pseudo_label_rc import B  # noqa: E402
from train_aligner import Aligner, refine  # noqa: E402
from vistacfusion.data.dataset import SimVisuoTactileDataset  # noqa: E402
from vistacfusion.data.transforms import fixed_center_crop  # noqa: E402
from vistacfusion.engine.train import load_checkpoint  # noqa: E402
from vistacfusion.models.model import build_model  # noqa: E402
from vistacfusion.utils.config import merge_configs  # noqa: E402

OUT = "/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2"; RAW = osp.join(ROOT, "new_real_test"); VID = osp.join(ROOT, "paper_RoboSoft", "video", "streams")
HALF_MM = 41.0; VMAX = 1.2; UP = 3; FPS = 4
MEAN = np.array([123.675, 116.28, 103.53], np.float32); STD = np.array([58.395, 57.12, 57.375], np.float32)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def cmap(d):
    """depth (mm, 224x224) -> viridis RGB uint8 at UP scale."""
    x = np.clip(d / VMAX, 0, 1); im = (cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS))[:, :, ::-1]
    return cv2.resize(np.ascontiguousarray(im), None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)


def up(rgb): return cv2.resize(rgb, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)


def outline(im, depth_mm, color, width=7):
    m = (depth_mm > 0).astype(np.uint8); cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cs = [(c * UP).astype(np.int32) for c in cs]
    cv2.drawContours(im, cs, -1, (0, 0, 0), width + 4); cv2.drawContours(im, cs, -1, color, width); return im


def glyph(im, theta, x_mm, y_mm, color=(255, 200, 0), L=18):
    """planar-pose glyph (full-view pixel coordinates of gt_depth_mm): object centre + orientation line."""
    S = 224; u = S / 2 - x_mm * 1e-3 * S / B.VIEW; v = S / 2 + y_mm * 1e-3 * S / B.VIEW
    c, s = math.cos(theta), math.sin(theta); p0 = (int(u * UP), int(v * UP)); p1 = (int((u - L * c) * UP), int((v - L * s) * UP))
    cv2.circle(im, p0, 9, (0, 0, 0), -1); cv2.circle(im, p0, 7, color, -1)
    cv2.arrowedLine(im, p0, p1, (0, 0, 0), 7, tipLength=0.25); cv2.arrowedLine(im, p0, p1, color, 4, tipLength=0.25); return im


def band(im, text, size=28, bottom=False):
    pil = Image.fromarray(im); d = ImageDraw.Draw(pil); f = ImageFont.truetype(FONT, size); h = size + 14
    y0 = pil.height - h if bottom else 0
    d.rectangle([0, y0, pil.width, y0 + h], fill=(255, 255, 255)); d.text((12, y0 + 6), text, fill=(0, 0, 0), font=f); return np.array(pil)


def titled(im, text, size=26):
    """title strip above a tile."""
    pil = Image.fromarray(im); W = pil.width; strip = Image.new("RGB", (W, size + 14), (255, 255, 255)); d = ImageDraw.Draw(strip)
    f = ImageFont.truetype(FONT, size); w = d.textlength(text, font=f); d.text(((W - w) / 2, 6), text, fill=(0, 0, 0), font=f)
    out = Image.new("RGB", (W, pil.height + strip.height)); out.paste(strip, (0, 0)); out.paste(pil, (0, strip.height)); return np.array(out)


def hstack(tiles, gap=8):
    h = max(t.shape[0] for t in tiles); w = sum(t.shape[1] for t in tiles) + gap * (len(tiles) - 1)
    out = np.full((h, w, 3), 255, np.uint8); x = 0
    for t in tiles:
        out[:t.shape[0], x:x + t.shape[1]] = t; x += t.shape[1] + gap
    return out


def write_clip(frames, name, fps=FPS):
    import shutil; shutil.rmtree(osp.join(VID, name), ignore_errors=True); os.makedirs(osp.join(VID, name))   # stale frames would be concatenated
    for i, f in enumerate(frames): Image.fromarray(f).save(osp.join(VID, name, f"{i:04d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", osp.join(VID, name, "%04d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p", "-r", "30", osp.join(VID, f"{name}.mp4")], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", osp.join(VID, name, "%04d.png"),
                    "-vf", f"fps={fps},scale=1280:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse", osp.join(VID, f"{name}.gif")], check=True)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--obj", required=True, help="real name, e.g. pattern_32"); ap.add_argument("--run", default="1")
    ap.add_argument("--iters", nargs="+", type=int, default=[2, 3]); ap.add_argument("--pad", type=int, default=5, help="frames kept before/after each contact")
    ap.add_argument("--pred-sec", nargs=2, type=float, default=None, help="keep this time range [s] of the 4 fps prediction clip")
    ap.add_argument("--label-sec", nargs=2, type=float, default=None, help="keep this time range [s] of the 4 fps labeling clip")
    ap.add_argument("--device", default="cuda:0"); a = ap.parse_args(); dev = a.device; os.chdir(ROOT)
    sim = B.OBJ_MAP[a.obj][0]
    cfg = merge_configs("ablation/encoder/tac_sitr_single.yaml", "ablation/pilot_robosoft/train_pilot_e50.yaml", "ablation/pilot_robosoft/data_T_k25_realonly.yaml")
    with contextlib.redirect_stdout(io.StringIO()):
        ds = SimVisuoTactileDataset(cfg, cfg.image_size, augment=False, split="val", data_section="real"); oid = torch.tensor([ds._obj_to_id[sim]], device=dev)
        M = {}
        for run, ck in (("T_k25_realonly", "best_depth"), ("T_k25_realonly", "best_pose"), ("T_k25_rf", "best_depth"), ("T_k25_rf", "best_pose")):
            m = build_model(cfg).to(dev).eval(); load_checkpoint(osp.join(OUT, "pilot150_" + run, ck + ".pt"), m, device=dev); M[(run, ck)] = m
    A = Aligner().to(dev).eval(); A.load_state_dict(torch.load("outputs/aligner_k25_realonly/aligner.pt", map_location=dev)); B.VIEW = 0.017502; B.get_zmap(sim)
    meta = json.load(open(f"{RAW}/{a.obj}/{a.run}/tactile_images/poses.json"))["frames"]
    keep = set()
    for it in a.iters:
        idx = [i for i, f in enumerate(meta) if f["iteration"] == it and f["is_at_or_below_contact"]]
        if idx: keep |= set(range(max(0, idx[0] - a.pad), min(len(meta), idx[-1] + a.pad + 1)))
    sel = sorted(keep); pred_frames, label_frames = [], []; t0 = meta[sel[0]]["capture_wall_time_sec"]
    for i in sel:
        f = meta[i]; rgb = np.array(Image.open(f"{RAW}/{a.obj}/{a.run}/tactile_images/{f['filename']}").convert("RGB"))
        x = torch.from_numpy(((fixed_center_crop(rgb.astype(np.float32), crop=0.816) - MEAN) / STD).transpose(2, 0, 1)).float()[None].to(dev)
        out = {k: m(x, x, config="tactile", object_ids=oid) for k, m in M.items()}
        d_ro = out[("T_k25_realonly", "best_depth")]["depth"][0, 0].float().clamp(min=0).cpu().numpy()
        d_al = out[("T_k25_rf", "best_depth")]["depth"][0, 0].float().clamp(min=0).cpu().numpy()
        se2 = out[("T_k25_rf", "best_pose")]["se2"][0].float().cpu(); th, px, py = math.atan2(se2[1], se2[0]), float(se2[2]) * HALF_MM, float(se2[3]) * HALF_MM
        contact = bool(f["is_at_or_below_contact"]); tsec = f["capture_wall_time_sec"] - t0
        # ---- prediction clip (cropped view, as the student sees it)
        t1 = up(rgb)                                                # the student's cropped view, no pose glyph
        side = int(224 * UP * 0.816); o = (224 * UP - side) // 2; t1 = cv2.resize(t1[o:o + side, o:o + side], (224 * UP, 224 * UP), interpolation=cv2.INTER_CUBIC)
        tiles = [titled(t1, "tactile stream"), titled(cmap(d_ro), "real only (K = 25)"), titled(cmap(d_al), "AlignTac student (K = 25)")]
        fr = hstack(tiles); fr = band(fr, f"{a.obj.replace('_', ' ')}   press {f['iteration']}   t = {tsec:5.1f} s   indentation {f['actual_press_depth_m'] * 1000:.2f} mm" + ("" if contact else "   (no contact)"), bottom=True)
        pred_frames.append(fr)
        # ---- pseudo-labeling clip (full view: teacher state -> aligned state)
        g1, g2, g3 = up(rgb), up(rgb), np.zeros((224 * UP, 224 * UP, 3), np.uint8)
        if contact:
            se2t = out[("T_k25_realonly", "best_pose")]["se2"][0].float().cpu(); dep = float(d_ro.max())
            p = torch.tensor([[math.atan2(se2t[1], se2t[0]), float(se2t[2]) * HALF_MM, float(se2t[3]) * HALF_MM, min(max(dep, 0.3), 1.6)]], device=dev)
            img = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1)[None].to(dev)
            q0 = p[0].cpu().numpy().tolist(); qT = refine(A, [sim], img, p, dev, iters=6)[0].cpu().numpy().tolist()
            d0 = B.gt_depth_mm(sim, *q0); dT = B.gt_depth_mm(sim, *qT)
            g1 = outline(g1, d0, (255, 60, 60)); g2 = outline(g2, dT, (60, 230, 90)); g3 = cmap(dT)
        tiles = [titled(g1, "teacher initial state"), titled(g2, "after aligner updates"), titled(g3, "pseudo-label (mesh depth)")]
        fr = hstack(tiles); fr = band(fr, f"{a.obj.replace('_', ' ')}   press {f['iteration']}   t = {tsec:5.1f} s" + ("" if contact else "   (no contact)"), bottom=True)
        label_frames.append(fr)
    tag = f"{a.obj}_run{a.run}"
    trim = lambda fr, sec: fr if sec is None else fr[int(round(sec[0] * FPS)):int(round(sec[1] * FPS))]
    write_clip(trim(pred_frames, a.pred_sec), f"pred_{tag}"); write_clip(trim(label_frames, a.label_sec), f"label_{tag}")
    print(tag, len(sel), "frames ->", VID)


if __name__ == "__main__":
    main()
