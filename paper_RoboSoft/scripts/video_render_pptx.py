"""Render a hand-edited deck (pptx) to the silent supplementary mp4, and check that every embedded clip plays.
Every movie object (mp4) and every animated-GIF picture in the deck is a clip: a poster copy of the deck replaces each clip by its
first frame (LibreOffice renders movie objects as placeholders), the poster deck is rendered to 1920x1080 backgrounds, and the
clips are overlaid and looped at their positions. Seconds per slide come from the notes ("<seconds>" or "<seconds> | ...").
usage (cwd = VisTacFusion): python paper_RoboSoft/scripts/video_render_pptx.py "paper_RoboSoft/video/AlignTac Video.pptx" --out paper_RoboSoft/video/AlignTac_video.mp4"""
import argparse, glob, json, os, os.path as osp, re, shutil, subprocess
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.util import Emu

NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main", "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
FADE = 0.3; DEFAULT_SEC = 16


def run(cmd): subprocess.run(cmd, check=True)


def group_tf(g, parent):
    """(scale_x, scale_y, off_x, off_y) mapping a child's coordinates of group g to slide EMU."""
    sx, sy, ox, oy = parent
    xf = g._element.xpath("./p:grpSpPr/a:xfrm")[0]; off = xf.find("a:off", NS); ext = xf.find("a:ext", NS); cho = xf.find("a:chOff", NS); che = xf.find("a:chExt", NS)
    gx, gy, gw, gh = int(off.get("x")), int(off.get("y")), int(ext.get("cx")), int(ext.get("cy"))
    cx, cy, cw, ch = int(cho.get("x")), int(cho.get("y")), int(che.get("cx")), int(che.get("cy"))
    kx, ky = (gw / cw if cw else 1.0), (gh / ch if ch else 1.0)
    return (sx * kx, sy * ky, ox + sx * (gx - kx * cx), oy + sy * (gy - ky * cy))


def flatten_groups(slide):
    """Move every grouped shape to the slide at its absolute position (LibreOffice misplaces children of groups whose child
    coordinate space differs from the group's own extent); z-order is kept by inserting the children where the group was."""
    def leaves(g, tf):
        for c in list(g.shapes):
            if c.shape_type == 6:
                yield from leaves(c, group_tf(c, tf))
            else:
                sx, sy, ox, oy = tf
                yield c, int(ox + sx * c.left), int(oy + sy * c.top), int(sx * c.width), int(sy * c.height)
    for g in [sh for sh in slide.shapes if sh.shape_type == 6]:
        for c, x, y, w, h in leaves(g, group_tf(g, (1.0, 1.0, 0.0, 0.0))):
            el = c._element; g._element.addprevious(el); c.left, c.top, c.width, c.height = x, y, w, h
        g._element.getparent().remove(g._element)


def collect(shapes, tf, out, slide_no, work):
    """Replace clips (movies, animated GIF pictures) by poster pictures in place; record their slide positions (inches) and clip files."""
    for sh in list(shapes):
        if sh.shape_type == 6:                               # group
            collect(sh.shapes, group_tf(sh, tf), out, slide_no, work); continue
        blob = ext = None
        if sh.shape_type == 16:                              # movie object
            vf = sh._element.xpath(".//a:videoFile")[0]; part = sh.part.related_part(vf.get(f"{{{NS['r']}}}link")); blob, ext = part.blob, "mp4"
        elif sh.shape_type == 13 and sh.image.ext.lower() == "gif":
            im = Image.open(__import__("io").BytesIO(sh.image.blob))
            if getattr(im, "n_frames", 1) > 1: blob, ext = sh.image.blob, "gif"
        if blob is None: continue
        k = len(out); src = osp.join(work, f"s{slide_no:02d}_c{k}.{ext}"); open(src, "wb").write(blob)
        clip = src if ext == "mp4" else src[:-4] + ".mp4"
        if ext == "gif":
            run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p", "-r", "30", clip])
        poster = src.rsplit(".", 1)[0] + "_poster.png"; run(["ffmpeg", "-y", "-loglevel", "error", "-i", clip, "-frames:v", "1", poster])
        crop = [float(getattr(sh, f"crop_{k}", 0) or 0) for k in ("left", "top", "right", "bottom")]      # the deck's picture crop (fractions)
        if any(crop):
            im = Image.open(poster); W_, H_ = im.size
            im.crop((int(W_ * crop[0]), int(H_ * crop[1]), int(W_ * (1 - crop[2])), int(H_ * (1 - crop[3])))).save(poster)
        sx, sy, ox, oy = tf; x, y = ox + sx * sh.left, oy + sy * sh.top; w, h = sx * sh.width, sy * sh.height
        out.append(dict(slide=slide_no, clip=clip, name=sh.name, crop=crop, x=Emu(int(x)).inches, y=Emu(int(y)).inches, w=Emu(int(w)).inches, h=Emu(int(h)).inches))
        new = shapes.add_picture(poster, sh.left, sh.top, sh.width, sh.height)      # same container coordinates, same z-order
        sh._element.addprevious(new._element); sh._element.getparent().remove(sh._element)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pptx"); ap.add_argument("--out", default="paper_RoboSoft/video/AlignTac_video.mp4"); ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--work", default="paper_RoboSoft/video/render_pptx"); a = ap.parse_args()
    work = a.work; shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
    prs = Presentation(a.pptx); W_IN = Emu(prs.slide_width).inches; PX = 1920 / W_IN
    placements, secs = [], []
    for i, s in enumerate(prs.slides, 1):
        flatten_groups(s); collect(s.shapes, (1.0, 1.0, 0.0, 0.0), placements, i, work)
        m = re.match(r"\s*(\d+)", s.notes_slide.notes_text_frame.text if s.has_notes_slide else ""); secs.append(int(m.group(1)) if m else DEFAULT_SEC)
    poster_pptx = osp.join(work, "poster.pptx"); prs.save(poster_pptx)
    run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", work, poster_pptx]); run(["pdftoppm", "-r", "144", "-png", osp.join(work, "poster.pdf"), osp.join(work, "bg")])
    bgs = sorted(glob.glob(osp.join(work, "bg-*.png"))); assert len(bgs) == len(secs), (len(bgs), len(secs))
    json.dump(dict(placements=placements, seconds=secs), open(osp.join(work, "placements.json"), "w"), indent=1)
    segs = []
    for i, (bg, sec) in enumerate(zip(bgs, secs), 1):
        clips = [p for p in placements if p["slide"] == i]; seg = osp.join(work, f"seg{i:02d}.mp4")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", str(a.fps), "-i", bg]
        for c in clips: cmd += ["-stream_loop", "-1", "-i", c["clip"]]
        fc = []; last = "0:v"
        for k, c in enumerate(clips, 1):
            w, h = int(round(c["w"] * PX)) // 2 * 2, int(round(c["h"] * PX)) // 2 * 2; x, y = int(round(c["x"] * PX)), int(round(c["y"] * PX))
            l, t, r, b = c["crop"]; cr = f"crop=iw*{1 - l - r:.4f}:ih*{1 - t - b:.4f}:iw*{l:.4f}:ih*{t:.4f}," if any(c["crop"]) else ""
            fc.append(f"[{k}:v]{cr}scale={w}:{h}:flags=lanczos,setsar=1[c{k}]"); fc.append(f"[{last}][c{k}]overlay={x}:{y}[o{k}]"); last = f"o{k}"
        fc.append(f"[{last}]scale=1920:1080,format=yuv420p,trim=duration={sec},setpts=PTS-STARTPTS,fade=t=in:st=0:d={FADE},fade=t=out:st={sec - FADE}:d={FADE}[v]")
        cmd += ["-filter_complex", ";".join(fc), "-map", "[v]", "-t", str(sec), "-r", str(a.fps), "-c:v", "libx264", "-preset", "medium", "-crf", "18", seg]
        run(cmd); segs.append(seg); print(f"slide {i:2d}: {sec:2d} s, {len(clips)} clip(s): {', '.join(c['name'] for c in clips)}", flush=True)
    lst = osp.join(work, "segments.txt")
    with open(lst, "w") as f:
        for seg in segs: f.write(f"file '{osp.abspath(seg)}'\n")
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", a.out])
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a.out], capture_output=True, text=True).stdout)
    print(f"wrote {a.out}  {dur:.1f} s, {osp.getsize(a.out) / 1e6:.1f} MB")
    # ---- playback check: every clip region must change between two moments of its slide
    t0 = 0; bad = 0
    for i, sec in enumerate(secs, 1):
        for c in [p for p in placements if p["slide"] == i]:
            fr = []
            for t in (t0 + 0.6, t0 + min(sec - 0.6, 3.1)):
                png = osp.join(work, "chk.png"); run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", a.out, "-frames:v", "1", png])
                im = np.asarray(Image.open(png).convert("L"), np.float32); x, y = int(c["x"] * PX), int(c["y"] * PX); fr.append(im[y:y + int(c["h"] * PX), x:x + int(c["w"] * PX)])
            d = float(np.abs(fr[0] - fr[1]).mean()); ok = d > 0.5; bad += not ok
            print(f"  slide {i:2d} {c['name']:<22s} frame change {d:5.1f}  {'plays' if ok else 'STATIC?'}")
        t0 += sec
    print("all clips play" if not bad else f"{bad} clip(s) look static")


if __name__ == "__main__":
    main()
