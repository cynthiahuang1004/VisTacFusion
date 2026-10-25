"""Render the silent supplementary video from the deck: slide backgrounds (paper_RoboSoft/video/render/bg-NN.png, the
poster deck rendered at 1920x1080) with the embedded clips overlaid and looped at their slide positions
(video/placements.json from video_deck.py), each slide held for its planned seconds, with 0.3 s fades (ffmpeg 4.2 has no xfade).
usage (cwd = VisTacFusion): python paper_RoboSoft/scripts/video_render.py [--out paper_RoboSoft/video/AlignTac_video.mp4]"""
import argparse, glob, json, os, os.path as osp, subprocess

V = "paper_RoboSoft/video"; R = f"{V}/render"; W_IN, H_IN = 13.333, 7.5; PX = 1920 / W_IN; FADE = 0.3


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{V}/AlignTac_video.mp4"); ap.add_argument("--fps", type=int, default=30); a = ap.parse_args()
    P = json.load(open(f"{V}/placements.json")); secs = P["seconds"]; bgs = sorted(glob.glob(f"{R}/bg-*.png")); assert len(bgs) == len(secs), (len(bgs), len(secs))
    segs = []
    for i, (bg, sec) in enumerate(zip(bgs, secs), 1):
        clips = [p for p in P["placements"] if p["slide"] == i]; seg = f"{R}/seg{i:02d}.mp4"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", str(a.fps), "-i", bg]
        for c in clips: cmd += ["-stream_loop", "-1", "-i", c["clip"]]
        fc = []; last = "0:v"
        for k, c in enumerate(clips, 1):
            w, h = int(round(c["w"] * PX)) // 2 * 2, int(round(c["h"] * PX)) // 2 * 2; x, y = int(round(c["x"] * PX)), int(round(c["y"] * PX))
            fc.append(f"[{k}:v]scale={w}:{h}:flags=lanczos,setsar=1[c{k}]"); fc.append(f"[{last}][c{k}]overlay={x}:{y}[o{k}]"); last = f"o{k}"
        fc.append(f"[{last}]format=yuv420p,trim=duration={sec},setpts=PTS-STARTPTS,fade=t=in:st=0:d={FADE},fade=t=out:st={sec - FADE}:d={FADE}[v]")
        cmd += ["-filter_complex", ";".join(fc), "-map", "[v]", "-t", str(sec), "-r", str(a.fps), "-c:v", "libx264", "-preset", "medium", "-crf", "18", seg]
        subprocess.run(cmd, check=True); segs.append(seg); print(f"slide {i:2d}: {sec:2d} s, {len(clips)} clip(s)", flush=True)
    lst = f"{R}/segments.txt"
    with open(lst, "w") as f:
        for seg in segs: f.write(f"file '{osp.abspath(seg)}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", a.out], check=True)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a.out], capture_output=True, text=True).stdout.strip()
    print("wrote", a.out, f"{float(dur):.1f} s,", f"{osp.getsize(a.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
