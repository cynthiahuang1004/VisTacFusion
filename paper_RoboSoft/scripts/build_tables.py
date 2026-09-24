"""Build the LaTeX table bodies of the RoboSoft paper from the result files.

Sources (all produced by ablation/pilot_robosoft/*):
  <OUT>/pilot150_<run>/history.json  depth_mse / normal_mse (best val depth epoch), rot / trans (best val rot epoch),
                                     and the same for the held-out TEST objects (A runs)
  masked_results.json                contact MAE, IoU, normal angular error (contact region) on val / TEST
  cross_session_results.json         curated cross-session set (68 frames)
  sensorB_results.json               GelSlim 4.0 test set (1620 frames)
  cross_sensor_results.json          zero-shot on GelSlim 4.0
  renderer_fidelity_all.log          contact-region L1 of every learned renderer
Writes paper_RoboSoft/data/tables.tex with one \\newcommand per table body and prints a summary.
usage: python paper_RoboSoft/scripts/build_tables.py
"""
import json
import os.path as osp
import re
import statistics as st

OUT = "/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2"
P = "ablation/pilot_robosoft"
MASKED = json.load(open(f"{P}/masked_results.json"))
CROSS = json.load(open(f"{P}/cross_session_results.json"))
SB = json.load(open(f"{P}/sensorB_results.json"))
XS = json.load(open(f"{P}/cross_sensor_results.json"))
GTF = json.load(open(f"{P}/cross_session_gtfree.json"))


def hist(run, prefix="pilot150_"):
    p = f"{OUT}/{prefix}{run}/history.json"
    if not osp.exists(p):
        return None
    h = json.load(open(p))
    g = lambda e, s: next(iter(e[s].values()))
    bd = min(h, key=lambda e: g(e, "val")["depth_mse"])
    br = min(h, key=lambda e: g(e, "val")["pose_rot_deg"])
    r = {"val": {"depth_mse": g(bd, "val")["depth_mse"], "normal_mse": g(bd, "val")["normal_mse"],
                 "rot": g(br, "val")["pose_rot_deg"], "trans": g(br, "val")["pose_trans"], "epochs": len(h)}}
    if "test" in bd and bd["test"]:
        r["TEST"] = {"depth_mse": g(bd, "test")["depth_mse"], "normal_mse": g(bd, "test")["normal_mse"],
                     "rot": g(br, "test")["pose_rot_deg"], "trans": g(br, "test")["pose_trans"]}
    return r


def row(run, split="val", prefix=None):
    """Merged metrics for one run: 150-ep if available else 50-ep pilot (flagged)."""
    for pre in ([prefix] if prefix else ["pilot150_", "pilot_"]):
        h = hist(run, pre)
        if h and h["val"]["epochs"] < 50:      # paused/incomplete 150-ep run -> fall back to the 50-ep pilot
            h = None
        if h and split in h:
            m = MASKED.get(f"{pre}{run}/{split}", {})
            return dict(h[split], c_mae=m.get("c_mae"), iou=m.get("iou"), nrm=m.get("normal_deg"),
                        peak=m.get("peak_err"), ep=h["val"]["epochs"], prefix=pre)
    return None


def f(v, nd=4):
    return "--" if v is None else f"{v:.{nd}f}"


def cell(v, nd, best=False, italic=False):
    s = f(v, nd)
    if italic:
        return f"\\textit{{{s}}}"
    return f"\\textbf{{{s}}}" if best else s


def table(rows, cols, bold_min=None, bold_max=()):
    """rows: list of (label, metrics dict|None, flags). cols: list of (key, nd). Bold best over non-leaky rows."""
    best = {}
    for key, _ in cols:
        vals = [(r[1][key], i) for i, r in enumerate(rows) if r[1] and r[1].get(key) is not None and not r[2].get("leaky") and not r[2].get("ref")]
        if vals:
            best[key] = (max if key in bold_max else min)(vals)[1]
    lines = []
    for i, (label, m, flags) in enumerate(rows):
        if m is None:
            if flags.get("sep"):
                lines.append(flags["sep"]); continue
            lines.append(f"{label} & " + " & ".join("--" for _ in cols) + r" \\"); continue
        cells = [cell(m.get(k), nd, best=(best.get(k) == i), italic=flags.get("leaky", False)) for k, nd in cols]
        if flags.get("leaky"):   # italicise only the text part of a "K & label" cell
            lab = " & ".join(p if j < label.count("&") else f"\\textit{{{p.strip()}}}" for j, p in enumerate(label.split("&"))) if "&" in label else f"\\textit{{{label}}}"
        else:
            lab = label
        lines.append(f"{lab} & " + " & ".join(cells) + (r" \\" if not flags.get("note") else f" \\\\ % {flags['note']}"))
    return "\n".join(lines)


def seeds(run, keys=("depth_mse", "c_mae", "iou", "rot")):
    rs = [row(run, prefix="pilot_")] + [row(f"{run}_seed{s}", prefix="pilot_") for s in (1, 2)]
    rs = [r for r in rs if r]
    out = {}
    for k in keys:
        v = [r[k] for r in rs if r.get(k) is not None]
        out[k] = (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v)) if v else None
    return out


T = {}
# ---------------- Table 1: leave-object-out (TEST = 3 unseen objects) ----------------
loo = [
    ("Real only (9 objects)", row("A_realonly", "TEST"), {}),
    ("+ Blender", row("A_blender", "TEST"), {}),
    ("+ Blender, bg-norm.", row("A_blender_bgsub", "TEST"), {}),
    ("+ Blender, 1 rotation / object", row("A_blender_cov1", "TEST"), {}),
    ("+ Blender, 2 rotations / object", row("A_blender_cov2", "TEST"), {}),
    ("+ GAN-LOO", row("A_ganloo", "TEST"), {}),
    ("\\quad w/o held-out meshes (control)", row("A_ganloo_9obj", "TEST"), {}),
    ("+ bg-cond.\\ GAN-LOO", row("A_bgcondloo", "TEST"), {}),
    ("+ GAN-LOO, rotate-then-render", row("A_ganloo_gr", "TEST"), {}),
    ("+ GAN-full (leaky)", row("A_ganfull", "TEST"), {"leaky": True}),
]
cols6 = [("depth_mse", 4), ("c_mae", 3), ("iou", 3), ("nrm", 1), ("rot", 1), ("trans", 3)]
T["loo"] = table(loo, cols6, bold_max=("iou",))
# ---------------- Table 2: K-shot ----------------
ks = []
for K in (0, 10, 25, 50, 100):
    first = True
    for label, run, flags in (("Real only", f"B_k{K}_realonly", {}), ("+ Blender", f"B_k{K}_blender", {}),
                              (f"+ GAN-{K}", f"B_k{K}_gank", {}), ("+ GAN-full (leaky)", f"B_k{K}_ganfull", {"leaky": True})):
        if K == 0 and label in ("Real only", "+ GAN-0"):
            continue
        r = row(run)
        if r is None:
            continue
        lab = (f"{K} & " if first else "  & ") + (label if K else label.replace("+ ", "") + " only")
        fl = dict(flags); fl["note"] = f"{r['prefix']}{run} ({r['ep']} ep)"
        ks.append((lab, r, fl)); first = False
    ks.append(("", None, {"sep": r"\midrule"}))
ks = ks[:-1]
cols5 = [("depth_mse", 4), ("c_mae", 3), ("iou", 3), ("nrm", 1), ("rot", 1)]
# bold per K block
blocks, cur = [], []
for r in ks:
    if r[1] is None:
        blocks.append(cur); cur = []
    else:
        cur.append(r)
blocks.append(cur)
T["kshot"] = "\n\\midrule\n".join(table(b, cols5, bold_max=("iou",)) for b in blocks)
# seeds (50 ep, 3 seeds) for K=25 and K=100
sd = []
for K in (25, 100):
    for label, run in (("Real only", f"B_k{K}_realonly"), (f"+ GAN-{K}", f"B_k{K}_gank")):
        s = seeds(run)
        sd.append(f"{K} & {label} & " + " & ".join(f"{s[k][0]:.{nd}f} $\\pm$ {s[k][1]:.{nd}f}" if s.get(k) else "--" for k, nd in (("depth_mse", 4), ("c_mae", 3), ("iou", 3), ("rot", 1))) + r" \\")
T["seeds"] = "\n".join(sd)
# ---------------- Table 3: cross-session (curated 68 frames) ----------------
def cs(run, pre="pilot150_"):
    r = CROSS.get(f"{pre}{run}/cross_session_curated") or CROSS.get(f"pilot_{run}/cross_session_curated")
    return dict(c_mae=r["c_mae"], iou=r["iou"], nrm=r.get("normal_deg"), rot=r["rot_deg"], rotc=r.get("rot_deg_calib")) if r else None
cross = [
    ("Real only (all objects, session 1)", cs("A_realonly"), {}),
    ("+ Blender", cs("A_blender"), {}),
    ("+ Blender, bg-norm.", cs("A_blender_bgsub"), {}),
    ("+ GAN-LOO", cs("A_ganloo"), {}),
    ("+ bg-cond.\\ GAN-LOO", cs("A_bgcondloo"), {}),
    ("+ GAN-full (leaky)", cs("A_ganfull"), {"leaky": True}),
    ("", None, {"sep": r"\midrule"}),
    ("Blender only ($\\Kreal=0$)", cs("B_k0_blender"), {}),
    ("$\\Kreal=25$ real only", cs("B_k25_realonly"), {}),
    ("$\\Kreal=25$ + GAN-25", cs("B_k25_gank"), {}),
    ("$\\Kreal=100$ real only", cs("B_k100_realonly"), {}),
    ("$\\Kreal=100$ + GAN-100", cs("B_k100_gank"), {}),
]
T["cross"] = table(cross, [("c_mae", 3), ("iou", 3), ("nrm", 1), ("rot", 1), ("rotc", 1)], bold_max=("iou",))
# ---------------- Table 4: sensor B (GelSlim 4.0) ----------------
def sb(run):
    r = SB.get(f"pilot_{run}/sensorB_test")
    return dict(depth_mse=r["full_mse"], c_mae=r["c_mae"], iou=r["iou"], nrm=r["normal_deg"], rot=r["rot_deg"], rotc=r["rot_deg_calib"]) if r else None
def xs(run, pre):
    r = XS.get(f"{pre}{run}/cross_sensor_gs40")
    return dict(depth_mse=r["full_mse"], c_mae=r["c_mae"], iou=r["iou"], nrm=r["normal_deg"], rot=r["rot_deg"], rotc=r["rot_deg_calib"]) if r else None
sbrows = [("0 & Sensor-A model, real only", xs("A_realonly", "pilot150_"), {"ref": True}),
          ("  & Sensor-A model, Blender only", xs("B_k0_blender", "pilot_"), {"ref": True}),
          ("", None, {"sep": r"\midrule"})]
for K in (10, 25, 50, 100):
    first = True
    for label, run in (("Real only", f"S_k{K}_realonly"), ("+ Blender (sensor-A calib.)", f"S_k{K}_blender"), (f"+ GAN-{K} (sensor B)", f"S_k{K}_gan40")):
        r = sb(run)
        if r is None:
            continue
        sbrows.append(((f"{K} & " if first else "  & ") + label, r, {})); first = False
    if K != 100 and any(sb(f"S_k{K}_{k}") for k in ("realonly", "blender", "gan40")):
        sbrows.append(("", None, {"sep": r"\midrule"}))
T["sensorb"] = table(sbrows, [("depth_mse", 4), ("c_mae", 3), ("iou", 3), ("nrm", 1), ("rotc", 1)], bold_max=("iou",))
# ---------------- Table 5: appearance-side variants (K=25 + LOO) ----------------
fid = {}
for line in open("outputs/pilot_logs/renderer_fidelity_all.log") if osp.exists("outputs/pilot_logs/renderer_fidelity_all.log") else []:
    m = re.match(r"(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
    if m:
        fid[m.group(1)] = float(m.group(4))
def var(label, k25run, loorun, fidk, fidloo, pre=None):
    a = row(k25run, prefix=pre) if k25run else None
    b = row(loorun, "TEST", prefix=pre) if loorun else None
    return (label, dict(fk=fid.get(fidk), c_mae=a["c_mae"] if a else None, fl=fid.get(fidloo), c_mae_loo=b["c_mae"] if b else None, rot_loo=b["rot"] if b else None), {})
variants = [
    var("Real only", "B_k25_realonly", "A_realonly", None, None),
    var("GAN-$\\Kreal$ / GAN-LOO", "B_k25_gank", "A_ganloo", "k25", "loo"),
    var("\\quad Blender-pretrained", "B_k25_hyb", "A_hybloo", "hyb_k25", "hyb_loo"),
    var("\\quad difference image", "B_k25_diff", "A_diffloo", "diff_k25", "diff_loo"),
    var("\\quad background-conditioned", "B_k25_bgcond", "A_bgcondloo", "bgcond_k25", "bgcond_loo"),
    var("\\quad physics-conditioned", None, None, "physcond_k25", None),
    var("\\quad physics residual", None, None, "physres_k25", None),
    var("\\quad physics + background", None, None, "physbgcond_k25", None),
    var("Blender", "B_k25_blender", "A_blender", None, None),
    var("Blender, bg-norm.", "B_k25_blender_bgsub", "A_blender_bgsub", None, None),
    var("Real only, mask head + grad.\\ loss", "B_k25_realonly_maskgrad", None, None, None),
    var("GAN-$\\Kreal$, mask head + grad.\\ loss", "B_k25_gank_maskgrad", None, None, None),
    var("Blender pretrain $\\to$ real fine-tune", "B_k25_ft", None, None, None),
    var("\\quad + L2-SP", "B_k25_ft_l2sp", None, None, None),
]
T["variants"] = table(variants, [("fk", 3), ("c_mae", 3), ("fl", 3), ("c_mae_loo", 3), ("rot_loo", 1)])
# ---------------- write ----------------
with open("paper_RoboSoft/data/tables.tex", "w") as fh:
    for k, v in T.items():
        fh.write(f"% ---- {k} ----\n\\newcommand{{\\tab{k}}}{{%\n{v}\n}}\n\n")
print(open("paper_RoboSoft/data/tables.tex").read())
