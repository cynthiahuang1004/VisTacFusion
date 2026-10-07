"""Build the LaTeX table bodies of the RoboSoft paper from the result files.

Sources (all produced by ablation/pilot_robosoft/*):
  <OUT>/pilot150_<run>/history.json  depth_mse / normal_mse (best val depth epoch), rot / trans (best val rot epoch),
                                     and the same for the held-out TEST objects (A runs)
  masked_results.json                contact MAE, IoU, normal angular error (contact region) on val / TEST
  cross_session_results.json         corrected cross-session set (854 frames; cross_session_fv)
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
    ("+ Blender (domain rand.)", row("A_dr", "TEST"), {}),
    ("+ Blender, bg-norm.", row("A_blender_bgsub", "TEST"), {}),
    ("+ Blender, 1 rotation / object", row("A_blender_cov1", "TEST"), {}),
    ("+ Blender, 2 rotations / object", row("A_blender_cov2", "TEST"), {}),
    ("+ GAN-LOO", row("A_ganloo", "TEST"), {}),
    ("\\quad w/o held-out meshes (control)", row("A_ganloo_9obj", "TEST"), {}),
    ("+ bg-cond.\\ GAN-LOO", row("A_bgcondloo", "TEST"), {}),
    ("+ GAN-LOO, rotate-then-render", row("A_ganloo_gr", "TEST"), {}),
    ("+ multi-sensor GAN-LOO", row("A_ms_sitr", "TEST", prefix="pilot_"), {}),
    ("+ Blender $\\oplus$ GAN-LOO", row("A_mixloo", "TEST"), {}),
    ("+ refiner (0 labels)", row("A_refloo", "TEST"), {}),
    ("+ refiner on domain rand.\\ (0 labels, no calib.)", row("A_refdrloo", "TEST"), {}),
    ("+ GAN-full (leaky)", row("A_ganfull", "TEST"), {"leaky": True}),
]
cols6 = [("depth_mse", 4), ("c_mae", 3), ("iou", 3), ("nrm", 1), ("rot", 1), ("trans", 3)]
T["loo"] = table(loo, cols6, bold_max=("iou",))
# ---------------- Table 2: K-shot ----------------
ks = []
for K in (0, 10, 25, 50, 100):
    first = True
    for label, run, flags in (("Real only", f"B_k{K}_realonly", {}), ("+ Blender", f"B_k{K}_blender", {}),
                              (f"+ GAN-{K}", f"B_k{K}_gank", {}),
                              (f"+ Blender $\\oplus$ GAN-{K}", f"B_k{K}_mix", {}),
                              (f"+ refiner-{K} (0 labels)", f"B_k{K}_ref", {}),
                              ("+ refiner-U (0 labels, all unlabeled)", f"B_k{K}_refufull", {}),
                              (f"+ self-trained GAN-{K} (renderer only)", f"B_k{K}_pseudo", {}),
                              ("+ teacher-label self-training, model only", f"B_k{K}_semipm", {}),
                              ("\\quad center window only (no stitching)", f"B_k{K}_semipm_c1", {}),
                              ("+ teacher-label self-training", f"B_k{K}_semipm_pseudo", {}),
                              ("\\quad round 2", f"B_k{K}_semipm_pseudo_r2", {}),
                              ("+ MeshTac, L1 search", f"B_k{K}_rcp", {}),
                              ("+ MeshTac (with renderer)", f"B_k{K}_al2", {}),
                              ("\\quad no synthetic frames", f"B_k{K}_al0", {}),
                              ("+ MeshTac (no renderer)", f"B_k{K}_rf", {}),
                              ("+ GAN-full (leaky)", f"B_k{K}_ganfull", {"leaky": True})):
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
    for label, run in (("Real only", f"B_k{K}_realonly"), (f"+ GAN-{K}", f"B_k{K}_gank"), ("+ teacher-label self-training", f"B_k{K}_semipm_pseudo"), ("+ MeshTac, L1 search", f"B_k{K}_rcp"), ("+ MeshTac (with renderer)", f"B_k{K}_al2"), ("+ MeshTac", f"B_k{K}_rf")):
        s = seeds(run)
        sd.append(f"{K} & {label} & " + " & ".join(f"{s[k][0]:.{nd}f} $\\pm$ {s[k][1]:.{nd}f}" if s.get(k) else "--" for k, nd in (("depth_mse", 4), ("c_mae", 3), ("iou", 3), ("rot", 1))) + r" \\")
T["seeds"] = "\n".join(sd)
# ---------------- verification ablation (K=25, 50 ep): label quality vs downstream ----------------
def labq(root, parts):
    """Quality of a label set against the withheld labels (pseudo_label_rc.py --eval-gt): mean over the given parts."""
    p = f"/media/hdd2/ihsuan/gs_blender/{root}/pseudo_manifest.json"
    if not osp.exists(p):
        return None
    q = json.load(open(p))["stats"]
    v = [q["q_" + k] for k in parts if "q_" + k in q]
    return [sum(x[i] for x in v) / len(v) for i in range(3)] if v else None
def abl(label, run, dep, pose, n=1):
    s = seeds(run) if n > 1 else None
    r = row(run, prefix="pilot_")
    if r is None:
        return None
    g = lambda k, nd: (f"{s[k][0]:.{nd}f}" if s and s.get(k) else f(r.get(k), nd))
    return (f"{label} & {f(dep[0], 2) if dep else '--'} & {f(dep[1], 2) if dep else '--'} & {f(pose, 1) if pose is not None else '--'} & "
            f"{g('depth_mse', 4)} & {g('c_mae', 3)} & {g('iou', 3)} & {g('rot', 1)}" + r" \\")
_t = labq("real_pseudo_k25_rc100", ["teacher (all)"]); _m = labq("real_pseudo_k25_rc100", ["mesh (kept)"])
_h = labq("real_pseudo_k25_rc", ["mesh (kept)", "teacher (rejected)"]); _hk = labq("real_pseudo_k25_rc", ["mesh (kept)"])
_a = labq("real_pseudo_k25_al", ["mesh (kept)"])
T["rcabl"] = "\n".join(x for x in (
    abl("teacher & --", "B_k25_semipm_pseudo", _t, None, 3),
    abl("teacher & teacher", "B_k25_tp", _t, _t[2] if _t else None),
    r"\midrule",
    abl("mesh$^{1/2}$ & L1$^{1/2}$", "B_k25_rc50", _h, _hk[2] if _hk else None),
    abl("mesh & L1", "B_k25_rc100", _m, _m[2] if _m else None),
    abl("mesh$^{1/2}$ & L1", "B_k25_rcp", _h, _m[2] if _m else None, 3),
    r"\midrule",
    abl("mesh & aligner$^{\\ddagger}$", "B_k25_al", _a, _a[2] if _a else None),
    abl("mesh & aligner$^{\\dagger}$", "B_k25_al2", _a, _a[2] if _a else None, 3),
) if x)
# ---------------- Table 3: cross-session (curated 68 frames) ----------------
def cs(run, pre="pilot150_"):
    # corrected session-2 labels only (scripts/fix_session2_view.py; tag cross_session_fv)
    r = CROSS.get(f"{pre}{run}/cross_session_fv") or CROSS.get(f"pilot_{run}/cross_session_fv")
    return dict(c_mae=r["c_mae"], iou=r["iou"], nrm=r.get("normal_deg"), rot=r["rot_deg"], rotc=r.get("rot_deg_calib")) if r else None
cross = [
    ("Real only (9 objects, session 1)", cs("A_realonly"), {}),
    ("+ Blender", cs("A_blender"), {}),
    ("+ Blender (domain rand.)", cs("A_dr"), {}),
    ("+ Blender, bg-norm.", cs("A_blender_bgsub"), {}),
    ("+ GAN-LOO", cs("A_ganloo"), {}),
    ("+ bg-cond.\\ GAN-LOO", cs("A_bgcondloo"), {}),
    ("+ Blender $\\oplus$ GAN-LOO", cs("A_mixloo"), {}),
    ("+ refiner (0 labels)", cs("A_refloo"), {}),
    ("+ GAN-full (leaky)", cs("A_ganfull"), {"leaky": True}),
    ("", None, {"sep": r"\midrule"}),
    ("Blender only ($\\Kreal=0$)", cs("B_k0_blender"), {}),
    ("$\\Kreal=25$ real only", cs("B_k25_realonly"), {}),
    ("$\\Kreal=25$ + Blender", cs("B_k25_blender"), {}),
    ("$\\Kreal=25$ + GAN-25", cs("B_k25_gank"), {}),
    ("$\\Kreal=25$ + refiner-25 (0 labels)", cs("B_k25_ref"), {}),
    ("$\\Kreal=25$ + self-trained GAN-25", cs("B_k25_pseudo"), {}),
    ("$\\Kreal=25$ + teacher-label self-training", cs("B_k25_semipm_pseudo"), {}),
    ("$\\Kreal=25$ + MeshTac, L1 search", cs("B_k25_rcp"), {}),
    ("$\\Kreal=25$ + MeshTac", cs("B_k25_al2"), {}),
    ("$\\Kreal=100$ real only", cs("B_k100_realonly"), {}),
    ("$\\Kreal=100$ + Blender", cs("B_k100_blender"), {}),
    ("$\\Kreal=100$ + GAN-100", cs("B_k100_gank"), {}),
    ("$\\Kreal=100$ + refiner-100 (0 labels)", cs("B_k100_ref"), {}),
    ("$\\Kreal=100$ + self-trained GAN-100", cs("B_k100_pseudo"), {}),
    ("$\\Kreal=100$ + teacher-label self-training", cs("B_k100_semipm_pseudo"), {}),
    ("$\\Kreal=100$ + MeshTac, L1 search", cs("B_k100_rcp"), {}),
    ("$\\Kreal=100$ + MeshTac", cs("B_k100_al2"), {}),
]
T["cross"] = table(cross, [("c_mae", 3), ("iou", 3), ("nrm", 1), ("rot", 1), ("rotc", 1)], bold_max=("iou",))
# ---------------- compact tables of the 8-page paper ----------------
km = []
for K in (10, 25, 50, 100):
    blk = []
    for label, run, flags in (("Real only", f"B_k{K}_realonly", {}), ("+ physical renderer", f"B_k{K}_blender", {}), (f"+ learned renderer", f"B_k{K}_gank", {}),
                              ("+ self-training", f"B_k{K}_semipm_pseudo", {}), ("+ MeshTac (ours)", f"B_k{K}_al2", {}), ("\\quad no renderer", f"B_k{K}_rf", {}),
                              ("+ leaky renderer", f"B_k{K}_ganfull", {"leaky": True})):
        r = row(run)
        if r is None:
            continue
        blk.append(((f"{K} & " if not blk else "  & ") + label, r, flags))
    km.append(table(blk, cols5, bold_max=("iou",)))
T["kshotmain"] = "\n\\midrule\n".join(km)
T["crossmain"] = table([
    ("$\\dagger$ & Real only", cs("A_realonly"), {}), ("  & + physical renderer", cs("A_blender"), {}), ("  & + learned renderer", cs("A_bgcondloo"), {}),
    ("  & + leaky renderer", cs("A_ganfull"), {"leaky": True}), ("", None, {"sep": r"\midrule"}),
    ("25 & Real only", cs("B_k25_realonly"), {}), ("  & + learned renderer", cs("B_k25_gank"), {}), ("  & + self-training", cs("B_k25_semipm_pseudo"), {}),
    ("  & + MeshTac (ours)", cs("B_k25_al2"), {}), ("  & \\quad no renderer", cs("B_k25_rf"), {}), ("", None, {"sep": r"\midrule"}),
    ("100 & Real only", cs("B_k100_realonly"), {}), ("  & + learned renderer", cs("B_k100_gank"), {}), ("  & + self-training", cs("B_k100_semipm_pseudo"), {}),
    ("  & + MeshTac (ours)", cs("B_k100_al2"), {}), ("  & \\quad no renderer", cs("B_k100_rf"), {}),
], [("c_mae", 3), ("iou", 3), ("nrm", 1), ("rot", 1)], bold_max=("iou",))
T["loomain"] = table([
    ("Real only (9 objects)", row("A_realonly", "TEST"), {}), ("+ physical renderer", row("A_blender", "TEST"), {}),
    ("+ learned renderer", row("A_ganloo", "TEST"), {}), ("\\quad w/o held-out meshes", row("A_ganloo_9obj", "TEST"), {}),
    ("\\quad background-conditioned", row("A_bgcondloo", "TEST"), {}), ("+ leaky renderer", row("A_ganfull", "TEST"), {"leaky": True}),
], [("depth_mse", 4), ("c_mae", 3), ("iou", 3), ("rot", 1)], bold_max=("iou",))
_r = labq("real_pseudo_k25_ro_al", ["mesh (kept)"]); _rt = labq("real_pseudo_k25_ro_al", ["teacher (all)"])
T["rcablmain"] = "\n".join(x for x in (
    abl("Teacher depth (self-training)", "B_k25_semipm_pseudo", _t, None, 3),
    abl("Teacher depth + teacher pose", "B_k25_tp", _t, _t[2] if _t else None),
    abl("Mesh at L1-search pose$^{1/2}$", "B_k25_rcp", _h, _m[2] if _m else None, 3),
    abl("Mesh at aligned pose (MeshTac)", "B_k25_al2", _a, _a[2] if _a else None, 3),
    abl("\\quad no renderer anywhere$^{\\ddagger}$", "B_k25_rf", _r, _r[2] if _r else None, 3),
) if x)
# ---------------- test-time pose refinement with the aligner ----------------
_pr = f"{P}/pose_refine_results.json"
PR = json.load(open(_pr)) if osp.exists(_pr) else {}
pr = []
for K in (10, 25, 50, 100):
    cells = []
    for split in ("val", "session2"):
        for run in (f"B_k{K}_realonly", f"B_k{K}_al2"):
            e = PR.get(f"pilot150_{run}/{split}")
            cells += [f(e["rot"], 1), f(e["rot_al"], 1)] if e else ["--", "--"]
    pr.append(f"{K} & " + " & ".join(cells) + r" \\")
T["poseref"] = "\n".join(pr)
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
    for label, run in (("Real only", f"S_k{K}_realonly"), ("+ Blender (sensor-A calib.)", f"S_k{K}_blender"), ("+ Blender (domain rand.)", f"S_k{K}_dr"), (f"+ GAN-{K} (sensor B)", f"S_k{K}_gan40"),
                       ("+ teacher-label self-training, model only", f"S_k{K}_semipm"), ("+ teacher-label self-training", f"S_k{K}_semipm_pseudo")):
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
    var("\\quad self-trained (+ pseudo pairs)", "B_k25_pseudo", None, "pseudo_k25", None),
    var("Refiner, unpaired (0 labels)", "B_k25_ref", "A_refloo", "ref_k25", "ref_loo"),
    var("\\quad + self-regularization", None, "A_refreg5loo", None, "ref_loo_reg5"),
    var("\\quad on domain-rand.\\ render", None, "A_refdrloo", None, "ref_drloo"),
    var("Blender (calibrated)", "B_k25_blender", "A_blender", None, "blender_phys"),
    var("Blender (domain rand.)", "B_k25_dr", "A_dr", None, None),
    var("Blender, bg-norm.", "B_k25_blender_bgsub", "A_blender_bgsub", None, None),
    var("Real only, mask head + grad.\\ loss", "B_k25_realonly_maskgrad", None, None, None),
    var("GAN-$\\Kreal$, mask head + grad.\\ loss", "B_k25_gank_maskgrad", None, None, None),
    var("Blender pretrain $\\to$ real fine-tune", "B_k25_ft", None, None, None),
    var("\\quad + L2-SP", "B_k25_ft_l2sp", None, None, None),
]
T["variants"] = table(variants, [("fk", 3), ("c_mae", 3), ("fl", 3), ("c_mae_loo", 3), ("rot_loo", 1)])
# ---------------- sensor B with mesh-registered labels: MeshTac ----------------
def sbm(run):
    r = SB.get(f"pilot_{run}/sensorB_mesh")
    return dict(depth_mse=r["full_mse"], c_mae=r["c_mae"], iou=r["iou"], nrm=r["normal_deg"], rot=r["rot_deg"]) if r else None
def sbm_seeds(run):
    """mean ± std over seeds 0,1,2 of a sensor-B run (tag sensorB_mesh); returns (dict of means, dict of stds, n)."""
    vs = [SB.get(f"pilot_{run}{sfx}/sensorB_mesh") for sfx in ("", "_seed1", "_seed2")]; vs = [v for v in vs if v]
    keys = {"depth_mse": "full_mse", "c_mae": "c_mae", "iou": "iou", "nrm": "normal_deg", "rot": "rot_deg"}
    if not vs:
        return None, None, 0
    return ({k: st.mean(v[kk] for v in vs) for k, kk in keys.items()},
            {k: (st.pstdev(v[kk] for v in vs) if len(vs) > 1 else 0.0) for k, kk in keys.items()}, len(vs))
def pm(run, key, nd):
    m, sd, n = sbm_seeds(run)
    return "--" if m is None else (f"{m[key]:.{nd}f}" if n < 2 else f"{m[key]:.{nd}f}\\,\\scalebox{{0.8}}{{$\\pm${sd[key]:.{nd}f}}}")
sbl = []
for K in (10, 25, 50, 100):
    blk = [(f"{K} & Real only", sbm(f"S_k{K}_realonly_m"), {})]
    if K == 25:
        blk += [("  & + learned renderer", sbm("S_k25_gan40m"), {}), ("  & + self-training", sbm("S_k25_semipm_m"), {}), ("  & + MeshTac (ours)", sbm("S_k25_mapl_m"), {})]
    blk.append(("  & " + ("\\quad no renderer" if K == 25 else "+ MeshTac (ours), no renderer"), sbm(f"S_k{K}_maplnr_m"), {}))
    sbl.append(table(blk, [("depth_mse", 4), ("c_mae", 3), ("iou", 3), ("nrm", 1), ("rot", 1)], bold_max=("iou",)))
T["sbmapl"] = "\n\\midrule\n".join(sbl)
# seeds version (mean ± std over three seeds) of the real-only vs renderer-free rows, all budgets
T["sbseeds"] = "\n".join(f"{K} & {lab} & " + " & ".join(pm(run, k, nd) for k, nd in (("depth_mse", 4), ("c_mae", 3), ("iou", 3), ("rot", 1))) + r" \\"
                          for K in (10, 25, 50, 100) for lab, run in (("Real only", f"S_k{K}_realonly_m"), ("MeshTac, no renderer", f"S_k{K}_maplnr_m")))
# K-frame selection robustness (K=25, 50 ep): seed-0 selection (evenly spaced) vs two random selections (sel1, sel2)
def mv(run):
    r = MASKED.get(f"pilot_{run}/val"); return (r["full_mse"], r["c_mae"], r["iou"]) if r else None
T["selection"] = "\n".join(f"{lab} & " + " & ".join(f"{a:.4f}" if i == 0 else f"{a:.3f}" for i, a in enumerate(mv(run))) + r" \\"
                            for lab, run in (("Real only, selection 0", "B_k25_realonly"), ("Real only, selection 1", "B_k25_realonly_sel1"), ("Real only, selection 2", "B_k25_realonly_sel2"),
                                             ("MeshTac (no renderer), selection 0", "B_k25_rf"), ("MeshTac (no renderer), selection 1", "B_k25_rf_sel1"), ("MeshTac (no renderer), selection 2", "B_k25_rf_sel2")) if mv(run))

# ---------------- 6-page paper: both sensors in one table, short cross-session table ----------------
ks = []
for K in (10, 25, 50, 100):
    blk = []
    for label, run, flags in (("Real only", f"B_k{K}_realonly", {}), ("+ learned renderer", f"B_k{K}_gank", {}), ("+ self-training", f"B_k{K}_semipm_pseudo", {}),
                              ("+ MeshTac (ours)", f"B_k{K}_al2", {}), ("+ leaky renderer", f"B_k{K}_ganfull", {"leaky": True})):
        r = row(run)
        if r is not None:
            blk.append(((f"{K} & " if not blk else "  & ") + label, r, flags))
    ks.append(table(blk, [("depth_mse", 4), ("c_mae", 3), ("iou", 3), ("rot", 1)], bold_max=("iou",)))
T["kshotshort"] = "\n\\midrule\n".join(ks)
sbs = []
for K in (10, 25, 50, 100):
    blk = [(f"{K} & Real only", sbm(f"S_k{K}_realonly_m"), {}), ("  & + MeshTac (ours)", sbm(f"S_k{K}_maplnr_m"), {})]
    sbs.append(table(blk, [("depth_mse", 4), ("c_mae", 3), ("iou", 3), ("rot", 1)], bold_max=("iou",)))
T["sbshort"] = "\n\\midrule\n".join(sbs)
T["crossshort"] = table([
    ("$\\dagger$ & Real only", cs("A_realonly"), {}), ("  & + learned renderer", cs("A_bgcondloo"), {}), ("", None, {"sep": r"\midrule"}),
    ("25 & Real only", cs("B_k25_realonly"), {}), ("  & + self-training", cs("B_k25_semipm_pseudo"), {}), ("  & + MeshTac (ours)", cs("B_k25_al2"), {}), ("", None, {"sep": r"\midrule"}),
    ("100 & Real only", cs("B_k100_realonly"), {}), ("  & + self-training", cs("B_k100_semipm_pseudo"), {}), ("  & + MeshTac (ours)", cs("B_k100_al2"), {}),
], [("c_mae", 3), ("iou", 3), ("rot", 1)], bold_max=("iou",))
T["crossshortb"] = table([
    ("$\\dagger$ & Real only", cs("A_realonly"), {}), ("", None, {"sep": r"\midrule"}),
    ("25 & Real only", cs("B_k25_realonly"), {}), ("  & Self-train.", cs("B_k25_semipm_pseudo"), {}), ("  & MeshTac", cs("B_k25_al2"), {}), ("", None, {"sep": r"\midrule"}),
    ("100 & Real only", cs("B_k100_realonly"), {}), ("  & Self-train.", cs("B_k100_semipm_pseudo"), {}), ("  & MeshTac", cs("B_k100_al2"), {}),
], [("c_mae", 3), ("iou", 3)], bold_max=("iou",))
def sb_pm_rows(K):
    rows = []
    mr, _, _ = sbm_seeds(f"S_k{K}_realonly_m"); mm, _, _ = sbm_seeds(f"S_k{K}_maplnr_m")
    for label, run, m, other in ((f"{K} & Real only", f"S_k{K}_realonly_m", mr, mm), ("  & MeshTac", f"S_k{K}_maplnr_m", mm, mr)):
        cells = []
        for key, nd, better in (("c_mae", 3, min), ("iou", 3, max)):
            c = pm(run, key, nd).replace("{$\\pm$0.", "{$\\pm$.").replace("scalebox{0.8}", "scalebox{0.72}")   # compact: ±.006
            if m and other and better(m[key], other[key]) == m[key]:
                c = f"\\textbf{{{c}}}"
            cells.append(c)
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    return "\n".join(rows)
T["sbshortb"] = "\n".join(sb_pm_rows(K) for K in (10, 25, 50, 100))
# Split the full result table at budget boundaries for readable two-column floats.
blocks = T["kshot"].split("\n\\midrule\n")
T["kshotfew"] = "\n\\midrule\n".join(blocks[:3])
T["kshotmore"] = "\n\\midrule\n".join(blocks[3:])
# Compact labels keep the ablation table readable without changing any metrics.
for old, new in [("Teacher depth (self-training)", "Teacher depth"),
                 ("Teacher depth + teacher pose", "Teacher depth + pose"),
                 ("Mesh at L1-search pose", "L1-search mesh"),
                 ("Mesh at aligned pose (MeshTac)", "MeshTac"),
                 ("no renderer anywhere", "no renderer")]:
    T["rcablmain"] = T["rcablmain"].replace(old, new)
# ---------------- write ----------------
with open("paper_RoboSoft/data/tables.tex", "w") as fh:
    for k, v in T.items():
        v = re.sub(r"(^|& )\+ (\w)", lambda m: m.group(1) + m.group(2).upper(), v, flags=re.M)   # rows read "Method", not "+ addition"
        v = re.sub(r"(^|& )(\\textit\{)\+ (\w)", lambda m: m.group(1) + m.group(2) + m.group(3).upper(), v, flags=re.M)
        fh.write(f"% ---- {k} ----\n\\newcommand{{\\tab{k}}}{{%\n{v}\n}}\n\n")
print(open("paper_RoboSoft/data/tables.tex").read())
