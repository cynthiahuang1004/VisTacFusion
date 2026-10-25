"""Fig. 6 with the four non-ridge sensor-A objects (button, edge, hex key, marble): image grids for two layouts.
Each row: frame | real only | AlignTac | GT, at the object's median AlignTac contact-MAE frame (K=25) unless a frame is
given in fig6_picks.json. The last row of every group shows surface normals (mean angular error over the GT contact in
degrees) instead of depth, as suggested by X. Xu (2026-10-25): depth MAE alone can hide a better reconstruction.
Data: outputs/robosoft_qual/qual_{A,B}.npz (dump_qual_preds.py) and qual_P4.npz (aligntac_newobj/eval_newobj.py).

  layout "wide": three groups side by side (sensor A ridge | sensor A non-ridge | sensor B), four rows each,
                 same height as the current figure.
  layout "tall": top = sensor A ridge | sensor B (two rows each), bottom = the four non-ridge objects (two per group).
usage (cwd = VisTacFusion): python paper_RoboSoft/scripts/fig_qual_plus4.py <out_dir> [wide|tall ...]   (paper: out_dir = paper_RoboSoft/figures/src, then rename image_grid_wide.pdf -> fig6_image_grid.pdf and run python build.py --figures)
writes <out_dir>/image_grid_<layout>.pdf (width 248.4 bp, the size figures/fig_qual.tex places)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
plt.rcParams["pdf.fonttype"] = 42          # as make_paper_figs.py (embedded TrueType)

Q = "outputs/robosoft_qual"; VMAX = 1.2; W_BP = 248.4
RUNS = {"A": ("T_k25_realonly", "T_k25_rf"), "B": ("S_k25_realonly_m", "S_k25_maplnr_m"), "P4": ("P4_k25_realonly", "P4_k25_rf")}
LAYOUTS = {
    "wide": [[[("A", "pattern_01_2_lines_angle_2"), ("A", "pattern_32"), ("A", "pattern_36"), ("A", "pattern_04_3_lines_angle_1")],
              [("P4", "button_v2"), ("P4", "hex_key_v2"), ("P4", "edge_v2"), ("P4", "marble_v2")],
              [("B", "pattern_04_3_lines_angle_1"), ("B", "pattern_33"), ("B", "pattern_06_5_lines_angle_1"), ("B", "pattern_36")]]],
    "tall": [[[("A", "pattern_01_2_lines_angle_2"), ("A", "pattern_36")], [("B", "pattern_04_3_lines_angle_1"), ("B", "pattern_06_5_lines_angle_1")]],
             [[("P4", "button_v2"), ("P4", "hex_key_v2")], [("P4", "edge_v2"), ("P4", "marble_v2")]]],
}
_D = {}


def data(name):
    if name not in _D:
        d = dict(np.load(f"{Q}/qual_{name}.npz"))
        if "obj" not in d:
            d["obj"] = np.array([str(k).split("/")[1] for k in d["keys"]])
        _D[name] = d
    return _D[name]


PICKS = "paper_RoboSoft/figures/src/fig6_picks.json"   # optional {"<set>/<object>": frame key} chosen by the authors


def median_frame(name, obj):
    picks = json.load(open(PICKS)) if os.path.exists(PICKS) else {}
    if f"{name}/{obj}" in picks:
        return list(map(str, data(name)["keys"])).index(picks[f"{name}/{obj}"])
    d = data(name); err = d[f"cmae_{RUNS[name][1]}"]          # select by AlignTac's own error, not by its gain
    ids = sorted(np.nonzero(d["obj"] == obj)[0], key=lambda j: err[j])
    return ids[len(ids) // 2]


def grid(layout, out):
    L = LAYOUTS[layout]; n_gr, n_gc = len(L), len(L[0]); n_r = len(L[0][0])
    gap_c, gap_r, sp = 0.05, 0.06, 0.012                       # group gaps / tile spacing, in tile units
    tile = W_BP / (n_gc * 4 + (n_gc - 1) * gap_c * 4 + n_gc * 3 * sp)
    h_bp = tile * (n_gr * n_r + (n_gr - 1) * gap_r * n_r + n_gr * (n_r - 1) * sp)
    fig = plt.figure(figsize=(W_BP / 72, h_bp / 72))
    outer = fig.add_gridspec(n_gr, n_gc, left=0, right=1, top=1, bottom=0, wspace=gap_c, hspace=gap_r)
    fs = 4.6 if layout == "tall" else 3.6
    for gi in range(n_gr):
        for gj in range(n_gc):
            inner = outer[gi, gj].subgridspec(n_r, 4, wspace=sp, hspace=sp * 4 / 4)
            for r, (name, obj) in enumerate(L[gi][gj]):
                d = data(name); i = median_frame(name, obj); r0, r1 = RUNS[name]; normals = r == n_r - 1
                vis = lambda nm: ((nm.astype(np.float32) + 1) / 2).clip(0, 1)                       # unit normal -> RGB
                ims = [d["img"][i]] + ([vis(d[f"normal_{r0}"][i]), vis(d[f"normal_{r1}"][i]), vis(d["gt_normal"][i])] if normals
                                       else [d[f"pred_{r0}"][i], d[f"pred_{r1}"][i], d["gt"][i]])
                for c, im in enumerate(ims):
                    ax = fig.add_subplot(inner[r, c])
                    ax.imshow(im, interpolation="none") if (c == 0 or normals) else ax.imshow(im.astype(np.float32), cmap="viridis", vmin=0, vmax=VMAX, interpolation="none")
                    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                    for s in ax.spines.values():
                        s.set_visible(False)
                    if c in (1, 2):
                        run = (r0, r1)[c - 1]
                        txt = f"{float(d[f'nerr_{run}'][i]):.0f}°" if normals else f"{float(d[f'cmae_{run}'][i]):.2f}"
                        ax.text(0.96, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=fs, color="white",
                                path_effects=[pe.withStroke(linewidth=0.8, foreground="black")])
                print(layout, name, obj, "frame", d["keys"][i], "real only %.3f AlignTac %.3f" % (d[f"cmae_{r0}"][i], d[f"cmae_{r1}"][i]),
                      "normals %.1f / %.1f deg" % (d[f"nerr_{r0}"][i], d[f"nerr_{r1}"][i]), "(normal row)" if normals else "")
    path = os.path.join(out, f"image_grid_{layout}.pdf"); fig.savefig(path, dpi=600); plt.close(fig)   # images embedded at native 224 px (interpolation="none")
    print(path, f"{W_BP:.1f} x {h_bp:.1f} bp")


if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    for lay in sys.argv[2:] or ["wide", "tall"]:
        grid(lay, out)
