import sys, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.patheffects as pe
out = sys.argv[1]; d = np.load("outputs/robosoft_qual/qual_P4.npz")
PICKS = json.load(open("paper_RoboSoft/figures/src/fig6_picks.json")); r0, r1 = "P4_k25_realonly", "P4_k25_rf"
names = {"button_v2": "button", "hex_key_v2": "hexkey", "edge_v2": "edge", "marble_v2": "marble"}; ids_all = {}
for obj, tag in names.items():
    ids = sorted(np.nonzero(d["obj"] == obj)[0], key=lambda j: d[f"cmae_{r1}"][j])
    pick = [ids[int(round(q * (len(ids) - 1)))] for q in np.linspace(0.03, 0.97, 16)]
    med = ids[len(ids) // 2]
    if med not in pick: pick[8] = med
    keys = list(map(str, d["keys"])); mine = [keys.index(v) for k, v in PICKS.items() if k == f"P4/{obj}"]
    for i in mine:                                   # the authors' earlier choice (made on the pre-fix GT), kept for comparison
        if i not in pick:
            q = ids.index(i); pick[int(round(q / (len(ids) - 1) * 15))] = i
    fig, axes = plt.subplots(8, 9, figsize=(9 * 1.3, 8 * 1.3), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 0.15, 1, 1, 1, 1], wspace=0.03, hspace=0.18))
    for a in axes.flat: a.axis("off")
    for n, i in enumerate(pick):
        r, c0 = n % 8, (0 if n < 8 else 5)
        for c, im in enumerate([d["img"][i], d[f"pred_{r0}"][i], d[f"pred_{r1}"][i], d["gt"][i]]):
            ax = axes[r, c0 + c]
            ax.imshow(im, interpolation="none") if c == 0 else ax.imshow(im.astype(np.float32), cmap="viridis", vmin=0, vmax=1.2, interpolation="none")
            if c in (1, 2):
                ax.text(0.95, 0.05, f"{d[f'cmae_{(r0, r1)[c-1]}'][i]:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="white", path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])
        lab = f"{tag[0].upper()}{n+1}" + (" (median)" if i == med else "") + (" (your pick)" if i in mine else "")
        axes[r, c0].set_title(lab, fontsize=10, loc="left", pad=2, color="blue" if i in mine else ("red" if i == med else "black"), fontweight="bold")
        ids_all[f"{tag[0].upper()}{n+1}"] = dict(obj=obj, index=int(i), key=str(d["keys"][i]))
    fig.suptitle(f"{tag}: frame | real only | AlignTac | GT   (sorted by AlignTac contact MAE; numbers = contact MAE mm)", fontsize=11, y=0.995)
    fig.savefig(f"{out}/cand_{tag}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
json.dump(ids_all, open(f"{out}/candidates.json", "w"), indent=0); print("ok")
