"""A copy of the real root whose TRAIN frame indices are randomly permuted (validation indices, idx % 10 == 0, keep their
frames).  Every component of the paper selects its K labeled frames as the first K evenly spaced TRAIN frames of each
unit, so pointing the configs at this root re-runs the whole pipeline with a different random K-frame selection
while keeping the validation split identical.
usage: python scripts/build_selection_root.py <seed>   -> /media/hdd2/ihsuan/gs_blender/real_filtered_sel<seed>
"""
import glob, os, os.path as osp, random, sys
SRC = "/media/hdd2/ihsuan/gs_blender/real_filtered"; seed = int(sys.argv[1]); DST = f"{SRC}_sel{seed}"
rng = random.Random(seed); n_units = 0
for unit in sorted(glob.glob(f"{SRC}/*/session_*/sensor_0000")):
    out = unit.replace(SRC, DST); os.makedirs(f"{out}/samples", exist_ok=True); os.makedirs(f"{out}/raw_data", exist_ok=True); os.makedirs(f"{out}/rgb", exist_ok=True)
    for name in ("calibration", "id_map.json"):
        if osp.lexists(f"{unit}/{name}") and not osp.lexists(f"{out}/{name}"): os.symlink(f"{unit}/{name}", f"{out}/{name}")
    sj = osp.join(osp.dirname(unit), "session.json")
    if osp.exists(sj) and not osp.lexists(osp.join(osp.dirname(out), "session.json")): os.symlink(sj, osp.join(osp.dirname(out), "session.json"))
    idx = sorted(int(osp.basename(p)[:4]) for p in glob.glob(f"{unit}/samples/*.png"))
    train = [i for i in idx if i % 10 != 0]; perm = train[:]; rng.shuffle(perm)
    mapping = {i: i for i in idx if i % 10 == 0}; mapping.update(dict(zip(train, perm)))   # new idx -> old idx
    for new, old in mapping.items():
        for sub, pat in (("samples", "{:04d}.png"), ("rgb", "{:04d}.png")):
            s = f"{unit}/{sub}/" + pat.format(old)
            if osp.exists(s): os.symlink(s, f"{out}/{sub}/" + pat.format(new))
        for p in glob.glob(f"{unit}/raw_data/{old:04d}*"):
            os.symlink(p, f"{out}/raw_data/{new:04d}" + osp.basename(p)[4:])
    n_units += 1
print(f"{DST}: {n_units} units, train indices permuted with seed {seed}, val frames unchanged")
