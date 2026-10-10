"""Session-2 press-level split (sensor A, second capture session).
Per object, the presses (run, iteration) are sorted and alternated: even -> UNLABELED POOL (all contact frames, no pose used),
odd -> TEST HALF (only the curated frames of real_filtered_new_curated_pred_r2_fv that belong to these presses).
Writes: s2_pool_presses.json (for build_real_filtered_new.py --keep-presses), the test-half mirror root
gs_blender/real_s2_testhalf (symlinks, re-indexed), and s2_poolcheck root (curated frames of POOL presses, label-quality check only)."""
import collections, glob, json, os, os.path as osp, shutil, sys
sys.path.insert(0, "scripts")
import build_real_filtered_new as B
T = "/media/hdd2/ihsuan/VisTacFusion/new_real_test"; G = "/media/hdd2/ihsuan/gs_blender"; FV = f"{G}/real_filtered_new_curated_pred_r2_fv"
sim2real = {v[0]: k for k, v in B.OBJ_MAP.items()}
allp = collections.defaultdict(set)                       # real_name -> {(run, iteration)} with at least one contact frame
for real_name in sorted(B.OBJ_MAP):
    for run in sorted((d for d in os.listdir(f"{T}/{real_name}") if d.isdigit()), key=int) if osp.isdir(f"{T}/{real_name}") else []:
        tp = f"{T}/{real_name}/{run}/tactile_images/poses.json"
        if osp.exists(tp):
            for f in json.load(open(tp))["frames"]:
                if f.get("actual_press_depth_m", 0) * 1000 >= 0.3:
                    allp[real_name].add((int(run), int(f["iteration"])))
pool, test = {}, collections.defaultdict(set)
for real_name, ps in allp.items():
    for i, (run, it) in enumerate(sorted(ps)):
        if i % 2 == 0:
            pool.setdefault(f"{real_name}/{run}", []).append(it)
        else:
            test[real_name].add((run, it))
json.dump(pool, open("ablation/pilot_robosoft/s2_pool_presses.json", "w"), indent=0)
def mirror(root, want_pool):
    shutil.rmtree(root, ignore_errors=True); n = collections.Counter()
    for unit in sorted(glob.glob(f"{FV}/*/session_*/sensor_0000")):
        obj, sess = unit.split("/")[-3], unit.split("/")[-2]; real_name = sim2real[obj]; k = 0
        for pj in sorted(glob.glob(f"{unit}/raw_data/*_pose.json")):
            s = json.load(open(pj))["source"]; key = (int(s["run"]), int(s["iteration"]))
            if (key in test[real_name]) == want_pool:
                continue
            idx = osp.basename(pj)[:4]; ou = f"{root}/{obj}/{sess}/sensor_0000"
            for sub in ("samples", "raw_data", "rgb"):
                os.makedirs(f"{ou}/{sub}", exist_ok=True)
            for src, dst in ((f"samples/{idx}.png", f"samples/{k:04d}.png"), (f"raw_data/{idx}_gt.npy", f"raw_data/{k:04d}_gt.npy"),
                             (f"raw_data/{idx}_pose.json", f"raw_data/{k:04d}_pose.json"), (f"rgb/{idx}.png", f"rgb/{k:04d}.png")):
                if osp.exists(f"{unit}/{src}"):
                    os.symlink(osp.realpath(f"{unit}/{src}"), f"{ou}/{dst}")
            k += 1; n[obj] += 1
        if k:
            sj = json.load(open(f"{osp.dirname(unit)}/session.json")); sj["NUM_OBJ_SAMPLES"] = k
            if "valid_cells" in sj:
                sj["valid_cells"] = sj["valid_cells"][:k]
            json.dump(sj, open(f"{root}/{obj}/{sess}/session.json", "w"))
    return n
nt = mirror(f"{G}/real_s2_testhalf", want_pool=False); npq = mirror(f"{G}/real_s2_poolcheck", want_pool=True)
print("presses per object (pool / test half):", {k: (sum(len(v) for kk, v in pool.items() if kk.startswith(k + "/")), len(test[k])) for k in allp})
print("test-half curated frames:", sum(nt.values()), dict(nt)); print("pool-press curated frames (label check only):", sum(npq.values()))
