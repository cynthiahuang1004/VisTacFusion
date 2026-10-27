#!/bin/bash
# Automated chain: wait for current runs to finish, then launch next steps.
cd /media/hdd2/ihsuan/VisTacFusion
PY=/home/shared/miniconda3/envs/vistacfusion/bin/python
OUT=/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2
M_CLS=ablation/encoder/tac_sitr_single_cls.yaml
T_CLS=ablation/pilot_robosoft/train_pilot_e50_cls.yaml

wait_run() { while ps -u shared -o args | grep -q "[o]utput-dir $OUT/pilot_$1\b"; do sleep 60; done; }
pick_gpu() { for g in 0 1 2 3; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ $m -lt 38000 ] && { echo $g; return; }; done; echo 0; }

echo "$(date +%m-%d\ %H:%M) chain started"

# ---- 1. Wait for the 6 classification runs to finish ----
for r in B_k0_blender_cls B_k25_realonly_cls B_k25_gank_cls B_k100_realonly_cls B_k100_gank_cls A_blender_cls; do wait_run $r; done
echo "$(date +%m-%d\ %H:%M) cls runs done"

# ---- 2. Eval classification runs (masked + cross-session + sensor-B) ----
CLS="B_k0_blender_cls B_k25_realonly_cls B_k25_gank_cls B_k100_realonly_cls B_k100_gank_cls A_blender_cls"
g=$(pick_gpu)
$PY ablation/pilot_robosoft/eval_masked.py --prefix pilot_ --device cuda:$g $CLS
$PY ablation/pilot_robosoft/eval_cross_session.py --device cuda:$g --batch 32 --tag cross_session_curated854 $(for r in $CLS; do echo -n "$r:pilot_ "; done)
$PY ablation/pilot_robosoft/eval_cross_session.py --device cuda:$g --batch 16 --root /media/hdd2/ihsuan/gs_blender/real_gelslim40_test --fixed-crop 1.0 --tag cross_sensor_gs40 --json ablation/pilot_robosoft/cross_sensor_results.json $(for r in $CLS; do echo -n "$r:pilot_ "; done)
echo "$(date +%m-%d\ %H:%M) cls eval done"

# ---- 3. Wait for S_k50 runs + eval ----
for r in S_k50_realonly S_k50_blender S_k50_gan40; do wait_run $r; done
echo "$(date +%m-%d\ %H:%M) S_k50 done"
g=$(pick_gpu)
$PY ablation/pilot_robosoft/eval_cross_session.py --device cuda:$g --batch 16 --root /media/hdd2/ihsuan/gs_blender/real_gelslim40_test --fixed-crop 1.0 --tag sensorB_test --json ablation/pilot_robosoft/sensorB_results.json S_k50_realonly:pilot_ S_k50_blender:pilot_ S_k50_gan40:pilot_
echo "$(date +%m-%d\ %H:%M) S_k50 eval done"

# ---- 4. Wait for multi-sensor GANs, eval fidelity ----
for d in outputs/tactile_gan_ms_sitr outputs/tactile_gan_ms_sitr_k25; do
  until [ -f $d/G_final.pt ]; do sleep 60; done
done
echo "$(date +%m-%d\ %H:%M) ms GANs done"
g=$(pick_gpu)
$PY scripts/tactile_render/eval_renderer.py --device cuda:$g ms_sitr=outputs/tactile_gan_ms_sitr ms_sitr_k25=outputs/tactile_gan_ms_sitr_k25 bgcond_loo=outputs/tactile_gan_bgcond_loo bgcond_k25=outputs/tactile_gan_bgcond_k25 | tee outputs/pilot_logs/renderer_fidelity_ms.txt
echo "$(date +%m-%d\ %H:%M) ms fidelity done"

# ---- 5. Generate ms_sitr sim images + downstream LOO run ----
$PY scripts/tactile_render/generate_tactile.py --ckpt outputs/tactile_gan_ms_sitr/G_final.pt --bg-cond --device cuda:$g --out-subdir samples_ms_sitr > outputs/pilot_logs/gen_ms_sitr.log 2>&1
echo "$(date +%m-%d\ %H:%M) ms_sitr samples generated"

# LOO run with the multi-sensor GAN sim (needs a data config)
$PY - <<'EOF'
import yaml, copy
c = yaml.safe_load(open("ablation/pilot_robosoft/data_A_bgcondloo.yaml"))
c["sim"]["tactile_subdir"] = "samples_ms_sitr"
yaml.safe_dump(c, open("ablation/pilot_robosoft/data_A_ms_sitr.yaml", "w"), sort_keys=False)
print("data_A_ms_sitr.yaml written")
EOF
g=$(pick_gpu)
CUDA_VISIBLE_DEVICES=$g nohup $PY -m vistacfusion.engine.train --model ablation/encoder/tac_sitr_single.yaml --train ablation/pilot_robosoft/train_pilot_e50.yaml --data ablation/pilot_robosoft/data_A_ms_sitr.yaml --output-dir $OUT/pilot_A_ms_sitr > outputs/pilot_logs/A_ms_sitr.log 2>&1 &
echo "$(date +%m-%d\ %H:%M) A_ms_sitr launched on GPU$g"

# ---- 6. Rebuild tables ----
wait_run A_ms_sitr
$PY ablation/pilot_robosoft/eval_masked.py --prefix pilot_ --device cuda:$(pick_gpu) A_ms_sitr
$PY ablation/pilot_robosoft/eval_cross_session.py --device cuda:$(pick_gpu) --batch 32 --tag cross_session_curated854 A_ms_sitr:pilot_
$PY paper_RoboSoft/scripts/build_tables.py
echo "$(date +%m-%d\ %H:%M) ALL DONE"
