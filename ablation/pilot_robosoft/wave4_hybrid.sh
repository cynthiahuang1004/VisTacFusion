#!/bin/bash
# Hybrid renderer: Blender-pretrained G/D -> fine-tune on K real pairs -> fidelity -> render sim -> downstream
cd /media/hdd2/ihsuan/VisTacFusion
PY=/home/shared/miniconda3/envs/vistacfusion/bin/python
HELD="pattern_04_3_lines_angle_2 pattern_31_rod pattern_33"
OBJS=$($PY -c "import yaml;print(' '.join(yaml.safe_load(open('ablation/pilot_robosoft/data_A_blender.yaml'))['sim']['include_objects']))")
until [ -f outputs/tactile_gan_simpre/D_final.pt ]; do sleep 60; done
ft(){ # gpu tag extra-args...
  g=$1; t=$2; shift 2
  $PY scripts/tactile_render/train_tactile_gan.py --out outputs/tactile_gan_hyb_$t --device cuda:$g \
     --init outputs/tactile_gan_simpre --lr 1e-4 --epochs 50 --min-steps 10000 "$@" > outputs/pilot_logs/gan_hyb_$t.log 2>&1
  $PY scripts/tactile_render/generate_tactile.py --ckpt outputs/tactile_gan_hyb_$t/G_final.pt --device cuda:$g \
     --out-subdir samples_h_$t --objects $OBJS > outputs/pilot_logs/gen_h_$t.log 2>&1
}
ft 0 k10 --per-object 10 &
ft 1 k25 --per-object 25 &
ft 2 k100 --per-object 100 &
ft 3 loo --exclude-objects $HELD &
wait
$PY scripts/tactile_render/eval_renderer.py --device cuda:0 full=outputs/tactile_gan simpre=outputs/tactile_gan_simpre \
  k10=outputs/tactile_gan_k10 hyb_k10=outputs/tactile_gan_hyb_k10 k25=outputs/tactile_gan_k25 hyb_k25=outputs/tactile_gan_hyb_k25 \
  k100=outputs/tactile_gan_k100 hyb_k100=outputs/tactile_gan_hyb_k100 2>&1 | grep -v -i warn > outputs/pilot_logs/fidelity_val.txt
$PY scripts/tactile_render/eval_renderer.py --device cuda:0 --objects $HELD full_leaky=outputs/tactile_gan \
  loo=outputs/tactile_gan_loo hyb_loo=outputs/tactile_gan_hyb_loo simpre=outputs/tactile_gan_simpre 2>&1 | grep -v -i warn > outputs/pilot_logs/fidelity_heldout.txt
ablation/pilot_robosoft/launch.sh 0 B_k10_hyb
ablation/pilot_robosoft/launch.sh 1 B_k25_hyb
ablation/pilot_robosoft/launch.sh 2 B_k100_hyb
ablation/pilot_robosoft/launch.sh 3 A_hybloo
echo "hybrid downstream launched"
