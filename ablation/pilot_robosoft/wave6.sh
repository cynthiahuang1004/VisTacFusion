#!/bin/bash
# Method chains: (gr) rotate-then-render; (ft) L2-SP fine-tune pair; (bc) bg-conditioned GAN x3.
cd /media/hdd2/ihsuan/VisTacFusion
PY=/home/shared/miniconda3/envs/vistacfusion/bin/python; T=ablation/pilot_robosoft; M=ablation/encoder
OUT=/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2; G=$T/gated_launch.sh
OBJS=$($PY -c "import yaml;print(' '.join(yaml.safe_load(open('$T/data_A_blender.yaml'))['sim']['include_objects']))")
case $1 in
  gr) until grep -q '^done' outputs/pilot_logs/gen_gr_loo.log; do sleep 120; done
      $G A_ganloo_gr A_ganloo_gr ;;
  ft) $G B_k25_ft B_k25_ft $M/tac_sitr_single.yaml $T/train_pilot_e50_ft.yaml --resume $OUT/pilot_B_k0_blender/best_depth.pt --finetune
      $G B_k25_ft_l2sp B_k25_ft $M/tac_sitr_single.yaml $T/train_pilot_e50_ft_l2sp.yaml --resume $OUT/pilot_B_k0_blender/best_depth.pt --finetune ;;
  bc) tag=$2; run=$3
      until [ -f outputs/tactile_gan_bgcond_$tag/D_final.pt ]; do sleep 120; done
      $PY scripts/tactile_render/generate_tactile.py --bg-cond --ckpt outputs/tactile_gan_bgcond_$tag/G_final.pt --device cuda:3 \
         --out-subdir samples_bc_$tag --objects $OBJS > outputs/pilot_logs/gen_bc_$tag.log 2>&1
      $G $run $run ;;
esac
echo "wave6 $* done"
