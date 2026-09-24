#!/bin/bash
# Physics-guided GANs (proposal #1), strict K-shot: K=25 two variants (+ K=10, K=50 later if positive).
#   physcond : G input = depth + xy + Blender render at the same pose (6 ch)
#   physres  : same input, output = Blender render + learned residual
# Then renderer fidelity vs the existing GAN-K25 on the real val pairs that have a physics render.
cd /media/hdd2/ihsuan/VisTacFusion; PY=/home/shared/miniconda3/envs/vistacfusion/bin/python
K=${K:-25}; MS=30000
pick(){ for g in 2 1 3 0; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ $m -lt 40000 ] && { echo $g; return; }; done; echo 2; }
g=$(pick); nohup $PY scripts/tactile_render/train_tactile_gan.py --phys-cond --per-object $K --min-steps $MS --device cuda:$g --out outputs/tactile_gan_physcond_k$K > outputs/pilot_logs/gan_physcond_k$K.log 2>&1 &
echo "$(date +%H:%M) physcond_k$K on GPU$g"; sleep 90
g=$(pick); nohup $PY scripts/tactile_render/train_tactile_gan.py --phys-cond --phys-residual --per-object $K --min-steps $MS --device cuda:$g --out outputs/tactile_gan_physres_k$K > outputs/pilot_logs/gan_physres_k$K.log 2>&1 &
echo "$(date +%H:%M) physres_k$K on GPU$g"
