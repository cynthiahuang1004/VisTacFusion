#!/bin/bash
# Evaluation chain for the runs finished on 2026-09-23 (K=50, seeds) + remaining GT-free cross-session rows.
cd /media/hdd2/ihsuan/VisTacFusion
PY=/home/shared/miniconda3/envs/vistacfusion/bin/python
D=cuda:0
echo "== masked eval =="
$PY ablation/pilot_robosoft/eval_masked.py --prefix pilot_ --device $D \
  B_k25_realonly_seed1 B_k25_realonly_seed2 B_k25_gank_seed1 B_k25_gank_seed2 \
  B_k100_realonly_seed1 B_k100_realonly_seed2 B_k100_gank_seed1 B_k100_gank_seed2
echo "== cross-session curated =="
$PY ablation/pilot_robosoft/eval_cross_session.py --device $D \
  B_k50_realonly:pilot_ B_k50_blender:pilot_ B_k50_gank:pilot_ \
  B_k25_realonly_seed1:pilot_ B_k25_realonly_seed2:pilot_ B_k25_gank_seed1:pilot_ B_k25_gank_seed2:pilot_ \
  B_k100_realonly_seed1:pilot_ B_k100_realonly_seed2:pilot_ B_k100_gank_seed1:pilot_ B_k100_gank_seed2:pilot_
echo "== cross-session GT-free =="
$PY ablation/pilot_robosoft/eval_cross_session_gtfree.py --device $D \
  B_k10_realonly:pilot150_ B_k10_blender:pilot150_ B_k10_gank:pilot_ \
  B_k25_realonly:pilot_ B_k25_blender:pilot_ B_k25_gank:pilot_ \
  B_k50_realonly:pilot_ B_k50_blender:pilot_ B_k50_gank:pilot_ \
  B_k100_realonly:pilot150_ B_k100_gank:pilot150_ B_k100_blender:pilot_ \
  A_blender_cov1:pilot_ A_blender_cov2:pilot_ A_ganloo_gr:pilot_ A_realonly:pilot_ A_blender:pilot_
echo "== chain done =="
