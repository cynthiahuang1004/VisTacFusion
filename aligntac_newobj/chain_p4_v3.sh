#!/bin/bash
# Four-GPU schedule for the 16-object re-run (replaces chain_p4_v2.sh after its seed-0 teacher had started on GPU 1).
#   GPU 1: real-only seed 0 (already running) -> teacher labels -> aligner -> aligned labels -> AlignTac seed 0 -> evaluation
#   GPU 2: real-only seed 1 -> AlignTac seed 1         GPU 3: real-only seed 2 -> AlignTac seed 2
#   GPU 0: self-training (as soon as the teacher labels exist)
# Teammates' jobs share GPUs 0, 2, 3; nothing of theirs is touched. usage: bash aligntac_newobj/chain_p4_v3.sh
cd /media/hdd2/ihsuan/VisTacFusion
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=/media/hdd2/ihsuan/gs_blender; L=outputs/pilot_logs; PY="conda run --no-capture-output -n vistacfusion python"; P=ablation/pilot_robosoft
M=ablation/encoder/tac_sitr_single.yaml; O=/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2; K=25
step() { echo "$(date '+%m-%d %H:%M') [p4v3] $*"; }
train() {  # <gpu> <run name> <train yaml> <data yaml>
  [ -f $O/pilot150_$2/final.pt ] && return 0
  CUDA_VISIBLE_DEVICES=$1 $PY -m vistacfusion.engine.train --model $M --train $3 --data $4 --output-dir $O/pilot150_$2 > $L/150ep_$2.log 2>&1
}
wait_final() { until [ -f $O/pilot150_$1/final.pt ]; do grep -q Traceback $L/150ep_$1.log 2>/dev/null && return 1; sleep 120; done; }
labels_ready() { [ -f $G/real_pseudo_p4_k25_ro_al/.done ]; }

step "teachers seed 1 (GPU 2), seed 2 (GPU 3)"
( train 2 P4_k25_realonly_seed1 $P/train_pilot_e150_pm_seed1.yaml $P/data_P4_k25_realonly.yaml || exit 1
  until labels_ready; do sleep 120; done; train 2 P4_k25_rf_seed1 $P/train_pilot_e150_pm_seed1.yaml $P/data_P4_k25_rf.yaml ) & s1=$!
( train 3 P4_k25_realonly_seed2 $P/train_pilot_e150_pm_seed2.yaml $P/data_P4_k25_realonly.yaml || exit 1
  until labels_ready; do sleep 120; done; train 3 P4_k25_rf_seed2 $P/train_pilot_e150_pm_seed2.yaml $P/data_P4_k25_rf.yaml ) & s2=$!

step "waiting for teacher seed 0 (GPU 1)"
wait_final P4_k25_realonly || { step "teacher FAILED"; exit 1; }
export CUDA_VISIBLE_DEVICES=1
step "teacher labels + pose (aligner source)"
$PY scripts/tactile_render/pseudo_label.py --run P4_k25_realonly --k $K --out-root $G/real_pseudo_p4_k25_ro_ng --device cuda:0 --pose --link-val --iou-thr 0 > $L/pseudo_p4_k25_ro_ng.log 2>&1 || { step "label FAILED"; exit 1; }
step "self-training (GPU 0)"
train 0 P4_k25_semipm $P/train_pilot_e150_pm.yaml $P/data_P4_k25_semipm.yaml & st=$!
step aligner
[ -f outputs/aligner_p4_k25_realonly/aligner.pt ] || $PY scripts/tactile_render/train_aligner.py --k $K --gan outputs/tactile_ganT --out outputs/aligner_p4_k25_realonly --steps 8000 --bs-real 64 --bs-sim 0 --real-root $G/real_filtered_plus4 > $L/aligner_p4_k25_realonly.log 2>&1 || { step "aligner FAILED"; exit 1; }
step "aligned mesh labels"
$PY scripts/tactile_render/pseudo_label_rc.py --src-root $G/real_pseudo_p4_k25_ro_ng --run P4_k25_realonly --gan outputs/tactile_ganT --aligner outputs/aligner_p4_k25_realonly/aligner.pt --out-root $G/real_pseudo_p4_k25_ro_al --keep-frac 1.0 --pose-all --eval-gt --real-root $G/real_filtered_plus4 --device cuda:0 > $L/pseudo_p4_k25_ro_al.log 2>&1 || { step "rc FAILED"; exit 1; }
touch $G/real_pseudo_p4_k25_ro_al/.done; grep -A3 "^labels" $L/pseudo_p4_k25_ro_al.log
step "AlignTac seed 0 (GPU 1); seeds 1, 2 start on GPUs 2, 3 when their teachers are done"
train 1 P4_k25_rf $P/train_pilot_e150_pm.yaml $P/data_P4_k25_rf.yaml || { step "student FAILED"; exit 1; }
for p in $s1 $s2 $st; do wait $p || { step "a training run FAILED (pid $p)"; exit 1; }; done
step "evaluation"
$PY aligntac_newobj/eval_newobj.py --device cuda:0 --json aligntac_newobj/eval_newobj_v2.json P4_k25_realonly P4_k25_realonly_seed1 P4_k25_realonly_seed2 P4_k25_rf P4_k25_rf_seed1 P4_k25_rf_seed2 P4_k25_semipm > $L/eval_newobj_v2.log 2>&1 || { step "eval FAILED"; exit 1; }
step "chain done"
