#!/bin/bash
# Re-run of the 16-object K=25 study after the z-map fix (exact rasterization of the *_v2 meshes, recalibrated placements).
# usage: chain_p4_v2.sh <gpu>      (run after calibrate_newobj.py and build_newobj.py)
# teachers (real only, seeds 0-2) -> teacher labels -> aligner -> aligned mesh labels -> students (AlignTac seeds 0-2, self-training seed 0)
# All outputs on /media/hdd2 (the root disk is full); the earlier runs were renamed *_facemin.
GPU=$1; cd /media/hdd2/ihsuan/VisTacFusion
export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=/media/hdd2/ihsuan/gs_blender; L=outputs/pilot_logs; PY="conda run --no-capture-output -n vistacfusion python"; P=ablation/pilot_robosoft
M=ablation/encoder/tac_sitr_single.yaml; O=/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2; K=25
step() { echo "$(date '+%m-%d %H:%M') [p4v2] $*"; }
train() {  # <run name> <train yaml> <data yaml>
  [ -f $O/pilot150_$1/final.pt ] && return 0
  $PY -m vistacfusion.engine.train --model $M --train $2 --data $3 --output-dir $O/pilot150_$1 > $L/150ep_$1.log 2>&1
}
step "teachers / real only (3 seeds)"
train P4_k25_realonly $P/train_pilot_e150.yaml $P/data_P4_k25_realonly.yaml & a=$!
train P4_k25_realonly_seed1 $P/train_pilot_e150_pm_seed1.yaml $P/data_P4_k25_realonly.yaml & b=$!
train P4_k25_realonly_seed2 $P/train_pilot_e150_pm_seed2.yaml $P/data_P4_k25_realonly.yaml & c=$!
wait $a || { step "teacher FAILED"; exit 1; }
step "teacher labels + pose (aligner source)"
$PY scripts/tactile_render/pseudo_label.py --run P4_k25_realonly --k $K --out-root $G/real_pseudo_p4_k25_ro_ng --device cuda:0 --pose --link-val --iou-thr 0 > $L/pseudo_p4_k25_ro_ng.log 2>&1 || { step "label FAILED"; exit 1; }
step aligner
[ -f outputs/aligner_p4_k25_realonly/aligner.pt ] || $PY scripts/tactile_render/train_aligner.py --k $K --gan outputs/tactile_ganT --out outputs/aligner_p4_k25_realonly --steps 8000 --bs-real 64 --bs-sim 0 --real-root $G/real_filtered_plus4 > $L/aligner_p4_k25_realonly.log 2>&1 || { step "aligner FAILED"; exit 1; }
step "aligned mesh labels"
$PY scripts/tactile_render/pseudo_label_rc.py --src-root $G/real_pseudo_p4_k25_ro_ng --run P4_k25_realonly --gan outputs/tactile_ganT --aligner outputs/aligner_p4_k25_realonly/aligner.pt --out-root $G/real_pseudo_p4_k25_ro_al --keep-frac 1.0 --pose-all --eval-gt --real-root $G/real_filtered_plus4 --device cuda:0 > $L/pseudo_p4_k25_ro_al.log 2>&1 || { step "rc FAILED"; exit 1; }
grep -A3 "^labels" $L/pseudo_p4_k25_ro_al.log
step "students (AlignTac 3 seeds, self-training)"
train P4_k25_rf $P/train_pilot_e150_pm.yaml $P/data_P4_k25_rf.yaml & d=$!
train P4_k25_rf_seed1 $P/train_pilot_e150_pm_seed1.yaml $P/data_P4_k25_rf.yaml & e=$!
train P4_k25_rf_seed2 $P/train_pilot_e150_pm_seed2.yaml $P/data_P4_k25_rf.yaml & f=$!
train P4_k25_semipm $P/train_pilot_e150_pm.yaml $P/data_P4_k25_semipm.yaml & g=$!
for p in $b $c $d $e $f $g; do wait $p || { step "a training run FAILED (pid $p)"; exit 1; }; done
step "evaluation"
$PY aligntac_newobj/eval_newobj.py --device cuda:0 --json aligntac_newobj/eval_newobj_v2.json P4_k25_realonly P4_k25_realonly_seed1 P4_k25_realonly_seed2 P4_k25_rf P4_k25_rf_seed1 P4_k25_rf_seed2 P4_k25_semipm > $L/eval_newobj_v2.log 2>&1 || { step "eval FAILED"; exit 1; }
step "chain done"
