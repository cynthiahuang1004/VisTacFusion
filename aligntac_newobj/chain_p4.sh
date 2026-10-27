#!/bin/bash
# K=25 AlignTac pipeline on the 16-object root (12 session-1 objects + edge/hex_key/marble/button), clean protocol
# (test frames in testA_plus4_frames.json are excluded from every component). usage: chain_p4.sh <gpu>
# teacher = real-only baseline -> teacher labels (+pose) -> aligner -> aligned mesh labels -> student
GPU=$1; cd /media/hdd2/ihsuan/VisTacFusion
export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=/media/hdd2/ihsuan/gs_blender; L=outputs/pilot_logs; PY="conda run -n vistacfusion python"; P=ablation/pilot_robosoft
M=ablation/encoder/tac_sitr_single.yaml; OUTT=/home/shared/vtf_outputs_T; LINK=/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2; K=25
step() { echo "$(date '+%m-%d %H:%M') [p4] $*"; }
if [ ! -f $OUTT/pilot150_P4_k25_realonly/final.pt ]; then
  step "teacher / real-only"; $PY -m vistacfusion.engine.train --model $M --train $P/train_pilot_e150.yaml --data $P/data_P4_k25_realonly.yaml --output-dir $OUTT/pilot150_P4_k25_realonly > $L/150ep_P4_k25_realonly.log 2>&1 || { step "teacher FAILED"; exit 1; }
fi
ln -sfn $OUTT/pilot150_P4_k25_realonly $LINK/pilot150_P4_k25_realonly
step "teacher labels + pose (aligner source)"
$PY scripts/tactile_render/pseudo_label.py --run P4_k25_realonly --k $K --out-root $G/real_pseudo_p4_k25_ro_ng --device cuda:0 --pose --link-val --iou-thr 0 > $L/pseudo_p4_k25_ro_ng.log 2>&1 || { step "label FAILED"; exit 1; }
if [ ! -f outputs/aligner_p4_k25_realonly/aligner.pt ]; then
  step aligner; $PY scripts/tactile_render/train_aligner.py --k $K --gan outputs/tactile_ganT --out outputs/aligner_p4_k25_realonly --steps 8000 --bs-real 64 --bs-sim 0 --real-root $G/real_filtered_plus4 > $L/aligner_p4_k25_realonly.log 2>&1 || { step "aligner FAILED"; exit 1; }
fi
step "aligned mesh labels"
$PY scripts/tactile_render/pseudo_label_rc.py --src-root $G/real_pseudo_p4_k25_ro_ng --run P4_k25_realonly --gan outputs/tactile_ganT --aligner outputs/aligner_p4_k25_realonly/aligner.pt --out-root $G/real_pseudo_p4_k25_ro_al --keep-frac 1.0 --pose-all --eval-gt --real-root $G/real_filtered_plus4 --device cuda:0 > $L/pseudo_p4_k25_ro_al.log 2>&1 || { step "rc FAILED"; exit 1; }
grep -A3 "^labels" $L/pseudo_p4_k25_ro_al.log
step student; $PY -m vistacfusion.engine.train --model $M --train $P/train_pilot_e150_pm.yaml --data $P/data_P4_k25_rf.yaml --output-dir $OUTT/pilot150_P4_k25_rf > $L/150ep_P4_k25_rf.log 2>&1 || { step "student FAILED"; exit 1; }
ln -sfn $OUTT/pilot150_P4_k25_rf $LINK/pilot150_P4_k25_rf; step "chain done"
