#!/usr/bin/env bash
# Baseline comparison in the crop-0.816 world (sim348 = 1:2.98, transfilt, sim RGB zoom 1.15, no real aug).
# GPU 1: TVL (ViT, tactile only) + TVL+MAE.   GPU 3: ViTaL & MViTac SSL pretrain (240 ep each, raw
# images, their own protocol) -> then their fine-tune runs. Outputs on /media/hdd, symlinked in outputs/.
set -uo pipefail
cd "$(dirname "$0")/.."
NEW=/media/hdd/ihsuan/VisTacFusion_outputs; TAG=c816_sim348
DATA=ablation/simqty_gtac/data_ratio_g3s_sim348_transfilt_zoom115_crop816.yaml
PY=/home/shared/miniconda3/envs/vistacfusion/bin/python; TRAIN=configs/train_bs32.yaml
link() { mkdir -p "$NEW/$1"; ln -sfn "$NEW/$1" "outputs/$1"; }
train() { # gpu name model_cfg
  out=base_$2_$TAG; link $out
  CUDA_VISIBLE_DEVICES=$1 nohup $PY -u -m vistacfusion.engine.train --model "$3" --train $TRAIN --data "$DATA" \
    --output-dir "$NEW/$out" > "$NEW/$out/train.log" 2>&1 &
  echo "GPU$1 $2 PID $! -> $NEW/$out"; }
# --- GPU 1: TVL baselines now ---
train 1 tvl_single ablation/encoder/tac_tvl_single.yaml
train 1 tvl_mae    ablation/encoder/tac_tvl_rgb_mae.yaml
# --- GPU 3: SSL pretrains (both in parallel), each followed by its fine-tune ---
for m in vital mvitac; do
  pre=${m}_pretrain_$TAG; link $pre
  ( CUDA_VISIBLE_DEVICES=3 $PY -u -m vistacfusion.engine.pretrain_$m --data-config "$DATA" --output-dir "$NEW/$pre" --device cuda:0 > "$NEW/$pre/pretrain.log" 2>&1
    if [[ -f $NEW/$pre/tactile_encoder.pt && -f $NEW/$pre/vision_encoder.pt ]]; then
      train 3 $m ablation/encoder/tac_${m}_$TAG.yaml
    else echo "$m pretrain FAILED (no encoder files)" >> "$NEW/$pre/pretrain.log"; fi ) &
  echo "GPU3 $m pretrain -> $NEW/$pre (fine-tune auto-follows)"
done
wait
