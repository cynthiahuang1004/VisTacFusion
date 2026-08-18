#!/usr/bin/env bash
# Launch encoder / method baselines on ONE chosen ratio-ladder data config.
# Everything (data, ratio, train recipe, val set) is identical to the VisTacFusion
# run on that config — only the model config (encoder / architecture) changes.
#
# Usage:
#   bash scripts/run_baselines.sh <data_config.yaml> <tag> <gpu_list> [step]
#     step = pretrain | baselines | all   (default: all)
#   e.g.
#   bash scripts/run_baselines.sh ablation/simqty_gtac/data_ratio_g3s_sim380.yaml g3s_sim380 0,1,2,3
#
# Step "pretrain": MViTac MoCo pretraining on EXACTLY the sim+real train images of
#   <data_config> (MViTac's own protocol = self-supervised on the target data).
#   -> outputs/mvitac_pretrain_<tag>/{tactile,vision}_encoder.pt   (~1-2 h)
# Step "baselines": one training run per model config in BASELINES, round-robin
#   over <gpu_list>, output dirs  outputs/base_<name>_<tag>.
#   MViTac needs the pretrain step first (its config is generated from the tag).
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${1:?data config}; TAG=${2:?tag}; GPUS=${3:?gpu list, e.g. 0,1,2}; STEP=${4:-all}
PY=/home/shared/miniconda3/envs/vistacfusion/bin/python
TRAIN=configs/train_bs32.yaml
IFS=, read -ra GPU_ARR <<< "$GPUS"

# ---- model configs to run (edit freely) -----------------------------------
# name              -> model yaml
declare -A BASELINES=(
  [mvitac]="ablation/encoder/tac_mvitac_${TAG}.yaml"     # generated below
  [tvl_single]="ablation/encoder/tac_tvl_single.yaml"    # TVL ViT-B, tactile only, no fusion
  [tvl_mae]="ablation/encoder/tac_tvl_rgb_mae.yaml"      # TVL tactile + MAE rgb, our fusion
  [dinov3_mae]="ablation/encoder/tac_dinov3_rgb_mae.yaml"
  [sparsh_mae]="ablation/encoder/tac_sparsh_rgb_mae.yaml"
  [t3_dinov3]="ablation/encoder/tac_t3_rgb_dinov3.yaml"
  [t3_clip]="ablation/encoder/tac_t3_rgb_clip.yaml"
  [t3_siglip]="ablation/encoder/tac_t3_rgb_siglip.yaml"
)
ORDER=(mvitac tvl_single tvl_mae dinov3_mae sparsh_mae t3_dinov3 t3_clip t3_siglip)
# ---------------------------------------------------------------------------

PRE_DIR=outputs/mvitac_pretrain_${TAG}

if [[ $STEP == pretrain || $STEP == all ]]; then
  echo "== MViTac pretrain on $DATA -> $PRE_DIR (GPU ${GPU_ARR[0]})"
  setsid nohup env CUDA_VISIBLE_DEVICES=${GPU_ARR[0]} $PY -m vistacfusion.engine.pretrain_mvitac \
      --data-config "$DATA" --output-dir "$PRE_DIR" --device cuda:0 \
      > outputs/mvitac_pretrain_${TAG}.log 2>&1 &
  if [[ $STEP == all ]]; then
    echo "   waiting for pretrain to finish before launching baselines..."
    until [[ -f $PRE_DIR/tactile_encoder.pt && -f $PRE_DIR/vision_encoder.pt ]]; do
      grep -qE "Traceback|Error" outputs/mvitac_pretrain_${TAG}.log 2>/dev/null && { echo "pretrain FAILED, see outputs/mvitac_pretrain_${TAG}.log"; exit 1; }
      sleep 60
    done
  fi
fi

if [[ $STEP == baselines || $STEP == all ]]; then
  # per-tag MViTac config pointing at the matching pretrain
  sed -e "s#tactile_ckpt: .*#tactile_ckpt: ${PRE_DIR}/tactile_encoder.pt#" \
      -e "s#vision_ckpt: .*#vision_ckpt: ${PRE_DIR}/vision_encoder.pt#" \
      ablation/encoder/tac_mvitac.yaml > "ablation/encoder/tac_mvitac_${TAG}.yaml"
  i=0
  for name in "${ORDER[@]}"; do
    mcfg=${BASELINES[$name]}
    [[ -f $mcfg ]] || { echo "skip $name (missing $mcfg)"; continue; }
    if [[ $name == mvitac && ! -f $PRE_DIR/tactile_encoder.pt ]]; then echo "skip mvitac (no pretrain at $PRE_DIR)"; continue; fi
    gpu=${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}; i=$((i+1))
    out=outputs/base_${name}_${TAG}
    echo "== $name  model=$mcfg  GPU $gpu  -> $out"
    setsid nohup env CUDA_VISIBLE_DEVICES=$gpu $PY -u -m vistacfusion.engine.train \
        --model "$mcfg" --train $TRAIN --data "$DATA" --output-dir "$out" \
        > ${out}_train.log 2>&1 &
  done
  echo "launched. monitor:  grep -H 'co-training\|Traceback' outputs/base_*_${TAG}_train.log"
fi
