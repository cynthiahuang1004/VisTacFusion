#!/bin/bash
# Real-data quantity ablation: real-only training with decreasing per-object caps
# Uses train_bs32.yaml (150 epochs), same real data handling as co-train

set -e

GPU=${1:-1}
MODEL="configs/model.yaml"
TRAIN="configs/train_bs32.yaml"

CAPS="all 350 300 200 100 50 25"

for cap in $CAPS; do
    DATA="ablation/realqty/data_realonly_${cap}.yaml"
    OUT="outputs/realqty_${cap}"
    LOG="outputs/realqty_${cap}_train.log"

    if [ -f "$OUT/history.json" ]; then
        echo "[SKIP] $OUT already has history.json"
        continue
    fi

    echo "=========================================="
    echo "  Real-only cap=$cap  ->  $OUT"
    echo "  GPU=$GPU, 150 epochs"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=$GPU \
    /home/shared/miniconda3/envs/vistacfusion/bin/python -u \
        -m vistacfusion.engine.train \
        --model "$MODEL" \
        --train "$TRAIN" \
        --data "$DATA" \
        --output-dir "$OUT" \
        > "$LOG" 2>&1

    echo "[DONE] cap=$cap  ->  $OUT"
    echo ""
done

echo "All real-qty ablation runs complete."
