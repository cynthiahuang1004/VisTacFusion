#!/bin/bash
# render_phys_chain.sh <gpu> <obj> [<obj> ...]  — Blender physics renders at real poses, one object after another.
gpu=$1; shift
cd /media/hdd2/ihsuan/gs_blender
for obj in "$@"; do
  echo "$(date +%m-%d\ %H:%M) start $obj on GPU$gpu"
  CUDA_VISIBLE_DEVICES=$gpu GELSIGHT_FIXED_PARAMS=bo_results/tactile_v2/best_params.json \
    /home/shared/blender-4.2.0-linux-x64/blender -t ${PHYS_THREADS:-8} --background gelsight_sampler.blend --python render_real_pose_tactile.py -- \
    --obj $obj --frames /media/hdd2/ihsuan/VisTacFusion/outputs/phys_frames/$obj.json 2>&1 | grep -E "\[phys\]|Traceback|Error"
done
echo "$(date +%m-%d\ %H:%M) chain done"
