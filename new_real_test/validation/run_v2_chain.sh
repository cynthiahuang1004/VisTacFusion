#!/bin/bash
cd /media/hdd2/ihsuan/VisTacFusion
PY=/home/shared/miniconda3/envs/vistacfusion/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "step 1: calibrate"
$PY scripts/calibrate_new_real_poses.py --v2 > new_real_test/validation/calibration_v2.log 2>&1
echo "calibrate exit: $?"

echo "step 2: build"
rm -rf /media/hdd2/ihsuan/gs_blender/real_filtered_new
$PY scripts/build_real_filtered_new.py --out /media/hdd2/ihsuan/gs_blender/real_filtered_new \
    --calib new_real_test/validation/pose_calibration_new.json > new_real_test/validation/build.log 2>&1
echo "build exit: $?"

echo "step 3: filter"
cd /media/hdd2/ihsuan/gs_blender
$PY filter_real_data.py --root /media/hdd2/ihsuan/gs_blender/real_filtered_new \
    --dry-run --workers 8 > /media/hdd2/ihsuan/VisTacFusion/new_real_test/validation/filter_dryrun.log 2>&1
echo "filter exit: $?"

echo "CHAIN_DONE" >> /media/hdd2/ihsuan/VisTacFusion/new_real_test/validation/build.log
echo "all done"
