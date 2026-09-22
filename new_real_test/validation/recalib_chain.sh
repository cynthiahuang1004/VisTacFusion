#!/bin/bash
cd /media/hdd2/ihsuan/VisTacFusion
PY=/home/shared/miniconda3/envs/vistacfusion/bin/python
rm -f new_real_test/validation/pose_calibration_new.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY scripts/calibrate_new_real_poses.py --workers 14 --mask blackhat > new_real_test/validation/calibration_blackhat.log 2>&1
rm -rf /media/hdd2/ihsuan/gs_blender/real_filtered_new
$PY scripts/build_real_filtered_new.py --out /media/hdd2/ihsuan/gs_blender/real_filtered_new --calib new_real_test/validation/pose_calibration_new.json > new_real_test/validation/build.log 2>&1
(cd /media/hdd2/ihsuan/gs_blender && $PY filter_real_data.py --root /media/hdd2/ihsuan/gs_blender/real_filtered_new --dry-run --workers 8 > /media/hdd2/ihsuan/VisTacFusion/new_real_test/validation/filter_dryrun.log 2>&1)
echo CHAIN_DONE >> new_real_test/validation/build.log
