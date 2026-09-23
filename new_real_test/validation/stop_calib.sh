#!/bin/bash
# Stop any running new-session calibration chain / workers. Kept in a file so the caller's
# command line never contains these patterns (pgrep -f would otherwise kill the caller).
pgrep -u shared -f 'recalib_chain_v[0-9]*[.]sh' | xargs -r kill
pgrep -u shared -f 'calibrate_new_real_pose[s][.]py' | xargs -r kill
sleep 2
echo "calib procs left: $(pgrep -u shared -f 'calibrate_new_real_pose[s][.]py' | wc -l)"
