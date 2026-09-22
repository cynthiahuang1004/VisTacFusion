#!/bin/bash
# Stop the formal-run queue and, optionally, our pilot150 runs matching $1 (run-name substring).
# Kept in a file so the calling shell's command line never contains the patterns (pkill -f
# would otherwise match and kill the caller).
pgrep -u shared -f 'ablation/pilot_robosoft/queue150[.]sh' | xargs -r kill
if [ -n "$1" ]; then
  pgrep -u shared -f "engine[.]train .*output-dir [^ ]*pilot150_$1" | xargs -r kill
fi
sleep 3
echo "queue procs: $(pgrep -u shared -f 'ablation/pilot_robosoft/queue150[.]sh' | wc -l)"
echo "live150: $(ps -u shared -o cmd | grep '[e]ngine.train --model' | grep -o 'pilot150_[A-Za-z0-9_]*' | sort | uniq -c | tr '\n' ' ')"
