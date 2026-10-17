C=/mnt/data/runs/mini-agi/w4-C
nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.sm,utilization.gpu --format=csv,noheader
for f in $C/gen-*; do echo "$(basename $f): $(grep -c '^\[' $f/read.log 2>/dev/null) lines, last: $(tail -1 $f/read.log | cut -c1-90) $(cat $f/end 2>/dev/null | head -1)"; done
for f in $C/replies.*.jsonl; do echo "$(basename $f) $(wc -l < $f)"; done 2>/dev/null
cat $C/../w4-*/guard.log $C/gen-*/guard.log 2>/dev/null
