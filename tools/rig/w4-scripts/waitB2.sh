W=/mnt/data/runs/mini-agi/w4-B-lrreset
for i in $(seq 1 40); do grep -q "before:" $W/read.log && break; [ -f $W/end ] && break; sleep 30; done
grep -vE "^\s*$" $W/read.log | tail -n 22 | cut -c1-200; cat $W/end 2>/dev/null
nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader
