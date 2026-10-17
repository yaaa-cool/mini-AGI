set -e
cd ~/train/mini-AGI
cmp config.yaml config.small.yaml
W=/mnt/data/runs/mini-agi/w4-A; L=/mnt/data/runs/mini-agi/w4-lanes; P=/home/train/train/mini-AGI/.venv/bin/python
mkdir -p $W
PYTHONUNBUFFERED=1 nohup /mnt/data/runs/mini-agi/guarded.sh $W $P train.py read $L/chat-train --save --weights-dir $W/weights --held-out $L/chat-val --minutes 180 --sample-every 10 --sample-log $W/samples.txt > $W/guard.out 2>&1 &
sleep 2; cat $W/pid
