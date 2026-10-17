set -e
OLD=/mnt/data/runs/mini-agi/w4-B
true
for i in $(seq 1 120); do [ -f $OLD/end ] && break; sleep 5; done; cat $OLD/end
cd ~/train/mini-AGI
grep -m1 "d_model" config.yaml
W=/mnt/data/runs/mini-agi/w4-B-lrreset; P=/home/train/train/mini-AGI/.venv/bin/python
mkdir -p $W && cp -a /mnt/data/runs/mini-agi/author/hf/weights $W/weights
PYTHONUNBUFFERED=1 nohup /mnt/data/runs/mini-agi/guarded.sh $W $P train.py read data/replay-chat --save --weights-dir $W/weights --held-out data/val --lr 1e-4 --lr-reset --minutes 120 --sample-every 10 --sample-log $W/samples.txt > $W/guard.out 2>&1 &
sleep 2; cat $W/pid
