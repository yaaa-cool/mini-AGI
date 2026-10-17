set -e
cd ~/train/mini-AGI
cp /mnt/data/runs/mini-agi/config.reference.yaml config.yaml
grep -m1 "d_model" config.yaml
W=/mnt/data/runs/mini-agi/w4-B; P=/home/train/train/mini-AGI/.venv/bin/python
[ -f $W/pid ] && { echo "B already started"; exit 1; }
PYTHONUNBUFFERED=1 nohup /mnt/data/runs/mini-agi/guarded.sh $W $P train.py read data/replay-chat --save --weights-dir $W/weights --held-out data/val --lr 1e-4 --minutes 120 --sample-every 10 --sample-log $W/samples.txt > $W/guard.out 2>&1 &
sleep 2; cat $W/pid
