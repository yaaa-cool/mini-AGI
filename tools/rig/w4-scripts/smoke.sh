set -e
cd ~/train/mini-AGI
cp -n config.yaml /mnt/data/runs/mini-agi/config.reference.yaml
cp config.small.yaml config.yaml
W=/mnt/data/runs/mini-agi/w4-smoke; L=/mnt/data/runs/mini-agi/w4-lanes; P=/home/train/train/mini-AGI/.venv/bin/python
mkdir -p $W
nohup /mnt/data/runs/mini-agi/guarded.sh $W bash -c "$P tools/device_check.py --window 2048 --chunk 2048 > $W/device_check.log 2>&1; $P train.py read $L/chat-train --save --weights-dir $W/weights --held-out $L/chat-val --minutes 15 --sample-every 5 --sample-log $W/samples.txt" > $W/guard.out 2>&1 &
sleep 2; cat $W/pid
