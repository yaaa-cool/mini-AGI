set -e
rsync -a --exclude .venv --exclude data --exclude "data_*" --exclude runs ~/train/mini-AGI/ ~/train/mini-AGI-small/
cd ~/train/mini-AGI-small && ln -sfn ~/train/mini-AGI/data data && cp config.small.yaml config.yaml && mkdir -p runs
L=/mnt/data/runs/mini-agi/w4-lanes
mkdir -p $L/chat-train $L/chat-val
cp -rs /home/train/train/mini-AGI/data/train/chat $L/chat-train/chat
cp -rs /home/train/train/mini-AGI/data/val/chat $L/chat-val/chat
echo links $(find $L/chat-train -type l | wc -l) src $(find /home/train/train/mini-AGI/data/train/chat -type f | wc -l) val $(find $L/chat-val -type l | wc -l)
grep -n "weights" config.yaml | head -3
