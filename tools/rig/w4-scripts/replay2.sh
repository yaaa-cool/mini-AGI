set -e
cd ~/train/mini-AGI
R=data/replay-chat; [ -e $R ] && { echo "exists"; exit 1; }
mkdir -p $R && cp -al data/train/persona $R/persona && cp -al data/replay/chat $R/chat
du -sb $R/* | awk '{printf "%s %.1fM\n",$2,$1/1e6}'; find $R -type f | wc -l
