set -e
cd ~/train/mini-AGI
R=data/replay; [ -e $R ] && { echo "replay exists"; exit 1; }
mkdir -p $R && cp -al data/train/persona $R/persona
budget=$(du -sb data/train/persona | cut -f1)
for lane in wikipedia stories chat reasoning arithmetic code chess; do
  mkdir -p $R/$lane
  find data/train/$lane -type f | sort | shuf --random-source=<(yes) | while read -r f; do
    [ "$(du -sb $R/$lane | cut -f1)" -ge "$budget" ] && break
    ln "$f" "$R/$lane/$(echo "${f#data/train/$lane/}" | tr / _)"
  done
done
du -sb $R/* | awk '{printf "%s %.1fM\n",$2,$1/1e6}'
