# genC.sh <name> <weights_dir> : 3 guarded generation shards for one model
set -e
NAME=$1; WD=$2
C=/mnt/data/runs/mini-agi/w4-C; P=/home/train/train/mini-AGI/.venv/bin/python
mkdir -p $C
if [ ! -f $C/prompts.2.jsonl ]; then
  split -n l/3 -d -a 1 --additional-suffix=.jsonl /mnt/data/runs/mini-agi/heldout.jsonl $C/prompts.
fi
wc -l $C/prompts.*.jsonl | tail -1
cd ~/train/mini-AGI
for k in 0 1 2; do
  R=$C/gen-$NAME-$k
  PYTHONUNBUFFERED=1 nohup /mnt/data/runs/mini-agi/guarded.sh $R $P tools/persona_generate.py --weights $WD --prompts $C/prompts.$k.jsonl --out $C/replies.$NAME.$k.jsonl --max-chars 400 > $R.out 2>&1 &
done
sleep 2; cat $C/gen-$NAME-*/pid
