#!/bin/bash
# eval-gen.sh <eval_dir> <shards> <max_chars> <prompts> <dispatch> <cards|auto> NAME=WEIGHTS... :
# W4's genC.sh for any number of models - the prompts split into <shards>, one
# guarded persona_generate.py per model and shard, all launched after one
# GPU-free check. <cards> is a comma list (auto: every free card); the jobs go
# round-robin over them, each pinned to its card. The weights' manifest sets
# the model's shape, so this does not touch config.yaml (W4 generated A with
# the reference config active).
# shellcheck source-path=SCRIPTDIR source=lib.sh
. "$(dirname "$0")/lib.sh"
E=$1; SHARDS=$2; MAXC=$3; PROMPTS=$4; DISPATCH=$5; GPU=$6
shift 6
pick_gpus "$GPU" all || exit 1
for m in "$@"; do
  NAME=${m%%=*}; WD=${m#*=}
  [ -d "$WD" ] || { echo "no weights at $WD for $NAME"; exit 1; }
  for k in $(seq 0 $((SHARDS - 1))); do
    fresh_dir "$E/gen-$NAME-$k" || exit 1
  done
done
mkdir -p "$E"
if [ -f "$E/shards" ] && [ "$(cat "$E/shards")" != "$SHARDS" ]; then
  echo "REFUSED: $E was split into $(cat "$E/shards") shards, not $SHARDS"; exit 1
fi
if [ ! -f "$E/shards" ]; then
  split -n "l/$SHARDS" -d -a 1 --additional-suffix=.jsonl "$PROMPTS" "$E/prompts." || exit 1
  echo "$SHARDS" > "$E/shards"
fi
wc -l "$E"/prompts.*.jsonl | tail -1
cd "$REPO" || exit 1
export MINAGI_DISPATCH=$DISPATCH
j=0
for m in "$@"; do
  NAME=${m%%=*}; WD=${m#*=}
  for k in $(seq 0 $((SHARDS - 1))); do
    launch_guarded "$E/gen-$NAME-$k" "${GPUS[j % ${#GPUS[@]}]}" "$PY" tools/persona_generate.py --weights "$WD" \
      --prompts "$E/prompts.$k.jsonl" --out "$E/replies.$NAME.$k.jsonl" --max-chars "$MAXC"
    j=$((j + 1))
  done
done
