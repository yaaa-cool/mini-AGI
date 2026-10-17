#!/bin/bash
# eval-status.sh <eval_dir> : W4's genstat.sh for any eval dir - the card,
# lines per shard, replies per file, guard hits, and how many shards ended.
E=$1
[ -d "$E" ] || { echo "no eval dir $E"; exit 1; }
nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.sm,utilization.gpu --format=csv,noheader
for f in "$E"/gen-*/; do
  f=${f%/}
  echo "$(basename "$f"): $(grep -c '^\[' "$f/read.log" 2>/dev/null) lines, last: $(tail -1 "$f/read.log" 2>/dev/null | cut -c1-90) $(head -1 "$f/end" 2>/dev/null)"
done
for f in "$E"/replies.*.jsonl; do
  [ -f "$f" ] && echo "$(basename "$f") $(wc -l < "$f")"
done
cat "$E"/gen-*/guard.log 2>/dev/null
echo "shards ended: $(find "$E" -mindepth 2 -maxdepth 2 -name end | wc -l) of $(find "$E" -mindepth 1 -maxdepth 1 -name 'gen-*' -type d | wc -l)"
