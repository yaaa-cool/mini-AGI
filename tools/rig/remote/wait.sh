#!/bin/bash
# wait.sh <run_dir> rate|end [max_minutes] : block on the rig until
#   rate - read.log has its first `rate x` line (or the run ended), then print
#          the lr lines to check (waitB2.sh's filter): RESET, rate x, subjects, before
#   end  - the run has an end file
W=$1; WHAT=$2; MAX=${3:-30}
for _ in $(seq 1 $((MAX * 2))); do
  [ -f "$W/end" ] && break
  [ "$WHAT" = rate ] && grep -q "rate x" "$W/read.log" 2>/dev/null && break
  sleep 30
done
if [ "$WHAT" = rate ]; then
  grep -vE "^\s*$" "$W/read.log" | grep -E "before|rate x|subjects|RESET" | cut -c1-200
  grep -q "rate x" "$W/read.log" || echo "no rate x line after $MAX min"
fi
[ -f "$W/end" ] && { echo "== END"; cat "$W/end"; }
exit 0
