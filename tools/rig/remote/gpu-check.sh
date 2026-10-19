#!/bin/bash
# gpu-check.sh <card|all> <max_temp> <max_used_mib> : per card (every card, or
# the one named) what it reads and what holds it, then the same check every
# GPU job runs first. Exit 0 if a card checked is free (with a card named:
# that one), 1 if none is.
# shellcheck source-path=SCRIPTDIR source=lib.sh
. "$(dirname "$0")/lib.sh"
CARD=$1; MAXT=$2; MAXMIB=$3
hostname
echo "strata-swift: $(systemctl is-active strata-swift)"
echo "config: $(active_config 2>/dev/null || echo unknown) ($(d_model "$REPO/config.yaml"))"
echo "live guarded runs:"
live_runs | while read -r d; do
  echo "  $d (card $(cat "$d/gpu" 2>/dev/null || echo '?'))"
done
if [ "$CARD" = all ]; then
  CARDS=$(nvidia-smi --query-gpu=index --format=csv,noheader) \
    || { echo "GPU CHECK FAILED: nvidia-smi did not answer"; exit 1; }
else
  CARDS=$CARD
fi
FREE=()
for c in ${CARDS// /}; do
  echo "== card $c"
  nvidia-smi -i "$c" --query-gpu=index,name,temperature.gpu,clocks.sm,memory.used,memory.total,power.limit,clocks_event_reasons.sw_thermal_slowdown --format=csv
  echo "compute apps:"
  card_apps "$(nvidia-smi -i "$c" --query-gpu=uuid --format=csv,noheader)" | sed 's/^/  /'
  gpu_free "$c" "$MAXT" "$MAXMIB" && FREE+=("$c")
done
echo "free cards: ${FREE[*]:-none}"
[ "${#FREE[@]}" -gt 0 ]
