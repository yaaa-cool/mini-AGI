#!/bin/bash
# gpu-check.sh <max_temp> <max_used_mib> : what holds the card, then the same
# check every GPU job runs first. Exit 1 if VRAM is held, a compute app runs,
# or the card is at max_temp.
# shellcheck source-path=SCRIPTDIR source=lib.sh
. "$(dirname "$0")/lib.sh"
hostname
nvidia-smi --query-gpu=name,temperature.gpu,clocks.sm,memory.used,memory.total,power.limit,clocks_event_reasons.sw_thermal_slowdown --format=csv
echo "compute apps:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/  /'
echo "strata-swift: $(systemctl is-active strata-swift)"
echo "config: $(active_config) ($(grep -m1 d_model "$REPO/config.yaml" | sed 's/^ *//'))"
echo "live guarded runs:"
live_runs | sed 's/^/  /'
gpu_free "$1" "$2"
