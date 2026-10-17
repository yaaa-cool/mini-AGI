#!/bin/bash
# guarded.sh <run_dir> <cmd...> : run cmd (cwd unchanged) with output to <run_dir>/read.log,
# sample nvidia-smi every 10 s into <run_dir>/gpu.csv, SIGTERM the run at >= 85 C.
W=$1; shift
mkdir -p "$W"
"$@" > "$W/read.log" 2>&1 &
PID=$!
echo $PID > "$W/pid"
date -Is > "$W/start"
echo "time,temp_c,sm_mhz,mem_used_mib,power_w,util_pct,sw_thermal,hw_thermal" > "$W/gpu.csv"
while kill -0 $PID 2>/dev/null; do
  L=$(nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,memory.used,power.draw,utilization.gpu,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown --format=csv,noheader,nounits)
  echo "$L" >> "$W/gpu.csv"
  T=$(echo "$L" | cut -d, -f2 | tr -d ' ')
  if [ "${T:-0}" -ge 85 ]; then
    echo "$(date -Is) TEMP GUARD: ${T}C >= 85, SIGTERM $PID" | tee -a "$W/guard.log"
    kill -TERM $PID
  fi
  sleep 10
done
wait $PID; echo "exit $?" > "$W/end"; date -Is >> "$W/end"
