#!/bin/bash
# guarded.sh <run_dir> <card> <cmd...> : run cmd (cwd unchanged) with output to <run_dir>/read.log,
# sample nvidia-smi card <card> every 10 s into <run_dir>/gpu.csv, SIGTERM the run at >= 85 C.
# W4's guarded.sh with the limit named once, and pinned to the run's own card:
# W4's query covered every GPU, which on the two-card rig gives two lines, a
# test that errors, and a guard that never fires. Only the run's own card
# trips it - the other card has its own run and its own guard, and gpu_free
# will not start a job on a hot card (both share the case: status prints both
# temps). gpu.csv keeps W4's columns, plus the card's index last. Writes pid,
# start, gpu.csv, guard.log (only if the guard fired) and end ("exit N", then
# the time).
GUARD_C=85
W=$1; CARD=$2; shift 2
mkdir -p "$W"
"$@" > "$W/read.log" 2>&1 &
PID=$!
echo $PID > "$W/pid"
date -Is > "$W/start"
echo "time,temp_c,sm_mhz,mem_used_mib,power_w,util_pct,sw_thermal,hw_thermal,gpu" > "$W/gpu.csv"
while kill -0 $PID 2>/dev/null; do
  L=$(nvidia-smi -i "$CARD" --query-gpu=timestamp,temperature.gpu,clocks.sm,memory.used,power.draw,utilization.gpu,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,index --format=csv,noheader,nounits)
  echo "$L" >> "$W/gpu.csv"
  T=$(echo "$L" | head -1 | cut -d, -f2 | tr -d ' ')
  if [ "${T:-0}" -ge "$GUARD_C" ]; then
    echo "$(date -Is) TEMP GUARD: card $CARD ${T}C >= ${GUARD_C}, SIGTERM $PID" | tee -a "$W/guard.log"
    kill -TERM $PID
  fi
  sleep 10
done
wait $PID; echo "exit $?" > "$W/end"; date -Is >> "$W/end"
