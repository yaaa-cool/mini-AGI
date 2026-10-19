#!/bin/bash
# smoke.sh <run_dir> <config> <dispatch> <card|auto> <minutes> <data> <held_out> :
# the thermal and speed gate, as W4's smoke.sh - device_check, then a short
# `train.py read --save` on the chat lanes, one guarded job on one card in a
# fresh dir.
# shellcheck source-path=SCRIPTDIR source=lib.sh
. "$(dirname "$0")/lib.sh"
W=$1; CONFIG=$2; DISPATCH=$3; GPU=$4; MIN=$5; DATA=$6; HELD=$7
pick_gpus "$GPU" || exit 1
fresh_dir "$W" || exit 1
cd "$REPO" || exit 1
select_config "$CONFIG" || exit $?
export MINAGI_DISPATCH=$DISPATCH
launch_guarded "$W" "${GPUS[0]}" bash -c "$PY tools/device_check.py --window 2048 --chunk 2048 > $W/device_check.log 2>&1; $PY train.py read $DATA --save --weights-dir $W/weights --held-out $HELD --minutes $MIN --sample-every 5 --sample-log $W/samples.txt"
