# shellcheck shell=bash
# lib.sh : sourced by the other rig scripts, never run on its own.
# rig.py puts the paths on the ssh command line as RIG_REPO, RIG_RUNS, RIG_PY;
# the defaults are the rig's. rig.py sends a ~/ path as "$HOME"/...; a
# RIG_REPO set by hand may still arrive as a literal "~/...".
REPO=${RIG_REPO:-$HOME/train/mini-AGI}
REPO=${REPO/#\~/$HOME}
RUNS=${RIG_RUNS:-/mnt/data/runs/mini-agi}
# shellcheck disable=SC2034  # used by the scripts that source this
PY=${RIG_PY:-/home/train/train/mini-AGI/.venv/bin/python}
RIG=$RUNS/.rig

# gpu_free [max_temp] [max_used_mib] : the check at the top of every GPU job.
# Fails if something holds VRAM (strata-swift holds about 11.6 GB), any
# compute app runs, or the card is already at max_temp.
gpu_free() {
  local max_t=${1:-85} max_mib=${2:-1024} row used temp apps bad=0
  row=$(nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader,nounits) \
    || { echo "GPU CHECK FAILED: nvidia-smi did not answer"; return 1; }
  used=$(echo "$row" | head -1 | cut -d, -f1 | tr -d ' ')
  temp=$(echo "$row" | head -1 | cut -d, -f2 | tr -d ' ')
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
  if [ "$used" -gt "$max_mib" ]; then
    echo "GPU BUSY: ${used} MiB of VRAM in use (limit ${max_mib}); strata-swift? (rig.py swift stop)"
    bad=1
  fi
  if [ -n "$apps" ]; then
    echo "GPU BUSY: compute apps running:"
    echo "$apps" | sed 's/^/  /'
    bad=1
  fi
  if [ "$temp" -ge "$max_t" ]; then
    echo "GPU HOT: ${temp} C >= ${max_t} C"
    bad=1
  fi
  [ "$bad" = 0 ] && echo "GPU free: ${used} MiB used, ${temp} C, no compute apps"
  return "$bad"
}

# live_runs : the guarded run dirs (runs and eval shards) whose process is
# still alive and that have no end file yet, one per line.
live_runs() {
  local p d
  for p in "$RUNS"/*/pid "$RUNS"/*/*/pid; do
    [ -f "$p" ] || continue
    d=$(dirname "$p")
    [ -f "$d/end" ] && continue
    kill -0 "$(cat "$p")" 2>/dev/null && echo "$d"
  done
}

# d_model FILE : the line holding the d_model key, not a comment that names it.
d_model() {
  grep -m1 -E '^[[:space:]]*d_model:' "$1" | sed 's/^ *//'
}

# active_config : the config this rig is meant to run, as recorded by the
# last select; before the first, what config.yaml's d_model says it is (384
# small, 512 reference - not a byte compare: config.small.yaml changes
# upstream, and an older small copy is still small). Anything else fails.
active_config() {
  local width
  if [ -f "$RIG/active-config" ]; then
    cat "$RIG/active-config"
    return 0
  fi
  width=$(d_model "$REPO/config.yaml" | sed -E 's/^d_model:[[:space:]]*([0-9]+).*/\1/')
  case $width in
    384) echo small ;;
    512) echo reference ;;
    *) echo "config.yaml has d_model '${width:-none}', neither small (384) nor reference (512): give rig.py sync --config small|reference" >&2
       return 2 ;;
  esac
}

# select_config small|reference|- : make <repo>/config.yaml that config.
# There is no --config flag in train.py and some keys are re-read mid-run, so
# config.yaml is never changed while ANY guarded run is live, whichever keys
# differ (the simple rule; A and B only ran side by side because theirs
# agreed). "-" keeps config.yaml as it is. The reference copy is the Mac's
# config.yaml, which `rig.py sync` ships to $RIG/config.reference.yaml.
select_config() {
  local want=$1 src live
  if [ -z "$want" ] || [ "$want" = - ]; then
    echo "config: $(active_config 2>/dev/null || echo unknown), kept ($(d_model "$REPO/config.yaml"))"
    return 0
  fi
  case $want in
    small) src=$REPO/config.small.yaml ;;
    reference) src=$RIG/config.reference.yaml ;;
    *) echo "unknown config '$want' (small or reference)"; return 2 ;;
  esac
  [ -f "$src" ] || { echo "no $src on the rig: run rig.py sync first"; return 2; }
  if ! cmp -s "$src" "$REPO/config.yaml"; then
    live=$(live_runs)
    if [ -n "$live" ]; then
      echo "REFUSED: config.yaml would change to $want while guarded runs are live:"
      echo "$live" | sed 's/^/  /'
      echo "(config.yaml is the config and some keys are re-read mid-run; wait for them to end)"
      return 3
    fi
    cp "$src" "$REPO/config.yaml"
  fi
  echo "$want" > "$RIG/active-config"
  echo "config: $want ($(d_model "$REPO/config.yaml"))"
}

# fresh_dir DIR : refuse a run dir that exists - a reused --weights-dir resumes.
fresh_dir() {
  if [ -e "$1" ]; then
    echo "REFUSED: $1 exists; every run gets a fresh dir (a reused --weights-dir resumes)"
    return 1
  fi
}

# launch_guarded RUN_DIR CMD... : the exact command into $W/cmd, the config it
# runs with into $W/config.yaml, then CMD nohup'd under guarded.sh from the
# current directory. MINAGI_DISPATCH must already be exported.
launch_guarded() {
  local w=$1; shift
  mkdir -p "$w"
  cp "$REPO/config.yaml" "$w/config.yaml"
  { echo "# $(date -Is) in $PWD"
    printf 'MINAGI_DISPATCH=%s PYTHONUNBUFFERED=1' "$MINAGI_DISPATCH"
    printf ' %q' "$@"
    echo; } > "$w/cmd"
  PYTHONUNBUFFERED=1 nohup bash "$RIG/guarded.sh" "$w" "$@" < /dev/null > "$w/guard.out" 2>&1 &
  sleep 2
  echo "launched $w pid $(cat "$w/pid" 2>/dev/null)"
}
