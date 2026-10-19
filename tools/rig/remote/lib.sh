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

# The rig has two cards and runs one job per card: every check, guard and
# launch names its card by nvidia-smi index (PCI bus order, which
# CUDA_DEVICE_ORDER=PCI_BUS_ID makes torch's order too).

# card_apps UUID : the compute apps on that card, "pid, name, used" per line.
card_apps() {
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
    | awk -F', *' -v u="$1" '$1 == u {print $2 ", " $3 ", " $4}'
}

# pinned_runs CARD : the live guarded runs launched on that card ($W/gpu), so a
# run still loading its model already holds its card.
pinned_runs() {
  local d
  live_runs | while read -r d; do
    [ "$(cat "$d/gpu" 2>/dev/null)" = "$1" ] && echo "$d"
  done
}

# gpu_free CARD [max_temp] [max_used_mib] : the check before every GPU job, on
# one card. Fails if something holds its VRAM (strata-swift holds about
# 11.6 GB), a compute app or a live guarded run is on it, or it is already at
# max_temp. The other card does not count.
gpu_free() {
  local card=$1 max_t=${2:-85} max_mib=${3:-1024} row uuid used temp apps runs bad=0
  row=$(nvidia-smi -i "$card" --query-gpu=uuid,memory.used,temperature.gpu --format=csv,noheader,nounits 2>&1) \
    || { echo "GPU $card CHECK FAILED: nvidia-smi -i $card did not answer: $row"; return 1; }
  IFS=, read -r uuid used temp <<< "$row"
  uuid=${uuid// /}; used=${used// /}; temp=${temp// /}
  apps=$(card_apps "$uuid")
  runs=$(pinned_runs "$card")
  if [ "$used" -gt "$max_mib" ]; then
    echo "GPU $card BUSY: ${used} MiB of VRAM in use (limit ${max_mib}); strata-swift? (rig.py swift stop)"
    bad=1
  fi
  if [ -n "$apps" ]; then
    echo "GPU $card BUSY: compute apps running:"
    echo "$apps" | sed 's/^/  /'
    bad=1
  fi
  if [ -n "$runs" ]; then
    echo "GPU $card BUSY: live guarded runs on it:"
    echo "$runs" | sed 's/^/  /'
    bad=1
  fi
  if [ "$temp" -ge "$max_t" ]; then
    echo "GPU $card HOT: ${temp} C >= ${max_t} C"
    bad=1
  fi
  [ "$bad" = 0 ] && echo "GPU $card free: ${used} MiB used, ${temp} C, no compute apps"
  return "$bad"
}

# pick_gpus SPEC [all] : sets GPUS to the cards a job runs on. SPEC is a card
# index or a comma list (each must be free), or auto: the first free card, or
# with "all" every free card. Prints each card's check; fails if a named card
# is not free or auto finds none.
pick_gpus() {
  local spec=$1 want=${2:-one} idx c out bad=0
  GPUS=()
  if [ "$spec" != auto ]; then
    IFS=, read -ra GPUS <<< "$spec"
    for c in "${GPUS[@]}"; do
      gpu_free "$c" || bad=1
    done
    return "$bad"
  fi
  idx=$(nvidia-smi --query-gpu=index --format=csv,noheader) \
    || { echo "GPU CHECK FAILED: nvidia-smi did not answer"; return 1; }
  for c in ${idx// /}; do
    if out=$(gpu_free "$c"); then
      echo "$out"
      GPUS+=("$c")
      [ "$want" = all ] || break
    else
      echo "$out"
    fi
  done
  if [ "${#GPUS[@]}" = 0 ]; then
    echo "NO FREE GPU: every card is busy or hot (rig.py gpu-check shows who holds them)"
    return 1
  fi
  echo "auto: card ${GPUS[*]}"
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

# d_model FILE : the line holding the d_model key, not a comment that names it,
# without its trailing comment.
d_model() {
  grep -m1 -E '^[[:space:]]*d_model:' "$1" | sed -e 's/^ *//' -e 's/[[:space:]]*#.*$//'
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
      echo "(config.yaml is the config and some keys are re-read mid-run; wait for them to end."
      echo " Both cards run from this one checkout: a second run side by side on the other"
      echo " card must use the active config, $(active_config 2>/dev/null || echo unknown))"
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

# launch_guarded RUN_DIR CARD CMD... : the card into $W/gpu, the exact command
# into $W/cmd, the config it runs with into $W/config.yaml, then CMD nohup'd
# under guarded.sh from the current directory, pinned to CARD (the job sees
# only that card, as cuda:0). MINAGI_DISPATCH must already be exported.
launch_guarded() {
  local w=$1 card=$2; shift 2
  mkdir -p "$w"
  echo "$card" > "$w/gpu"
  cp "$REPO/config.yaml" "$w/config.yaml"
  { echo "# $(date -Is) in $PWD, card $card"
    printf 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=%s MINAGI_DISPATCH=%s PYTHONUNBUFFERED=1' \
      "$card" "$MINAGI_DISPATCH"
    printf ' %q' "$@"
    echo; } > "$w/cmd"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$card PYTHONUNBUFFERED=1 \
    nohup bash "$RIG/guarded.sh" "$w" "$card" "$@" < /dev/null > "$w/guard.out" 2>&1 &
  sleep 2
  echo "launched $w on card $card, pid $(cat "$w/pid" 2>/dev/null)"
}
