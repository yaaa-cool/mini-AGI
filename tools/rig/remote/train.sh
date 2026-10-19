#!/bin/bash
# train.sh <run_dir> <config|-> <dispatch> <card|auto> <from_weights|-> <mix_src|-> <mix_spec|-> <train.py args...>
# One guarded `train.py read` on one card in a fresh run dir: GPU-free check
# (picks the card for auto), config select, a fresh cp -a of the starting
# weights, the replay mix folder, then the launch. rig.py builds the train.py
# arguments; this only runs them.
# shellcheck source-path=SCRIPTDIR source=lib.sh
. "$(dirname "$0")/lib.sh"
W=$1; CONFIG=$2; DISPATCH=$3; GPU=$4; FROM=$5; MIX_SRC=$6; MIX=$7
shift 7

# build_mix OUT SRC LANE:N... : the replay folder, as replay.sh/replay2.sh did.
# read gives every top-level folder an equal share, so a lane with N shares
# becomes N folders (lane.1 .. lane.N). The first lane is linked whole, split
# over its folders; every other folder gets files up to the size of one of
# those, in replay.sh's fixed shuffled order. Hard links where the corpus is on
# the run dir's disk, symbolic links otherwise - read follows both (w4-lanes).
build_mix() {
  local out=$1 src spec lane n i got size f r dir budget=0
  src=$(cd "$2" && pwd -P) || return 1
  shift 2
  for spec in "$@"; do
    lane=${spec%%:*}; n=${spec#*:}
    [ -d "$src/$lane" ] || { echo "no lane $src/$lane"; return 1; }
    for i in $(seq 1 "$n"); do
      dir=$lane; [ "$n" = 1 ] || dir=$lane.$i
      mkdir -p "$out/$dir"
    done
    i=1; got=0
    if [ "$budget" = 0 ]; then
      budget=$(( $(du -sb "$src/$lane" | cut -f1) / n ))
      order() { sort -k2; }
    else
      order() { sort -k2 | shuf --random-source=<(yes); }
    fi
    while read -r size f; do
      dir=$lane; [ "$n" = 1 ] || dir=$lane.$i
      r=${f#"$src/$lane"/}
      ln "$f" "$out/$dir/${r//\//_}" 2>/dev/null || ln -s "$f" "$out/$dir/${r//\//_}"
      got=$((got + size))
      if [ "$got" -ge "$budget" ]; then
        i=$((i + 1)); got=0
        [ "$i" -gt "$n" ] && break
      fi
    done < <(find "$src/$lane" -type f -printf '%s %p\n' | order)
  done
  du -sbL "$out"/* | awk '{printf "  mix %s %.1fM\n", $2, $1/1e6}'
}

pick_gpus "$GPU" || exit 1
fresh_dir "$W" || exit 1
if [ "$FROM" != - ] && [ ! -d "$FROM" ]; then
  echo "no weights at $FROM"; exit 1
fi
cd "$REPO" || exit 1
select_config "$CONFIG" || exit $?
mkdir -p "$W"
if [ "$FROM" != - ]; then
  # ALWAYS a fresh copy: a reused weights dir resumes, and the run writes it
  cp -a "$FROM" "$W/weights" || exit 1
  echo "weights: fresh copy of $FROM"
fi
if [ "$MIX" != - ]; then
  IFS=, read -ra LANES <<< "$MIX"   # "persona:1,chat:1" from rig.py
  build_mix "$W/mix" "$MIX_SRC" "${LANES[@]}" || exit 1
fi
export MINAGI_DISPATCH=$DISPATCH
launch_guarded "$W" "${GPUS[0]}" "$PY" "$@"
