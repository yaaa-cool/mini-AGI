#!/bin/bash
# status.sh <run_dir> : read-only status of a guarded run - W4's status.sh plus
# held-out per lane, speed, ETA and state. The last line is always "state: ...".
W=$1
[ -d "$W" ] || { echo "no run dir $W"; echo "state: missing"; exit 1; }
echo "== $W"
if [ -f "$W/gpu.csv" ]; then
  echo "== gpu ($(($(wc -l < "$W/gpu.csv") - 1)) samples, every 10 s; last:)"
  tail -1 "$W/gpu.csv"
  awk -F, 'NR>1 && $2+0>0 {t=$2+0; s=$3+0; n++; st+=t; ss+=s; if(t>mt)mt=t; if(s<ms||ms==0)ms=s;
    if($7~/Active/ && $7!~/Not/)sw++; if($8~/Active/ && $8!~/Not/)hw++}
    END{if(n) printf "maxT=%d meanT=%.1f meanSM=%.0f minSM=%d swThermalActive=%d hwThermalActive=%d\n",mt,st/n,ss/n,ms,sw+0,hw+0}' "$W/gpu.csv"
fi
HITS=0; [ -f "$W/guard.log" ] && HITS=$(grep -c "TEMP GUARD" "$W/guard.log")
echo "guard hits: $HITS"
[ -f "$W/guard.log" ] && tail -3 "$W/guard.log"
if [ -f "$W/read.log" ]; then
  echo "== speed (latest): $(grep "char/s" "$W/read.log" | tail -1 | sed 's/^ *//' | cut -c1-200)"
  echo "== log (gate lines)"
  grep -E "char/s|vram|after|before|held-out|saved|eval|rate x|RESET|TEMP GUARD|Error|Traceback" "$W/read.log" | tail -10 | cut -c1-200
fi
S=$W/samples.txt
if [ -f "$S" ]; then
  # post-574a397 scale: the banner is the start's evaluation spread over the
  # whole set, and every check is reported at that level ("level" = what the
  # check's own blocks read differently)
  echo "== held-out, nats/char"
  echo "start (spread over the whole set):"
  grep "^# held-out at the start, spread over the whole set:" "$S" | tail -1 \
    | sed 's/^# held-out at the start, spread over the whole set: /  /'
  echo "latest check (at the start's level):"
  awk '/^held-out loss/{h=$0; l=""; if((getline n)>0 && n~/^  /)l=n} END{if(h)print "  " h; if(l)print l}' "$S"
fi
if [ -f "$W/start" ]; then
  START=$(cat "$W/start")
  S0=$(date -d "$START" +%s)
  echo "started $START, $(( ($(date +%s) - S0) / 60 )) min ago"
  MIN=$(tr '\\' ' ' < "$W/cmd" 2>/dev/null | grep -oE -- "--minutes +[0-9.]+" | tail -1 | awk '{print $2}')
  if [ -n "$MIN" ] && [ ! -f "$W/end" ]; then
    echo "ETA ~$(date -d "@$(( S0 + ${MIN%.*} * 60 ))" -Is) (--minutes $MIN from the start, then the closing eval and save)"
  fi
  # --max-chars counts this session's characters; samples.txt's "step ... 12.3M"
  # counts from the weights' first session, so the earlier sessions come off
  MAXC=$(tr '\\' ' ' < "$W/cmd" 2>/dev/null | grep -oE -- "--max-chars +[0-9.eE+]+" | tail -1 | awk '{print $2}')
  if [ -n "$MAXC" ] && [ ! -f "$W/end" ]; then
    BASE=$(grep -m1 "characters read in earlier sessions" "$W/read.log" 2>/dev/null | awk '{print $1+0}')
    EST=$(awk -v maxc="$MAXC" -v base="${BASE:-0}" -v s0="$S0" '
      /^step [0-9,]+ +[0-9.]+M/ {c=$3+0; for(i=4;i<NF;i++) if($(i+1)=="min") m=$i+0}
      match($0, /reading [0-9,]+ char\/s/) {r=substr($0, RSTART+8, RLENGTH-15); gsub(",", "", r); r+=0}
      END{if(c=="" || r<=0) exit; d=(c-base)*1e6; left=maxc-d; if(left<0) left=0
          printf "%d %.1f %d\n", s0+m*60+left/r, d/1e6, r}' "$S" 2>/dev/null)
    if [ -n "$EST" ]; then
      read -r T DONE RATE <<< "$EST"
      echo "ETA ~$(date -d "@$T" -Is) (an estimate: --max-chars $MAXC, ${DONE}M read at the last check, at its $RATE char/s; then the closing eval and save)"
    else
      echo "ETA: --max-chars $MAXC; no estimate until samples.txt has a check with a reading rate"
    fi
  fi
fi
if [ -f "$W/end" ]; then
  echo "state: ended, $(head -1 "$W/end") at $(tail -1 "$W/end")"
elif [ -f "$W/pid" ] && kill -0 "$(cat "$W/pid")" 2>/dev/null; then
  echo "state: running, pid $(cat "$W/pid")"
else
  echo "state: dead - no end file and no live pid"
fi
