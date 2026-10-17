# status.sh <run_dir> : read-only status of a guarded run
W=$1
echo "== gpu (n=$(($(wc -l < $W/gpu.csv)-1)))"; tail -2 $W/gpu.csv
awk -F, 'NR>1{t=$2+0; s=$3+0; if(t>mt)mt=t; n++; ss+=s; if(s<ms||ms==0)ms=s; if($7~/Active/ && $7!~/Not/)a++} END{printf "maxT=%d meanSM=%.0f minSM=%d swThermalActive=%d\n",mt,ss/n,ms,a+0}' $W/gpu.csv
[ -f $W/guard.log ] && cat $W/guard.log
[ -f $W/end ] && { echo "== END"; cat $W/end; }
echo "== log"; tail -c 1200 $W/read.log
