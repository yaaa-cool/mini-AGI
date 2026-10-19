#!/bin/bash
# swift.sh stop|start|status : strata-swift (the llama server on :8080, about
# 11.6 GB of VRAM). Runs as the admin user (cc), which has sudo; rig.py feeds
# it on stdin (`bash -s`) because that user cannot see train's upload dir.
case $1 in
  stop)
    sudo systemctl stop strata-swift && sleep 5
    echo "strata-swift: $(systemctl is-active strata-swift)"
    nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv,noheader ;;
  start)
    sudo systemctl start strata-swift && sleep 3
    echo "strata-swift: $(systemctl is-active strata-swift)" ;;
  status)
    echo "strata-swift: $(systemctl is-active strata-swift)"
    nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv,noheader ;;
  *)
    echo "usage: swift.sh stop|start|status"; exit 2 ;;
esac
