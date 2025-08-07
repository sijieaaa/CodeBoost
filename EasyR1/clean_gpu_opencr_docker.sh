#!/bin/bash


for gpu in {0..7}; do
  echo "=== GPU $gpu ==="
  pids=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits | grep -E "^ *[0-9]+" | awk -v g=$gpu 'NR==1{next} {print $1}' | xargs)

  for pid in $pids; do
    if [[ -n "$pid" ]]; then
      echo "Killing PID $pid ..."
      kill -9 $pid 2>/dev/null
    fi
  done
done
echo "🧼 GPU RAM cleaned ✅"



pids=$(ps -u root -o pid=,cmd= | grep -E 'python|ray' | grep -v grep | awk '{print $1}')
for pid in $pids; do
  echo "Killing PID $pid (python or ray match)..."
  kill -9 $pid 2>/dev/null
done
echo "🧼 Python + Ray cleaned ✅"
