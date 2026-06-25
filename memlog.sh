#!/bin/bash
for i in $(seq 1 600); do
  read u a < <(free -m | awk '/Mem:/{print $3, $7}')
  echo "$(date +%H:%M:%S) used=${u}MB avail=${a}MB" >> ~/joana/NeoVerse/freeze_mem.log
  sync
  sleep 1
done
