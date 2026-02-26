#!/bin/bash
# monitor_memory.sh — sample RSS every 100ms while vio-metal runs
PID=$1
OUTPUT="memory_log.csv"
echo "timestamp_ms,rss_mb,vsz_mb" > $OUTPUT

while kill -0 $PID 2>/dev/null; do
    RSS=$(ps -o rss= -p $PID | tr -d ' ')
    VSZ=$(ps -o vsz= -p $PID | tr -d ' ')
    TS=$(python3 -c "import time; print(int(time.time()*1000))")
    echo "$TS,$((RSS/1024)),$((VSZ/1024))" >> $OUTPUT
    sleep 0.1
done