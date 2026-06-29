#!/usr/bin/env bash
# Download short YouTube clips for the SAM3-labeled training pilot.
# Reads data/sources.txt; downloads each clip into examples/videos/training_pilot/.
#
# data/sources.txt format (one entry per line, comma-separated):
#   <youtube-url>,<start-time>,<end-time>,<short-name>
# Lines starting with # are ignored.
# Example:
#   https://www.youtube.com/watch?v=abcDEF12345,0:30,0:42,campus_walk_01
#
# Requirements:
#   pip install yt-dlp
#
# Usage:
#   bash scripts/download_training_clips.sh

set -e
SRC="data/sources.txt"
OUT_DIR="examples/videos/training_pilot"
mkdir -p "$OUT_DIR"

if [ ! -f "$SRC" ]; then
    echo "Missing $SRC. Create it with URL,start,end,name lines."
    exit 1
fi

while IFS=, read -r url start end name; do
    # skip comments / blanks
    case "$url" in
        \#*|"") continue ;;
    esac
    # strip surrounding whitespace
    url=$(echo "$url" | xargs)
    start=$(echo "$start" | xargs)
    end=$(echo "$end" | xargs)
    name=$(echo "$name" | xargs)
    out="$OUT_DIR/${name}.mp4"
    if [ -f "$out" ]; then
        echo "SKIP $name (already downloaded)"
        continue
    fi
    echo "GET  $name  ($start - $end)"
    yt-dlp -f "best[height<=720][ext=mp4]" \
        --download-sections "*${start}-${end}" \
        --no-warnings \
        -o "$out" \
        "$url"
done < "$SRC"

echo ""
echo "All clips in $OUT_DIR/"
ls -lh "$OUT_DIR/"
