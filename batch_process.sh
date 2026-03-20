#!/bin/bash
# Process all videos in the videos/ directory through golf_tracer.py
#
# Usage:
#   bash batch_process.sh [--all-trails] [--debug] [--show]
#
# Output goes to output/ directory with same filename as input.

INPUT_DIR="videos"
OUTPUT_DIR="output"
MODEL="best_new_v1.pt"
CONF="0.4"
EXTRA_ARGS="$@"

mkdir -p "$OUTPUT_DIR"

count=0
failed=0

for video in "$INPUT_DIR"/*.{MOV,mov,mp4,MP4,avi,AVI}; do
    [ -f "$video" ] || continue

    name=$(basename "$video")
    stem="${name%.*}"
    output="$OUTPUT_DIR/${stem}_traced.mp4"

    echo ""
    echo "══════════════════════════════════════"
    echo "  Processing: $name"
    echo "  Output:     $output"
    echo "══════════════════════════════════════"

    python golf_tracer.py "$video" \
        --model "$MODEL" \
        --conf "$CONF" \
        -o "$output" \
        $EXTRA_ARGS

    if [ $? -eq 0 ]; then
        count=$((count + 1))
    else
        echo "  FAILED: $name"
        failed=$((failed + 1))
    fi
done

echo ""
echo "══════════════════════════════════════"
echo "  Done! $count processed, $failed failed"
echo "  Output directory: $OUTPUT_DIR/"
echo "══════════════════════════════════════"
