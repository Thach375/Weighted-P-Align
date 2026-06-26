#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

: "${MODEL_PATH:?Set MODEL_PATH to the evaluator model path}"
: "${INPUT_FILE:?Set INPUT_FILE to the raw JSONL path}"
: "${OUTPUT_FILE:?Set OUTPUT_FILE to the prefix JSONL output path}"

mkdir -p output/log "$(dirname "$OUTPUT_FILE")"

nohup python src/binary_search.py \
    --model "$MODEL_PATH" \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    > output/log/prefix.log 2>&1 &
