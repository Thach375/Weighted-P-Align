#!/bin/bash
set -euo pipefail

: "${INPUT_FILE:?Set INPUT_FILE to the generated JSONL path}"
: "${OUTPUT_FILE:?Set OUTPUT_FILE to the evaluated JSONL output path}"

mkdir -p "$(dirname "$OUTPUT_FILE")"

python src/evaluation.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE"
