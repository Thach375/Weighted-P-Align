#!/bin/bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the model path}"
: "${INPUT_FILES:?Set INPUT_FILES to space-separated input JSONL paths}"
: "${OUTPUT_FILES:?Set OUTPUT_FILES to space-separated output JSONL paths}"

read -r -a INPUT_FILE_ARRAY <<< "$INPUT_FILES"
read -r -a OUTPUT_FILE_ARRAY <<< "$OUTPUT_FILES"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
mkdir -p output/log

nohup python src/test.py \
    --model "$MODEL_PATH" \
    --input_files "${INPUT_FILE_ARRAY[@]}" \
    --output_files "${OUTPUT_FILE_ARRAY[@]}" \
    --batch_size "${BATCH_SIZE:-1000}" \
    --n "${N_SAMPLES:-3}" \
    --temperature "${TEMPERATURE:-0.6}" \
    --top_p "${TOP_P:-0.9}" \
    --max_tokens "${MAX_TOKENS:-4096}" \
    > output/log/result.log 2>&1 &

echo "Inference job started."
