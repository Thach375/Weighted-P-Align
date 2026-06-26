#!/bin/bash

NPROC_PER_NODE=1
NNODES=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29330
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

: "${TRAIN_CONFIG:?Set TRAIN_CONFIG to the LLaMA-Factory training YAML path}"

if ! command -v llamafactory-cli >/dev/null 2>&1; then
    echo "llamafactory-cli not found. Install LLaMA-Factory before running training." >&2
    exit 1
fi

llamafactory-cli train "$TRAIN_CONFIG" > output.log 2>&1 &
