#!/bin/bash
set -e
set -x

PROJECT_ROOT="<PATH_TO_YOUR_PROJECT_ROOT>"
MODEL_PATH="<PATH_TO_YOUR_LLADA2_MINI_WEIGHTS>"
BASE_OUTPUT_PATH="${PROJECT_ROOT}/outputs/baseline_gsm8k"

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

LENGTH=256
STEPS=32
BLOCK=32
TASK="gsm8k"
TYPE="math"
NAME="baseline_n1"

mkdir -p "${BASE_OUTPUT_PATH}/${NAME}"

accelerate launch evaluation_script.py \
    --model LLaDA2 \
    --tasks ${TASK} \
    --batch_size 1 \
    --model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning> " \
    --gen_kwargs "use_hts=True,hts_N=1,hts_mode=False,steps=${STEPS},block_length=${BLOCK},gen_length=${LENGTH},task_type=${TYPE},temperature=0.7,realtime_output=${BASE_OUTPUT_PATH}/${NAME}/baseline.jsonl" \
    --num_fewshot 0 \
    --output_path "${BASE_OUTPUT_PATH}/${NAME}"