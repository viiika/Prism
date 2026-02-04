#!/bin/bash
set -e
set -x

PROJECT_ROOT="<PATH_TO_YOUR_ROOT>"
MODEL_PATH="<PATH_TO_YOUR_LLaDA_8B_INSTRUCT_WEIGHTS>"
BASE_OUTPUT_PATH="${PROJECT_ROOT}/outputs/results_humaneval"

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://hf-mirror.com

LENGTH=512
STEPS=32   
BLOCK=32
TASK="humaneval"
NAME="baseline"

mkdir -p "${BASE_OUTPUT_PATH}/${NAME}"

accelerate launch evaluation_script.py \
    --model LLaDA \
    --tasks ${TASK} \
    --batch_size 1 \
    --model_args "pretrained=${MODEL_PATH},mask_id=126336,assistant_prefix=<reasoning>" \
    --gen_kwargs "use_hts=True,hts_N=1,hts_mode=False,steps=${STEPS},block_length=${BLOCK},gen_length=${LENGTH},task_type=code,temperature=0.7,realtime_output=${BASE_OUTPUT_PATH}/${NAME}/baseline.jsonl" \
    --num_fewshot 0 \
    --confirm_run_unsafe_code \
    --output_path "${BASE_OUTPUT_PATH}/${NAME}"