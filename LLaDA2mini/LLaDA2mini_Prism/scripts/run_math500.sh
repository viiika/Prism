#!/bin/bash
set -e
set -x

PROJECT_ROOT="<PATH_TO_YOUR_PROJECT_ROOT>"
MODEL_PATH="<PATH_TO_YOUR_LLADA2_MINI_WEIGHTS>"
BASE_OUTPUT_PATH="${PROJECT_ROOT}/outputs/llada2_math500"

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

LENGTH=256
STEPS=32
BLOCK=32
TASK="math500"
TYPE="math"
NAME="win_0.1-0.6_s2_k4"

mkdir -p "${BASE_OUTPUT_PATH}/${NAME}"

accelerate launch evaluation_script.py \
    --model LLaDA2 \
    --tasks ${TASK} \
    --batch_size 1 \
    --model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning> " \
    --gen_kwargs "use_hts=True,hts_N=16,final_K=4,hts_survivor_k=2,hts_mode=True,hts_start_pct=0.1,hts_end_pct=0.6,pruning_interval=3,decay_factor=1.8,reward_mode=svf,task_type=${TYPE},steps=${STEPS},block_length=${BLOCK},gen_length=${LENGTH},temperature=0.7,realtime_output=${BASE_OUTPUT_PATH}/${NAME}/res.jsonl" \
    --num_fewshot 0 \
    --output_path "${BASE_OUTPUT_PATH}/${NAME}"