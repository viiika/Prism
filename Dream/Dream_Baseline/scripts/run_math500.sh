#!/bin/bash
set -e
set -x

PROJECT_ROOT="<PATH_TO_YOUR_DREAM_ROOT>"
MODEL_PATH="<PATH_TO_YOUR_DREAM_V0_INSTRUCT_7B>"
BASE_OUTPUT_PATH="${PROJECT_ROOT}/outputs/baseline_dream_math500"

cd ${PROJECT_ROOT}
export CUDA_VISIBLE_DEVICES=0,1
export HF_ENDPOINT=https://hf-mirror.com
export HF_ALLOW_CODE_EVAL=1
export PYTHONPATH=.  


TASK="math500"      
LENGTH=256      
STEPS=256          
PORT=12334
NAME="baseline_n1"

mkdir -p "${BASE_OUTPUT_PATH}/${NAME}"

accelerate launch --main_process_port ${PORT} -m lm_eval\
    --model diffllm \
    --tasks ${TASK} \
    --batch_size 1 \
    --model_args "pretrained=${MODEL_PATH},trust_remote_code=True,dtype=bfloat16,max_new_tokens=${LENGTH},diffusion_steps=${STEPS}" \
    --gen_kwargs "use_hts=True,initial_N=1,final_K=1,hts_mode=False,task_type=math,temperature=0.7,realtime_output=${BASE_OUTPUT_PATH}/${NAME}/baseline.jsonl" \
    --num_fewshot 0 \
    --output_path "${BASE_OUTPUT_PATH}/${NAME}"