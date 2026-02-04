#!/bin/bash
set -e
set -x

PROJECT_ROOT="<PATH_TO_YOUR_DREAM_ROOT>"
MODEL_PATH="<PATH_TO_YOUR_DREAM_V0_INSTRUCT_7B>"
BASE_OUTPUT_PATH="${PROJECT_ROOT}/outputs/dream_mbpp"

cd ${PROJECT_ROOT}
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_ALLOW_CODE_EVAL=1
export PYTHONPATH=.  

TASK="mbpp"      
LENGTH=512       
STEPS=512         
PORT=12334
NAME="win_0.1-0.6_s2_k4"

mkdir -p "${BASE_OUTPUT_PATH}/${NAME}"

accelerate launch --main_process_port ${PORT} -m lm_eval\
    --model diffllm \
    --tasks ${TASK} \
    --batch_size 1 \
    --model_args "pretrained=${MODEL_PATH},trust_remote_code=True,dtype=bfloat16,max_new_tokens=${LENGTH},diffusion_steps=${STEPS}" \
    --gen_kwargs "use_hts=True,initial_N=16,final_K=4,hts_survivor_k=2,hts_mode=True,hts_start_pct=0.1,hts_end_pct=0.6,pruning_interval=3,decay_factor=1.8,reward_mode=svf,task_type=code,temperature=0.7,realtime_output=${BASE_OUTPUT_PATH}/${NAME}/res.jsonl" \
    --num_fewshot 0 \
    --confirm_run_unsafe_code \
    --output_path "${BASE_OUTPUT_PATH}/${NAME}"