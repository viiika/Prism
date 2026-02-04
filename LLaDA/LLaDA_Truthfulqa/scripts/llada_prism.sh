#!/bin/bash
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export CUDA_VISIBLE_DEVICES=0
BASE_OUT="./outputs/truthfulqa/llada_prism"


for seed in {0..19}
do
    export MASTER_PORT=$(shuf -i 48000-55000 -n 1)
    output_path="${BASE_OUT}/n4_s1_k2/seed-${seed}"
    mkdir -p "$output_path"

    echo "Running n4-s1-k2 | Seed: ${seed} | Port: ${MASTER_PORT}"

    accelerate launch --num_processes 1 --main_process_port $MASTER_PORT eval_llada_prism.py \
        --seed $seed \
        --tasks truthfulqa_gen \
        --model llada_dist \
        --confirm_run_unsafe_code \
        --output_path "$output_path" \
        --model_args \
model_path='<PATH_TO_YOUR_LLaDA_8B_INSTRUCT_WEIGHTS>',\
mask_length=32,\
sampling_steps=32,\
sampler='hts',\
task_type='qa',\
hts_initial_n=4,\
final_K=2,\
hts_survivor_k=1,\
hts_decay_factor=1.8,\
hts_reward_mode='svf',\
hts_start_pct=0.1,\
hts_end_pct=0.6,\
pruning_interval=3

    if ls ./*.json 1> /dev/null 2>&1; then
        cp ./*.json "$output_path/"
        rm ./*.json
    fi
    sleep 5
done