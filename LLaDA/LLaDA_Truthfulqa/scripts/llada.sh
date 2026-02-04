
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

pkill -9 python
sleep 2

for seed in {0..19}
do
    
    output_path="./outputs/truthfulqa/llada_conf/genlen-32_T-32_blocksize-32_seed-${seed}"
    mkdir -p $output_path

    accelerate launch --num_processes 1 --main_process_port 0 eval_llada.py \
        --seed $seed \
        --tasks truthfulqa_gen \
        --model llada_dist \
        --confirm_run_unsafe_code \
        --output_path $output_path \
        --model_args model_path='<PATH_TO_YOUR_LLaDA_8B_INSTRUCT_WEIGHTS>',mask_length=32,sampling_steps=32,block_size=32,sampler='llada_conf'

    if ls ./*.json 1> /dev/null 2>&1; then
        cp ./*.json "$output_path/"
        rm ./*.json
    fi
done