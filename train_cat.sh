#!/bin/bash

## CAT training
python cat_train.py \
    --model LLaMA-1B \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --adapter LoRA \
    --data_path data/skill_datasets/mix/metamath_code_alpaca_10k.json \
    --output_dir models/math_code_cat \
    --batch_size 10 \
    --micro_batch_size 6 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --cutoff_len 256 \
    --val_set_size 120 \
    --eval_step 80 \
    --save_step 80 \
    --lora_weights models/math models/code
