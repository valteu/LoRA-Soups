#!/bin/bash

# Example of fine-tuning on math with MetaMathQA
python skill_finetune.py \
    --model LLaMA-7B \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --adapter LoRA \
    --data_path data/skill_datasets/metamath.json \
    --output_dir models/math \
    --batch_size 16 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --lora_r 32 \
    --lora_alpha 64 \
    --cutoff_len 256 \
    --val_set_size 120 \
    --eval_step 80 \
    --save_step 80
