#!/bin/bash

# skill evaluation
python evaluate.py \
    --task math \
    --dataset gsm8k-hard \
    --model LLaMA-7B\
    --adapter LoRA\
    --base_model yahma/llama-7b-hf \
    --batch_size 1 \
    --lora_weights models/qa_lora  \
    --outfile bio_qa

## MoE  evaluation
# pass --moe and specify the experts in --lora_weights

