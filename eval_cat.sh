#!/bin/bash

## CAT evaluation
python cat_eval.py \
	--dataset gsm8k-hard \
    --task math \
	--model LLaMA-7B\
	--adapter LoRA\
	--base_model yahma/llama-7b-hf \
	--batch_size 1 \
	--lora_weights models/math_code_cat/adapter_0 models/math_code_cat/adapter_1  \
	--outfile bio_qa_cat

## TIES evaluation
# python cat_eval.py \
#     --dataset gsm8k_hard \
#     --task math \
#     --model LLaMA-7B\
#     --adapter LoRA\
#     --base_model yahma/llama-7b-hf \
#     --batch_size 1 \
#     --lora_weights models/math models/code \
#     --outfile math_coder_gsm8k-hard_ties \
#     --lora_mix_mode ties \
#     --mix_weights 1.0 1.0