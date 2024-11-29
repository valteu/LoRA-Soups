## CAT Utils
import os
import sys
import torch
from peft import PeftModel


from learnable_lora.peft_model import (
    CustomPeftModel
)
from try_llama import LinearLlamaModel, LinearLlamaForCausalLM

import re
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')
    if not args.lora_weights:
        raise ValueError(f'can not find lora weight, the value is')
    lora_weights = args.lora_weights[0]

    load_8bit = args.load_8bit
    if "LLaMA" in args.model:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
        
    ## add special tokens
    if args.add_special_toks:
        tokenizer.add_tokens(['[START]', '[END]'], special_tokens=True)
        x = tokenizer('[START] [END]')
        print(x)
        
    if device == "cuda":
        model = LinearLlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
        ) # fix zwq
        if args.lora_mix_mode:
            raise NotImplementedError("LoRA mix not supported")
            # Convert from transformers to PEFT model and add 1st adapter
            model = CustomPeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map={"":0},
                adapter_name='adapter_0'
            )
            adapters_mix = args.lora_weights
            adapters_mix_names = [f"adapter_{str(i)}" for i in range(len(adapters_mix))]
            num_adapters = len(adapters_mix)
            # load remaining adapters
            for i in range(1,num_adapters):
                model.load_adapter(adapters_mix[i], adapter_name=adapters_mix_names[i])
            # equal weights for all adapters
            adapters_mix_weights = [1/num_adapters for _ in range(num_adapters)]
            assert args.lora_mix_mode in ["linear", "cat", "svd"], "wrong LoRA mix method"
            combination_name = args.lora_mix_mode
            model.add_weighted_adapter(adapters=adapters_mix_names,
                                    weights=adapters_mix_weights,
                                    combination_type=combination_name,
                                    adapter_name=f"adapter_{args.lora_mix_mode}_mix",
                                    )
            model.set_adapter(f"adapter_{args.lora_mix_mode}_mix")
            print(model.active_adapters)
            print(adapters_mix_weights)
            print(f"Added mix adapter using {args.lora_mix_mode} with {num_adapters} adapters")
        else :
            # Load single LoRA adapter
            model = CustomPeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float32,
                device_map={"":0},
                adapter_name='adapter_0'
            )
            print("using custom peft")
            adapters_mix = args.lora_weights
            adapters_mix_names = [f"adapter_{str(i)}" for i in range(len(adapters_mix))]
            num_adapters = len(adapters_mix)
            # load remaining adapters
            for i in range(1,num_adapters):
                model.load_adapter(adapters_mix[i], adapter_name=adapters_mix_names[i])
            model.base_model.set_adapter(adapters_mix_names)
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

    ## add special tokens
    if args.add_special_toks:
        model.resize_token_embeddings(len(tokenizer))
    model.eval()
    
    print("-"*10)
    print(model)
    print("-"*10)

    print(f"Active adapters {model.active_adapters}")
    for name,p in model.named_parameters():
        p.requires_grad = False
    
    return tokenizer, model



def load_mixed_model(args):
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')
    if not args.lora_weights:
        raise ValueError(f'can not find lora weight, the value is')
    lora_weights = args.lora_weights[0]

    load_8bit = args.load_8bit
    if "LLaMA" in args.model:
        tokenizer = LlamaTokenizer.from_pretrained(base_model, token=TOKEN)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )

    ## add special tokens
    if args.add_special_toks:
        tokenizer.add_tokens(['[START]', '[END]'], special_tokens=True)
        x = tokenizer('[START] [END]')
        print(x)
    
    lora_weights = args.lora_weights[0]

    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            token = TOKEN
        )
    model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map={"":0},
                adapter_name='adapter_0'
            )
    adapters_mix = args.lora_weights
    adapters_mix_names = [f"adapter_{str(i)}" for i in range(len(adapters_mix))]
    num_adapters = len(adapters_mix)
    # load remaining adapters
    for i in range(1,num_adapters):
        _ = model.load_adapter(adapters_mix[i], adapter_name=adapters_mix_names[i])

    # method params
    weights = args.mix_weights
    adapter_name = "merge"
    density = 0.75
    if adapter_name in model.peft_config:
        model.delete_adapter(adapter_name)
    
    model.add_weighted_adapter(adapters_mix_names, weights, adapter_name, combination_type=args.lora_mix_mode, density=density)
    model.eval()
    model.set_adapter("merge")
    print(model.active_adapters)
    
    return tokenizer, model


def extract_answer(args, sentence: str) -> float:
    dataset = args.dataset
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'natural_questions':
        sentence_ = sentence.strip()
        if "|" in sentence_:
            sentence_ = sentence_.split("|")
        else:
            sentence_ = [sentence_]
        return sentence_
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        pred_options = re.findall(r'answer1|answer2', sentence_)
        if pred_answers:
            return pred_answers[0]
        if pred_options:
            return pred_options[0]
        return ""
    elif dataset in ['ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        pred_options = re.findall(r'option1|option2|option3|option4|option5', sentence_) 
        if pred_answers:
            return pred_answers[0]
        if pred_options:
            return pred_options[0]
        return ""
    elif dataset == 'social_i_qa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3', sentence_)
        pred_options = re.findall(r'option1|option2|option3', sentence_) 
        if pred_answers:
            return pred_answers[0]
        if pred_options:
            return pred_options[0]
        return ""
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        pred_options = re.findall(r'option1|option2|option3|option4', sentence_)
        if pred_answers:
            return pred_answers[0]
        if pred_options:
            return pred_options[0]
        return ""
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]


def extract_answer_number(args, sentence: str) -> float:
    dataset = args.dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp", "formats10", "gsm8k-hard"]:
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def extract_answer_letter(args, sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        if not pred_answers:
            return ''
        return pred_answers[0]
    else:
        return ''
