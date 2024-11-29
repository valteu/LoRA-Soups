from dotenv import load_dotenv
import os
load_dotenv()
os.environ['WANDB_MODE']="offline"
TOKEN = os.getenv("HF_TOKEN")

import sys
from typing import List
import argparse
import fire
import torch
import transformers
from datasets import load_dataset
from typing import List, Optional, Union
from accelerate.utils import DistributedType


import peft
from peft import (  # noqa: E402
    LoraConfig,
    # BottleneckConfig,
    # PrefixTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainerState,
    TrainerControl,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
    LlamaTokenizer,
    TrainerCallback
)

from peft.utils import  _get_submodules


from learnable_lora.peft_model import (
    CustomPeftModel
)

from try_llama import LinearLlamaModel, LinearLlamaForCausalLM

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
print(peft.__version__)

transformers.logging.set_verbosity_info()

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def train(
        # model/data params
        base_model: str = "",  # the only required argument
        model: str = "",
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        adapter: str = "lora",
        load_8bit : bool = False,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        use_gradient_checkpointing: bool = False,
        eval_step: int = 200,
        save_step: int = 200,
        # lora hyperparams
        lora_weights: List = [],
        lora_mix_mode: str = "", 
        num_virtual_tokens: int = 30,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_weights: {lora_weights}\n"
        f"lora_mix_mode: {lora_mix_mode}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if ddp and torch.cuda.is_available():
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    print("gradient_accumulation_steps", gradient_accumulation_steps)
    model_path=base_model   
    model = LinearLlamaForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
        token=TOKEN
    )
    model = prepare_model_for_int8_training(model, use_gradient_checkpointing=None)

    lora_weights_names = [f"adapter_{str(i)}" for i in range(len(lora_weights))]    
    num_adapters = len(lora_weights)
    model = CustomPeftModel.from_pretrained(
                model,
                lora_weights[0],
                torch_dtype=torch.float16,
                device_map={"":0},
                adapter_name='adapter_0',
            )
    for i in range(1,num_adapters):
        model.load_adapter(lora_weights[i], adapter_name=lora_weights_names[i])
    model.base_model.set_adapter(lora_weights_names)
    print(f"Active adapters {model.active_adapters}")
    
    model.train()
    model = model.to(device)
    for name,p in model.named_parameters():
        if "lora_learnable" not in name:
            p.requires_grad = False
        else:
            p.requires_grad = True
    print(f"Active adapters after {model.active_adapters}")

    tokenizer = LlamaTokenizer.from_pretrained(base_model, token=TOKEN)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference    

    
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None


    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    #https://github.com/huggingface/peft/issues/96
    class SavePeftModelCallback(TrainerCallback):
        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            ) 
            peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
            kwargs["model"].save_pretrained(peft_model_path)
            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            return control
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        callbacks=[SavePeftModelCallback],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=2,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    for name,p in trainer.model_wrapped.named_parameters():
        print(name,p.dtype)
    model.model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['LLaMA-7B','LLaMA-3B',  "LLaMA-13B",'BLOOM-7B', 'GPT-j-6B'], required=True)
    parser.add_argument('--base_model', default='openlm-research/open_llama_3b_v2', required=True)
    
    ## data
    parser.add_argument('--data_path', default="",
                        required=True)
    parser.add_argument('--micro_batch_size', default=4, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--cutoff_len', default=256, type=int)
    parser.add_argument('--val_set_size', default=120, type=int)
    parser.add_argument('--eval_step', default=80, type=int)
    parser.add_argument('--save_step', default=80, type=int)
    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--num_epochs', default=3, type=int)
    
    ## lora
    parser.add_argument('--adapter', choices=['LoRA', 'AdapterP', 'AdapterH', 'Parallel'],
                        required=True)
    parser.add_argument('--lora_weights', required=True, nargs='+', type=str, default=[])
    parser.add_argument('--lora_mix_mode', default=None)
    
    # 8bit
    parser.add_argument('--load_8bit', action='store_true', default=False)
    parser.add_argument('--output_dir', default="", type=str)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args.lora_weights, type(args.lora_weights))
    
    train(base_model=args.base_model,  # the only required argument
        model=args.model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        adapter=args.adapter,
        load_8bit=args.load_8bit,
        # training hyperparams
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        cutoff_len=args.cutoff_len,
        val_set_size=args.val_set_size,
        eval_step=args.eval_step,
        save_step=args.save_step,
        # lora hyperparams
        lora_weights=args.lora_weights,
        lora_mix_mode=args.lora_mix_mode,
        )

