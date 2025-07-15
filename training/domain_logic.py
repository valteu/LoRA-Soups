import argparse
import re
from argparse import Namespace as parserArguments
from pathlib import Path

import torch
from transformers import TrainingArguments
from peft import LoraConfig

from benchmark.evaluation.energy_evaluation import extrapolate_training_flops
from benchmark.profiler import NvidiaProfiler

from . import DataCollator, Trainer, MCCallback, RougeCallback
from tp_helper import get_adapter_path, ModelConfig, prepare_datasets, save_experiment_config


def parse_args():
    parser = argparse.ArgumentParser(description="Training CLI")

    parser.add_argument("--batch-size", type=int, help="Batch size for training", default=16)
    parser.add_argument("--dataset", type=str, required=True, help="DatasetID indicating where the dataset is located inside of the shared cluster storage")
    parser.add_argument("--train-split", type=str, required=True, help="Train split name (e.g. 'qa_train')")
    parser.add_argument("--validation-method", type=str, required=True, choices=["mc", "rouge"], help="Validation method")
    parser.add_argument("--validation-split", type=str, help="Validation split (only if validation is 'mc')", default="mc_validation")
    parser.add_argument("--rank", type=int, help="Rank parameter (optional)", default=16)
    parser.add_argument("--lora-alpha", type=int, help="LoRA alpha value (optional)", default=32)
    parser.add_argument("--fine-tuning-method", type=str, required=True, choices=["lora", "ff"], help="Fine-tuning method")
    parser.add_argument("--epochs", type=int, help="Number of training epochs (optional)", default=3)
    parser.add_argument("--learning-rate", type=float, help="Learning rate (optional)", default=2e-5)
    parser.add_argument("--lora-dropout", type=float, help="LoRA dropout", default=0.05)
    parser.add_argument("--logging-steps", type=int, help="Logging steps", default=20)
    parser.add_argument("--evaluation-steps", type=int, help="Evaluation steps", default=20)
    parser.add_argument("--gradient-accumulation-steps", type=int, help="Gradient accumulation steps", default=1)
    parser.add_argument("--save-steps", type=int, help="Save steps. Hint: Choose multiple of evaluation steps to ensure checkpoint is evaluated", default=40)
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--silent", dest="verbose", action="store_false", help="Disable verbose output")
    parser.set_defaults(verbose=True)

    args = parser.parse_args()

    return args


def get_lora_config(args: parserArguments):
    return LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "k_proj",
            "q_proj",
            "gate_proj",
            "up_proj",
            "o_proj",
            "v_proj",
            "down_proj"
        ],
        bias="none",
        use_rslora=True,
        task_type="CAUSAL_LM",
        lora_dropout=args.lora_dropout,
    )


def parse_input_args_to_training_args(args: parserArguments):

    output_path = get_adapter_path() / get_versioned_adapter_name(args.dataset, args.fine_tuning_method)

    return TrainingArguments(
        output_dir=output_path,
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.evaluation_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )


def get_versioned_adapter_name(dataset_name: str, fine_tuning_method: str) -> str:
    base_path = get_adapter_path()
    base_name = f"{dataset_name}_{fine_tuning_method}"
    pattern = re.compile(rf"^{re.escape(base_name)}_v(\d+)$")

    versions = []
    for path in base_path.iterdir():
        if path.is_dir():
            match = pattern.match(path.name)
            if match:
                versions.append(int(match.group(1)))

    next_version = max(versions, default=0) + 1
    return f"{base_name}_v{next_version}"


def get_valudation_method_callback(validation_method):
    callbacks = {
        "mc": MCCallback(),
        "rouge": RougeCallback()
    }
    return callbacks.get(validation_method, None) if validation_method in callbacks else None


def main(training_args: TrainingArguments, lora_config: LoraConfig = None, verbose: bool = True):
    model_config = ModelConfig(lora_config)
    model = model_config.get_model()
    tokenizer = model_config.get_tokenizer()
    collator = DataCollator(tokenizer)

    prepared_train_dataset, validation_dataset = prepare_datasets(
        args.dataset,
        args.validation_split,
        tokenizer
    )
    callbacks = [get_valudation_method_callback(args.validation_method)]

    trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset=prepared_train_dataset,
        validation_dataset=validation_dataset,
        tokenizer=tokenizer,
        collator=collator,
        callbacks=callbacks,
        verbose=verbose
    )

    total_flops, total_energy = None, None
    profiler_cache_path = Path(training_args.output_dir) / "profiler_cache.csv"
    with NvidiaProfiler(
        interval=1000,
        cache_file=profiler_cache_path,
        force_cache=True,
    ) as profiler:
        profiler.record_step("finetuning_start")
        trainer.train()
        profiler.record_step("finetuning_end")

    if torch.cuda.device_count() <= 1:
        total_flops = extrapolate_training_flops(
            model,
            tokenizer,
            num_training_samples=len(prepared_train_dataset),
            num_tokens_per_sequence=265,
            batch_size=training_args.per_device_train_batch_size,
            num_epochs=training_args.num_train_epochs,
            num_gradient_accumulation_steps=1
        )

    tmp_prof = NvidiaProfiler.from_cache(cache_file=profiler_cache_path)
    total_energy = tmp_prof.get_total_energy(record_steps=["finetuning_start"])

    save_experiment_config(trainer.state.logs, training_args, lora_config, total_flops, total_energy)

    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    training_args = parse_input_args_to_training_args(args)
    lora_config = get_lora_config(args)

    main(
        training_args=training_args,
        lora_config=lora_config,
        verbose=args.verbose,
    )
