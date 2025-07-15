import json
from pathlib import Path
import numpy as np

from transformers import TrainingArguments
from peft import LoraConfig
from dataclasses import asdict
from enum import Enum


def make_json_serializable(configuration):
    if isinstance(configuration, Enum):
        return configuration.value
    elif isinstance(configuration, (np.float64, np.int64)):
        return configuration.item()
    elif isinstance(configuration, set):
        return list(configuration)
    elif isinstance(configuration, (dict, list, tuple)):
        if isinstance(configuration, dict):
            return {k: make_json_serializable(v) for k, v in configuration.items()}
        return [make_json_serializable(v) for v in configuration]
    else:
        return configuration


def save_experiment_config(
    trainer_logs: list,
    training_args: TrainingArguments,
    lora_config: LoraConfig,
    total_flops: int,
    total_energy: int
):

    experiment_config = {
        "training_args": training_args.to_dict(),
        "lora_config": asdict(lora_config) if lora_config else None,
        "total_flops": total_flops,
        "total_energy": total_energy,
        "logs": trainer_logs
    }

    serializable_config = make_json_serializable(experiment_config)

    with open(Path(training_args.output_dir) / "experiment_config.json", "w") as f:
        json.dump(serializable_config, f, indent=4)
