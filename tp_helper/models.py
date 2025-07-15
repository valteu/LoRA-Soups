import json
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import DataCollatorForCompletionOnlyLM

from tp_helper import get_adapter_path

load_dotenv()


class ModelConfig:
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    padding_token = "<|finetune_right_pad_id|>"

    def __init__(self, lora_config: LoraConfig = None):
        self.lora_config = lora_config
        self.load_model()
        self._apply_lora_adapter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, truncation=False)
        self.tokenizer.add_special_tokens({"pad_token": self.padding_token})
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

    def reset_to_base_model(self):
        del self.model
        self.load_model()
        self._apply_lora_adapter()

    def _apply_lora_adapter(self):
        if self.lora_config is not None:
            self.model = get_peft_model(self.model, self.lora_config)

    def get_best_adapter_step(self, adapter_name: str):
        adapter_path = get_adapter_path() / adapter_name
        logs = {}
        with open(adapter_path / "experiment_config.json", "r") as f:
            logs = json.load(f).get("logs", {})
        log_types = list(set([log.get("type", "info") for log in logs]))
        eval_type = "mc" if "mc" in log_types else "rouge"

        def extract_score_from_log(log: dict):
            eval_content = log.get("content", {})
            if eval_type == "mc":
                return eval_content.get("accuracy", 0)
            elif eval_type == "rouge":
                return eval_content.get("rougeL", {}).get("fmeasure", 0)
            else:
                raise ValueError(f"Unknown evaluation type: {eval_type}")

        best_step = None
        best_score = 0
        for log in logs:
            if log.get("type", "info") == eval_type:
                if best_step is None or extract_score_from_log(log) > best_score:
                    best_step = log["step"]
                    best_score = extract_score_from_log(log)
        return best_step

    def load_best_adapter_checkpoint(self, adapter_name: str):
        adapter_path = get_adapter_path() / adapter_name
        best_step = self.get_best_adapter_step(adapter_name)

        checkpoints = list(adapter_path.glob("checkpoint-*"))
        if best_step is None:
            raise ValueError("No best step found in logs")
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {adapter_path}")
        if f"checkpoint-{best_step}" in [checkpoint.name for checkpoint in checkpoints]:
            print(f"Loading best checkpoint at step {best_step} from {adapter_path}")
            print(adapter_path / f"checkpoint-{best_step}")
            self.model.load_adapter(adapter_path / f"checkpoint-{best_step}", adapter_name=adapter_name)
        else:
            next_checkpoints = sorted(
                [checkpoint for checkpoint in checkpoints if int(checkpoint.name.split("-")[-1]) > best_step],
                key=lambda x: int(x.name.split("-")[-1])
            )
            if not next_checkpoints:
                raise ValueError(f"No checkpoints found after step {best_step} in {adapter_path}")
            print(f"Loading next checkpoint after step {best_step}: {next_checkpoints[0]}")
            self.model.load_adapter(next_checkpoints[0], adapter_name=adapter_name)


def setup_collator(
    tokenizer: PreTrainedTokenizer,
):
    training_marker = calculate_encoded_assistant_header_tokens(tokenizer)

    return DataCollatorForCompletionOnlyLM(
        response_template=training_marker,
        tokenizer=tokenizer,
    )


def calculate_encoded_assistant_header_tokens(
    tokenizer: PreTrainedTokenizer,
):
    training_marker_text = "<|start_header_id|>assistant<|end_header_id|>"
    return tokenizer.encode(training_marker_text, add_special_tokens=False)[1:]
