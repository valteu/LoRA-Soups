import warnings
import requests
import json

from transformers import AutoTokenizer
from pathlib import Path
from datasets import DatasetDict, Dataset, load_from_disk
from . import get_dataset_path
import copy

####################################
#        Dataset Creation          #
####################################


def _write_split_config(config: dict, sample_count: int, source_type: str, seed: int, split_name: str, source: str, full_name: str):
    config.setdefault("splits", {})[split_name] = {
        "sample_count": sample_count,
        "source_type": source_type,
        "source": source,
        "full_name": full_name,
        "seed": seed
    }


def write_all_split_configs(config: dict, dataset_type: str, sample_count: dict, source_type: str, seed: int, source: str, full_name: str):
    _write_split_config(config, sample_count["train"], source_type, seed, f"{dataset_type}_train", source, full_name)
    _write_split_config(config, sample_count["test"] - sample_count["train"], source_type, seed, f"{dataset_type}_test", source, full_name)
    _write_split_config(config, sample_count["validation"] - sample_count["test"], source_type, seed, f"{dataset_type}_validation", source, full_name)


def load_dataset_from_allen_ai(task_name: str):
    _allen_ai_task_url = "https://raw.githubusercontent.com/allenai/natural-instructions/refs/heads/master/tasks"
    dataset_url = f"{_allen_ai_task_url}/{task_name}.json"
    get_request = requests.get(dataset_url)
    try:
        get_request.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch dataset: {e}") from e

    response_data = get_request.json()["Instances"]
    return response_data, _allen_ai_task_url


def calculate_split_offsets(sample_count: int):
    split_offsets = {
        "train": 20000,
        "test": 22000,
        "validation": 24000
    }

    if (sample_count < sum(split_offsets.values())):
        split_offsets["train"] = int(0.8 * sample_count)
        split_offsets["test"] = split_offsets["train"] + int(0.1 * sample_count)
        split_offsets["validation"] = split_offsets["test"] + int(0.1 * sample_count)
    return split_offsets


def create_dataset_directory_in_share(task_name: str, task: str):
    base_path = get_dataset_path() / task_name
    try:
        base_path.mkdir(parents=False, exist_ok=False)
    except FileNotFoundError as e:
        warnings.warn(f"For {base_path} we ran into {e}, so we are skipping the split creation for {task}")
        return
    except FileExistsError:
        warnings.warn(f"We already created {task_name}, so we are skipping the split creation for {task}")
        return
    return base_path


def save_raw_split(base_path: Path, train_data: list, test_data: list, validation_data: list):
    train_path = base_path / "raw_train"
    test_path = base_path / "raw_test"
    validation_path = base_path / "raw_validation"

    train_path.mkdir()
    test_path.mkdir()
    validation_path.mkdir()

    with open(train_path / "data.json", "w") as f:
        json.dump(train_data, f, indent=4)
    with open(test_path / "data.json", "w") as f:
        json.dump(test_data, f, indent=4)
    with open(validation_path / "data.json", "w") as f:
        json.dump(validation_data, f, indent=4)


def get_dataset(dataset_id):
    return load_from_disk(get_dataset_path() / dataset_id)

####################################
#        Dataset Formatting        #
####################################


def generate_training_dict(question: str, answer: str) -> dict:
    chat = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return {"messages": chat}


def get_question_answer_splits(dataset_dictionary: DatasetDict, train_data: list, test_data: list, validation_data: list):
    qa_train_data = {
        "question": [d["input"] for d in train_data],
        "answer": [" ".join(d["output"]) for d in train_data]
    }
    qa_test_data = {
        "question": [d["input"] for d in test_data],
        "answer": [" ".join(d["output"]) for d in test_data]
    }
    qa_validation_data = {
        "question": [d["input"] for d in validation_data],
        "answer": [" ".join(d["output"]) for d in validation_data]
    }

    dataset_dictionary["qa_train"] = Dataset.from_dict(qa_train_data)
    dataset_dictionary["qa_test"] = Dataset.from_dict(qa_test_data)
    dataset_dictionary["qa_validation"] = Dataset.from_dict(qa_validation_data)

    return dataset_dictionary


def reformat_qa_dataset_to_chat_format(dataset: Dataset):

    return dataset.map(
        lambda data: generate_training_dict(
            data["question"],
            data["answer"],
        ),
    )


def prepare_datasets(
    dataset_id: str,
    validation_split: str,
    tokenizer: AutoTokenizer
):
    dataset_dict = get_dataset(dataset_id)
    train_data = dataset_dict["qa_train"]
    validation_data = dataset_dict[validation_split]

    formatted_train_data = reformat_qa_dataset_to_chat_format(train_data)

    tokenized_train_data = tokenize_dataset(formatted_train_data, tokenizer)

    return tokenized_train_data, validation_data


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    tokenized_dataset = [tokenize(sample, tokenizer) for sample in dataset]
    return Dataset.from_list(tokenized_dataset)


def _apply_chat_template(chat_subset, add_generation_prompt, continue_final_message, tokenizer):
    text = tokenizer.apply_chat_template(
        chat_subset,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
    )
    tokens = tokenizer([text])["input_ids"][0]
    return tokens


def tokenize(sample: dict, tokenizer: AutoTokenizer):
    formatted_chat_sample = sample["messages"]

    already_processed_input_ids = []
    already_processed_chat = []
    assistant_mask = []

    for i, turn in enumerate(formatted_chat_sample):
        is_last = i == len(formatted_chat_sample) - 1
        is_user = turn["role"] == "user"

        input_ids_with_turn = _apply_chat_template(
            already_processed_chat + [turn],
            add_generation_prompt=is_user and is_last,
            continue_final_message=not is_last,
            tokenizer=tokenizer
        )

        added_tokens_in_turn = len(input_ids_with_turn) - len(already_processed_input_ids)

        assistant_mask.extend([0] * added_tokens_in_turn if is_user else [1] * added_tokens_in_turn)

        already_processed_input_ids = input_ids_with_turn
        already_processed_chat.append(turn)

    input_ids = already_processed_input_ids
    labels = copy.deepcopy(input_ids)

    loss_mask_token = -100
    labels = [loss_mask_token if mask == 0 else label for mask, label in zip(assistant_mask, labels)]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids)
    }
