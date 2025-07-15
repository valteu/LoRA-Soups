import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def _process_batch(
    stems: list,
    options_list: list,
    target_options: list,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM
):

    amount_correct_answers_in_batch = 0

    prompts = [_apply_mc_question_template(stem, tokenizer) for stem in stems]
    option_ids = [tokenizer.convert_tokens_to_ids(o) for o in options_list]
    outputs = _batched_single_token_inference(model, tokenizer, prompts, option_ids)

    for output, target_option in zip(outputs, target_options):
        predicted_token = tokenizer.decode(output, skip_special_tokens=True).strip()

        is_correct = predicted_token.lower() == target_option.lower()

        if is_correct:
            amount_correct_answers_in_batch += 1

    return amount_correct_answers_in_batch


def evaluate_by_multiple_choice(
    model,
    tokenizer,
    dataset: Dataset,
    batch_size: int,
):
    model.eval()
    dataset_size = len(dataset["stem"])

    dataset = dataset.add_column("target_option", [
        dataset["options"][i][dataset["correct_option_indices"][i][0]]
        for i in range(dataset_size)
    ])

    total_amount_correct_answers = 0

    for batch_start_index in range(0, dataset_size, batch_size):
        batch_end_index = min(batch_start_index + batch_size, dataset_size)

        stems = dataset["stem"][batch_start_index:batch_end_index]
        options_list = dataset["options"][batch_start_index:batch_end_index]
        target_options = dataset["target_option"][batch_start_index:batch_end_index]

        total_amount_correct_answers += _process_batch(
            stems=stems,
            options_list=options_list,
            target_options=target_options,
            tokenizer=tokenizer,
            model=model
        )

    accuracy = total_amount_correct_answers / dataset_size if dataset_size > 0 else 0.0

    return {
        "accuracy": accuracy,
        "total_correct": total_amount_correct_answers,
        "total_questions": dataset_size,
    }


def _apply_mc_question_template(stem: str, tokenizer: AutoTokenizer):
    chat = [{"role": "user", "content": stem}]
    text = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    return text


def _batched_single_token_inference(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    option_ids: list
):
    device = model.device
    tokenizer.padding_side = "left"

    tokenized_inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**tokenized_inputs)
        logits = outputs.logits
        last_token_logits = logits[:, -1, :]

        option_ids_tensor = torch.tensor(option_ids, device=device)
        option_logits = last_token_logits.gather(1, option_ids_tensor)
        predicted_token_ids = torch.argmax(option_logits, dim=-1)

        chosen_option_ids = option_ids_tensor[torch.arange(len(option_ids_tensor)), predicted_token_ids]

    return chosen_option_ids.cpu().tolist()
