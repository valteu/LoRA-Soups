from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset


def evaluate_by_rouge(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    batch_size: int = 16
):
    model.eval()
    rouge_scores = []
    dataset_size = len(dataset["question"])
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL', 'rouge1', 'rouge2'], use_stemmer=True)

    for start_index in range(0, dataset_size, batch_size):
        end_index = min(start_index + batch_size, dataset_size)

        prompt_batch = dataset["question"][start_index:end_index]
        target_answer_list = dataset["answer"][start_index:end_index]
        generated_answers = _batched_inference(model, tokenizer, prompt_batch)

        for generated_answer, target_answer in zip(generated_answers, target_answer_list):

            if not generated_answer or not target_answer:
                continue

            scores = rouge_scorer_instance.score(target_answer, generated_answer)
            rouge_scores.append(scores)

    return {
        rouge_type: {
            measure: sum(getattr(score_dict[rouge_type], measure) for score_dict in rouge_scores) / len(rouge_scores)
            for measure in ['precision', 'recall', 'fmeasure']
        }
        for rouge_type in rouge_scores[0]
    }


def _apply_chat_template(question: str, tokenizer: AutoTokenizer):
    chat = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    return text


def _batched_inference(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt_batch: list):
    prompts = [
        _apply_chat_template(question=question, tokenizer=tokenizer)
        for question in prompt_batch
    ]
    device = model.device
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    result_ids = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=128,
        pad_token_id=tokenizer.pad_token_id,
    )
    print(result_ids.shape)
    input_dim = inputs.input_ids.shape[1]
    masked_result = result_ids[:, input_dim:]
    decoded_result = tokenizer.batch_decode(masked_result, skip_special_tokens=True)
    return decoded_result


if __name__ == "__main__":
    from tp_helper import ModelConfig, get_adapter_path, get_dataset
    model_config = ModelConfig()
    # model_config.load_best_adapter_checkpoint("qa_mmlu_elementary_mathematics_lora_v1")
    # model_config.load_best_adapter_checkpoint("nlg_rocstories_title_answer_generation_lora_v1")
    model_config.load_best_adapter_checkpoint("merged_nlg_math_rocstory_titles_lora_v1")
    model = model_config.get_model()
    # model.load_adapter(get_adapter_path() / "linear_nlg_math_rocstory_titles")
    # model.load_adapter(get_adapter_path() / "cat_nlg_math_rocstory_titles")
    # model.load_adapter(get_adapter_path() / "dareties_nlg_math_rocstory_titles")
    tokenizer = model_config.get_tokenizer()
    # dataset = get_dataset("qa_mmlu_elementary_mathematics")
    # dataset = get_dataset("nlg_rocstories_title_answer_generation")
    dataset = get_dataset("merged_nlg_math_rocstory_titles")
    rouge_results = evaluate_by_rouge(model, tokenizer, dataset["qa_validation"], batch_size=16)
    print(rouge_results)
