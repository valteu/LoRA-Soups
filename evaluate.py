from dotenv import load_dotenv
import os
load_dotenv()
os.environ['WANDB_MODE']="offline"
TOKEN = os.getenv("HF_TOKEN")
import copy
import json


import re
import argparse
import torch
from tqdm import tqdm
from transformers import GenerationConfig, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM

from utils import load_model, extract_answer, extract_answer_letter, extract_answer_number 

from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

list_math_ds = ['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP', 'formats10', 'gsm8k-hard']

def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        share_gradio: bool = False,
):
    args = parse_args()
    print(args)

    def evaluate(
            instructions,
            input=None,
            temperature=0.01,
            top_p=0.95,
            num_beams=1,
            max_new_tokens=None,
            **kwargs,
    ):
        

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_beams=num_beams,
            pad_token_id=0,
            **kwargs,
        )

        if args.dataset in list_math_ds:
            prompt = generate_prompt(instructions, args.task, input)
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            max_new_tokens=100
        else:
            prompts = [generate_prompt(instruction, args.task, input) for instruction in instructions]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(device)
            max_new_tokens=150
        
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        if args.dataset in list_math_ds:
            s = s[0]
            outputs = tokenizer.decode(s)
            outputs = outputs.split("### Response:")[1].strip()
        else:
            outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
            outputs = [o.split("Now write a Response for this Instruction. Output only the correct answer.\n\n")[1].split("### Response:")[1].strip() for o in outputs][0]
        return outputs

    save_file = f'experiment/{args.outfile}-{args.dataset}.json'
    create_dir('experiment/')

    dataset = load_data(args)
    if args.dataset in list_math_ds:
        batches = dataset
        total = len(dataset)
    else:
        batches = create_batch(dataset, args.batch_size)
        total = len(batches)
    tokenizer, model = load_model(args)
    correct = 0
    miss = 0.001
    current = 0
    exact_correct = 0
    output_data = []
    pbar = tqdm(total=total)
    for idx, batch in enumerate(batches):
        if args.dataset in list_math_ds:
            instructions = batch.get('instruction')
        else:
            current += len(batch)
            instructions = [data.get('instruction') for data in batch]
        outputs = evaluate(instructions)
        
        if args.dataset in list_math_ds:
            label = batch.get('answer')
            # # chnage
            data = batch
            flag = False
            if args.dataset.lower() in ['aqua']:
                predict = extract_answer_letter(args, outputs)
                if label == predict:
                    correct += 1
                    flag = True
            else:
                if isinstance(label, str):
                    label = float(label)
                predict = extract_answer_number(args, outputs)
                if abs(label - predict) <= miss:
                    correct += 1
                    flag = True
        else:
            for data, output in zip(batch, outputs):
                label = data.get('answer')
                flag = False
                
                if args.dataset=="natural_questions":
                    predict_list = extract_answer(args, output)
                    if "|" in label:
                        label = label.split("|")
                    else:
                        label = [label]

                    max_sim = 0
                    predict = predict_list[0] if predict_list else ""
                    for pred in predict_list:
                        for lbl in label:
                            tfidf = vect.fit_transform([pred, lbl])
                            similarity = tfidf * tfidf.T
                            sim = similarity.toarray()[0][1]
                            if sim > max_sim:
                                predict = pred
                                max_sim = sim
                    if max_sim > 0.3:
                        correct +=1
                        flag = True

                elif args.dataset in ["ARC-Challenge", "ARC-Easy", "social_i_qa", "hellaswag", "boolq", "piqa", "winogrande"]:
                    predict = extract_answer(args, output)
                    if label == predict:
                        exact_correct += 1
                        flag = True
                    if predict and label[-1] == predict[-1]:
                        correct += 1
                elif args.dataset in ["squad", "qgen", "mbpp", "bioasq", "knowledge"]:
                    predict = extract_answer(args, output)
                    if label == predict:
                        correct += 1
                        flag = True
                else:
                    predict = extract_answer(args, output)
                    if label == predict:
                        correct += 1
                        flag = True
        new_data = copy.deepcopy(data)
        new_data['output_pred'] = outputs
        new_data['pred'] = predict
        new_data['flag'] = flag
        output_data.append(new_data)
        print(data["instruction"])
        print(outputs)
        print('prediction:', predict)
        print('label:', label)
        print('---------------')
        if args.dataset in list_math_ds:
            print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / (idx + 1)}', flush=True)
        else:
            print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current}', flush=True)
            print(f'\rtest:{idx + 1}/{total} | exact acc {exact_correct}  {exact_correct / current}', flush=True)

        print('---------------')
        with open(save_file, 'w+') as f:
            json.dump(output_data, f, indent=4)
        pbar.update(1)
    pbar.close()
    print('\n')
    print('test finished')


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_math_prompt(instruction, input=None):
    with open("data/few_shot_demo/math.json", "r") as f:
        cot_data = json.load(f)
    prompt = """Let's use Python to solve math problems step by step. Below are a few Instruction-Response pairs on how to do it."""
    prompt += "\n\n"
    for data in cot_data:
        prompt += f"### Instruction:\n{data['instruction']}\n\n### Response:\n{data['output']}\n\n"
    prompt += "Now write a function 'solution' encolsed in ``` in Python to solve this Instruction. Write only a code block. Write only valid Python code without using any units with the numerical values and any invalid symbols.\n\n"
    prompt += f"### Instruction:\n{instruction}\n\n### Response:\n"
    return prompt

def generate_qa_prompt(instruction, input=None):
    with open("data/few_shot_demo/qa.json", "r") as f:
        cot_data = json.load(f)
    cot_data = cot_data[:1]
    prompt = """Below are a few Instruction-Response pairs on answering questions."""
    prompt += "\n\n"
    for data in cot_data:
        prompt += f"### Instruction:\n{data['instruction']}\n\n### Response:\n{data['output']}\n\n"
    prompt += "Now write a Response for this Instruction. Output only the correct answer.\n\n"
    prompt += f"### Instruction:\n{instruction}\n\n### Response:\n"
    return prompt

def generate_rc_prompt(instruction, input=None):
    with open("data/few_shot_demo/rc.json", "r") as f:
        cot_data = json.load(f)
    prompt = """Below are a few Instruction-Response pairs on solving reading comprehension."""
    prompt += "\n\n"
    for data in cot_data:
        prompt += f"### Instruction:\n{data['instruction']}\n\n### Response:\n{data['output']}\n\n"
    prompt += "Now write a Response for this Instruction. Output only the correct answer.\n\n"
    prompt += f"### Instruction:\n{instruction}\n\n### Response:\n"
    return prompt

def generate_prompt(instruction, task, input=None):
    if task == 'rc':
        return generate_rc_prompt(instruction, input)
    elif task == 'math':
        return generate_math_prompt(instruction, input)
    elif task == 'qa':
        return generate_qa_prompt(instruction, input)
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f"data/{args.task}/test.json"
    if args.task == "prompt":
        file_path = f"data/{args.task}/test_instructions_format3.json"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data

def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', required=True)
    parser.add_argument('--model', choices=['LLaMA-7B', "LLaMA-13B",'BLOOM-7B', 'GPT-j-6B'], required=True)
    parser.add_argument('--adapter', choices=['LoRA', 'AdapterP', 'AdapterH', 'Parallel'],
                        required=True)
    parser.add_argument('--task', choices=['math', 'qa', 'rc', 'prompt'],
                        required=True, help="the task you want to evaluate")
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--lora_weights', required=True, nargs='+', type=str, default=[])
    parser.add_argument('--lora_mix_mode', default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--load_8bit', action='store_true', default=False)
    parser.add_argument('--outfile', type=str, default=None)
    parser.add_argument('--add_special_toks', action='store_true', default=False)
    parser.add_argument('--no_lora', action='store_true', default=False)
    parser.add_argument('--moe', action='store_true', help="Evaluate a trained MoE model", default=False)
    return parser.parse_args()



def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction



if __name__ == "__main__":
    main()
