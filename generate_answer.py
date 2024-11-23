import os
import re
import json
from tqdm import tqdm
import torch
import time
import argparse
import transformers
from transformers import AutoTokenizer
import openai
from transformers import StoppingCriteria, StoppingCriteriaList
import tiktoken
import sys
from template import *
from modelscope import AutoModelForCausalLM, AutoTokenizer

# max_length = 8192
# context_length = 7168
max_length = 131072
context_length = 10000

templates = {"cot_system": general_cot_system, "cot_prompt": general_cot,
                    "medrag_system": general_medrag_system, "medrag_prompt": general_medrag,
                    "simple_system":simple_medrag_system,"simple_prompt":simple_medrag_prompt,
                    "innate_system":simple_generate_system,"innate_prompt":simple_generate_prompt,
                    "accessment_system":self_accessment_system,"accessment_prompt":self_accessment_prompt}
                    

model_name = '/home/sda/wangzhijun/MedicalGPT/jbi/llama3.1-8b-instrcut'


def generate_answer(model_name):
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype= torch.float16,
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = templates["innate_prompt"]

    res = []
    with open('/home/sda/wangzhijun/MedicalGPT/jbi/data/MedMCQA.jsonl','r',encoding='utf-8') as file:
        for line in file:
            dictionary = json.loads(line)
            res.append(dictionary)
    id = 0
    with open('/home/sda/wangzhijun/MedicalGPT/jbi/result/innate/MedMCQA-llama3.jsonl', 'w', encoding='utf-8') as jsonl_file:
        for example in tqdm(res,desc="Processing examples", unit="example", dynamic_ncols=True):
            # start_time = time.time()
            id += 1
            per_prompt = prompt.render(question=example['question'], options=example['options'])
            messages = [
                {"role": "system", "content": templates["innate_system"]},
                {"role": "user", "content": per_prompt}
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens = max_length,
                temperature = 0.2
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            response_dict = {'id':id,'answer':response}
            jsonl_file.write(json.dumps(response_dict) + '\n') 

            # elapsed_time = time.time() - start_time
            # tqdm.write(f"Processed question: {example['question']} | Response time: {elapsed_time:.2f}s")

# generate_answer(model_name)



def vllm_generate_answer(model_name):

    from vllm import LLM,SamplingParams

    prompt = templates["simple_prompt"]
    system_prompt = templates["simple_system"]
    sampling_params = SamplingParams(temperature=0.02, top_p=0.95)

    res = []
    with open('/root/autodl-tmp/merge.jsonl','r',encoding='utf-8') as file:
        for line in file:
            dictionary = json.loads(line)
            res.append(dictionary)

    example = res[0]

    per_prompt = prompt.render(question=example['question'], options=example['options'])

    full_prompt = f"{system_prompt}\n{per_prompt}"

    model = LLM(model_name)

    outputs = model.generate(full_prompt,sampling_params=sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(generated_text)
    


def test(model_name):
    res = []
    with open('/root/autodl-tmp/MedQA.jsonl','r',encoding='utf-8') as file:
        for line in file:
            dictionary = json.loads(line)
            res.append(dictionary)
   
    prompt = templates["innate_prompt"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype= torch.float16,
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

   
    for example in res[0:10]:
        per_prompt = prompt.render(question=example['question'], options=example['options'])
        messages = [
            {"role": "system", "content": templates["innate_system"]},
            {"role": "user", "content": per_prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_length,
            temperature = 0.2
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        

def llama3_generate(model_name,out_path,rag=False):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = transformers.pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16,
                # torch_dtype=torch.bfloat16,
                device_map="cuda"
            )
    
    prompt = templates["medrag_prompt"]

    res = []
    with open('/home/sda/wangzhijun/MedicalGPT/jbi/data/MMLU-Med-docs.jsonl','r',encoding='utf-8') as file:
        for line in file:
            dictionary = json.loads(line)
            res.append(dictionary)

    id = 0

    if not rag:

        with open(out_path, 'w', encoding='utf-8') as jsonl_file:
            for example in tqdm(res,desc="Processing examples", unit="example", dynamic_ncols=True):

                id += 1

                per_prompt = prompt.render(question=example['question'], options=example['options'])

                messages = [
                    {"role": "system", "content": templates["innate_system"]},
                    {"role": "user", "content": per_prompt}
                ]

                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                response = model(
                                text,
                                temperature=0.05,
                                eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                                pad_token_id=tokenizer.eos_token_id,
                                max_new_tokens = 2000,
                                truncation=True
                            )
                
                ans = response[0]["generated_text"][len(text):]
                response_dict = {'id':id,'answer':ans}
                jsonl_file.write(json.dumps(response_dict) + '\n') 

    else:
            with open(out_path, 'w', encoding='utf-8') as jsonl_file:
                for example in tqdm(res,desc="Processing examples", unit="example", dynamic_ncols=True):
    
                    id += 1
                    # rag
                    retrieved_snippets = example['docs']
                    contexts = ["Document [{:d}]  {:s}".format(idx, retrieved_snippets[idx]["content"]) for idx in range(len(retrieved_snippets))]
                    contexts = [tokenizer.decode(tokenizer.encode("\n".join(contexts), add_special_tokens=False)[:context_length])]
                    

                    per_prompt = prompt.render(context= contexts,question=example['question'], options=example['options'])

                    messages = [
                        {"role": "system", "content": templates["medrag_system"]},
                        {"role": "user", "content": per_prompt}
                    ]

                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    response = model(
                                    text,
                                    # temperature=0.2,
                                    do_sample = True,
                                    eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                                    pad_token_id=tokenizer.eos_token_id,
                                    max_new_tokens = 2000,
                                    truncation=True
                                )
                    
                    ans = response[0]["generated_text"][len(text):]
                    response_dict = {'id':id,'answer':ans}
                    jsonl_file.write(json.dumps(response_dict) + '\n')


llama3_generate(model_name,out_path='/home/sda/wangzhijun/MedicalGPT/jbi/result/rag/llama3-8b-instruct/MMLU-Med-cot-docs32-new_param.jsonl',rag=True)

# llamafactory-cli train examples/train_qlora/llama3_lora_sft_otfq.yaml

# 使用cot第一个模板是61多
# 0.6446

# huggingface-cli download --token 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 --resume-download meta-llama/Llama-3.1-8B --local-dir /home/sda/wangzhijun/MedicalGPT/jbi/llama-3.1-8b