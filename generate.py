import os
import re
import json
from tqdm import tqdm
import torch
import transformers
from transformers import AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import tiktoken
import sys
from template import *

class MedRAG_innate:
    def __init__(self,rag = False,llm_name = '/home/sda/wangzhijun/MedicalGPT/jbi/llama3.1-8b-instrcut',self_accessment = False):
        
        self.templates = {"cot_system": general_cot_system_a, "cot_prompt": general_cot_a,
                            "medrag_system": general_medrag_system, "medrag_prompt": general_medrag,
                            "simple_system":simple_medrag_system,"simple_prompt":simple_medrag_prompt,
                            "innate_system":simple_generate_system,"innate_prompt":simple_generate_prompt,
                            "accessment_system":self_accessment_system,"accessment_prompt":self_accessment_prompt}

        self.max_length = 131072
        self.context_length = 128000

        self.model_name = llm_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)


        self.model = transformers.pipeline(
                        "text-generation",
                        model=self.model_name,
                        # torch_dtype=torch.float16,
                        torch_dtype=torch.bfloat16,
                        device_map="cuda"
                    )
        self.rag = rag
        self.self_accessment = self_accessment

        if not self.self_accessment:
            self.prompt = self.templates["simple_prompt"] if rag else self.templates["innate_prompt"]
        else:
            self.prompt = self.templates["accessment_prompt"]

    def generate(self,example):

        rag = self.rag
        if not self.self_accessment:
            if rag:

                retrieved_snippets = example['docs']
                contexts = ["Document [{:d}]  {:s}".format(idx, retrieved_snippets[idx]["content"]) for idx in range(len(retrieved_snippets))]
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts), add_special_tokens=False)[:self.context_length])]
                

                per_prompt = self.prompt.render(context= contexts,question=example['question'], options=example['options'])

                messages = [
                    {"role": "system", "content": self.templates["simple_system"]},
                    {"role": "user", "content": per_prompt}
                ]
                
            
            else:

                per_prompt = self.prompt.render(question=example['question'], options=example['options'])

                messages = [
                    {"role": "system", "content": self.templates["innate_system"]},
                    {"role": "user", "content": per_prompt}
                ]
        else:

            per_prompt = self.prompt.render(question=example['question'], options=example['options'])

            messages = [
                {"role": "system", "content": self.templates["accessment_system"]},
                {"role": "user", "content": per_prompt}
            ]


        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        response = self.model(
                        prompt,
                        temperature = 1,
                        eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                        pad_token_id=self.tokenizer.eos_token_id,
                        max_new_tokens = 2000,
                        truncation=True
                    )

        ans = response[0]["generated_text"][len(prompt):]

        return ans
    
    def medrag_answer(self,data_path,output_path):

        res = []
        
        with open(data_path,'r',encoding='utf-8') as file:
            for line in file:
                dictionary = json.loads(line)
                res.append(dictionary)

        id = 0


        with open(output_path, 'w', encoding='utf-8') as jsonl_file:
            for example in tqdm(res,desc="Processing examples", unit="example", dynamic_ncols=True):

                id += 1
                ans = self.generate(example)
                response_dict = {'id':id,'answer':ans}
                jsonl_file.write(json.dumps(response_dict) + '\n')
        print("sucess!")

class MedRAG:
    def __init__(self,rag = False,llm_name = '/home/sda/wangzhijun/MedicalGPT/jbi/llama3.1-8b-instrcut'):
        
        self.templates = {"cot_system": general_cot_system, "cot_prompt": general_cot,
                            "medrag_system": general_medrag_system, "medrag_prompt": general_medrag}

        self.max_length = 131072
        self.context_length = 128000

        self.model_name = llm_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)


        self.model = transformers.pipeline(
                        "text-generation",
                        model=self.model_name,
                        # torch_dtype=torch.float16,
                        torch_dtype=torch.bfloat16,
                        device_map="cuda"
                    )
        self.rag = rag

        self.prompt = self.templates["medrag_prompt"] if rag else self.templates["cot_prompt"]


    def generate(self,example):

        rag = self.rag

        if rag:

            retrieved_snippets = example['docs']
            contexts = ["Document [{:d}]  {:s}".format(idx, retrieved_snippets[idx]["content"]) for idx in range(len(retrieved_snippets))]
            contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts), add_special_tokens=False)[:self.context_length])]
            

            per_prompt = self.prompt.render(context= contexts,question=example['question'], options=example['options'])

            messages = [
                {"role": "system", "content": self.templates["medrag_system"]},
                {"role": "user", "content": per_prompt}
            ]
            
        
        else:

            per_prompt = self.prompt.render(question=example['question'], options=example['options'])

            messages = [
                {"role": "system", "content": self.templates["cot_system"]},
                {"role": "user", "content": per_prompt}
            ]
            
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        response = self.model(
                        prompt,
                        # temperature = 0.2,
                        do_sample = True,
                        eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                        pad_token_id=self.tokenizer.eos_token_id,
                        max_new_tokens = 2000,
                        truncation=True
                    )

        ans = response[0]["generated_text"][len(prompt):]

        return ans
    
    def medrag_answer(self,data_path,output_path):

        res = []

        with open(data_path,'r',encoding='utf-8') as file:
            for line in file:
                dictionary = json.loads(line)
                res.append(dictionary)

        id = 0


        with open(output_path, 'w', encoding='utf-8') as jsonl_file:
            for example in tqdm(res,desc="Processing examples", unit="example", dynamic_ncols=True):

                id += 1
                ans = self.generate(example)
                response_dict = {'id':id,'answer':ans}
                jsonl_file.write(json.dumps(response_dict) + '\n')
        print("sucess!")

# medrag = MedRAG(rag=False)
medrag = MedRAG_innate(rag=False,self_accessment=True)
data_path = "/home/sda/wangzhijun/MedicalGPT/jbi/data/merge.jsonl"
output_path = "/home/sda/wangzhijun/MedicalGPT/jbi/result/self-accessment/merge_1.jsonl"
medrag.medrag_answer(data_path,output_path)
