import os
import re
import json
import tqdm
import torch
import time
import argparse
import transformers
from transformers import AutoTokenizer
import openai
from transformers import StoppingCriteria, StoppingCriteriaList
import tiktoken
import sys
sys.path.append("src")
from utils import RetrievalSystem, DocExtracter
from template import *

from config import config

openai.api_type = openai.api_type or os.getenv("OPENAI_API_TYPE") or config.get("api_type")
openai.api_version = openai.api_version or os.getenv("OPENAI_API_VERSION") or config.get("api_version")
openai.api_key = openai.api_key or os.getenv('OPENAI_API_KEY') or config["api_key"]

if openai.__version__.startswith("0"):
    openai.api_base = openai.api_base or os.getenv("OPENAI_API_BASE") or config.get("api_base")
    if openai.api_type == "azure":
        openai_client = lambda **x: openai.ChatCompletion.create(**{'engine' if k == 'model' else k: v for k, v in x.items()})["choices"][0]["message"]["content"]
    else:
        openai_client = lambda **x: openai.ChatCompletion.create(**x)["choices"][0]["message"]["content"]
else:
    if openai.api_type == "azure":
        openai.azure_endpoint = openai.azure_endpoint or os.getenv("OPENAI_ENDPOINT") or config.get("azure_endpoint")
        openai_client = lambda **x: openai.AzureOpenAI(
            api_version=openai.api_version,
            azure_endpoint=openai.azure_endpoint,
            api_key=openai.api_key,
        ).chat.completions.create(**x).choices[0].message.content
    else:
        openai_client = lambda **x: openai.OpenAI(
            api_key=openai.api_key,
        ).chat.completions.create(**x).choices[0].message.content

class MedRAG:

    def __init__(self, llm_name="OpenAI/gpt-3.5-turbo-16k", rag=True, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus", cache_dir=None):
        self.llm_name = llm_name
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.docExt = None
        if rag:
            self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir)
        else:
            self.retrieval_system = None
        self.templates = {"cot_system": general_cot_system, "cot_prompt": general_cot,
                    "medrag_system": general_medrag_system, "medrag_prompt": general_medrag}
        if self.llm_name.split('/')[0].lower() == "openai":
            self.model = self.llm_name.split('/')[-1]
            if "gpt-3.5" in self.model or "gpt-35" in self.model:
                self.max_length = 16384
                self.context_length = 15000
            elif "gpt-4" in self.model:
                self.max_length = 32768
                self.context_length = 30000
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        elif "gemini" in self.llm_name.lower():
            import google.generativeai as genai
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
            self.model = genai.GenerativeModel(
                model_name=self.llm_name.split('/')[-1],
                generation_config={
                    "temperature": 0,
                    "max_output_tokens": 2048,
                }
            )
            if "1.5" in self.llm_name.lower():
                self.max_length = 1048576
                self.context_length = 1040384
            else:
                self.max_length = 30720
                self.context_length = 28672
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            self.max_length = 4096
            self.context_length = 3072
            self.max_new_tokens = 1024
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
            if "mixtral" in llm_name.lower():
                self.tokenizer.chat_template = open('./templates/mistral-instruct.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 32768
                self.context_length = 30000
            elif "llama-2" in llm_name.lower():
                self.max_length = 4096
                self.context_length = 3072  #! 设置为3072 有可能导致输出长度不够，内容不完整，无法输出最后的选项
                self.max_new_tokens = 1024
            elif "llama-3" in llm_name.lower():
                # self.max_length = 4096
                # self.context_length = 3072
                # self.max_new_tokens = 1024
                self.max_length = 8192
                self.context_length = 6000
                self.max_new_tokens = 2000
                if ".1" in llm_name or ".2" in llm_name:
                    # self.max_length = 131072
                    # self.context_length = 128000
                    self.max_length = 12000
                    self.context_length = 10000
                    self.max_new_tokens = 2000
            elif "meditron" in llm_name.lower():
                self.tokenizer.chat_template = open('./templates/meditron.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 4096
                self.context_length = 3072
                self.templates["cot_prompt"] = meditron_cot
                self.templates["medrag_prompt"] = meditron_medrag
            elif "pmc_llama" in llm_name.lower():
                self.tokenizer.chat_template = open('./templates/pmc_llama.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 2048
                self.context_length = 1024
            elif "mistral" in llm_name.lower():
                self.tokenizer.chat_template = open('./templates/mistral-instruct.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 4096
                self.context_length = 3072
                self.max_new_tokens = 1024
            self.model = transformers.pipeline(
                "text-generation",
                model=self.llm_name,
                # torch_dtype=torch.float16,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                model_kwargs={"cache_dir":self.cache_dir},
            )

    def answer(self, question, options=None, k=32, rrf_k=100, save_dir = None, snippets=None, snippets_ids=None):
        '''
        question (str): question to be answered
        options (Dict[str, str]): options to be chosen from
        k (int): number of snippets to retrieve
        rrf_k (int): parameter for Reciprocal Rank Fusion
        save_dir (str): directory to save the results
        snippets (List[Dict]): list of snippets to be used
        snippets_ids (List[Dict]): list of snippet ids to be used
        '''

        if options is not None:
            options = '\n'.join([key+". "+options[key] for key in sorted(options.keys())])
        else:
            options = ''

        # retrieve relevant snippets
        if self.rag:
            if snippets is not None:
                retrieved_snippets = snippets[:k]
                scores = []
            elif snippets_ids is not None:
                if self.docExt is None:
                    self.docExt = DocExtracter(db_dir=self.db_dir, cache=False, corpus_name=self.corpus_name)
                retrieved_snippets = self.docExt.extract(snippets_ids[:k])
                scores = []
            else:
                assert self.retrieval_system is not None
                retrieved_snippets, scores = self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k)
            # print(retrieved_snippets)
            # print("==============")
            # print(len(retrieved_snippets))
            # print("==============")
            contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, retrieved_snippets[idx]["title"], retrieved_snippets[idx]["content"]) for idx in range(len
            (retrieved_snippets))]
            # print("==============")
            # print(contexts)
            if len(contexts) == 0:
                contexts = [""]
            if "openai" in self.llm_name.lower():
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts))[:self.context_length])]
            elif "gemini" in self.llm_name.lower():
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts))[:self.context_length])]
            else:
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts), add_special_tokens=False)[:self.context_length])]
                # print("===========len of contexts =========",len(contexts))
                # print("===========context===========",contexts[0])
                # print(len(self.tokenizer.encode(contexts[0])))  #! 3073
                #! tokenizer 会将内容truncate 到 设置的context length 长度即3072
        else:
            retrieved_snippets = []
            scores = []
            contexts = []

        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # generate answers
        answers = []
        if not self.rag:
            #* 使用liquid 包的 Template.render
            #* 返回值为字符串
            prompt_cot = self.templates["cot_prompt"].render(question=question, options=options)
            messages = [
                {"role": "system", "content": self.templates["cot_system"]},
                {"role": "user", "content": prompt_cot}
            ]
            ans = self.generate(messages)
            print("==========ans==========", ans)
            answers.append(re.sub("\s+", " ", ans))
        else:
            for context in contexts:
                prompt_medrag = self.templates["medrag_prompt"].render(context=context, question=question, options=options)
                messages=[
                        {"role": "system", "content": self.templates["medrag_system"]},
                        {"role": "user", "content": prompt_medrag}
                ]
                ans = self.generate(messages)
                print("==========ans==========", ans)
                answers.append(re.sub("\s+", " ", ans))
                # print("==========answer=========", answers[-1])
        if save_dir is not None:
            with open(os.path.join(save_dir, "snippets.json"), 'w') as f:
                json.dump(retrieved_snippets, f, indent=4)
            with open(os.path.join(save_dir, "response.json"), 'w') as f:
                json.dump(answers, f, indent=4)
        
        return answers[0] if len(answers)==1 else answers, retrieved_snippets, scores
            
    def custom_stop(self, stop_str, input_len=0):
        stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(stop_str, self.tokenizer, input_len)])
        return stopping_criteria

    def generate(self, messages):
        '''
        generate response given messages
        '''
        if "openai" in self.llm_name.lower():
            ans = openai_client(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
        elif "gemini" in self.llm_name.lower():
            response = self.model.generate_content(messages[0]["content"] + '\n\n' + messages[1]["content"])
            ans = response.candidates[0].content.parts[0].text
        else:
            stopping_criteria = None
            # print(self.tokenizer.default_chat_template)
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # print(prompt)
            #! add_generation_prompt 用来在尾部添加 assistant 这样的提示， 如果模型的tokenizer有设置的话
            if "meditron" in self.llm_name.lower():
                # stopping_criteria = custom_stop(["###", "User:", "\n\n\n"], self.tokenizer, input_len=len(self.tokenizer.encode(prompt_cot, add_special_tokens=True)))
                stopping_criteria = self.custom_stop(["###", "User:", "\n\n\n"], input_len=len(self.tokenizer.encode(prompt, add_special_tokens=True)))
            if "llama-3" in self.llm_name.lower():
                response = self.model(
                    prompt,
                    do_sample=True,
                    temperature =0.2,
                    eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                    pad_token_id=self.tokenizer.eos_token_id,
                    # max_length=self.max_length,                    
                    max_new_tokens = self.max_new_tokens, 
                    truncation=True,
                    stopping_criteria=stopping_criteria
                )
            elif "mistral" in self.llm_name.lower():
                response = self.model(
                    prompt,
                    do_sample=False,
                    eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                    pad_token_id=self.tokenizer.eos_token_id,
                    # max_length=self.max_length,
                    max_new_tokens = self.max_new_tokens, 
                    truncation=True,
                    stopping_criteria=stopping_criteria
                )
            else:
                response = self.model(
                    prompt,
                    do_sample=True, #* 尝试do_sample
                    # num_beams=2,#* 尝试 beam search
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    # max_length=self.max_length,
                    max_new_tokens = self.max_new_tokens, 
                    truncation=True,
                    stopping_criteria=stopping_criteria
                )
            #! 这里的model在初始化时设置为了 text_generation 的pipeline
            #! 在返回的时候也会包含输入部分
            input_len=len(self.tokenizer.encode(prompt, add_special_tokens=True))
            # print(self.tokenizer.model_max_length)
            # print(input_len) # 3340  #* 添加了系统指令，与问题后的长度， 并添加了special token
            prompt_len = len(prompt)
            # print(prompt_len)
            ans = response[0]["generated_text"]
            # ans = response[0]["generated_text"][prompt_len:]

            #! 当输入超出了预设的长度时，返回的generated_text里不会包含超出输入的部分，因此最好将指令发在上下文之前
        return ans

class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, input_len=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops_words = stop_words
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        tokens = self.tokenizer.decode(input_ids[0][self.input_len:])
        return any(stop in tokens for stop in self.stops_words)