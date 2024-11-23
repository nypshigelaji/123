import json
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('/home/sda/wangzhijun/MedicalGPT/jbi/llama3-8b-instruct')

res = []
with open('/home/sda/wangzhijun/MedicalGPT/jbi/data/MMLU-Med-docs.jsonl','r',encoding='utf-8') as file:
    for line in file:
        dictionary = json.loads(line)
        res.append(dictionary)

for data in res:
    retrieved_snippets = data['docs']
    contexts = ["Document [{:d}]  {:s}".format(idx, retrieved_snippets[idx]["content"]) for idx in range(len(retrieved_snippets))]
    contexts = [tokenizer.decode(tokenizer.encode("\n".join(contexts), add_special_tokens=False)[:11000])]
    print(contexts)
    break

# huggingface-cli download --token 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 --resume-download meta-llama/Llama-3.1-8B-Instruct --local-dir /home/sda/wangzhijun/MedicalGPT/jbi/llama3.1-8b-instruct