import json
import random
from template import self_accessment_system_1,self_accessment_prompt
know_res = []
with open('/home/sda/wangzhijun/MedicalGPT/jbi/result/innate/llama3-8b-instruct/merge_know.jsonl','r',encoding='utf-8') as file:
    for line in file:
        dictionary = json.loads(line)
        know_res.append(dictionary)

unknow_res = []
with open('/home/sda/wangzhijun/MedicalGPT/jbi/result/innate/llama3-8b-instruct/merge_unknow.jsonl','r',encoding='utf-8') as file1:
    for line in file1:
        dictionary = json.loads(line)
        unknow_res.append(dictionary)

res = []
prompt = self_accessment_prompt
for data in know_res:
    per_dict = {}
    per_dict["system"] = self_accessment_system_1
    per_list = []
    per_user_dict = {"from":"human"}
    per_gpt_dict = {"from":"gpt"}
    per_prompt = prompt.render(question=data['question'], options=data['options'])
    per_user_dict["value"] = per_prompt
    answer_option =  list(data["answer"].keys())[0]
    answer  = answer_option + ": " + data["answer"][answer_option]
    per_gpt_dict["value"] = answer
    per_list.append(per_user_dict)
    per_list.append(per_gpt_dict)
    per_dict["conversations"] = per_list
    res.append(per_dict)

for data in unknow_res:
    per_dict = {}
    per_dict["system"] = self_accessment_system_1
    per_list = []
    per_user_dict = {"from":"human"}
    per_gpt_dict = {"from":"gpt"}
    per_prompt = prompt.render(question=data['question'], options=data['options'])
    per_user_dict["value"] = per_prompt
    per_gpt_dict["value"] = "###I am not very sure about the answer to this question."
    per_list.append(per_user_dict)
    per_list.append(per_gpt_dict)
    per_dict["conversations"] = per_list
    res.append(per_dict)


random.shuffle(res)


with open('/home/sda/wangzhijun/MedicalGPT/jbi/result/innate/llama3-8b-instruct/merge_train.jsonl', 'w', encoding='utf-8') as file:
    # 将列表写入文件
    json.dump(res, file, ensure_ascii=False, indent=4)