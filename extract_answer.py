import json
import os
import re

import numpy as np

# 匹配答案中的花括号
def braces_match(text):
    match = re.search(r'\{([^{}]*)\}', text)
    if match:
        content = match.group(0)  # 提取整个花括号内容，包括大括号
        return content
    else:
        return '#404'
    
def extract_ans(gold_path,generate_path):

    # 匹配答案
    pattern = re.compile(r'["\']?(?:answer_choice)?["\']?:?\s*["\']?([A-D]):?')

    gold = []
    options = []
    with open(gold_path,'r',encoding='utf-8') as file:

        for line in file:
            dictionary = json.loads(line)
            gold.append(dictionary['answer'])
            options.append(list(dictionary['options'].values()))
    
    correct_num = 0
    total_num = len(gold)

    uk = 0

    with open(generate_path,'r',encoding='utf-8') as file:

        for i,line in enumerate(file):

            # 正确答案 A,B,C,D
            current_gold = list(gold[i].keys())[0]
            dictionary = json.loads(line)

            # 花括号中的内容
            generation = braces_match(dictionary['answer'])

            if generation != '#404':

                # 匹配的答案

                match = pattern.search(generation)

                if match:

                    prediction = match.group(1)

                    if prediction.lower() == current_gold.lower():
                        correct_num += 1
                    else:
                        if gold[i][current_gold] in generation:
                            correct_num += 1

                else:
                    if gold[i][current_gold] in generation:
                        correct_num += 1
            else:
                current_option = options[i]
                if gold[i][current_gold] in current_option:
                    current_option.remove(gold[i][current_gold])
                    num = 0
                    if gold[i][current_gold] in dictionary['answer']:
                        for item in current_option:
                            if item not in dictionary['answer']:
                                num += 1
                        if num == 3:
                            correct_num += 1
            
    
    precion = correct_num / total_num
    print(f"correct_num:{correct_num},total_num:{total_num},precion:{precion:.4f}")

def knoworunknow_data(docs_path,gold_merge_path):
    doc_path = '/home/sda/wangzhijun/MedicalGPT/jbi/result/innate/llama3-8b-instruct'
    fnames = sorted([fname for fname in os.listdir(doc_path) if fname.endswith(".jsonl") and 'merge' in fname])

    pattern = re.compile(r'["\']?(?:answer_choice)?["\']?:?\s*["\']?([A-D]):?')


    gold = []
    options = []
    
    with open(gold_merge_path,'r',encoding='utf-8') as file:

        for line in file:
            dictionary = json.loads(line)
            gold.append(dictionary['answer'])
            options.append(list(dictionary['options'].values()))
    
    total_num = len(gold)

    result = [[0 for _ in range(len(fnames))] for _ in range(total_num)]
    
    for id,fname in enumerate(fnames):
        fname = os.path.join(doc_path,fname)

        with open(fname,'r',encoding='utf-8') as file:

            for i,line in enumerate(file):

            # 正确答案 A,B,C,D
                current_gold = list(gold[i].keys())[0]
                dictionary = json.loads(line)

                # 花括号中的内容
                generation = braces_match(dictionary['answer'])

                if generation != '#404':

                    # 匹配的答案

                    match = pattern.search(generation)

                    if match:

                        prediction = match.group(1)

                        if prediction.lower() == current_gold.lower():
                            result[i][id] = 1
                        else:
                            if gold[i][current_gold] in generation:
                                result[i][id] = 1

                    else:
                        if gold[i][current_gold] in generation:
                            result[i][id] = 1
                else:
                    current_option = options[i]
                    if gold[i][current_gold] in current_option:
                        current_option.remove(gold[i][current_gold])
                        num = 0
                        if gold[i][current_gold] in dictionary['answer']:
                            for item in current_option:
                                if item not in dictionary['answer']:
                                    num += 1
                            if num == 3:
                                result[i][id] = 1
           
    res = [False for _ in range(total_num)]

    for i in range(total_num):
        if sum(result[i]) == len(fnames):
            res[i] = True
    
    know_list = []
    unknow_list = []
    
    with open(gold_merge_path,'r',encoding='utf-8') as file:

        for i,line in enumerate(file):
            dictionary = json.loads(line)
            if res[i]:
                know_list.append(dictionary)
            else:
                unknow_list.append(dictionary)
    
    know_path = os.path.join(docs_path,'merge_know.jsonl')
    unknow_path = os.path.join(doc_path,'merge_unknow.jsonl')
    with open(know_path, 'w', encoding='utf-8') as jsonl_file:
        for dict in know_list:
            jsonl_file.write(json.dumps(dict) + '\n') 

    with open(unknow_path, 'w', encoding='utf-8') as jsonl_file:
        for dict in unknow_list:
            jsonl_file.write(json.dumps(dict) + '\n') 
    print('sucess!')
# docs_path = '/home/sda/wangzhijun/MedicalGPT/jbi/result/innate/llama3-8b-instruct'
# gold_merge_path = '/home/sda/wangzhijun/MedicalGPT/jbi/data/merge.jsonl'
# knoworunknow_data(docs_path,gold_merge_path)

def self_accessment_generate_answer(gold_path,generate_path):

    tag = "###I am not very sure about the answer to this question."
    gold = []
    gold_dict = []
    know_dict = []
    unknow_dict = []
    know_generation = []
    unknow_generation = []
    pattern = r"^[A-Za-z0-9]+[：:]\s?[\s\S]*$"
    with open(gold_path,'r',encoding='utf-8') as file:

        for line in file:
            dictionary = json.loads(line)
            gold_dict.append(dictionary)
            gold_answer = list(dictionary['answer'].keys())[0]
            print(gold_answer)
            gold.append(gold_answer)


    with open(generate_path,'r',encoding='utf-8') as file:

        for line in file:
            dictionary = json.loads(line)
            answer = dictionary['answer']
            if re.match(pattern, answer):
                know_generation.append(dictionary)
                know_dict.append(gold_dict[dictionary['id']-1])
            elif tag == answer or "###" in answer:
                unknow_generation.append(dictionary)
                unknow_dict.append(gold_dict[dictionary['id']-1])
            else:
                unknow_generation.append(dictionary)
                unknow_dict.append(gold_dict[dictionary['id']-1])

    unknow_path = os.path.join('/'.join(generate_path.split('/')[:-1]),generate_path.split('/')[-1].split('.')[0]+'_'+'unknow.jsonl')
    know_path = os.path.join('/'.join(generate_path.split('/')[:-1]),generate_path.split('/')[-1].split('.')[0]+'_''know.jsonl')

    with open(unknow_path, 'w', encoding='utf-8') as jsonl_file:
        for dict in unknow_dict:
            jsonl_file.write(json.dumps(dict) + '\n') 
    with open(know_path, 'w', encoding='utf-8') as jsonl_file:
        for dict in know_dict:
            jsonl_file.write(json.dumps(dict) + '\n') 
    
    true_labels = 0
    total_labels = len(know_generation)
    for data in know_generation:
        label = data['answer'].split(':')[0]
        if label.lower() == gold[data['id']-1].lower():
            true_labels += 1
    
    print(true_labels / total_labels)
    print(len(know_generation))
    print(len(gold))

gold_path = '/home/sda/wangzhijun/MedicalGPT/jbi/data/MedMCQA.jsonl'
generate_path = '/home/sda/wangzhijun/MedicalGPT/jbi/result/innate/llama3-8b-instruct/MedMCQA.jsonl'


# extract_ans(gold_path,generate_path)

# self_accessment_generate_answer(gold_path,generate_path)    


def locate_answer(sentence:str):

    ans = re.findall("^\s*(A|B|C|D)$", sentence)
    if len(ans) > 0:
        return ans[0].upper()
    
    ans = re.findall("^\s*(A|B|C|D) or", sentence)
    if len(ans) > 0:
        return ans[0].upper()
    
    ans = re.findall("^\s*(A|B|C|D) and", sentence)
    if len(ans) > 0:
        return ans[0].upper()
        
    ans = re.findall("^\s*(A|B|C|D)/", sentence)
    if len(ans) > 0:
        return ans[0].upper()
    
    ans = re.findall("^\s*(A|B|C|D),", sentence)
    if len(ans) > 0:
        return ans[0].upper()
    
    ans = re.findall("[Oo]ption (A|B|C|D)", sentence)
    if len(ans) > 0:
        return ans[0]

    ans = re.findall(":\s*(A|B|C|D)", sentence)
    if len(ans) > 0:
        return ans[0].upper()

    ans = re.findall("^\s*(A|B|C|D)\.", sentence)
    if len(ans) > 0:
        return ans[0].upper()

    ans = re.findall("^\s*(A|B|C|D)\"", sentence)
    if len(ans) > 0:
        return ans[0].upper()
    
    ans = re.findall("^\s*(A|B|C|D):", sentence)
    if len(ans) > 0:
        return ans[0].upper()

    return "A"

def extract_answer(gold_path,generate_path):

    gold_list = []

    with open(gold_path,'r',encoding='utf-8') as file:

        for line in file:
            dictionary = json.loads(line)
            gold_answer = list(dictionary['answer'].keys())[0]
            gold_list.append(gold_answer)

    generate_list = []

    with open(generate_path,'r',encoding='utf-8') as file:

        for line in file:
            dictionary = json.loads(line)
            answer = dictionary['answer']
            answer = answer.split('"answer_choice": "')[-1].strip()
            print(locate_answer(answer))
            generate_list.append(locate_answer(answer))
    
    correct_num = 0
    total_num = len(gold_list)

    for gold,gene in zip(gold_list,generate_list):
        if gold.lower() == gene.lower():
            correct_num += 1
    
    acc = correct_num / total_num
    print(acc)

extract_ans(gold_path,generate_path)