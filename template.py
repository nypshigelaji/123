from liquid import Template

general_cot_system = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{your answer}}. Your responses will be used for research purposes only, so please have a definite answer.'''

general_cot = Template('''
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json:
''')

general_cot_system_a = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''

general_cot_a = Template('''
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json:
''')

general_medrag_system = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{your answer}}. Your responses will be used for research purposes only, so please have a definite answer.'''

general_medrag = Template('''
Here are the relevant documents:
{{context}}

Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json:
''')

general_medrag_system_a = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''

general_medrag_a = Template('''
Here are the relevant documents:
{{context}}

Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json:
''')

meditron_cot = Template('''
### User:
Here is the question:
...

Here are the potential choices:
A. ...
B. ...
C. ...
D. ...
X. ...

Please think step-by-step and generate your output in json.

### Assistant:
{"step_by_step_thinking": ..., "answer_choice": "X"}

### User:
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json.

### Assistant:
''')

meditron_medrag = Template('''
Here are the relevant documents:
{{context}}

### User:
Here is the question:
...

Here are the potential choices:
A. ...
B. ...
C. ...
D. ...
X. ...

Please think step-by-step and generate your output in json.

### Assistant:
{"step_by_step_thinking": ..., "answer_choice": "X"}

### User:
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json.

### Assistant:
''')

simple_medrag_system = '''You are a helpful medical expert, and your task is to answer a medical question using the relevant documents. You do not need to provide an analysis; simply give the correct option directly. Organize your output in a json formatted as Dict{"answer_choice": Str{your answer}}. Your responses will be used for research purposes only, so please have a definite answer.'''
simple_medrag_prompt = Template('''Here are the relevant documents:\n{{context}}\nHere is the question:\n{{question}}\nHere are the potential choices:\n{{options}}''')

i_medrag_system = '''You are a helpful medical assistant, and your task is to answer the given question following the instructions given by the user. '''

follow_up_instruction_ask = '''Please first analyze all the information in a section named Analysis (## Analysis). Then, use key terms from previous answers to form specific and direct questions. Generate {}  concise, context-specific queries to search for additional information in an external knowledge base, in a section named Queries (## Queries). Each query should be simple and focused, directly relating to the key terms used in the answers. Wait for responses from the user before proceeding.'''
follow_up_instruction_answer = '''Please first think step-by-step to analyze all the information in a section named Analysis (## Analysis). Then, please provide your answer choice in a section named Answer (## Answer).'''

simple_generate_system = '''You are a helpful medical expert, and your task is to answer a medical question. You do not need to provide an analysis; simply give the correct option directly. Organize your output in a json formatted as Dict{"answer_choice": Str{your answer}}. Your responses will be used for research purposes only, so please have a definite answer.'''
simple_generate_prompt = Template('''Here is the question:\n{{question}}\nHere are the potential choices:\n{{options}}''')

simple_generate_system_a = '''You are a helpful medical expert, and your task is to answer a medical question. You do not need to provide an analysis; simply give the correct option directly. Organize your output in a json formatted as Dict{"answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''
simple_generate_prompt_a = Template('''Here is the question:\n{{question}}\nHere are the potential choices:\n{{options}}''')


self_accessment_system_1 = '''You are a helpful medical expert, and your task is to answer a medical question. You do not need to provide an analysis; simply give the correct option directly. If you are entirely confident in the solution to the problem, directly output the corresponding option. If you are not very certain about the solution, please respond with, "###I am not very sure about the answer to this question.". Your responses will be used for research purposes only, so please have a definite answer.'''
self_accessment_system = '''You are a helpful medical expert, and your task is to answer a medical question. You do not need to provide an analysis; simply give the correct option directly. If you are entirely confident in the solution to the problem, organize your output in a json formatted as Dict{"answer_choice": Str{your answer}}. If you are not very certain about the solution, please respond with, "###I am not very sure about the answer to this question.". Your responses will be used for research purposes only, so please have a definite answer.'''
self_accessment_prompt = Template('''Here is the question:\n{{question}}\nHere are the potential choices:\n{{options}}''')