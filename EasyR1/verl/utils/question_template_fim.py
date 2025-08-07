import libcst as cst
import datasets
from datasets import load_dataset
import json

from copy import deepcopy
import random
import string
# from utils.utils_libcst import (
#     ReplaceIfConditionWithNegation,
#     ReplaceBinaryOp,
#     ReplaceBooleanOp,
#     ReplaceUnaryOp,
#     ReplaceAugmentedOp,
#     ReplaceComparisonOp,
# )
import difflib
import ast
import sys
import platform
# import shlex

import random
import tokenize
import io
import re
import multiprocessing



from openai import Client, OpenAI
import os
import time

import re


python3_version = platform.python_version()

# ==== OpenAI ====
client_openai = Client(
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ==== Ali ====
client_ali = Client(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=1000000,
)

# ==== siliconflow
client_siliconflow = Client(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
    timeout=1000000,
)




def extract_md_anyblock(text):
    pattern = r'```\s*?([a-zA-Z0-9]*?)\s*?\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches




def extract_md_block_withtag(text, tag):
    # pattern = r'```python\s*?\n(.*?)\n```'
    # pattern = r'```' + tag + r'\s*?\n(.*?)\n```'
    pattern = r'```' + re.escape(tag) + r'\s*\n(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    codetexts = []
    for i, match in enumerate(matches):
        # codetext = match.strip()  # 代码内容
        codetext = match
        codetexts.append(codetext)
    # print(codetexts)
    codetexts = sorted(codetexts) # strlength-Increasing-order  
    if len(codetexts) > 0:
        codetext = codetexts[-1]
    elif len(codetexts) == 0:
        codetext = None
    return codetext



def build_code_str_fim(code_str, max_num_corrupted_lines=2):
    code_str_list = code_str.splitlines()
    non_empty_line_ids = [
        line_id for line_id in range(len(code_str_list)) if code_str_list[line_id].strip()
    ]
    num_corrupted_lines = random.randint(1, max_num_corrupted_lines)
    corrupted_line_ids = random.sample(non_empty_line_ids, num_corrupted_lines)
    corrupted_line_ids = sorted(corrupted_line_ids)

    for i, currupted_line_id in enumerate(corrupted_line_ids):
        code_str_list[currupted_line_id] = "MASKED_LINE_" + str(i)
    code_str_corrupted = "\n".join(code_str_list)

    return code_str_corrupted, num_corrupted_lines





def call_llms(supplier, model, input_text):
    if supplier == 'openai':
        client = client_openai
    elif supplier == 'ali':
        client = client_ali
    elif supplier == 'siliconflow':
        client = client_siliconflow
    else:
        raise NotImplementedError
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": input_text},
        ],
        stream=False,
    )
    output_text = response.choices[0].message.content
    output_dict = {
        'output_text': output_text,
        'response': response
    }
    return output_dict




question_template_fim = '''Belows are the context information of a corrupted code project. 
Project tree structure:
```
{tree_str}
```

The project contains the following files:

file_abspath: 
{file_abspath}
file_content:
```{language}
{file_content}
```       

There are {num_corrupted_lines} line(s) corrupted, where they are masked by MASKED_LINE_$ID. E.g., 
MASKED_LINE_0, MASKED_LINE_1, MASKED_LINE_2, etc.

Originally, I run the project in the following environment:
cwd: {project_root_dir}
python3_version: {python3_version}
datetime_start: {datetime_start}
datetime_end: {datetime_end}
bash_command:
```bash
{command}
```

Then, the exact printed stdout and stderr in the terminal are:
```stdout
{stdout}
```
```stderr
{stderr}
```

What should be the original contents of each corrupted line? Necessary indents should be preserved.
Let's think step by step, and then output the final answer in ```answer_MASKED_LINE_$ID```, even there is only 1 line masked. E.g., 
```answer_MASKED_LINE_0
    for i in range(10):
```

```answer_MASKED_LINE_1
print("Hello, World!")
```

```answer_MASKED_LINE_2
        a = 4
```


'''





if __name__ == "__main__":
    from opencoderunner import run as opencr_run
    from opencoderunner import RunInfo


    code_str = '''
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return "Cannot divide by zero"

def multiply(a, b):
    return a * b

a = 12
b = 4

print("Add:", add(a, b))
print("Subtract:", subtract(a, b))
print("Multiply:", multiply(a, b))
print("Divide:", divide(a, b))

for i in range(1, 6):
    print(f"{i} * {b} = {multiply(i, b)}")

if a > b:
    print("a is greater than b")
else:
    print("a is not greater than b")

'''
    run_info = RunInfo(
        code_str=code_str,
        language="python",
        delete_after_run=True,
        timeout=5,
        pre_command=""
    )
    result_info = opencr_run(run_info)
    gt = {
        "stdout": result_info.stdout_str,
        "stderr": result_info.stderr_str,
    }

    max_num_corrupted_lines = 2
    code_str_fim, num_corrupted_lines = build_code_str_fim(
        code_str, 
        max_num_corrupted_lines=max_num_corrupted_lines
    )
    # code_str_list = code_str.splitlines()
    # num_corrupted_lines = 0
    # for line_id in range(len(code_str_list)):
    #     if not code_str_list[line_id].strip():
    #         continue
    #     if random.random() < prob:
    #         code_str_list[line_id] = "MASKED_LINE_" + str(num_corrupted_lines)
    #         num_corrupted_lines += 1
    # code_str_corrupted = "\n".join(code_str_list)
    # assert len(code_str_list) == len(code_str_corrupted.splitlines())


    question_fim = question_template_fim.format(
        file_abspath=run_info.file_infos[0].file_abspath,
        file_content=code_str_fim,
        language=run_info.language,
        project_root_dir=run_info.project_root_dir,
        python3_version=python3_version,
        command=run_info.command,
        tree_str=result_info.tree_str,
        datetime_start=result_info.datetime_start,
        datetime_end=result_info.datetime_end,
        stdout=result_info.stdout_str,
        stderr=result_info.stderr_str,
        num_corrupted_lines=num_corrupted_lines,
    )
    if num_corrupted_lines == 0:
        exit(0)
    print("-"*20)
    print(question_fim)
    
    
    supplier = 'siliconflow'
    # model = "Qwen/Qwen2.5-Coder-7B-Instruct"
    model = "Qwen/Qwen2.5-Coder-32B-Instruct"
    output_dict = call_llms(supplier, model, question_fim)
    output_text = output_dict['output_text']
    print("-"*20)
    print(output_text)


    fim_answers = []
    for i in range(num_corrupted_lines):
        tag = "answer_MASKED_LINE_" + str(i)
        answer = extract_md_block_withtag(output_text, tag)
        fim_answers.append(
            [tag, answer]
        )
    
    code_str_fim_reconstructed = deepcopy(code_str_fim)
    for tag, answer in fim_answers:
        tag = tag.replace("answer_", "")
        if answer is not None:
            code_str_fim_reconstructed = code_str_fim_reconstructed.replace(tag, answer)
    print("-"*20)
    print(code_str_fim_reconstructed)

    run_info_reconstructed = deepcopy(run_info)
    run_info_reconstructed.file_infos[0].file_content = code_str_fim_reconstructed
    result_info_reconstructed = opencr_run(run_info_reconstructed)
    print(result_info)
    print(result_info_reconstructed)
    pred = {
        "stdout": result_info_reconstructed.stdout_str,
        "stderr": result_info_reconstructed.stderr_str,
    }

    if gt['stdout'] == pred['stdout']:
        print("stdout is correct!")
    else:
        print("stdout is wrong!")
    
    if gt['stderr'] == pred['stderr']:
        print("stderr is correct!")
    else:
        print("stderr is wrong!")

    None