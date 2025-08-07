import libcst as cst
import datasets
from datasets import load_dataset
import json

from copy import deepcopy
import random
import string

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
from opencoderunner import run as opencr_run
from opencoderunner import RunInfo  
from opencoderunner import ResultInfo  
from datetime import datetime, timezone

from openai import Client, OpenAI
import os
import time
import concurrent
import concurrent.futures


python3_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
datetime_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC%z")




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



def extract_md_block_withtag(text, tag):
    # pattern = r'```python\s*?\n(.*?)\n```'
    # pattern = r'```' + tag + r'\s*?\n(.*?)\n```'
    pattern = r'```' + re.escape(tag) + r'\s*\n(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    codetexts = []
    for i, match in enumerate(matches):
        codetext = match.strip()  # 代码内容
        codetexts.append(codetext)
    # print(codetexts)
    codetexts = sorted(codetexts) # strlength-Increasing-order  
    if len(codetexts) > 0:
        codetext = codetexts[-1]
    elif len(codetexts) == 0:
        codetext = None
    return codetext




def call_llms(supplier, model, input_text):
    if supplier == 'openai':
        client = client_openai
    elif supplier == 'ali':
        client = client_ali
    else:
        raise NotImplementedError
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": input_text},
        ],
    )
    output_text = response.choices[0].message.content
    output_dict = {
        'output_text': output_text,
        'response': response
    }
    return output_dict





question_template_reward = '''Belows are a question input for a code LLM, marked within QUESTION START and QUESTION END.

QUESTION START
====
Belows are the context information of a code project. 
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


What are the exact printed stdout and stderr in the terminal if I run the below bash command?:
cwd: {project_root_dir}
python3_version: {python3_version}
datetime_start: {datetime_start}
datetime_end: {datetime_end}
bash_command:
```bash
{command}
```

Let's think step by step and output the final answer in ```answer_stdout``` for stdout and ```answer_stderr``` for stderr even they are blank. E.g., 
```answer_stdout
Hello, World!
```

```answer_stdout
```

```answer_stderr
Traceback (most recent call last):\n  File "/home/code1.py", line 15, in <module>\n    main()\n  File "/home/code1.py", line 8, in main\n    n = int(input[idx])\nIndexError: list index out of range
```

```answer_stderr
```
====
QUESTION END


Then the code LLM's prediction is:
```answer_stdout
{answer_stdout}
```

```answer_stderr
{answer_stderr}
```

The ground truth is:
```gt_stdout
{gt_stdout}
```

```gt_stderr
{gt_stderr}
```

Due to some non-deterministic factors, the code execution may not always produce the same output.
Therefore, the code LLM's prediction may not match the ground truth exactly, but the main components can still show some correctness.
Please evaluate the code LLM's prediction and give a float score for the correctness between 0 to 1, where 0 means completely wrong and 1 means completely correct. 
Reason step by step, and then MUST give your score in ```llm_score```. I.e,
```llm_score
YOUR SCORE HERE 
```


'''


if __name__ == "__main__":

    code_str = '''
import random
import os
n = random.randint(1, 10)
print(f"Random number: {n}")
import sys
print(f"Python version: {sys.version}")
from datetime import datetime, timezone
print(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC%z"))
'''
    run_info = RunInfo(
        code_str=code_str,
        session_name="example_session",
        project_root_name="example_project",
        language="python",
        timeout=5,
        delete_after_run=True,
        pre_command="unset DISPLAY; "
    )
    result_info = opencr_run(run_info)
    # print(result_info)
    None


    # Example usage
    # tree_str = "root\n├── src\n│   ├── main.py\n│   └── utils.py\n└── tests\n    └── test_main.py"
    # file_abspath = "/home/user/project/src/main.py"
    # file_content = "print('Hello, World!')"
    # language = "python"
    # project_root_dir = "/home/user/project"
    # python3_version = "3.8.10"
    # datetime = "2023-10-01 12:00:00"
    # command = "python3 main.py"
    # answer_stdout = "Hello, World!"
    # answer_stderr = ""
    # gt_stdout = "Hello, World!"
    # gt_stderr = ""

    answer_stdout = "Random number: 15\nPython version: 3.10.16 (main, Dec 11 2024, 16:24:50) [GCC 11.2.0]\n2025-07-08 15:33:40 UTC+0000"
    answer_stderr = ""

    question = question_template_reward.format(
        tree_str=result_info.tree_str,
        file_abspath=run_info.file_infos[0].file_abspath,
        file_content=run_info.file_infos[0].file_content,
        language=run_info.language,
        project_root_dir=run_info.project_root_dir,
        python3_version=python3_version,
        datetime_start=result_info.datetime_start,
        datetime_end=result_info.datetime_end,
        command=result_info.command,
        answer_stdout=answer_stdout,
        answer_stderr=answer_stderr,
        gt_stdout=result_info.stdout_str,
        gt_stderr=result_info.stderr_str,
    )

    supplier = 'ali'
    model = 'qwen2.5-coder-7b-instruct'
    questions = [question] * 5
    num_theads = min(len(questions), os.cpu_count() // 2)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_theads) as executor:
        futures = [executor.submit(call_llms, supplier=supplier, model=model, input_text=e) for e in questions]
        llm_output_dicts = [future.result() for future in futures]
    print(f"Sync execution time: {time.time() - t0:.2f} seconds\n")

    # llm_output_dict = call_llms(supplier=supplier, model=model, input_text=question)
    # output_text = llm_output_dict['output_text']
    llm_output_dict = llm_output_dicts[0]

    output_text = llm_output_dict['output_text']
    print(output_text)

    llm_score = extract_md_block_withtag(output_text, 'llm_score')
    print(llm_score)

    None

