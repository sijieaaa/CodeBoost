import libcst as cst
import datasets
from datasets import load_dataset
import json

from copy import deepcopy
import random
import string
from .utils_libcst import (
    ReplaceIfConditionWithNegation,
    ReplaceBinaryOp,
    ReplaceBooleanOp,
    ReplaceUnaryOp,
    ReplaceAugmentedOp,
    ReplaceComparisonOp,
)
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

def augment_stdin_char(stdin_str, prob=0.1):
    stdin_str_list = list(stdin_str)
    length = len(stdin_str_list)
    if length == 0:
        return ""
    # 先找到所有可替换的索引
    valid_indices = [i for i, c in enumerate(stdin_str_list) if c.isdigit() or c.isalpha()]
    if not valid_indices:
        return stdin_str  # 无可替换字符，返回原始字符串
    for i in valid_indices:
        old_char = stdin_str_list[i]
        if random.random() <= prob:
            if old_char.isdigit():
                stdin_str_list[i] = str(random.randint(0, 9))
            elif old_char.isalpha():
                stdin_str_list[i] = random.choice(string.ascii_letters)
    output = ''.join(stdin_str_list)
    return output



def augment_code_str_digit(code_str, prob=0.1):
    code_str_list = list(code_str)
    protected_indices = set()
    # 匹配所有变量名、函数名、类名（含数字）
    for match in re.finditer(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code_str):
        if any(c.isdigit() for c in match.group()):
            # 记录这些标识符中数字的索引位置
            for i, c in enumerate(match.group()):
                if c.isdigit():
                    protected_indices.add(match.start() + i)

    # 执行替换
    for i, c in enumerate(code_str_list):
        if i in protected_indices:
            continue
        if c.isdigit() and random.random() < prob:
            # Do not use zero "0" to avoid leading zero issues
            code_str_list[i] = random.choice([d for d in '123456789' if d != c])

    return ''.join(code_str_list)
    



def augment_code_str_logical(code_str, prob=0.1, max_retries=3):
    module = cst.parse_module(code_str)
    possible_classes = [
        ReplaceBinaryOp,
        ReplaceBooleanOp,
        ReplaceUnaryOp,
        ReplaceAugmentedOp,
        ReplaceComparisonOp,
        ReplaceIfConditionWithNegation
    ]
    for ReplaceClass in possible_classes:
        module = module.visit(ReplaceClass(prob=prob))

    code_augmented = module.code
    if judge_has_syntax_error(code_augmented):
        # 限制递归深度避免死循环
        if max_retries <= 0:
            return code_str  # fallback to original if too many retries
        return augment_code_str_logical(code_str, prob=prob, max_retries=max_retries - 1)
    return code_augmented




def judge_has_syntax_error(code_str: str) -> tuple[bool, str | None]:
    try:
        compile(code_str, "<string>", "exec")  # 检查语法和部分静态引用
        return False
    except SyntaxError as e:
        return True
    except Exception as e:
        return True




def load_func_from_str(code_str: str, func_path: str):
    """
    Load a function from a code string.
    """
    namespace = {}
    try:
        exec(code_str, namespace)
        parts = func_path.split('.')
        # function only
        if len(parts) == 1:
            return namespace[func_path]
        # class function
        cls_name, method_name = parts
        cls = namespace[cls_name]
        obj = cls()  # instantiate
        return getattr(obj, method_name)
    except Exception as e:
        print(f"Error loading {func_path}: {e}")
        return None



def run_func_with_timeout(func:callable, args=[], kwargs={}, timeout=5):
    """
    Run a function with a timeout.
    """
    if kwargs is None:
        kwargs = {}

    def target_func(q):
        try:
            result = func(*args, **kwargs)
            q.put(("ok", result))
        except Exception as e:
            q.put(("err", e))

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=target_func, args=(q,))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        raise TimeoutError(f"Function call timed out after {timeout} seconds")

    if q.empty():
        raise RuntimeError("Function crashed without output")

    status, value = q.get()
    if status == "ok":
        return value
    else:
        raise value  # re-raise the caught exception





question_template_predoutput_noinput = '''Belows are the context information of a code project. 
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

'''


question_template_predoutput_stdin = '''Belows are the context information of a code project. 
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

'''






question_template_predoutput_function = '''Belows are the context information of a code project. 
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

'''



if __name__ == "__main__":
    from opencoderunner import run as opencr_run
    from opencoderunner import RunInfo
    dataset = datasets.load_dataset(
        "csv", 
        data_files="/data/sijie/llm/verl_prepare_datasets/datasets/open-thoughts@OpenThoughts-114k@metadata_row-1_line10_len30_time-1.csv"
        # data_files="/home/runner/verl_prepare_datasets/datasets/open-thoughts@OpenThoughts-114k@metadata_row-1_line10_len30_time-1.csv"
    )
    dataset_train = dataset["train"]
    row_dict = random.choice(dataset_train)
    test_type = row_dict['test_type']
    if not test_type == "stdin":
        print(f"Skip the row with test_type: {test_type}")
        exit(0)
    test_cases_str = row_dict['test_cases']
    test_cases = json.loads(test_cases_str)
    for i, each_input in enumerate(test_cases['inputs']):
        if i >= 3:
            break
        
        each_input_augmented = deepcopy(each_input)
        # -- augment input
        stdin_augment_prob = 0.1
        each_input_augmented = augment_stdin_char(each_input_augmented, prob=stdin_augment_prob)


        code_str_augmented = deepcopy(row_dict['code_str'])

        # -- augment code logical
        code_str_logical_augment_prob = 0.1
        code_str_augmented = augment_code_str_logical(code_str_augmented, prob=code_str_logical_augment_prob)
        # diff = difflib.ndiff(row_dict['code_str'].splitlines(), code_str_augmented.splitlines())
        # print('\n'.join(diff)) 


        # -- augment code digit
        code_str_digit_augment_prob = 0.1
        code_str_augmented = augment_code_str_digit(code_str_augmented, prob=code_str_digit_augment_prob)
        
        
        diff = difflib.ndiff(row_dict['code_str'].splitlines(), code_str_augmented.splitlines())
        print('\n'.join(diff))


        # -- check if has syntax error
        has_syntax_error = judge_has_syntax_error(code_str_augmented)
        if has_syntax_error:
            print(f"Syntax error in the augmented code. Skipping...")
            continue
        

        run_info = RunInfo(
            code_str=code_str_augmented,
            project_root_name="project_root_name",
            session_name="session_name",
            language="python",
            timeout=3,
            input_content=each_input_augmented,
            pre_command="unset DISPLAY; "

        )
        result_info = opencr_run(run_info=run_info, is_run=True)
        print(result_info)
        None


        question_predoutstdin = question_template_predoutput_stdin.format(
            file_abspath=run_info.file_infos[0].file_abspath,
            file_content=run_info.file_infos[0].file_content,
            language=run_info.language,
            project_root_dir=run_info.project_root_dir,
            python3_version=platform.python_version(),
            test_case_stdin_repr=repr(each_input_augmented),
            datetime_start=result_info.datetime_start,
            datetime_end=result_info.datetime_end,
            # test_case_stdin_repr=repr(str(each_input_augmented)) 
            # test_case_stdin_repr=shlex.quote(each_input_augmented)
        )
        print(question_predoutstdin)
        # print(repr(question_predoutstdin))
        print("=" * 20)

