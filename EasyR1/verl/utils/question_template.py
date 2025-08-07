

import sys
import string
import random
from opencoderunner import RunInfo, FileInfo
from opencoderunner import run as opencr_run
from copy import deepcopy
import difflib

def augment_code_str_char(code_str, n=1):
    code_str = deepcopy(code_str)  # 确保不修改原始代码
    # 所有可能的字符集合
    charset = string.ascii_letters + string.digits + string.punctuation + " "
    for _ in range(n):
        random_char = random.choice(charset)
        flag = random.choice(['add', 'delete', 'replace'])  # 随机选一种操作
        random_pos = random.randint(0, len(code_str)-1)  # 随机位置
        if flag == 'add':
            # 在 pos 位置插入字符
            code_str = code_str[:random_pos] + random_char + code_str[random_pos:]
        elif flag == 'delete':
            # 删除 pos 位置的字符
            if len(code_str) > 1:
                code_str = code_str[:random_pos] + code_str[random_pos+1:]
        elif flag == 'replace':
            # 替换 pos 位置的字符
            code_str = code_str[:random_pos] + random_char + code_str[random_pos+1:]
    return code_str




# ==== Predict output
question_template_predoutput = """Belows are the context information of a code project. The project contains the following files:

file_abspath: 
{file_abspath}
file_content:
```{language}
{file_content}
```        


Based on the above project information, what is the exact output printed in the termial if I run the below bash command?:
cwd: {project_root_dir}
python3_version: {python3_version}
```bash
{command}
```
Let's think step by step and output the final answer in ```answer```.
"""



# ==== Predict corrupted code
question_template_predcodeaug = """Belows are the context information of a code project. The project contains the following files:

file_abspath:
{file_abspath}
file_content:
```{language}
{file_content}
```


Based on the above project information, I run the below bash command:
cwd: {project_root_dir}
python3_version: {python3_version}
```bash
{command}
```
The exact output printed in the terminal is:
```
{output_original}
```


Now, I randomly corrupt the code at most {n_char_augment} individual character positions, by character-level adding, deleting, or replacing.
After the corruption, the exact output printed in the terminal is:
```
{output_corrupted}
```


Based on the above information, give the corrupted code.
Let's think step by step and output the final answer in ```answer_code```.
"""




# ==== predoutputfunc
question_template_predoutputfunc = """Belows are the context information of a code project. The project contains the following files:

file_abspath:
{file_abspath}
file_content:
```{language}
{file_content}
```




I will call a function in this code with the following information:
stdin: 
```
{stdin}
```
entry_file_abspath: {entry_file_abspath}
entry_func_kwargs: 
```
{entry_func_kwargs}
```
entry_func_name: 
```
{entry_func_name}
```
python3_version: {python3_version}


Questions:
What is the exact content printed in the terminal? Write your answer in ```answer_output```

What is the return value of the entry function? Keep in Python format. Write your answer in ```answer_return```
"""





if __name__ == "__main__":


    python3_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    code_str = '''
import sys
def greet(name: str):
    """Prints a greeting message to the user."""
    print(f"Hello, {name}!")

def main(name: str = "World"):
    # This is a simple Python program that prints a personalized greeting.
    greet(name)
    stdin_content = sys.stdin.read().strip()
    print(f"Received from stdin: {stdin_content}")

if __name__ == "__main__":
    main("Alice")  # 你可以在这里改名字
'''



    # ==== Question template predict output function
    run_info = RunInfo(
        code_str=code_str,
        project_root_name="project_root_name",
        session_name="session_name",
        language="python",
        timeout=3,
        entry_file_relpath="__code_str__.py",  # The file where the entry function is defined
        entry_func_name="main",  # The function to call
        entry_func_kwargs={"name": "Alice"},  # The kwargs to pass to the function
        input_content="STDIN"  # The input to the function, if any
    )
    result_info = opencr_run(run_info=run_info, is_run=True)
    question_predoutputfunc = question_template_predoutputfunc.format(
        file_abspath=run_info.file_infos[0].file_abspath,
        file_content=run_info.file_infos[0].file_content,
        language=run_info.language,
        entry_file_abspath=run_info.entry_file_abspath,
        entry_func_name=run_info.entry_func_name,
        entry_func_kwargs=run_info.entry_func_kwargs,
        stdin=run_info.input_content,
        python3_version=python3_version
    )
    None


    # ==== Question template predict output
    run_info = RunInfo(
        code_str=augment_code_str_char(code_str, n=3),  # Corrupting the code
        project_root_name="project_root_name",
        session_name="session_name",
        language="python",
        timeout=3
    )
    result_info = opencr_run(run_info=run_info, is_run=True)
    stdout = result_info.stdout if isinstance(result_info.stdout, str) else result_info.stdout.decode()
    stderr = result_info.stderr if isinstance(result_info.stderr, str) else result_info.stderr.decode()
    stdout_stderr = "\n".join([stdout, stderr])
    question_predoutput = question_template_predoutput.format(
        file_abspath=run_info.file_infos[0].file_abspath,
        file_content=run_info.file_infos[0].file_content,
        language=run_info.language,
        project_root_dir=run_info.project_root_dir,
        command=run_info.command,
        python3_version=python3_version,
    )
    print(question_predoutput)
    print(stdout_stderr)
    





    # ==== Question template predict code
    run_info_original = RunInfo(
        code_str=code_str,
        project_root_name="project_root_name",
        session_name="session_name",
        language="python",
        timeout=3
    )
    result_info_original = opencr_run(run_info=run_info_original, is_run=True)
    stdout_original = result_info_original.stdout if isinstance(result_info_original.stdout, str) else result_info_original.stdout.decode()
    stderr_original = result_info_original.stderr if isinstance(result_info_original.stderr, str) else result_info_original.stderr.decode()
    stdout_stderr_original = "\n".join([stdout_original, stderr_original])


    n_char_augment = 1  # Number of characters to augment
    code_str_corrupted = augment_code_str_char(code_str, n=n_char_augment)  # Corrupting the code
    
    linediff = list(difflib.ndiff(code_str.splitlines(), code_str_corrupted.splitlines()))

    run_info_corrupted = RunInfo(
        code_str=code_str_corrupted,  # Corrupting the code
        language="python",
        project_root_name="project_root_name",
        session_name="session_name",
        timeout=3
    )
    result_info_corrupted = opencr_run(run_info=run_info_corrupted, is_run=True)
    stdout_corrupted = result_info_corrupted.stdout if isinstance(result_info_corrupted.stdout, str) else result_info_corrupted.stdout.decode()
    stderr_corrupted = result_info_corrupted.stderr if isinstance(result_info_corrupted.stderr, str) else result_info_corrupted.stderr.decode()
    stdout_stderr_corrupted = "\n".join([stdout_corrupted, stderr_corrupted])
        

    question_predcode = question_template_predcodeaug.format(
        file_abspath=run_info_corrupted.file_infos[0].file_abspath,
        file_content=run_info_corrupted.file_infos[0].file_content,
        language=run_info_corrupted.language,
        project_root_dir=run_info_corrupted.project_root_dir,
        command = run_info_corrupted.command,
        output_corrupted=stdout_stderr_corrupted,
        output_original=stdout_stderr_original,
        n_char_augment=n_char_augment,
        python3_version=python3_version
    )
    print(question_predcode)