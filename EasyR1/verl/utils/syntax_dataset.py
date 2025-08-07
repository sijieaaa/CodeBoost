# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
import datasets
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF


import opencoderunner as opencr
from opencoderunner.run import run as opencr_run
from opencoderunner.file_info import FileInfo
from opencoderunner.run_info import RunInfo
from opencoderunner.result_info import ResultInfo
import json
import sys
import re

import random
import string
from copy import deepcopy
# from .question_template import question_template_predoutput
# from .question_template import question_template_predcodeaug
from .question_template_predoutput_noinput_stdin_function import question_template_predoutput_stdin 
from .question_template_predoutput_noinput_stdin_function import question_template_predoutput_function
from .question_template_predoutput_noinput_stdin_function import question_template_predoutput_noinput
from .question_template_fim import question_template_fim
from .question_template_fim import build_code_str_fim

import libcst as cst
from .utils_libcst import (
    ReplaceBinaryOp,
    ReplaceBooleanOp,
    ReplaceUnaryOp,
    ReplaceAugmentedOp,
    ReplaceComparisonOp,
    ReplaceIfConditionWithNegation
)
from .python_builtin_exceptions import python_builtin_exceptions

import sys
import warnings
import difflib
import networkx as nx
from collections import Counter
from ray.experimental.tqdm_ray import tqdm
import torch
import time
import networkit as nk


import io
import tokenize
from datetime import datetime, timezone
from pprint import pprint

python3_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"







def stitch_testcase_func_into_codestr(
    code_str: str,
    func_name: str, # 'func', 'cls().func'
    func_args,
    func_kwargs,
    func_return_name: str = "__result__",
    use_opencr_run: bool = False,
) -> Dict[str, Any]:
    arg_list  = [repr(a) for a in func_args]
    kwarg_list = [f"{k}={repr(v)}" for k, v in func_kwargs.items()]
    call_args = ", ".join(arg_list + kwarg_list)
    
    call_line = f"{func_return_name} = {func_name}({call_args})"

    pattern = re.compile(r'^\s*if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:\s*$', re.MULTILINE)
    match = pattern.search(code_str)

    if match:
        indent = re.match(r'^\s*', match.group()).group() + " " * 4
        insertion_line = f"\n{indent}{call_line}"
        print_line = f"\n{indent}print({func_return_name})"
        full_code = code_str + insertion_line + print_line
    else:
        insertion = f'''
if __name__ == "__main__":
    {call_line}
    print({func_return_name})        
'''
        full_code = code_str.rstrip() + insertion

    output_dict = {
        'full_code': full_code,
    }

    if use_opencr_run:
        session_name = f"session_" + ''.join(random.sample(string.ascii_letters + string.digits, 8))
        run_info = RunInfo(
            code_str=full_code,
            language="python",
            session_name=session_name,
            project_root_name="project_root_name",
            timeout=5,
            pre_command="unset DISPLAY; "
        )
        result_info = opencr_run(run_info=run_info, is_run=True)
        output_dict["result_info"] = result_info
        output_dict["run_info"] = run_info

    # # func return value
    # ns: Dict[str, Any] = {"__name__": "__main__"}  
    # exec(full_code, ns)
    # func_return_value = ns.get(func_return_name)

    # output_dict = {
    #     "full_code": full_code,
    #     # "func_return_value": func_return_value,
    # }
    return output_dict





def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}



def augment_code_str_char(code_str, n=1, charset=None):
    code_str = deepcopy(code_str)  # Avoid modifying the original string
    # 所有可能的字符集合
    if charset == "basic":
        charset = string.ascii_letters + string.digits + string.punctuation + " "
    elif charset == "printable":
        charset = string.printable
    else:
        try:
            charset = getattr(string, charset)
        except Exception as e:
            print(e)
            raise NotImplementedError

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



def augment_stdin_char(stdin_str, prob=0.1):
    stdin_str_list = list(stdin_str)
    length = len(stdin_str_list)
    if length == 0:
        return ""
    # 先找到所有可替换的索引
    valid_indices = [i for i, c in enumerate(stdin_str_list) if isinstance(c, str)]
    # valid_indices = [i for i, c in enumerate(stdin_str_list) if (c.isdigit() or c.isalpha())]
    valid_indices = [i for i in valid_indices if stdin_str_list[i].isdigit() or stdin_str_list[i].isalpha()]
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




def augment_any(data: Any, prob):
    data = deepcopy(data)  # Ensure we don't modify the original data

    if isinstance(data, str):
        data = list(data)
        for i in range(len(data)):
            if random.random() < prob:
                if data[i].isdigit():
                    data[i] = random.choice(string.digits)
                elif data[i].isalpha():
                    data[i] = random.choice(string.ascii_letters)
        data = ''.join(data)

    elif isinstance(data, int):
        choice = random.choice(["add", "scale", "negate"])
        if choice == "add":
            return data + random.randint(-5, 5)
        elif choice == "scale":
            data = data * random.uniform(0.5, 1.5)
            data = int(data) 
            return data
        elif choice == "negate":
            return -data

    elif isinstance(data, float):
        choice = random.choice(["add", "scale", "negate"])
        if choice == "add":
            return data + random.uniform(-5.0, 5.0)
        elif choice == "scale":
            return data * random.uniform(0.5, 1.5)
        elif choice == "negate":
            return -data

    return data  



def augment_function_input(data, prob):
    data = deepcopy(data)  
    if isinstance(data, list):
        for i in range(len(data)):
            if random.random() < prob:
                data[i] = augment_function_input(data[i], prob=prob)
    elif isinstance(data, dict):
        for k in list(data.keys()):
            if random.random() < prob:
                data[k] = augment_function_input(data[k], prob=prob)
    elif isinstance(data, str) or isinstance(data, int) or isinstance(data, float):
        if random.random() < prob:
            data = augment_any(data, prob=prob)

    return data



def augment_code_str_digit(code_str, prob=0.1):
    tokens = list(tokenize.generate_tokens(io.StringIO(code_str).readline))
    new_tokens = []
    for i, (tok_type, tok_str, start, end, line) in enumerate(tokens):
        new_tok_str = tok_str
        if tok_type == tokenize.NUMBER:
            prev_tok = tokens[i - 1] if i > 0 else None
            next_tok = tokens[i + 1] if i + 1 < len(tokens) else None
            if prev_tok and prev_tok[1] == '[':
                is_index = True
            elif next_tok and next_tok[1] == ']':
                is_index = True
            else:
                is_index = False
            if not is_index:
                new_tok = list(tok_str)
                for j, ch in enumerate(new_tok):
                    if ch.isdigit() and random.random() < prob:
                        new_tok[j] = random.choice([d for d in '123456789' if d != ch])
                new_tok_str = ''.join(new_tok)
        new_tokens.append((tok_type, new_tok_str, start, end, line))

    return tokenize.untokenize(new_tokens)






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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)  # 忽略语法警告
            compile(code_str, "<string>", "exec")
        return False, None
    except Exception as e:
        return True, e
    


def fix_spaces_around_keywords(code: str) -> str:
    keywords = ["if", "elif", "for", "while", "with", "except", "return", "assert", "del", "yield"]
    for kw in keywords:
        code = re.sub(rf'\b{kw}\(', rf'{kw} (', code)
    return code



class ImageProcessMixin:
    max_pixels: int
    min_pixels: int

    def process_image(self, image: Union[Dict[str, Any], ImageObject]) -> ImageObject:
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image


class SyntaxDataset(Dataset, ImageProcessMixin):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,

        ppo_config = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        # self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.filter_overlong_prompts = filter_overlong_prompts

        # -- control
        self.ppo_config = ppo_config
        # global cfg
        # cfg = fullconfig


        # -- random initiate predtype
        if random.random() <= self.ppo_config.predoutput_ratio:
            self.predtype = "predoutput"
        else:
            self.predtype = "predcodeaug"
        print(f"⚠️ Initiate predtype: {self.predtype}")




        # if "@" in data_path:
        #     data_path, data_split = data_path.split("@")
        # else:
        #     data_split = "train"

        # if os.path.isdir(data_path):
        #     # when we use dataset builder, we should always refer to the train split
        #     self.dataset = load_dataset("parquet", data_dir=data_path, split="train")
        # elif os.path.isfile(data_path):
        #     self.dataset = load_dataset("parquet", data_files=data_path, split="train")
        # else:
        #     # load remote dataset from huggingface hub
        #     self.dataset = load_dataset(data_path, split=data_split)
        
        # breakpoint()
        # -- split by comma
        if "," in data_path:
            if data_path.startswith(","):
                data_path = data_path[1:]
            data_path = data_path.split(",")
            for e in data_path:
                assert os.path.exists(e), f"{e} does not exist."
        features = datasets.Features({
            "data_source":   datasets.Value("string"),
            "language":      datasets.Value("string"),
            "code_str":      datasets.Value("string"),
            "len":           datasets.Value("int64"),
            "test_cases_str":    datasets.Value("string"),
            "starter_code":  datasets.Value("string"),
            "is_executable": datasets.Value("bool"),
            "use_stdin":     datasets.Value("bool"),
        })
        dataset_list = []
        for e in data_path:
            print(f"⚠️ {e}")
            dataset = datasets.Dataset.from_csv(
                path_or_paths=e,
                features=features,
            )
            if "opc-sft-stage1" in e:
                opc_ddownsample = self.ppo_config.opc_ddownsample
                indices = range(0, len(dataset), opc_ddownsample)
                print(f"OPC ddownsampled dataset size: {len(dataset)} -> {len(indices)}")
                dataset = dataset.select(indices)
            dataset_list.append(dataset)
        self.dataset = datasets.concatenate_datasets(dataset_list)


        # self.dataset = datasets.Dataset.from_csv(
        #     path_or_paths=data_path,
        #     features=features,
        # )



        # -- dataset ratio
        num_selected_samples = int(len(self.dataset) * ppo_config.dataset_ratio)
        indices = np.linspace(0, len(self.dataset) - 1, num=num_selected_samples, dtype=int)
        self.dataset = self.dataset.select(indices.tolist())
        print(f"⚠️ Dataset select {len(self.dataset)} samples. Ratio: {ppo_config.dataset_ratio}")

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        # -- maximal clique
        # breakpoint()
        # time_start = time.time()
        # max_clique_ids = []
        # size_org = len(self.dataset)
        # strs_list = self.dataset['code_str']
        # sub_list_size = 400
        # num_sub_lists = len(strs_list) // sub_list_size + 1
        # for i in tqdm(range(num_sub_lists), total=num_sub_lists, desc="Processing sublists"):
        #     start_id = i * sub_list_size
        #     end_id = min((i + 1) * sub_list_size, len(strs_list)-1)
        #     strs_list_sub = strs_list[start_id:end_id]
        #     print(f"start_id: {start_id}    end_id: {end_id}    len: {len(strs_list_sub)}")
        #     _, sub_max_clique_ids = find_max_clique_by_diff(strs_list_sub, 
        #                                                     threshold=ppo_config.clique_threshold, 
        #                                                     num_workers=0)
        #     sub_max_clique_ids = [start_id + i for i in sub_max_clique_ids] # align to the original indices
        #     max_clique_ids.extend(sub_max_clique_ids)
        # assert len(set(max_clique_ids)) == len(max_clique_ids), "Duplicate indices found in max clique ids"
        # self.dataset = self.dataset.select(max_clique_ids)
        # print(f"Total time taken: {time.time() - time_start:.2f} seconds")
        # print(f"⚠️ clique size: {len(self.dataset)} / org size: {size_org}")





        # -- timed out samples
        # breakpoint()
        self.num_timed_out_samples = 0


        # -- other error samples
        # self.supported_error_types = [
        #     # "syntax error",
        #     # "attribute error", # not used
        #     "index error",
        #     "value error",
        #     "name error",
        #     "type error",
        #     "key error",
        #     "zero division error",
        # ]
        self.supported_error_types = self.ppo_config.supported_error_types
        self.supported_error_types = [e.strip() for e in self.supported_error_types.split(",") if e.strip()]
        print(f"⚠️ Supported error types: {self.supported_error_types}")
        self.num_error_samples = {error_type: 0 for error_type in self.supported_error_types}
        self.num_error_samples['total'] = 0

        # self.notsupported_error_types = [
        #     "not implemented error",
        #     "module not found error",
        #     "name error",
        #     "attribute error"
        # ]
        # self.num_error_samples = {}
        # self.num_error_samples['total'] = 0

        # -- Filter dataset by test_type
        # self.dataset = self.dataset.filter(lambda x: x["test_type"] == "stdin")
        # print(f"⚠️ Dataset select {len(self.dataset)} samples with test_type 'stdin'.")


    # def _build_messages_predoutput(self, example: Dict[str, Any], ppo_config) -> List[Dict[str, Any]]:
    #     code_str = example["code_str"]
    #     session_name = "session_" + "".join(random.sample(string.ascii_letters + string.digits, 10))
    #     n_char_augment = ppo_config.n_char_augment
    #     code_str_corrupted = augment_code_str_char(code_str, n=n_char_augment, charset=ppo_config.charset)   
        
    #     run_info = RunInfo(
    #         code_str=code_str_corrupted,  
    #         project_root_name="project_root_name",
    #         session_name=session_name,
    #         language="python",
    #         timeout=3
    #     )

    #     result_info = opencr_run(run_info=run_info, is_run=True)
    #     stdout = result_info.stdout if isinstance(result_info.stdout, str) else result_info.stdout.decode()
    #     stderr = result_info.stderr if isinstance(result_info.stderr, str) else result_info.stderr.decode()
    #     stdout_stderr = "\n".join([stdout, stderr])
    #     question_predoutput = question_template_predoutput.format(
    #         file_abspath=run_info.file_infos[0].file_abspath,
    #         file_content=run_info.file_infos[0].file_content,
    #         language=run_info.language,
    #         project_root_dir=run_info.project_root_dir,
    #         command=run_info.command,
    #         python3_version=python3_version,
    #     )
    #     ground_truth = stdout_stderr
    #     # -- Add tag to ground truth
    #     # ground_truth = "##<|predtype|>predoutput##\n" + ground_truth 
        

    #     miscinfo_dict = {
    #         "predtype": "predoutput",
    #         "question": question_predoutput,
    #         "ground_truth": ground_truth,
    #         "n_char_augment": n_char_augment,
    #         "run_info": run_info,
    #         "result_info": result_info,
    #         "stdout_stderr": stdout_stderr,
    #         "code_str": code_str,
    #     }
    #     return [{"role": "user", "content": question_predoutput}], miscinfo_dict


    # TODO
    def _build_messages_predoutput_stdin_function(
            self, example: Dict[str, Any], ppo_config,
            predtype
        ):
        code_str = example["code_str"]
        has_syntax_error, stderr = judge_has_syntax_error(code_str)
        if has_syntax_error:
            print(f"Syntax error in code_str. {stderr}. Skipping...")
            return [], {}
        
        session_name = "session_" + "".join(random.sample(string.ascii_letters + string.digits, 8))


        test_cases_str = example['test_cases_str'] 
        try:
            test_cases = json.loads(test_cases_str) # Should be a list after json.loads
        except:
            print(f"Error json.loads: {test_cases_str}")
            return [], {}
        if not isinstance(test_cases, list):
            print(f"Error: test_cases should be list: {test_cases}")
            return [], {}
        



        code_str_augmented = deepcopy(code_str)

        # -- augment code logical
        if self.ppo_config.use_logical_augment:
            code_str_logical_augment_prob = self.ppo_config.prob
            try:
                code_str_augmented = fix_spaces_around_keywords(code_str_augmented)
                code_str_augmented = augment_code_str_logical(code_str_augmented, prob=code_str_logical_augment_prob)
            except Exception as e:
                print(f"Error in augment_code_str_logical. {e}. Skipping... {code_str_augmented}")
                return [], {}


        # -- augment code digit
        if self.ppo_config.use_digit_augment:
            code_str_digit_augment_prob = self.ppo_config.prob
            code_str_augmented = augment_code_str_digit(code_str_augmented, prob=code_str_digit_augment_prob)
            has_syntax_error, stderr = judge_has_syntax_error(code_str_augmented)
            if has_syntax_error:
                print(f"Syntax error in augment_code_str_digit. {stderr}. Skipping...")
                return [], {}




        # ========== if len(test_cases) == 0
        # no STDIN, no function call args or kwargs
        if len(test_cases) == 0:
            run_info = RunInfo(
                code_str=code_str_augmented,  
                session_name=session_name,
                project_root_name="project_root_name",
                language="python",
                timeout=5,
                delete_after_run=True,
                pre_command="unset DISPLAY; "
            )
            result_info = opencr_run(run_info=run_info, is_run=True)
            ground_truth = {
                "stdout": result_info.stdout_str,
                "stderr": result_info.stderr_str
            }
            

            if predtype == "predoutput":
                question = question_template_predoutput_noinput.format(
                    file_abspath=run_info.file_infos[0].file_abspath,
                    file_content=run_info.file_infos[0].file_content,
                    language=run_info.language,
                    project_root_dir=run_info.project_root_dir,
                    python3_version=python3_version,
                    command=run_info.command,
                    tree_str=result_info.tree_str,
                    datetime_start=result_info.datetime_start,
                    datetime_end=result_info.datetime_end,
                )

            elif predtype == 'fim':
                code_str_fim, num_corrupted_lines = build_code_str_fim(
                    code_str=run_info.file_infos[0].file_content,
                    max_num_corrupted_lines=ppo_config.max_num_corrupted_lines,
                )
                question = question_template_fim.format(
                    file_abspath=run_info.file_infos[0].file_abspath,
                    file_content=code_str_fim,
                    language=run_info.language,
                    project_root_dir=run_info.project_root_dir,
                    python3_version=python3_version,
                    command=run_info.command,
                    tree_str=result_info.tree_str,
                    datetime_start=result_info.datetime_start,
                    datetime_end=result_info.datetime_end,
                    num_corrupted_lines=num_corrupted_lines,
                    stdout=result_info.stdout_str,
                    stderr=result_info.stderr_str,
                )

            miscinfo_dict = {
                "predtype": predtype,
                "inputtype": "noinput",
                "question": question,
                "ground_truth": ground_truth,
                "run_info": run_info,
                "result_info": result_info,
                "code_str": code_str,
                "num_corrupted_lines": num_corrupted_lines if predtype == 'fim' else None,
                "code_str_fim": code_str_fim if predtype == 'fim' else None,
            }
            return [{"role": "user", "content": question}], miscinfo_dict



        # ========== if len(test_cases) > 0 

        elif len(test_cases) > 0:
            test_case = random.choice(test_cases)
            if not isinstance(test_case, dict):
                print(f"Error: test_case should be dict: {test_case}")
                return [], {}

            each_input = test_case['input'] # str + list + dict
            each_input_augmented = deepcopy(each_input)


            # -- STDIN: augment input
            if isinstance(each_input_augmented, str): # STDIN
                if self.ppo_config.use_digit_augment:
                    stdin_augment_prob = self.ppo_config.prob
                    each_input_augmented = augment_stdin_char(each_input_augmented, prob=stdin_augment_prob)

                run_info = RunInfo(
                    code_str=code_str_augmented,  
                    session_name=session_name,
                    project_root_name="project_root_name",
                    language="python",
                    timeout=5,
                    input_content=each_input_augmented,
                    delete_after_run=True,
                    pre_command="unset DISPLAY; "
                )

                result_info = opencr_run(run_info=run_info, is_run=True)
                ground_truth = {
                    "stdout": result_info.stdout_str,
                    "stderr": result_info.stderr_str
                }
                

                if predtype == "predoutput":
                    question = question_template_predoutput_stdin.format(
                        file_abspath=run_info.file_infos[0].file_abspath,
                        file_content=run_info.file_infos[0].file_content,
                        language=run_info.language,
                        project_root_dir=run_info.project_root_dir,
                        python3_version=python3_version,
                        command=run_info.command,
                        tree_str=result_info.tree_str,
                        datetime_start=result_info.datetime_start,
                        datetime_end=result_info.datetime_end,
                    )
                elif predtype == 'fim':
                    code_str_fim, num_corrupted_lines = build_code_str_fim(
                        code_str=run_info.file_infos[0].file_content,
                        max_num_corrupted_lines=ppo_config.max_num_corrupted_lines,
                    )
                    question = question_template_fim.format(
                        file_abspath=run_info.file_infos[0].file_abspath,
                        file_content=code_str_fim,
                        language=run_info.language,
                        project_root_dir=run_info.project_root_dir,
                        python3_version=python3_version,
                        command=run_info.command,
                        tree_str=result_info.tree_str,
                        datetime_start=result_info.datetime_start,
                        datetime_end=result_info.datetime_end,
                        num_corrupted_lines=num_corrupted_lines,
                        stdout=result_info.stdout_str,
                        stderr=result_info.stderr_str,
                    )

                miscinfo_dict = {
                    "predtype": predtype,
                    "inputtype": "stdin",
                    "question": question,
                    "ground_truth": ground_truth,
                    "run_info": run_info,
                    "result_info": result_info,
                    "code_str": code_str,
                    "num_corrupted_lines": num_corrupted_lines if predtype == 'fim' else None,
                    "code_str_fim": code_str_fim if predtype == 'fim' else None,
                }
                return [{"role": "user", "content": question}], miscinfo_dict


            # -- Function call args
            elif isinstance(each_input_augmented, list) or isinstance(each_input_augmented, dict): # Function call args
                func_name = test_case['func_name']  # func   cls().func

                if self.ppo_config.use_digit_augment:
                    function_input_augment_prob = self.ppo_config.prob
                    each_input_augmented = augment_function_input(each_input_augmented, prob=function_input_augment_prob)

                full_code_info = stitch_testcase_func_into_codestr(
                    code_str=code_str_augmented,
                    func_name=func_name,
                    func_args = each_input_augmented,
                    func_kwargs = {},
                    use_opencr_run=True
                )
                code_str = full_code_info['full_code']
                run_info = full_code_info['run_info']
                result_info = full_code_info['result_info']
                ground_truth = {
                    "stdout": result_info.stdout_str,
                    "stderr": result_info.stderr_str
                }

                if predtype == "predoutput":
                    question = question_template_predoutput_function.format(
                        file_abspath=run_info.file_infos[0].file_abspath,
                        file_content=run_info.file_infos[0].file_content,
                        language=run_info.language,
                        project_root_dir=run_info.project_root_dir,
                        python3_version=python3_version,
                        command=run_info.command,   
                        tree_str=result_info.tree_str,
                        datetime_start=result_info.datetime_start,
                        datetime_end=result_info.datetime_end,
                    )
                elif predtype == 'fim':
                    code_str_fim, num_corrupted_lines = build_code_str_fim(
                        code_str=run_info.file_infos[0].file_content,
                        max_num_corrupted_lines=ppo_config.max_num_corrupted_lines,
                    )
                    question = question_template_fim.format(
                        file_abspath=run_info.file_infos[0].file_abspath,
                        file_content=code_str_fim,
                        language=run_info.language,
                        project_root_dir=run_info.project_root_dir,
                        python3_version=python3_version,
                        command=run_info.command,
                        tree_str=result_info.tree_str,
                        datetime_start=result_info.datetime_start,
                        datetime_end=result_info.datetime_end,
                        num_corrupted_lines=num_corrupted_lines,
                        stdout=result_info.stdout_str,
                        stderr=result_info.stderr_str,
                    )

                
                miscinfo_dict = {
                    "predtype": predtype,
                    "inputtype": "function",
                    "question": question,
                    "ground_truth": ground_truth,
                    "run_info": run_info,
                    "result_info": result_info,
                    "code_str": code_str,
                    "num_corrupted_lines": num_corrupted_lines if predtype == 'fim' else None,
                    "code_str_fim": code_str_fim if predtype == 'fim' else None,
                }
                return [{"role": "user", "content": question}], miscinfo_dict



            # -- Function call kwargs
            elif isinstance(each_input_augmented, dict): # Function call kwargs
                func_name = test_case['func_name']  # func   cls().func
                
                if self.ppo_config.use_digit_augment:
                    function_input_augment_prob = self.ppo_config.prob
                    each_input_augmented = augment_function_input(each_input_augmented, prob=function_input_augment_prob)

                full_code_info = stitch_testcase_func_into_codestr(
                    code_str=code_str_augmented,
                    func_name=func_name,
                    func_args = [],
                    func_kwargs = each_input_augmented,
                    use_opencr_run=True,
                )
                code_str = full_code_info['full_code']
                run_info = full_code_info['run_info']
                result_info = full_code_info['result_info']
                ground_truth = {
                    "stdout": result_info.stdout_str,
                    "stderr": result_info.stderr_str
                }

                if predtype == "predoutput":
                    question = question_template_predoutput_function.format(
                        file_abspath=run_info.file_infos[0].file_abspath,
                        file_content=run_info.file_infos[0].file_content,
                        language=run_info.language,
                        project_root_dir=run_info.project_root_dir,
                        python3_version=python3_version,
                        command=run_info.command,
                        tree_str=result_info.tree_str,
                        datetime_start=result_info.datetime_start,
                        datetime_end=result_info.datetime_end,
                    )
                elif predtype == 'fim':
                    code_str_fim, num_corrupted_lines = build_code_str_fim(
                        code_str=run_info.file_infos[0].file_content,
                        max_num_corrupted_lines=ppo_config.max_num_corrupted_lines,
                    )
                    question = question_template_fim.format(
                        file_abspath=run_info.file_infos[0].file_abspath,
                        file_content=code_str_fim,
                        language=run_info.language,
                        project_root_dir=run_info.project_root_dir,
                        python3_version=python3_version,
                        command=run_info.command,
                        tree_str=result_info.tree_str,
                        datetime_start=result_info.datetime_start,
                        datetime_end=result_info.datetime_end,
                        num_corrupted_lines=num_corrupted_lines,
                        stdout=result_info.stdout_str,
                        stderr=result_info.stderr_str,
                    )

                miscinfo_dict = {
                    "predtype": predtype,
                    "inputtype": "function",
                    "question": question,
                    "ground_truth": ground_truth,
                    "run_info": run_info,
                    "result_info": result_info,
                    "code_str": code_str,
                    "num_corrupted_lines": num_corrupted_lines if predtype == 'fim' else None,
                    "code_str_fim": code_str_fim if predtype == 'fim' else None,
                }
                return [{"role": "user", "content": question}], miscinfo_dict






    # def _build_messages_predcodeaug(self, example: Dict[str, Any], ppo_config) -> List[Dict[str, Any]]: 
    #     code_str = example["code_str"]
    #     session_name = "session_" + "".join(random.sample(string.ascii_letters + string.digits, 10))
    
    #     run_info_original = RunInfo(
    #         code_str=code_str,
    #         project_root_name="project_root_name",
    #         session_name=session_name,
    #         language="python",
    #         timeout=3
    #     )
    #     result_info_original = opencr_run(run_info=run_info_original, is_run=True)
    #     stdout_original = result_info_original.stdout if isinstance(result_info_original.stdout, str) else result_info_original.stdout.decode()
    #     stderr_original = result_info_original.stderr if isinstance(result_info_original.stderr, str) else result_info_original.stderr.decode()
    #     stdout_stderr_original = "\n".join([stdout_original, stderr_original])


    #     n_char_augment = ppo_config.n_char_augment
    #     code_str_corrupted = augment_code_str_char(code_str, n=n_char_augment, charset=ppo_config.charset)  
    #     run_info_corrupted = RunInfo(
    #         code_str=code_str_corrupted,  # Corrupting the code
    #         language="python",
    #         project_root_name="project_root_name",
    #         session_name=session_name,
    #         timeout=3
    #     )

    #     result_info_corrupted = opencr_run(run_info=run_info_corrupted, is_run=True)
    #     stdout_corrupted = result_info_corrupted.stdout if isinstance(result_info_corrupted.stdout, str) else result_info_corrupted.stdout.decode()
    #     stderr_corrupted = result_info_corrupted.stderr if isinstance(result_info_corrupted.stderr, str) else result_info_corrupted.stderr.decode()
    #     stdout_stderr_corrupted = "\n".join([stdout_corrupted, stderr_corrupted])



    #     question_predcodeaug = question_template_predcodeaug.format(
    #         file_abspath=run_info_corrupted.file_infos[0].file_abspath,
    #         file_content=run_info_corrupted.file_infos[0].file_content,
    #         language=run_info_corrupted.language,
    #         project_root_dir=run_info_corrupted.project_root_dir,
    #         command = run_info_corrupted.command,
    #         output_corrupted=stdout_stderr_corrupted,
    #         output_original=stdout_stderr_original,
    #         n_char_augment=n_char_augment,
    #         python3_version=python3_version,
    #     )
    #     # The ground truth is the corrupted code string
    #     ground_truth = code_str_corrupted  


    #     miscinfo_dict = {
    #         "predtype": "predcodeaug",
    #         "question": question_predcodeaug,
    #         "ground_truth": ground_truth,
    #         "n_char_augment": n_char_augment,
    #         "run_info_original": run_info_original,
    #         "result_info_original": result_info_original,
    #         "run_info_corrupted": run_info_corrupted,
    #         "result_info_corrupted": result_info_corrupted,
    #         "stdout_stderr_original": stdout_stderr_original,
    #         "stdout_stderr_corrupted": stdout_stderr_corrupted,
    #         "code_str_original": code_str,
    #         "code_str_corrupted": code_str_corrupted,
    #     }

    #     return [{"role": "user", "content": question_predcodeaug}], miscinfo_dict


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        
        example: dict = self.dataset[index]


        # -- Sample-level predtype
        if random.random() <= self.ppo_config.predoutput_ratio:
            predtype = "predoutput"
        else:
            predtype = 'fim'

        # -- predoutputstdin
        # messages, miscinfo_dict = self._build_messages_predoutput_stdin_function(
        #     example, ppo_config=self.ppo_config,
        #     predtype=predtype
        # )
        try:
            messages, miscinfo_dict = self._build_messages_predoutput_stdin_function(
                example, ppo_config=self.ppo_config,
                predtype=predtype
            )
        except Exception as e:
            print(f"⚠️ Error in _build_messages_predoutput_stdin: {e}")
            print(f"Skipping index {index}...")
            return self.__getitem__(random.randint(0, len(self.dataset) - 1))


        # test_type = example["test_type"]
        # if not test_type == "stdin":
        #     print(f"Skip the row with test_type: {test_type}")
        #     return self.__getitem__(random.randint(0, len(self.dataset) - 1))
        # try:
        #     messages, miscinfo_dict = self._build_messages_predoutput_stdin(example, ppo_config=self.ppo_config)
        # except Exception as e:
        #     print(f"Error in _build_messages_predoutput_stdin: {e}")
        #     print(f"Skipping sample {index}...")
        #     return self.__getitem__(random.randint(0, len(self.dataset) - 1))


        # # -- Batch-level predtype
        # if self.predtype == "predoutput":
        #     messages, miscinfo_dict = self._build_messages_predoutput(example, ppo_config=self.ppo_config)
        # elif self.predtype == "predcodeaug":
        #     messages, miscinfo_dict = self._build_messages_predcodeaug(example, ppo_config=self.ppo_config)
        # else:
        #     raise NotImplementedError



        if len(messages) == 0:
            print(f"Sample {index} has no messages. Skipping...")
            return self.__getitem__(random.randint(0, len(self.dataset) - 1))


        if "timed out" in miscinfo_dict["ground_truth"]["stderr"].lower():
            print_interval = 10
            self.num_timed_out_samples += 1
            if self.num_timed_out_samples % print_interval == 1:
                print(f"Sample {index} timed out. Total {self.num_timed_out_samples}. skipping...")
            return self.__getitem__(random.randint(0, len(self.dataset) - 1))
        

        # # -- other errors 
        # # value error   index error     type error    syntax error   name error
        # print_interval = 10
        # stderr_lower = miscinfo_dict["ground_truth"]["stderr"].lower()
        # if "value error" in stderr_lower or "valueerror" in stderr_lower:
        #     self.num_value_error_samples += 1
        #     if self.num_value_error_samples % print_interval == 1:
        #         print(f"Sample {index} has ValueError. Total {self.num_value_error_samples}. Skipping...")
        #         print("="*10)
        #     return self.__getitem__(random.randint(0, len(self.dataset) - 1))
        
        # if "index error" in stderr_lower or "indexerror" in stderr_lower:
        #     self.num_index_error_samples += 1
        #     if self.num_index_error_samples % print_interval == 1:
        #         print(f"Sample {index} has IndexError. Total {self.num_index_error_samples}. Skipping...")
        #         print("="*10)
        #     return self.__getitem__(random.randint(0, len(self.dataset) - 1))
        
        # if "type error" in stderr_lower or "typeerror" in stderr_lower:
        #     self.num_typed_error_samples += 1
        #     if self.num_typed_error_samples % print_interval == 1:
        #         print(f"Sample {index} has TypeError. Total {self.num_typed_error_samples}. Skipping...")
        #         print("="*10)
        #     return self.__getitem__(random.randint(0, len(self.dataset) - 1))
        
        
        # if "name error" in stderr_lower or "nameerror" in stderr_lower:
        #     self.num_name_error_samples += 1
        #     if self.num_name_error_samples % print_interval == 1:
        #         print(f"Sample {index} has NameError. Total {self.num_name_error_samples}. Skipping...")
        #         print("="*10)
        #     return self.__getitem__(random.randint(0, len(self.dataset) - 1))


        # -- Maintain supported error types
        print_interval = 25
        stderr_lower = miscinfo_dict["ground_truth"]["stderr"].lower()
        if "error" in stderr_lower or "warning" in stderr_lower:
            has_supported_error = False
            for error_type in self.supported_error_types:
                if error_type in stderr_lower or error_type.replace(" ", "") in stderr_lower:
                    self.num_error_samples[error_type] += 1
                    self.num_error_samples['total'] += 1
                    has_supported_error = True
                    break
            if not has_supported_error:
                return self.__getitem__(random.randint(0, len(self.dataset) - 1))
            if self.num_error_samples['total'] % print_interval == print_interval:
                pprint(self.num_error_samples)
                print("-"*10)


        # # -- Exclude not supported error types + only builtin
        # print_interval = 50
        # stderr_lower = miscinfo_dict["ground_truth"]["stderr"].lower()
        # if "error" in stderr_lower or "warning" in stderr_lower:
        #     has_builtin_error = any(e.lower() in stderr_lower for e in python_builtin_exceptions)
        #     if not has_builtin_error:
        #         print(f"Sample {index} has no builtin error in stderr: {stderr_lower}. Skipping...")
        #         return self.__getitem__(random.randint(0, len(self.dataset) - 1))
        #     else:
        #         has_notsupported_error = any(
        #             e.lower() in stderr_lower or e.lower().replace(" ", "") in stderr_lower for e in self.notsupported_error_types
        #         )
        #         if has_notsupported_error:
        #             print(f"Sample {index} has not supported error in stderr: {stderr_lower}. Skipping...")
        #             return self.__getitem__(random.randint(0, len(self.dataset) - 1))




        # # -- Exclude not supported error types
        # print_interval = 50
        # stderr_lower = miscinfo_dict["ground_truth"]["stderr"].lower()
        # if "error" in stderr_lower or "warning" in stderr_lower:
        #     has_notsupported_error = any(
        #         e.lower() in stderr_lower or e.lower().replace(" ", "") in stderr_lower for e in self.notsupported_error_types
        #     )
        #     if has_notsupported_error:
        #         print(f"Sample {index} has not supported error in stderr: {stderr_lower}. Skipping...")
        #         return self.__getitem__(random.randint(0, len(self.dataset) - 1))





        # # -- exclude all error samples
        # print_interval = 50
        # stderr_lower = miscinfo_dict["ground_truth"]["stderr"].lower()
        # if "error" in stderr_lower or "warning" in stderr_lower:
        #     self.num_error_samples += 1
        #     if self.num_error_samples % print_interval == 1:
        #         # print(f"Sample {index} has error in stderr: {stderr_lower}. Skipping...")
        #         # print(f"Sample {index} has error in stderr. Total {self.num_error_samples}. Skipping...")
        #         print(f"Sample {index} has error in stderr. Total {self.num_error_samples}. {stderr_lower}. Skipping...")
        #         print("-"*10)
        #     return self.__getitem__(random.randint(0, len(self.dataset) - 1))



        example['question'] = miscinfo_dict["question"]
        example["ground_truth"] = miscinfo_dict["ground_truth"]
        example["miscinfo_dict"] = miscinfo_dict

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = [self.process_image(image) for image in example.pop(self.image_key)]
            model_inputs = self.processor(images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"image": images}
            example["multi_modal_inputs"] = dict(model_inputs)
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            # qwen2vl mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        # example["ground_truth"] = example.pop(self.answer_key)
        example["prompt"] = prompt

        # # -- Run and obtain ground truth
        # result_info = opencr_run(run_info=run_info,
        #                             # host="172.16.142.130",
        #                             # host="192.168.246.128",
        #                             # host="0.0.0.0",
        #                             # port=8000,
        #                             )
        
        # if isinstance(result_info.stdout, bytes):
        #     result_info.stdout = result_info.stdout.decode()
        # if isinstance(result_info.stderr, bytes):
        #     result_info.stderr = result_info.stderr.decode()
        # gt_stdout_stderr = "\n".join([result_info.stdout, result_info.stderr])
        # example["ground_truth"] = gt_stdout_stderr

        return example
