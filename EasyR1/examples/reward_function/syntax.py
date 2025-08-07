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

import re
from typing import Dict, List

# from mathruler.grader import extract_boxed_content, grade_answer
import difflib
from copy import deepcopy
import numpy as np
from collections import Counter

from opencoderunner import run as opencr_run
from opencoderunner import RunInfo

import ast
from verl.utils.question_template_reward import question_template_reward
from datetime import datetime, timezone
import sys
import os
import time
from openai import Client
import concurrent
import concurrent.futures
import random

python3_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"



non_determine_strs = [
    # process/thread
    # "thread", "process", "multiprocessing", "threading", "concurrent", "asyncio",

    # Time
    "date", "time",

    # Random
    "rand", "random", "randint", "choice", "shuffle", "sample", "seed", "randomize",

    # Web
    "website", 'github', "git", "url", "http", 
    "web", "browser", "internet", "network", "socket",
    "request", "api", 'user', "mail", "send", "receive",
    "download", "upload", 'ping', 'fetch', 'wget', 'curl',

    # # file
    # "csv", "pandas", "pd", "file", "json", "xml", "yaml", "open", "read", "write",

    # # Visualization 
    # "plt", "tk", "pygame", "draw", "image", "video", "audio", "sound", "chart", "graph", "plot", "figure", "canvas", "matplotlib", "cv2", "opencv", "pillow", "PIL", "tkinter",
    # "graphics", 'visual', "viz", "tkinter", "turtle", "imageio",

    # # AI/ML: This will use GPU
    # "torch", "tensorflow", "pytorch", "bert", "gpt", "llm", "chatgpt", "openai", "huggingface", "hf", "transformers", "dgl", "nn", "detectron", "timm", "accelerate", "datasets", "torchvision", "diffusers", "pytorch_lightning", "cuda", "gpu"
]


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



def judge_is_number_str(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


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



def judge_non_determine(code_str: str, keywords: list[str]) -> bool:
    "both left and right char should be non-letter"
    for kw in keywords:
        pattern = rf"(?<![a-zA-Z]){re.escape(kw)}(?![a-zA-Z])"
        if re.search(pattern, code_str):
            return True
    return False




def extract_md_anyblock(text):
    pattern = r'```\s*?([a-zA-Z0-9]*?)\s*?\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches




def extract_md_block_withtag(text, tag):
    pattern = r'```' + re.escape(tag) + r'\s*\n(.*?)\s*```' # exclude whitespace
    # pattern = r'```' + re.escape(tag) + r'\n(.*?)```' # maintain whitespace
    # pattern = r'```' + re.escape(tag) + r'(.*?)\n```' # maintain whitespace
    matches = re.findall(pattern, text, re.DOTALL)
    codetexts = []
    for i, match in enumerate(matches):
        # codetext = match.strip()  # exclude whitespace
        codetext = match
        codetexts.append(codetext)
    # print(codetexts)
    codetexts = sorted(codetexts) # strlength-Increasing-order  
    if len(codetexts) > 0:
        codetext = codetexts[-1]
    elif len(codetexts) == 0:
        codetext = None
    return codetext



def clip_min_max(value, min_value=0, max_value=1):
    output = min(max(value, min_value), max_value)
    return output








def compute_multiline_counter_iou(gt:list, pred:list):
    # use Counter
    gt_counter = Counter(gt)
    pred_counter = Counter(pred)
    intersec_counter = gt_counter & pred_counter
    union_counter = gt_counter | pred_counter
    len_intersec = sum(intersec_counter.values())  # 计算交集大小
    len_union = sum(union_counter.values())        # 计算并集大小
    if len_union == 0:
        iou = 1
    else:
        iou = len_intersec / len_union
        iou = clip_min_max(iou, 0, 1)  # Clip IoU to [0, 1]
    return iou, intersec_counter, union_counter



def judge_answer(gt, pred):
    if isinstance(gt, str):
        assert isinstance(pred, str)
        gt_list = list(gt.splitlines())
        pred_list = list(pred.splitlines())
        iou, intersec_counter, union_counter = compute_multiline_counter_iou(gt_list, pred_list)

    elif isinstance(gt, list):
        assert isinstance(pred, list)
        gt_list = deepcopy(gt)
        pred_list = deepcopy(pred)
        iou, intersec_counter, union_counter = compute_multiline_counter_iou(gt_list, pred_list)

    else:
        raise NotImplementedError
    
    if max(len(pred_list), len(gt_list)) == 0:
        length_score = 1.0
    else:
        length_score = 1 - abs(len(pred_list) - len(gt_list)) / max(len(pred_list), len(gt_list))
    length_score = clip_min_max(length_score, 0, 1)

    output_dict = {
        "iou": iou,
        "num_correct_lines": len(intersec_counter),
        "num_gt_lines": len(gt_list),
        "num_pred_lines": len(pred_list),
        "length_score": length_score,
    }

    return output_dict


def judge_answer_hard(gt:str, pred:str, ignore_first_last_newline=True):
    # To avoid leading/trailing newlines
    if ignore_first_last_newline:
        if gt == pred:
            is_correct = True
        elif gt == pred + '\n':
            is_correct = True
        elif gt == '\n' + pred:
            is_correct = True
        elif pred == gt + '\n':
            is_correct = True
        elif pred == '\n' + gt:
            is_correct = True
        else:
            is_correct = False
    else:
        if gt == pred:
            is_correct = True
        else:
            is_correct = False
    return is_correct


def judge_answer_triplet(gt, pred, ref):
    linediff_ref_gt = list(difflib.ndiff(gt.splitlines(), ref.splitlines()))
    linediffonly_ref_gt = [e for e in linediff_ref_gt if e[0] in ['?', '+', '-']]
    linediffonly_ref_gt = list(linediffonly_ref_gt)

    linediff_ref_pred = list(difflib.ndiff(pred.splitlines(), ref.splitlines()))
    linediffonly_ref_pred = [e for e in linediff_ref_pred if e[0] in ['?', '+', '-']]
    linediffonly_ref_pred = list(linediffonly_ref_pred)
    output_dict_linediff = judge_answer(gt=linediffonly_ref_gt, pred=linediffonly_ref_pred)

    return output_dict_linediff



# -- Pred output
def compute_score_predoutput(gt, pred, ppo_config): 
    output_dict = judge_answer(gt=gt, pred=pred)
    iou_score = output_dict["iou"]
    length_score = output_dict["length_score"]
    length_score = length_score * ppo_config.predoutput_length_weight

    score_dict = {
        "iou": iou_score,
        "length": length_score,
    }
    return score_dict




# -- Pred output stdin
def compute_score_predoutput_fim(gt, pred, ppo_config):
    # -- Judge stdout
    gt_stdout = gt["stdout"]
    # pred_stdout = "" if pred["stdout"] is None else pred["stdout"]
    pred_stdout = pred["stdout"]
    if pred_stdout is None:
        is_correct_stdout = False
    else:
        is_correct_stdout = judge_answer_hard(gt=gt_stdout, pred=pred_stdout, ignore_first_last_newline=True)
    None



    # -- judge stderr. pred_stderr == None, then pred_stderr = ""
    # gt_stderr = gt["stderr"]
    # pred_stderr = pred["stderr"]
    # if pred_stderr is None:
    #     pred_stderr = ""
    # output_dict_stderr = judge_answer(gt=gt_stderr, pred=pred_stderr)
    # iou_score = output_dict_stderr["iou"]
    # iou_score = iou_score * ppo_config.stderr_weight


    # -- judge stderr. pred_stderr == None, then iou_score = 0
    gt_stderr = gt["stderr"]
    pred_stderr = pred["stderr"]
    if pred_stderr is None:
        iou_score = 0
    else:
        output_dict_stderr = judge_answer(gt=gt_stderr, pred=pred_stderr)
        iou_score = output_dict_stderr["iou"]
        iou_score = iou_score * ppo_config.stderr_weight

    if ppo_config.reward_stderr:
        score_dict = {
            "iou_stderr": iou_score,
            "is_correct_stdout": int(is_correct_stdout),
        }
    else:
        score_dict = {
            "is_correct_stdout": int(is_correct_stdout),
        }

    return score_dict




# -- Pred code
def compute_score_predcodeaug(gt, pred, code_str_original, ppo_config, miscinfo_dict):

    # -- judge result
    run_info_corrupted_gt = miscinfo_dict["run_info_corrupted"]
    stdouterr_corrupted_gt = miscinfo_dict["stdout_stderr_corrupted"]
    run_info_corrupted_pred = RunInfo(
        code_str=pred,  # Corrupting the code
        language="python",
        project_root_name=run_info_corrupted_gt.project_root_name,
        session_name=run_info_corrupted_gt.session_name,
        timeout=3
    )
    result_info_corrupted_pred = opencr_run(run_info_corrupted_pred)
    stdouterr_corrupted_pred = result_info_corrupted_pred.stdout_stderr
    output_dict_stdouterr = judge_answer(gt=stdouterr_corrupted_gt, pred=stdouterr_corrupted_pred)
    if output_dict_stdouterr["iou"] == 1:
        result_score = 1
    else:
        result_score = 0





    # -- judge code_str
    output_dict_code_str_linediff = judge_answer_triplet(gt=gt, pred=pred, ref=code_str_original)
    code_str_linediff_iou_score = output_dict_code_str_linediff["iou"]
    code_str_linediff_length_score = output_dict_code_str_linediff["length_score"]
    code_str_linediff_length_score = code_str_linediff_length_score * ppo_config.predoutput_length_weight


    score_dict = {
        "diff_iou": code_str_linediff_iou_score,
        "diff_length": code_str_linediff_length_score,
        "result_score": result_score,
    }

    return score_dict





def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.1,
                #   questions=None, 
                  miscinfo_dicts=None, ppo_config=None) -> List[Dict[str, float]]:

        

    
    # ppo_config = ppo_config
    # breakpoint()
    scores = []
    answers = []
    for predict, ground_truth, miscinfo_dict in zip(predicts, ground_truths, miscinfo_dicts):
        # # -- Find predtype + exclude predtype
        # predtype, ground_truth_pure = extract_and_exclude_predtype(ground_truth)
        predtype = miscinfo_dict["predtype"]
        # question = miscinfo_dict["question"]


        # -- Predict output
        # if predtype == 'predoutput':
        #     answer = extract_md_block_withtag(predict, 'answer')
        #     if answer is None:
        #         score_dict = {
        #             "overall": 0,
        #             "format": 0,
        #             "accuracy": 0,
        #         }
        #         scores.append(score_dict)
        #         continue
        #     format_score = 1
        #     is_correct_dict = compute_score_predoutput(gt=ground_truth, 
        #                                                pred=answer, 
        #                                                ppo_config=ppo_config
        #                                                )
        #     accuracy_score = sum(is_correct_dict.values()) / len(is_correct_dict)
        #     overall_score = (1 - format_weight) * accuracy_score + format_weight * format_score
        #     score_dict = {
        #         "overall": overall_score,
        #         "format": format_score,
        #         "accuracy": accuracy_score,
        #         **is_correct_dict
        #     }
        #     scores.append(score_dict)


        # # -- predict code corrupted
        # elif predtype == 'predcodeaug': 
        #     answer = extract_md_block_withtag(predict, 'answer_code')
        #     if answer is None:
        #         score_dict = {
        #             "overall": 0,
        #             "format": 0,
        #             "accuracy": 0,
        #         }
        #         scores.append(score_dict)
        #         continue
        #     format_score = 1
        #     code_str_original = miscinfo_dict["code_str_original"]
        #     subscore_dict = compute_score_predcodeaug(gt=ground_truth, pred=answer, 
        #                                                code_str_original=code_str_original,
        #                                                ppo_config=ppo_config,
        #                                                miscinfo_dict=miscinfo_dict
        #                                                )
            
        #     accuracy_score = sum(subscore_dict.values()) / len(subscore_dict)
        #     overall_score = (1-format_weight) * accuracy_score + format_weight*format_score

        #     score_dict = {
        #         "overall": overall_score,
        #         "format": format_score,
        #         "accuracy": accuracy_score,
        #         **subscore_dict
        #     }
        #     scores.append(score_dict)


        # ==== Predict output stdin function noinput 
        if predtype in ['predoutput']:
            answer_stderr = extract_md_block_withtag(predict, 'answer_stderr')
            answer_stdout = extract_md_block_withtag(predict, 'answer_stdout')
            answer = {
                "stdout": answer_stdout,
                "stderr": answer_stderr
            }
            answers.append(answer)
            format_score = sum([1 if not e == None else 0 for e in answer.values()]) / len(answer)
            if format_score == 0:
                score_dict = {
                    "overall": 0,
                    "format": 0,
                    "accuracy": 0,
                }
                scores.append(score_dict)
                continue
            is_correct_dict = compute_score_predoutput_fim(gt=ground_truth,
                                                       pred=answer,
                                                       ppo_config=ppo_config
                                                       )
            accuracy_score = sum(is_correct_dict.values()) / len(is_correct_dict)
            overall_score = (1 - format_weight) * accuracy_score + format_weight * format_score
            score_dict = {
                "overall": overall_score,
                "format": format_score,
                "accuracy": accuracy_score,
                **is_correct_dict
            }
            scores.append(score_dict)


        # ==== fim
        elif predtype == 'fim':
            # breakpoint()
            num_corrupted_lines = miscinfo_dict["num_corrupted_lines"]
            code_str_fim = miscinfo_dict["code_str_fim"]
            question = miscinfo_dict["question"]
            run_info = miscinfo_dict["run_info"]

            # fim_answers = []
            fim_answer_dict = {}
            for i in range(num_corrupted_lines):
                tag = "answer_MASKED_LINE_" + str(i)
                answer_masked_line = extract_md_block_withtag(predict, tag)
                tag_pure = tag.replace("answer_", "")
                fim_answer_dict[tag_pure] = answer_masked_line
            format_score = sum([1 if not e is None else 0 for e in fim_answer_dict.values()]) / len(fim_answer_dict)
            if format_score == 0:
                score_dict = {
                    "overall": 0,
                    "format": 0,
                    "accuracy": 0,
                }
                scores.append(score_dict)
                continue
            
            code_str_fim_reconstructed = deepcopy(code_str_fim)
            for tag_pure, answer_masked_line in fim_answer_dict.items():
                if answer_masked_line is not None:
                    code_str_fim_reconstructed = code_str_fim_reconstructed.replace(tag_pure, answer_masked_line)
            # print("-"*20)
            # print(code_str_fim_reconstructed)
            run_info_reconstructed = deepcopy(run_info)
            run_info_reconstructed.file_infos[0].file_content = code_str_fim_reconstructed
            result_info_reconstructed = opencr_run(run_info_reconstructed)
            # print(result_info)
            # print(result_info_reconstructed)
            answer = {
                "stdout": result_info_reconstructed.stdout_str,
                "stderr": result_info_reconstructed.stderr_str,
            }
            is_correct_dict = compute_score_predoutput_fim(gt=ground_truth,
                                                              pred=answer,
                                                              ppo_config=ppo_config
                                                              )
            accuracy_score = sum(is_correct_dict.values()) / len(is_correct_dict)
            overall_score = (1 - format_weight) * accuracy_score + format_weight * format_score
            score_dict = {
                "overall": overall_score,
                "format": format_score,
                "accuracy": accuracy_score,
                **is_correct_dict
            }
            scores.append(score_dict)


        else:
            raise NotImplementedError
        






    # ---- Non-deterministic code 
    use_llm_score = False
    if use_llm_score:
        non_determine_ids = []
        for i, miscinfo_dict in enumerate(miscinfo_dicts):
            code_str = miscinfo_dict["code_str"]
            if judge_non_determine(code_str, non_determine_strs):
                non_determine_ids.append(i)
                continue
        non_determine_miscinfo_dicts = [ miscinfo_dicts[i] for i in non_determine_ids ]
        if len(non_determine_miscinfo_dicts) > 0: # has non-deterministic code
            None
            # breakpoint()
            questions = []
            for non_determine_id in non_determine_ids:
                miscinfo_dict = miscinfo_dicts[non_determine_id]
                run_info = miscinfo_dict["run_info"]
                result_info = miscinfo_dict["result_info"]
                answer = answers[non_determine_id]
                answer_stdout = answer["stdout"]
                answer_stderr = answer["stderr"]
                score_dict = scores[non_determine_id]
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
                questions.append(question)
            
            # - Run LLM
            # supplier = 'ali'
            # model = 'qwen2.5-coder-7b-instruct'
            supplier = 'siliconflow'
            model = 'Pro/Qwen/Qwen2.5-Coder-7B-Instruct'
            num_threads = min(len(questions), int(os.cpu_count()*0.75))
            t0 = time.time()
            print(f"⚠️ Calling LLMs for non-deterministic code blocks, num_threads={num_threads}, num_questions={len(questions)}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Submit LLM calls
                futures = []
                for question in questions:
                    try:
                        future = executor.submit(call_llms, supplier=supplier, model=model, input_text=question)
                    except Exception as e:
                        print(f"Error submitting LLM call: {e}")
                        future = None
                    futures.append(future)
                # Collect LLM outputs
                llm_output_dicts = []
                for i, future in enumerate(futures):
                    try:
                        llm_output_dict = future.result(timeout=60)  # Set a timeout for each LLM call
                    except Exception as e:
                        print(f"Error collecting LLM result: {e}")
                        llm_output_dict = {'output_text': "", 'response': None}
                    llm_output_dicts.append(llm_output_dict)
            print(f"⚠️ END calling LLMs for non-deterministic code blocks {time.time()-t0:.2f} ")

            # - Collect LLM scores
            llm_scores = []
            for i in range(len(llm_output_dicts)):
                llm_output_dict = llm_output_dicts[i]
                question = questions[i]
                miscinfo_dict = non_determine_miscinfo_dicts[i]

                output_text = llm_output_dict['output_text']
                llm_score = extract_md_block_withtag(output_text, 'llm_score')
                if llm_score is None:
                    llm_score = None
                else:
                    is_number_str = judge_is_number_str(llm_score)
                    if is_number_str:
                        llm_score = float(llm_score)
                        if llm_score < 0 or llm_score > 1:
                            print(f"Warning: LLM score {llm_score} is out of range [0, 1].")
                            llm_score = None
                    else:
                        llm_score = None
                llm_scores.append(llm_score)

            # - Update scores
            num_replaced = 0
            for i, non_determine_id in enumerate(non_determine_ids):
                llm_score = llm_scores[i]
                if llm_score is not None:
                    score_dict_org = scores[non_determine_id]
                    format_score = score_dict_org["format"]
                    accuracy_score = llm_score
                    overall_score = (1 - format_weight) * accuracy_score + format_weight * format_score
                    # Replace overall + accuracy
                    score_dict = {'format': format_score}
                    score_dict["overall"] = overall_score
                    score_dict["accuracy"] = accuracy_score
                    scores[non_determine_id] = score_dict
                    num_replaced += 1
            print(f"⚠️ Replaced {num_replaced} / {len(non_determine_ids)} non-deterministic samples to LLM scores.")
            None



    return scores







if __name__ == "__main__":
    gt = """
"""

    pred = """
```answer
123
```
"""
#     pred = """
# ```answer 123
# 123
# ```
# """
    answer = extract_md_block_withtag(pred, 'answer')
    print(repr(answer))
    scores = compute_score([pred], [gt])
    print(scores)
    None


    pred = """
"""
    answer = extract_md_block_withtag(pred, 'answer')
    print(repr(answer))

