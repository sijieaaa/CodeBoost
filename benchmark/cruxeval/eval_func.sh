#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh

set -e

if [ $# -lt 2 ]; then
  echo "Usage: $0 <global_step> <exp_name>"
  exit 1
fi

global_step=$1
exp_name=$2

# llm_dir=/home/users/ntu/sijiewan/scratch/llm
# llm_dir=/data/sijie/llm
easyr1_dir=$llm_dir/EasyR1_v4.14
lcb_dir=$llm_dir/LiveCodeBench
bcb_dir=${HOME}/llm/bigcodebench
plus_dir=$llm_dir/evalplus


chmod -R 777 $lcb_dir
chmod -R 777 $bcb_dir
chmod -R 777 $plus_dir



global_step_dir=$easyr1_dir/ckpts/$exp_name/$global_step
local_actor_dir=$global_step_dir/actor
local_model_path=$local_actor_dir/hf
model_repr=$exp_name-$global_step


# TEST
local_model_path="/data/sijie/llm/hf_models/Qwen/Qwen2.5-Coder-3B-Instruct"
model_repr="model_repr_123"



echo "⚠️ [Step 4] Evaluating with repobench..."
repobench_dir=${llm_dir}/repobench
cd ${repobench_dir}
# -- install tree-sitter
pip uninstall tree-sitter -y
pip uninstall tree-sitter-python -y
pip install tree-sitter
pip install tree-sitter-python
# -- Generate
export HF_DATASETS_OFFLINE=0
python run.py \
    --model_name $local_model_path \
    --dataset_name "tianyang/repobench_python_v1.1" \
    --start_date "2023-12-01" \
    --end_date "2023-12-31" \
    --language "python" \
    --max_token_nums 8192 \
    --levels "2k" "4k" \
    --temperature 0.2 \
    --top_p 0.95 \
    --max_new_tokens 128 \
    --batch_size 16 \
    --model_repr "$model_repr" \
    

# -- Evaluate
python eval.py --path "/data/sijie/llm/repobench/4k/Qwen2.5-Coder-3B-Instruct-python" --language "python" 2>&1 | tee >>"$log_path"