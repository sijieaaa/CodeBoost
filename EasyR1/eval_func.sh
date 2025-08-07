#!/bin/bash
# source ~/miniconda3/etc/profile.d/conda.sh

set -e

if [ $# -lt 3 ]; then
  echo "Usage: $0 <global_step> <exp_name> <lmstyle>"
  exit 1
fi

global_step=$1
exp_name=$2
lmstyle=$3


easyr1_dir=$cb_dir/EasyR1

lcb_dir=$cb_dir/benchmark/LiveCodeBench_v3
bcb_dir=$cb_dir/benchmark/bigcodebench
plus_dir=$cb_dir/benchmark/evalplus
cruxeval_dir=$cb_dir/benchmark/cruxeval


chmod -R 777 $lcb_dir
chmod -R 777 $bcb_dir
chmod -R 777 $plus_dir

# exp_name=Qwen2.5-Coder-3B-Instruct-a100-ep10-30k-ngpu4-b2-g32-roll256-max4096
global_step_dir=$easyr1_dir/ckpts/$exp_name/$global_step
local_actor_dir=$global_step_dir/actor
local_model_path=$local_actor_dir/hf
model_repr=$exp_name-$global_step




conda activate easyr1


echo "⚠️ [Step 0] Merging model..."
cd $easyr1_dir
python3 scripts/model_merger.py --local_dir $local_actor_dir




echo "⚠️ [Step 1.1] Evaluating with bigcodebench-complete..."
cd ${bcb_dir}
# uv pip install -e . --index-url https://pypi.tuna.tsinghua.edu.cn/simple
SUBSET=hard
SPLIT=complete
gen_path_rel=$(bigcodebench.generate --model $local_model_path --split $SPLIT --subset $SUBSET --backend vllm --resume False | tail -n 1)
# bcb-Evaluate-e2b
gen_path=${bcb_dir}/${gen_path_rel}
echo "----START"
echo ${gen_path}
log_path=${global_step_dir}/bcb-${SUBSET}-${SPLIT}--${model_repr}.log
echo ${log_path}
bigcodebench.evaluate --gradio_endpoint https://bigcode-bigcodebench-evaluator.hf.space/ \
  --model "" --split $SPLIT --subset $SUBSET --backend vllm --execution gradio \
  --samples ${gen_path} --log_path ${log_path}
echo "----END"



echo "⚠️ [Step 1.2] Evaluating with bigcodebench-instruct..."
cd ${bcb_dir}
# uv pip install -e . --index-url https://pypi.tuna.tsinghua.edu.cn/simple
SUBSET=hard
SPLIT=instruct
gen_path_rel=$(bigcodebench.generate --model $local_model_path --split $SPLIT --subset $SUBSET --backend vllm --resume False | tail -n 1)
# bcb-Evaluate-e2b
gen_path=${bcb_dir}/${gen_path_rel}
echo "----START"
echo ${gen_path}
log_path=${global_step_dir}/bcb-${SUBSET}-${SPLIT}--${model_repr}.log
bigcodebench.evaluate --gradio_endpoint https://bigcode-bigcodebench-evaluator.hf.space/ \
  --model "" --split $SPLIT --subset $SUBSET --backend vllm --execution gradio \
  --samples ${gen_path} --log_path ${log_path}
echo "----END"




echo "⚠️ [Step 2] Evaluating with LiveCodeBench..."
cd $lcb_dir
# pip install -e . --no-deps
# lmstyle="GenericBase"
# lmstyle="LLaMa3"
# lmstyle="CodeQwenInstruct"
# lmstyle="DeepSeekCodeInstruct"
log_path=${global_step_dir}/lcb-${lmstyle}--${model_repr}.log
python -m lcb_runner.runner.main --scenario codegeneration --evaluate --trust_remote_code \
  --start_date 2025-01-01 --end_date 2025-05-01 \
  --model $model_repr \
  --model_repr "$model_repr" \
  --local_model_path "$local_model_path" \
  --lmstyle $lmstyle \
  --num_process_evaluate 32 \
  --max_tokens 4096 \
  --n 10 2>&1 | tee >(tail -n 20 > "$log_path")




echo "⚠️ [Step 3.2] Evaluating with evalplus-mbpp..."
cd $plus_dir
log_path=${global_step_dir}/evalplus-mbpp--${model_repr}.log
evalplus.evaluate --model ${local_model_path} --dataset mbpp --backend vllm --tp 1 --greedy 2>&1 | tee >(tail -n 20 > "$log_path")





echo "⚠️ [Step 4] Evaluating with cruxeval..." # always out of space
unset DISPLAY
cd $cruxeval_dir
log_path=${global_step_dir}/cruxeval--${model_repr}.log
OUTPUT_DIR=${cruxeval_dir}/cruxeval_results/${model_repr}
mkdir -p ${OUTPUT_DIR}
# - DEBUG
# cruxeval_dir=${cb_dir}/cruxeval
# cd $cruxeval_dir
# local_model_path="Qwen/Qwen2.5-Coder-7B-Instruct"
# OUTPUT_DIR=${cruxeval_dir}/cruxeval_results/${local_model_path}
# log_path=${OUTPUT_DIR}/cruxeval--.log
# --
bash test.sh ${local_model_path} 1 ${OUTPUT_DIR} 2>&1 | tee ${log_path};
tmpfile=$(mktemp)
grep "pass" "${log_path}" > "$tmpfile"
if [ ! -s "$tmpfile" ]; then
    echo "[No matching 'pass' found in log]" > "$tmpfile"
fi
mv "$tmpfile" "${log_path}"


