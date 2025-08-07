

# instruct_dir="/data/sijie/llm/Qwen2.5-Coder/qwencoder-eval/instruct"
cruxeval_dir="/data/sijie/llm/Qwen2.5-Coder/qwencoder-eval/instruct/cruxeval"
local_model_path=meta-llama/Llama-3.1-8B-Instruct
cd ${cruxeval_dir}


# -- cruxeval 
export CUDA_VISIBLE_DEVICES=0
unset DISPLAY
conda activate easyr1
pip install -r requirements.txt
OUTPUT_DIR=${cruxeval_dir}/cruxeval_results
unset DISPLAY
bash test.sh ${local_model_path} ${TP} ${OUTPUT_DIR}

