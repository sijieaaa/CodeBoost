export VLLM_USE_MODELSCOPE=false

easyr1_dir=$cb_dir/EasyR1
cd $easyr1_dir

# sijieaaa/verl_prepare_datasets
huggingface-cli  download sijieaaa/verl_prepare_datasets  --repo-type dataset --local-dir ${cb_dir}/verl_prepare_datasets/datasets



code_str_lines_thd=10
code_str_length_thd=30
timeout=-1
downsample=1
opc_downsample=2
opc_ddownsample=2
clique_threshold=0.5
num_iters=10



# -- after maximum clique
train_files=""
train_files+=",$cb_dir/verl_prepare_datasets/datasets/OpenCoder-LLM@opc-sft-stage1@realuser_instruct_line$((code_str_lines_thd * 2))_len$((code_str_length_thd * 2))_time${timeout}_down${opc_downsample}_clique${clique_threshold}_i$((num_iters * 1)).csv"
train_files+=",$cb_dir/verl_prepare_datasets/datasets/open-thoughts@OpenThoughts-114k@metadata_line$((code_str_lines_thd * 1))_len$((code_str_length_thd * 1))_time${timeout}_down${downsample}_clique${clique_threshold}_i$((num_iters * 1)).csv"
train_files+=",$cb_dir/verl_prepare_datasets/datasets/open-r1@codeforces-cots@solutions_py_decontaminated_line$((code_str_lines_thd * 1))_len$((code_str_length_thd * 1))_time${timeout}_down${downsample}_clique${clique_threshold}_i$((num_iters * 1)).csv"
train_files="${train_files#,}" # remove leading comma if exists





export PYTHONUNBUFFERED=1
export RAY_DEBUG_POST_MORTEM=0 # ray debug

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
n_gpus_per_node=8
tensor_parallel_size=1




# MODEL_PATH=Qwen/Qwen2.5-Coder-1.5B-Instruct  
MODEL_PATH=Qwen/Qwen2.5-Coder-3B-Instruct  
# MODEL_PATH=Qwen/Qwen2.5-Coder-7B-Instruct 
# MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct  
# MODEL_PATH=ByteDance-Seed/Seed-Coder-8B-Reasoning-bf16
# MODEL_PATH=01-ai/Yi-Coder-9B-Chat

huggingface-cli download ${MODEL_PATH}


save_freq=15
total_epochs=2
global_batch_size=128                             
rollout_batch_size=512
strategy=adamw
micro_batch_size_per_device_for_update=2
micro_batch_size_per_device_for_experience=16
max_prompt_length=4096
max_response_length=4096
max_num_batched_tokens=$((max_prompt_length + max_response_length))
predoutput_ratio=0.75 # 0.75
predoutput_length_weight=0.2
n_char_augment=1
charset=basic
dataset_ratio=1 # 1
stderr_weight=0.5
prob=0.1

max_num_corrupted_lines=1
use_logical_augment=true # true
use_digit_augment=true # true
reward_stderr=true # true
supported_error_types="syntax error, index error, value error, name error, type error, key error, zero division error"


load_checkpoint_path=""


exp_name=${MODEL_PATH##*/}-Main
exp_name+=-nv$n_gpus_per_node
exp_name+=-tp$tensor_parallel_size  
exp_name+=-g$global_batch_size
exp_name+=-ro$rollout_batch_size
exp_name+=-max$max_prompt_length-$max_response_length
exp_name+=-err$stderr_weight
exp_name+=-p$prob

exp_name+=-cliq$clique_threshold
exp_name+=-opcD${opc_downsample}D${opc_ddownsample}
exp_name+=-maxl$max_num_corrupted_lines
exp_name+=-pout$predoutput_ratio




printf $exp_name
num_threads=8
export OMP_NUM_THREADS=$num_threads 
export MKL_NUM_THREADS=$num_threads  
export NUMEXPR_NUM_THREADS=$num_threads  
export PYTORCH_NUM_THREADS=$num_threads  
python3 -m verl.trainer.main \
    worker.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    use_logical_augment=$use_logical_augment \
    use_digit_augment=$use_digit_augment \
    reward_stderr=$reward_stderr \
    supported_error_types="$supported_error_types" \
    opc_ddownsample=$opc_ddownsample \
    max_num_corrupted_lines=$max_num_corrupted_lines \
    stderr_weight=$stderr_weight \
    prob=$prob \
    dataset_ratio=$dataset_ratio \
    trainer.load_checkpoint_path=$load_checkpoint_path \
    predoutput_ratio=$predoutput_ratio \
    predoutput_length_weight=$predoutput_length_weight \
    n_char_augment=$n_char_augment \
    charset=$charset \
    trainer.total_epochs=$total_epochs \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.val_before_train=false \
    trainer.val_freq=-1  \
    trainer.save_freq=$save_freq  \
    trainer.save_limit=20 \
    data.rollout_batch_size=$rollout_batch_size \
    data.val_batch_size=1024 \
    worker.actor.optim.strategy=$strategy \
    worker.actor.global_batch_size=$global_batch_size \
    worker.actor.micro_batch_size_per_device_for_update=$micro_batch_size_per_device_for_update \
    worker.actor.micro_batch_size_per_device_for_experience=$micro_batch_size_per_device_for_experience \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    worker.reward.reward_function=./examples/reward_function/syntax.py:compute_score \
    config=examples/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    data.train_files=$train_files \
    data.val_files=$train_files \
    worker.rollout.tensor_parallel_size=$tensor_parallel_size 

