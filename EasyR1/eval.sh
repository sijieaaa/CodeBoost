

# Example for evaluation
#  <global_step>  <exp_name>  <lmstyle>
cd $cb_dir/EasyR1;  conda activate easyr1;  export CUDA_VISIBLE_DEVICES=0
bash ./eval_func_lcb_bcb_plus.sh  g_step_15   Qwen2.5-Coder-7B-Instruct-UseLogicSyntaxError-nv4-g128-ro512-line10-len30-err0.5-p0.1-cliq0.5-opcD2D2-maxl1-pout0.75   CodeQwenInstruct  
bash ./eval_func_lcb_bcb_plus.sh  g_step_30   Qwen2.5-Coder-7B-Instruct-UseLogicSyntaxError-nv4-g128-ro512-line10-len30-err0.5-p0.1-cliq0.5-opcD2D2-maxl1-pout0.75   CodeQwenInstruct  

