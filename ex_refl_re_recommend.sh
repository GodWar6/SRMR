dataset_name='Amazon_Arts'
baseline_name='SRMR_w_reflect'
EP_reflect_name='EP_B1_maxFM_cluster_user_first_t1.0_te_new'
CF_reflect_name='CF_B1_maxFM_cluster_user_first_t1.0_te_new'
IP_reflect_name='IP_B1_maxFM_cluster_user_first_t1.0_te_new'
iter_i=0
arm_size=4
epoch=500
ppo_model_name="run_ppo_lr_a_0.001_lc_c_0.001_rt_0.9_bs_64_4_arm_0_iter_4_train_model_epoch_500.pth"
refl_ver=3
ex="PPO_v1_1_${baseline_name}_vllm_nds_ne_greedy_np_inp_cssp_all_rv_${refl_ver}_iter_${iter_i}_a_${arm_size}_ep_${epoch}_v2_fuse_fix" # remember to change
mkdir -p log
CUDA_VISIBLE_DEVICES=0  \
nohup  python -u refl_re_recommend.py  \
    -dn $dataset_name -ex $ex \
    -bs 12 \
    --vllm \
    --num_prompt \
    --baseline_name $baseline_name \
    --inters_num_prompt \
    --candidate_set_set_prompt \
    --ppo_model_path /root/autodl-tmp/SRMR-Rec/rl_select/Meta-Reflect/${dataset_name}_${ppo_model_name} \
    --refl_result_dir /root/autodl-tmp/SRMR-Rec/iter_pool/${dataset_name}_${EP_reflect_name} \
    --CF_refl_path /root/autodl-tmp/SRMR-Rec/iter_pool/${dataset_name}_${CF_reflect_name} \
    --IP_refl_path /root/autodl-tmp/SRMR-Rec/iter_pool/${dataset_name}_${IP_reflect_name} \
    --EP_refl_path /root/autodl-tmp/SRMR-Rec/iter_pool/${dataset_name}_${EP_reflect_name} \
    --iter_i ${iter_i}\
    --SRMR_w_reflect_version ${refl_ver}\
    -mil 20\
    --waiting_min 0\
    --gpu_memory_utilization 0.95\
    --arm_size ${arm_size}\
    > log/${ex}_${dataset_name}-1-2026-08-08.log 2>&1 &
