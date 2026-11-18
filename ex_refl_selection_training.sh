dataset_name="Amazon_Arts"
iter_num=4
gpu_id=0
random_thresh=0.9
batch_size=64
reflect_act_dim=4
lr_a=0.001
lr_c=0.001

valid_sample_path="/root/autodl-tmp/SRMR-Rec/prepare_cluster_user/datasets/Amazon_Arts/new_valid_samples.json"
valid_base_ndcg_path="/root/autodl-tmp/SRMR-Rec/iter_pool/Amazon_Arts_CF_B1_maxFM_cluster_user_first_t1.0_te_new"
CF_refl_path="/root/autodl-tmp/SRMR-Rec/iter_pool/Amazon_Arts_CF_B1_maxFM_cluster_user_first_t1.0_te_new"
EP_refl_path="/root/autodl-tmp/SRMR-Rec/iter_pool/Amazon_Arts_EP_B1_maxFM_cluster_user_first_t1.0_te_new"
IP_refl_path="/root/autodl-tmp/SRMR-Rec/iter_pool/Amazon_Arts_IP_B1_maxFM_cluster_user_first_t1.0_te_new"

ex="run_ppo_lr_a_${lr_a}_lc_c_${lr_c}_rt_${random_thresh}_bs_${batch_size}_${reflect_act_dim}_arm_0_iter_${iter_num}"

mkdir -p rl_select/logs

nohup python -u refl_selection_training.py \
    -dn ${dataset_name} \
    -in ${iter_num} \
    -vsp ${valid_sample_path} \
    -vbnp ${valid_base_ndcg_path} \
    -crp ${CF_refl_path} \
    -erp ${EP_refl_path} \
    -irp ${IP_refl_path} \
    -gi ${gpu_id} \
    -bs ${batch_size} \
    -mbs ${batch_size} \
    --lr_a ${lr_a} \
    --lr_c ${lr_c} \
    -rad ${reflect_act_dim} \
    --savedir ${dataset_name}_${ex}_train_model \
    -rt ${random_thresh} \
> rl_select/logs/${dataset_name}_${ex}-$(date +%Y%m%d).log 2>&1 &
