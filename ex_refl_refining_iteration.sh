# ---------- CF ----------
dataset_name='Amazon_Instruments'
ex='CF_B1_maxFM_cluster_t1.0_te_new' # remember to change
CUDA_VISIBLE_DEVICES=7  \
nohup  $PY -u refl_refining_iteration.py  -dn $dataset_name -ex $ex \
    --refl_type 'CF' \
    -bs 12 -bs2 3 \
    --max_iter 7 \
    --fs_do_sample \
    --fs_temperature 1.0 \
    --fusion_mode 'max' \
    --use_user_cluster \
    > log/new_iter_${dataset_name}_${ex}-2026-08-05.log 2>&1 &

# ---------- IP  ----------
dataset_name='Amazon_Instruments'
ex='IP_B1_maxFM_cluster_t1.0_te_new' # remember to change
CUDA_VISIBLE_DEVICES=6  \
nohup  $PY -u refl_refining_iteration.py  -dn $dataset_name -ex $ex \
    --refl_type 'IP' \
    -bs 12 -bs2 3 \
    --max_iter 7 \
    --fs_do_sample \
    --fs_temperature 1.0 \
    --fusion_mode 'max' \
    --use_user_cluster \
    > log/new_iter_${dataset_name}_${ex}-2026-08-05.log 2>&1 &

# ---------- EP  ----------
dataset_name='Amazon_Instruments'
ex='EP_B1_maxFM_cluster_user_first_t1.0_te_new' # remember to change
CUDA_VISIBLE_DEVICES=5  \
nohup  $PY -u refl_refining_iteration.py  -dn $dataset_name -ex $ex \
    --refl_type 'EP' \
    -bs 12 -bs2 4 \
    --max_iter 7 \
    --fs_do_sample \
    --fs_temperature 1.0 \
    --fusion_mode 'max' \
    --use_user_cluster \
    --user_first \
    > log/new_iter_${dataset_name}_${ex}-2026-08-05.log 2>&1 &
