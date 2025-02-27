pkill -9 -f redis-server
pkill -9 -f wandb-service
pkill -9 -f train

AREA="i18n"
if echo "${ARNOLD_REGION}" | grep -q -e "CN"; then
    AREA="cn"
fi
echo "current area = ${AREA}"

set -ex
export HF_DATASETS_OFFLINE=1
# =========================== 重要配置 ===========================
MOUNT_DIR="/mnt/hdfs/mount_dir"
CODE_DIR="/opt/tiger/olmo"

# 以下为可修改的几个配置, run_name: 任务名, DEBUG_FLAG: 是否为debug环境(0 or 1, 防止hdfs上产生太多无效目录)
export run_name="dense_baseline_1B2_sdd_64H100"
DEBUG_FLAG=0
CONFIG_PATH=${CODE_DIR}/configs/exps/LLaMA-3.2-1B-like-stage1.yaml
# =========================== 重要配置 ===========================

# =========================== 环境初始化(无需改动) ===========================
if [ "$DEBUG_FLAG" = "1" ]; then
    mkdir -p ${CODE_DIR}/olmo2_exps
    SAVE_DIR="${CODE_DIR}/olmo2_exps/${run_name}"
else
    SAVE_DIR="${MOUNT_DIR}/olmo2_exps/${run_name}"
fi

if [ -d /mnt/bn/mount_nas ]; then
    echo "/mnt/bn/mount_nas exists, skip ..."
else
    sudo mkdir -p /mnt/bn/mount_nas
    sudo ln -s ${MOUNT_DIR}/corpus /mnt/bn/mount_nas/datasets
fi

if [ "$AREA" = "cn" ]; then
    REMOTE_HDFS_DIR="hdfs://haruna/home/byte_data_seed/ssd_hldy/user/yourname"
else
    REMOTE_HDFS_DIR="hdfs://harunava/home/byte_arnold_va_mlsys/user/yourname"
fi

echo "MOUNT_DIR = ${MOUNT_DIR}"
echo "CODE_DIR = ${CODE_DIR}"
echo "REMOTE_HDFS_DIR = ${REMOTE_HDFS_DIR}"
# =========================== 环境初始化(无需改动) ===========================

# =========================== 运行脚本 ===========================
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 以下为自动resume的配置脚本, 如果需要中途修改load_path, 需要去修改hdfs目录下的latest_checkpointed_iteration.txt文件
if [ -e ${SAVE_DIR}/latest_checkpointed_iteration.txt ]; then
    read -r CUR_STEP < ${SAVE_DIR}/latest_checkpointed_iteration.txt
    echo "current latest_checkpointed_iteration = ${CUR_STEP}"
    if [ -d ${CODE_DIR}/${CUR_STEP} ]; then
        echo "${CODE_DIR}/${CUR_STEP} already exists, skip downloading ..."
    else
        # cp -r ${MOUNT_DIR}/olmo2_exps/${run_name}/${CUR_STEP} ${CODE_DIR}/
        hdfs dfs -get -t 16 ${REMOTE_HDFS_DIR}/olmo2_exps/${run_name}/${CUR_STEP} ${CODE_DIR}/
    fi
    CUR_CKPT_PATH="${CODE_DIR}/${CUR_STEP}"
else
    CUR_CKPT_PATH="auto"
fi

echo "trial_load_path = ${CUR_CKPT_PATH}"

# 以下为hotfix常用配置
# CUR_BRANCH="olmoe"
# CUR_COMMIT="22ce4b63b3c4cafdea0a84327507f6fe6beb3a36"
# git stash && git fetch origin ${CUR_BRANCH} && git checkout ${CUR_BRANCH} && git reset --hard ${CUR_COMMIT}

sh launch.sh ${CONFIG_PATH} \
--save_folder=${SAVE_DIR} \
--run_name=${run_name} \
--save_overwrite=true \
--mount_common_hdfs=true \
--fsdp.sharding_strategy=FULL_SHARD \
--fsdp.wrapping_strategy=null \
--canceled_check_interval=9999999 \
--global_indices_file=${CODE_DIR}/global_indices.npy \
--load_path=${CUR_CKPT_PATH} \
--model.weight_tying=true \
--model.d_model=2048 \
--model.n_heads=32 \
--model.n_kv_heads=8 \
--model.n_layers=16 \
--model.mlp_hidden_size=16384 \
--model.attention_layer_norm=true \
--model.attention_layer_norm_with_affine=true \
--model.convert2fp32=false \
--model.init_std=0.0013975424859373685 \
--model.init_fn="full_megatron" \
--max_duration=2e12T \
--scheduler.t_warmup=8388608000 \
--scheduler.t_max=2e12 \
--global_train_batch_size=1024 \
--device_train_microbatch_size=4 \
--save_interval=2000 \
--eval_interval=1000 \
--save_num_checkpoints_to_keep=-1

# 如果显存开销太大, 可以设置--activation_checkpointing=fine_grained