pkill -9 -f redis
pkill -9 -f wandb
pkill -9 -f train

# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

proxy()
{
    export HTTP_PROXY=""
    export http_proxy=""
    export https_proxy=""
    export no_proxy=""
}
unproxy()
{
    unset HTTP_PROXY
    unset http_proxy
    unset https_proxy
}

CUR_DEVICE_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader)
CUR_DEVICE_TYPE=$(echo "${CUR_DEVICE_TYPE}" | tr '[:upper:]' '[:lower:]')
if echo "${CUR_DEVICE_TYPE}" | grep -q -e "h100" -e "h800"; then
    echo "Using NVIDIA H-Series GPU setting"
    DEVICE_SERIES="H"
    PKG_SUFFIX=".h800"
else
    echo "Using NVIDIA A-Series GPU setting"
    DEVICE_SERIES="A"
    PKG_SUFFIX=""
fi

set -ex
export HF_DATASETS_OFFLINE=1
# =========================== Mount Configuration ===========================
MOUNT_DIR="/mnt/hdfs/mount_dir"
CODE_DIR="/opt/tiger/olmoe"

# 以下为可修改的几个配置
export run_name="olmoe_1bin7b_sdd_64H100"
CONFIG_PATH=${CODE_DIR}/configs/exps/OLMoE-1B-7B-0906_reproduce.yml
SAVE_DIR="${MOUNT_DIR}/olmoe_exps/${run_name}"
# ===========================  Mount Configuration ===========================

# =========================== Environment ===========================
if [ -d /mnt/bn/mount_nas ]; then
    echo "/mnt/bn/mount_nas exists, skip ..."
else
    sudo mkdir -p /mnt/bn/mount_nas
    sudo ln -s ${MOUNT_DIR}/corpus /mnt/bn/mount_nas/datasets
fi
echo "MOUNT_DIR = ${MOUNT_DIR}"
echo "CODE_DIR = ${CODE_DIR}"


OLMO_ENV_INIT_FLAG_FILE="${CODE_DIR}/OLMO_ENV_INITIALIZED.lock"
if [ -e "${OLMO_ENV_INIT_FLAG_FILE}" ]; then
    echo "olmo env initialized, skip ... (for hotfix)"
else
if echo "${ARNOLD_REGION}" | grep -q -e "CN"; then
    proxy
    echo "Using proxy"
    fi
    # 这两个包的安装顺序需要保证: 先olmoe -> 后zloss
    pip3 install /mnt/bn/mount_nas/datasets/olmoe_related/pkgs/megablocks-0.5.1+olmoe${PKG_SUFFIX}-cp39-cp39-linux_x86_64.whl
    pip3 install /mnt/bn/mount_nas/datasets/olmoe_related/pkgs/megablocks-0.5.1+zloss${PKG_SUFFIX}-cp39-cp39-linux_x86_64.whl
        # pip3 install git+https://github.com/Muennighoff/megablocks.git@4a25bc7b5665bcb9da93d72d5ad0c14d41e1a351 && \
    # pip3 install git+https://github.com/Muennighoff/megablocks.git@e430ad707bed4d45016f315da9372e16acb55a1c
if echo "${ARNOLD_REGION}" | grep -q -e "CN"; then
    unproxy
echo "Unset proxy"
    fi
    mkdir -p ~/.cache
    pushd ~/.cache
    tar --keep-newer-files -xzf /mnt/bn/mount_nas/datasets/olmoe_related/huggingface_cache_v3.tar.gz
    popd
    touch ${OLMO_ENV_INIT_FLAG_FILE}
fi
# =========================== Environment ===========================

# =========================== Script ===========================
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [ -e ${SAVE_DIR}/latest_checkpointed_iteration.txt ]; then
    read -r CUR_STEP < ${SAVE_DIR}/latest_checkpointed_iteration.txt
    echo "current latest_checkpointed_iteration = ${CUR_STEP}"
    if [ -d ${CODE_DIR}/${CUR_STEP} ]; then
        echo "${CODE_DIR}/${CUR_STEP} already exists, skip downloading ..."
    else
        cp -r ${MOUNT_DIR}/olmoe_exps/${run_name}/${CUR_STEP} ${CODE_DIR}/
    fi
    CUR_CKPT_PATH="${CODE_DIR}/${CUR_STEP}"
else
    CUR_CKPT_PATH="auto"
fi

echo "trial_load_path = ${CUR_CKPT_PATH}"

# git stash && git fetch origin ${CUR_BRANCH} && git checkout ${CUR_BRANCH} && git reset --hard ${CUR_COMMIT}

sh launch.sh ${CONFIG_PATH} \
--save_folder=${SAVE_DIR} \
--run_name=${run_name} \
--save_overwrite=true \
--mount_common_hdfs=true \
--fsdp.sharding_strategy=FULL_SHARD \
--canceled_check_interval=9999999 \
--load_path=${CUR_CKPT_PATH} \
--global_indices_file=${CODE_DIR}/global_indices.npy \
--device_train_microbatch_size=1 \
--global_train_batch_size=1024 \
--model.convert2fp32=false \
--save_interval=1000 \
--eval_interval=1000 \
--activation_checkpointing=fine_grained \
--model.init_fn="full_megatron" \
--save_num_checkpoints_to_keep=20 \
--model.init_std=0.0013975424859373685 \
--model.activation_type="swiglu" \
--model.mlp_hidden_size=2048 \
--max_duration=5e11T