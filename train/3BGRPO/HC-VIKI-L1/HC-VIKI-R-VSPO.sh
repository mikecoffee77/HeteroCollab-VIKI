#!/bin/bash
# VIKI-R training script with VSPO semantic validation integration (Step 2: Fine-tuning from zero-shot checkpoint)
set -x
ray stop --force 2>/dev/null || true
sleep 2
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_TMPDIR=/root/autodl-tmp/qwen2.5vl/ray_tmp
export RAY_BACKEND_LOG_LEVEL=warning
export RAY_LOG_TO_STDERR=1
export PYTHONUNBUFFERED=1
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_DEDUP_LOGS_AGG_WINDOW_S=5
export RAY_DASHBOARD_HOST=0.0.0.0
export RAY_memory_usage_threshold=0.98
export PYTHONPATH=/opt/conda/envs/roboviki/lib/python3.10/site-packages:/app/verl:$PYTHONPATH
mkdir -p /tmp/ray_tmp

# Step 1 实验目录与 Step 2 输出
ZERO_EXP_NAME='qwen2_5_vl_3b_VIKI_L1_rl_zero_vspo'
ZERO_OUTPUT_DIR="/root/autodl-tmp/qwen2.5vl/saves/${ZERO_EXP_NAME}"
EXP_NAME='qwen2_5_vl_3b_VIKI_L1_rft_vspo'
OUTPUT_DIR="/root/autodl-tmp/qwen2.5vl/saves/${EXP_NAME}"

# 根据是否已有 Step 2 的 checkpoint 决定从哪里恢复：
# - 若 Step 2 目录已有 latest_checkpointed_iteration.txt，则从 Step 2 自己接着训（auto）
# - 否则视为第一次运行，从 Step 1 的最新 global_step_* 恢复（resume_path）
RFT_LATEST_TXT="${OUTPUT_DIR}/latest_checkpointed_iteration.txt"
if [[ -f "${RFT_LATEST_TXT}" ]]; then
  echo "Found existing Step 2 checkpoint at ${RFT_LATEST_TXT}, using resume_mode=auto."
  RESUME_OPTS="trainer.resume_mode=auto"
else
  echo "No Step 2 checkpoint found, will resume from Step 1."
  ZERO_LATEST_TXT="${ZERO_OUTPUT_DIR}/latest_checkpointed_iteration.txt"
  if [[ ! -f "${ZERO_LATEST_TXT}" ]]; then
    echo "Error: Step 1 checkpoint not found. Run VIKI-R-zero-VSPO.sh first and ensure ${ZERO_LATEST_TXT} exists."
    exit 1
  fi
  ZERO_LATEST_STEP=$(cat "${ZERO_LATEST_TXT}")
  RESUME_FROM_PATH="${ZERO_OUTPUT_DIR}/global_step_${ZERO_LATEST_STEP}"
  if [[ ! -d "${RESUME_FROM_PATH}" ]]; then
    echo "Error: Resume path not found: ${RESUME_FROM_PATH}"
    exit 1
  fi
  echo "Resuming first Step 2 run from Step 1: ${RESUME_FROM_PATH}"
  RESUME_OPTS="trainer.resume_mode=resume_path trainer.resume_from_path=${RESUME_FROM_PATH}"
fi

# VSPO Configuration（与 zero 一致）
VSPO_ENABLED=${VSPO_ENABLED:-true}
VSPO_WEIGHT=${VSPO_WEIGHT:-0.1}
VSPO_MODEL_NAME=${VSPO_MODEL_NAME:-all-MiniLM-L6-v2}
VSPO_THRESHOLD=${VSPO_THRESHOLD:-0.7}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/root/autodl-tmp/qwen2.5vl/VIKI-R/viki/VIKI-L1/train.parquet \
    data.val_files=/root/autodl-tmp/qwen2.5vl/VIKI-R/viki/VIKI-L1/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.reward_fn_key=data_source \
    actor_rollout_ref.model.path=/root/autodl-tmp/qwen2.5vl/sft/qwen2.5_vl-3b/full/viki_1_sft_2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.save_freq=100 \
    trainer.test_freq=300 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='VIKI-L1_3b_VSPO' \
    trainer.experiment_name=${EXP_NAME} \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.ray_wait_register_center_timeout=900 \
    ${RESUME_OPTS} \
    trainer.total_epochs=5 \
    +reward_model.reward_kwargs.vspo_enabled=${VSPO_ENABLED} \
    +reward_model.reward_kwargs.vspo_weight=${VSPO_WEIGHT} \
    +reward_model.reward_kwargs.vspo_model_name=${VSPO_MODEL_NAME} \
    +reward_model.reward_kwargs.vspo_threshold=${VSPO_THRESHOLD} \
    $@
