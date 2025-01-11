#!/usr/bin/bash

params=""
if [ $# -ne 0 ]; then
    params="$*"
fi

# use envs as local params for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./train.sh
NGPU=${NGPU:-"8"}
LOG_RANK=${LOG_RANK:-0}

: '
Usage:

bash train.sh -h

Training a 340M model:

bash train.sh \
  --job.config_file train.toml \
  --job.dump_folder exp/transformer-340M-10B/batch32.seqlen2048.warmup1024.update1.steps20480.lr3e-4 \
  --model.config configs/transformer_340M.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.lr 3e-4 \
  --optimizer.scheduler cosine \
  --training.batch_size 32 \
  --training.seq_len 2048 \
  --training.warmup_steps 1024 \
  --training.gradient_accumulation_steps 1 \
  --training.steps 20480 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name default \
  --training.dataset_split train \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --checkpoint.interval 2048 \
  --checkpoint.load_step -1 \
  --metrics.log_freq 32
'

echo "Launching training..."

set -x
path=$(grep -oP '(?<=--job.dump_folder )[^ ]+' <<< "$params")
mkdir -p $path
cp * $path
cp -r configs $path
cp -r flame   $path
cp -r 3rdparty/flash-linear-attention/fla $path
cp -r 3rdparty/torchtitan/torchtitan $path

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
if [ "$date" == "" ]; then
  date=$(date +%Y%m%d%H%M)
fi
export WANDB_RESUME=allow
if [[ -z "${WANDB_NAME}" ]]; then
  export WANDB_NAME="$(basename $path)"
fi
if [[ -z "${WANDB_PROJECT}" ]]; then
  export WANDB_PROJECT="fla"
fi
if [[ -z "${WANDB_RUN_ID}" ]]; then
  export WANDB_RUN_ID="$WANDB_NAME-$date"
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} \
  --rdzv_backend c10d \
  --rdzv_endpoint "localhost:0" \
  --local-ranks-filter ${LOG_RANK} \
  --role rank \
  --tee 3 \
  train.py \
  $params

echo "RUNNING DONE!"