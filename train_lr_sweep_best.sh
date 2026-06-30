#!/usr/bin/bash

set -u

usage() {
  cat <<'EOF'
Usage:
  NNODE=1 NGPU=8 LOG_RANK=0 bash train_lr_sweep_best.sh \
    --lrs 1e-4,3e-4,1e-3 \
    --job.config_file flame/models/fla.toml \
    --job.dump_folder exp/transformer-1M-multi-stack-ops/128.seqlen1024.context1024.warmup1250.update1.steps12500.lr1e-3.cosine.vocab256 \
    --model.config configs/transformer_1M.json \
    --model.tokenizer_path tasklets/tasklets_tokenizer_256 \
    --eval-split test \
    --eval-batch-size 8 \
    --eval-device auto \
    --eval-dtype auto \
    ... other training args ...

Notes:
- The script runs one training job per LR in --lrs.
- It reads each run's wandb summary and selects the run with the lowest validation/loss.
- It converts only the best run to HF format.
- Then it runs tasklets/evaluate.py, and you can override evaluate args with:
  --eval-task, --eval-dataset-name, --eval-split, --eval-batch-size,
  --eval-max-samples, --eval-device, --eval-dtype, --eval-revision,
  --eval-trust-remote-code, --eval-results-csv.
EOF
}

extract_arg_value() {
  local key="$1"
  shift
  local prev=""
  for tok in "$@"; do
    if [[ "$prev" == "$key" ]]; then
      echo "$tok"
      return 0
    fi
    prev="$tok"
  done
  return 1
}

replace_or_append_lr_path() {
  local base_path="$1"
  local lr="$2"
  python - "$base_path" "$lr" <<'PY'
import re
import sys
base_path = sys.argv[1]
lr = sys.argv[2]
out = re.sub(r"lr[0-9eE.+-]+", f"lr{lr}", base_path, count=1)
if out == base_path:
    out = f"{base_path}.lr{lr}"
print(out)
PY
}

infer_task_from_exp_group() {
  local exp_group="$1"
  python - "$exp_group" <<'PY'
import sys
exp_group = sys.argv[1]
known = [
    "single-recall",
    "multi-recall",
    "fuzzy-recall",
    "noisy-recall",
    "full-copy",
    "reverse-copy",
    "selective-copy",
    "memorization",
    "single-stack-ops",
    "multi-stack-ops",
    "flip-flop",
    "dyck-language",
    "anbncn-language",
    "sorting",
    "counting",
]
for task in known:
    if exp_group == task or exp_group.endswith("-" + task):
        print(task.replace("-", "_"))
        raise SystemExit(0)
print("")
PY
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

LR_LIST=""
TRAIN_ARGS=()
EVAL_TASK=""
EVAL_DATASET_NAME=""
EVAL_SPLIT="test"
EVAL_BATCH_SIZE="8"
EVAL_MAX_SAMPLES=""
EVAL_DEVICE="auto"
EVAL_DTYPE="auto"
EVAL_REVISION=""
EVAL_TRUST_REMOTE_CODE="0"
EVAL_RESULTS_CSV=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lrs)
      LR_LIST="$2"
      shift 2
      ;;
    --lrs=*)
      LR_LIST="${1#*=}"
      shift
      ;;
    --eval-task)
      EVAL_TASK="$2"
      shift 2
      ;;
    --eval-task=*)
      EVAL_TASK="${1#*=}"
      shift
      ;;
    --eval-dataset-name)
      EVAL_DATASET_NAME="$2"
      shift 2
      ;;
    --eval-dataset-name=*)
      EVAL_DATASET_NAME="${1#*=}"
      shift
      ;;
    --eval-split)
      EVAL_SPLIT="$2"
      shift 2
      ;;
    --eval-split=*)
      EVAL_SPLIT="${1#*=}"
      shift
      ;;
    --eval-batch-size)
      EVAL_BATCH_SIZE="$2"
      shift 2
      ;;
    --eval-batch-size=*)
      EVAL_BATCH_SIZE="${1#*=}"
      shift
      ;;
    --eval-max-samples)
      EVAL_MAX_SAMPLES="$2"
      shift 2
      ;;
    --eval-max-samples=*)
      EVAL_MAX_SAMPLES="${1#*=}"
      shift
      ;;
    --eval-device)
      EVAL_DEVICE="$2"
      shift 2
      ;;
    --eval-device=*)
      EVAL_DEVICE="${1#*=}"
      shift
      ;;
    --eval-dtype)
      EVAL_DTYPE="$2"
      shift 2
      ;;
    --eval-dtype=*)
      EVAL_DTYPE="${1#*=}"
      shift
      ;;
    --eval-revision)
      EVAL_REVISION="$2"
      shift 2
      ;;
    --eval-revision=*)
      EVAL_REVISION="${1#*=}"
      shift
      ;;
    --eval-trust-remote-code)
      EVAL_TRUST_REMOTE_CODE="1"
      shift
      ;;
    --eval-results-csv)
      EVAL_RESULTS_CSV="$2"
      shift 2
      ;;
    --eval-results-csv=*)
      EVAL_RESULTS_CSV="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      TRAIN_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$LR_LIST" ]]; then
  echo "Missing required --lrs argument"
  usage
  exit 1
fi

IFS=',' read -r -a LRS <<< "$LR_LIST"
if [[ ${#LRS[@]} -eq 0 ]]; then
  echo "No learning rates parsed from --lrs=$LR_LIST"
  exit 1
fi

BASE_PATH=$(extract_arg_value --job.dump_folder "${TRAIN_ARGS[@]}")
CONFIG=$(extract_arg_value --model.config "${TRAIN_ARGS[@]}")
TOKENIZER=$(extract_arg_value --model.tokenizer_path "${TRAIN_ARGS[@]}")

# train.sh parses args from a flat string and expects a single --job.dump_folder
# and --optimizer.lr. Remove any existing values and inject per-run overrides later.
SANITIZED_TRAIN_ARGS=()
skip_next="0"
for tok in "${TRAIN_ARGS[@]}"; do
  if [[ "$skip_next" == "1" ]]; then
    skip_next="0"
    continue
  fi

  case "$tok" in
    --job.dump_folder|--optimizer.lr)
      skip_next="1"
      ;;
    --job.dump_folder=*|--optimizer.lr=*)
      ;;
    *)
      SANITIZED_TRAIN_ARGS+=("$tok")
      ;;
  esac
done

if [[ -z "${BASE_PATH:-}" ]]; then
  echo "Missing required --job.dump_folder in training args"
  exit 1
fi
if [[ -z "${CONFIG:-}" ]]; then
  echo "Missing required --model.config in training args"
  exit 1
fi
if [[ -z "${TOKENIZER:-}" ]]; then
  echo "Missing required --model.tokenizer_path in training args"
  exit 1
fi

NNODE=${NNODE:-"1"}
NGPU=${NGPU:-"8"}
LOG_RANK=${LOG_RANK:-0}

BEST_VAL=""
BEST_LR=""
BEST_PATH=""

for lr in "${LRS[@]}"; do
  lr_trimmed=$(echo "$lr" | xargs)
  if [[ -z "$lr_trimmed" ]]; then
    continue
  fi

  run_path=$(replace_or_append_lr_path "$BASE_PATH" "$lr_trimmed")
  run_stamp="$(date +%Y%m%d%H%M%S)-lr${lr_trimmed//./p}"

  echo "============================================================"
  echo "Starting run for lr=$lr_trimmed"
  echo "Run folder: $run_path"

  export date="$run_stamp"
  bash train.sh "${SANITIZED_TRAIN_ARGS[@]}" --optimizer.lr "$lr_trimmed" --job.dump_folder "$run_path"
  train_rc=$?

  if [[ $train_rc -ne 0 ]]; then
    echo "Training failed for lr=$lr_trimmed (exit code: $train_rc). Skipping this lr."
    continue
  fi

  summary_path=$(find "$run_path"/tb -type f -path "*/wandb/latest-run/files/wandb-summary.json" 2>/dev/null | sort | tail -n 1)
  if [[ -z "$summary_path" ]]; then
    summary_path=$(find "$run_path"/tb -type f -path "*/wandb/run-*/files/wandb-summary.json" 2>/dev/null | sort | tail -n 1)
  fi

  if [[ -z "$summary_path" ]]; then
    echo "Could not find wandb-summary.json for lr=$lr_trimmed under $run_path/tb"
    continue
  fi

  val_loss=$(python - "$summary_path" <<'PY'
import json
import math
import sys
p = sys.argv[1]
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)
# Preferred metric key from training code.
if "validation/loss" in d:
    v = d["validation/loss"]
    print(v if isinstance(v, (int, float)) else "nan")
    raise SystemExit(0)
# Fallback: any key that contains both validation and loss.
cands = []
for k, v in d.items():
    lk = k.lower()
    if "validation" in lk and "loss" in lk and isinstance(v, (int, float)):
        cands.append((k, float(v)))
if cands:
    cands.sort(key=lambda x: x[1])
    print(cands[0][1])
else:
    print("nan")
PY
)

  if [[ -z "$val_loss" || "$val_loss" == "nan" ]]; then
    echo "No numeric validation loss found in $summary_path for lr=$lr_trimmed"
    continue
  fi

  echo "lr=$lr_trimmed validation/loss=$val_loss"

  if [[ -z "$BEST_VAL" ]]; then
    BEST_VAL="$val_loss"
    BEST_LR="$lr_trimmed"
    BEST_PATH="$run_path"
  else
    is_better=$(python - "$val_loss" "$BEST_VAL" <<'PY'
import sys
curr = float(sys.argv[1])
best = float(sys.argv[2])
print("1" if curr < best else "0")
PY
)
    if [[ "$is_better" == "1" ]]; then
      BEST_VAL="$val_loss"
      BEST_LR="$lr_trimmed"
      BEST_PATH="$run_path"
    fi
  fi

done

if [[ -z "$BEST_PATH" ]]; then
  echo "No successful run with a valid validation/loss metric was found."
  exit 1
fi

echo "============================================================"
echo "Best run selected:"
echo "  lr=$BEST_LR"
echo "  validation/loss=$BEST_VAL"
echo "  path=$BEST_PATH"

echo "Converting best run checkpoint to HF format..."
python -m flame.utils.convert_dcp_to_hf \
  --path "$BEST_PATH" \
  --step -1 \
  --config "$CONFIG" \
  --tokenizer "$TOKENIZER"
convert_rc=$?
if [[ $convert_rc -ne 0 ]]; then
  echo "Conversion failed for best run (exit code: $convert_rc)"
  exit $convert_rc
fi

exp_group=$(basename "$(dirname "$BEST_PATH")")
task="$EVAL_TASK"
if [[ -z "$task" ]]; then
  task=$(infer_task_from_exp_group "$exp_group")
fi

if [[ -z "$task" ]]; then
  echo "Could not infer task from experiment group '$exp_group'."
  echo "Skipping evaluation. Run manually with:"
  echo "python tasklets/evaluate.py --model-name '$BEST_PATH' --task <task_name> --dataset-name <dataset_or_path>"
  exit 1
fi

dataset_path="$EVAL_DATASET_NAME"
if [[ -z "$dataset_path" ]]; then
  dataset_path="tasklets/data/$task"
fi
if [[ ! -d "$dataset_path" && ! -f "$dataset_path" ]]; then
  echo "Dataset path does not exist: $dataset_path"
  echo "Set --eval-dataset-name to a valid local path or dataset id."
  exit 1
fi

eval_cmd=(
  python tasklets/evaluate.py
  --model-name "$BEST_PATH"
  --task "$task"
  --dataset-name "$dataset_path"
  --split "$EVAL_SPLIT"
  --batch-size "$EVAL_BATCH_SIZE"
  --device "$EVAL_DEVICE"
  --dtype "$EVAL_DTYPE"
)
if [[ -n "$EVAL_MAX_SAMPLES" ]]; then
  eval_cmd+=(--max-samples "$EVAL_MAX_SAMPLES")
fi
if [[ -n "$EVAL_REVISION" ]]; then
  eval_cmd+=(--revision "$EVAL_REVISION")
fi
if [[ "$EVAL_TRUST_REMOTE_CODE" == "1" ]]; then
  eval_cmd+=(--trust-remote-code)
fi
if [[ -n "$EVAL_RESULTS_CSV" ]]; then
  eval_cmd+=(--results-csv "$EVAL_RESULTS_CSV")
fi

echo "Running evaluation..."
echo "  task=$task"
echo "  dataset=$dataset_path"
echo "  split=$EVAL_SPLIT batch_size=$EVAL_BATCH_SIZE device=$EVAL_DEVICE dtype=$EVAL_DTYPE"
"${eval_cmd[@]}"

echo "ALL DONE!"
