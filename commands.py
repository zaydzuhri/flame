#!/usr/bin/env python3
from pathlib import Path
import itertools

ARCHITECTURES = ["abc", "comba", "delta_net", "forgetting_transformer", "gated_deltanet", "gated_deltaproduct", "gla", "hgrn", "hgrn2", "kda", "lightnet", "mamba", "mamba2", "mesa_net", "mla", "mom", "path_attn", "retnet", "rodimus", "rwkv6", "rwkv7", "samba", "transformer"]
TASKS = ["multi-recall", "fuzzy-recall", "noisy-recall", "full-copy", "reverse-copy", "selective-copy", "memorization", "counting", "sorting", "single-stack-ops", "multi-stack-ops", "flip-flop", "dyck-language", "anbncn-language"]

OUT_FILE = "commands.txt"

TEMPLATE = """WANDB_PROJECT=tasklets NGPU=1 bash train_lr_sweep_best.sh --lrs 1e-4,1e-3,1e-2 --job.config_file flame/models/fla.toml --job.dump_folder exp/{dump_arch}-1M-{task}/128.seqlen1024.context1024.warmup1250.update1.steps12500.cosine --model.config configs/{arch}_1M.json --model.tokenizer_path zaydzuhri/tasklets_tokenizer_256 --optimizer.name AdamW --optimizer.eps 1e-15 --lr_scheduler.warmup_steps 1250 --lr_scheduler.lr_min 0.1 --lr_scheduler.decay_type cosine --training.batch_size 128 --training.seq_len 1024 --training.context_len 1024 --training.gradient_accumulation_steps 1 --training.steps 12500 --training.max_norm 1.0 --training.skip_nan_inf --training.data_backend tasklets --training.dataset /workspace/.cache/zaydzuhri___{task}/default --training.dataset_split train --training.validation_interval 100 --training.validation_dataset_split test --training.enable_early_stopping --training.early_stopping_patience 2 --training.early_stopping_threshold 5e-4 --training.num_workers 4 --training.prefetch_factor 2 --training.seed 42 --training.compile --checkpoint.interval 2000 --checkpoint.load_step -1 --checkpoint.keep_latest_k 2 --metrics.log_freq 5 --eval-task {eval_task} --eval-dataset-name zaydzuhri/{task} --eval-batch-size 16 --eval-device cuda --eval-dtype bfloat16 --eval-results-csv results/{arch}.csv"""

def make_command(arch: str, task: str) -> str:
    return TEMPLATE.format(
        arch=arch,
        dump_arch=arch.replace("_", "-"),
        task=task,
        eval_task=task.replace("-", "_")
    )

def main():
    commands = [
        make_command(arch, task)
        for arch, task in itertools.product(ARCHITECTURES, TASKS)
    ]

    Path(OUT_FILE).write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"Wrote {len(commands)} commands to {OUT_FILE}")

if __name__ == "__main__":
    main()