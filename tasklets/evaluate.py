import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import fla

TASK_CHOICES = [
    "single_recall", # works
    "multi_recall", # works
    "fuzzy_recall", # works
    "noisy_recall", # works
    "full_copy", # works but should be generated not teacher-forced
    "reverse_copy", # works but should be generated not teacher-forced
    "reversal", # will remove
    "selective_copy", # works but should be generated not teacher-forced
    "memorization", # works
    "single_stack_ops", # works
    "multi_stack_ops", # works
    "flip_flop", # works
    "dyck_language", # works
    "anbncn_language", # works
    "sorting", # works but should be generated not teacher-forced
    "counting", # works
]

SINGLE_TARGET_TASKS = {
    "single_recall",
    "counting",
    "dyck_language",
    "anbncn_language",
}

SEQUENCE_EXACT_MATCH_TASKS = {
    "full_copy",
    "reverse_copy",
    "reversal",
    "selective_copy",
    "fuzzy_recall",
    "sorting",
}

SPARSE_TOKEN_TASKS = {
    "memorization",
    "single_stack_ops",
    "multi_stack_ops",
    "flip_flop",
}

QUERY_EXACT_MATCH_TASKS = {
    "multi_recall",
    "noisy_recall",
}

DTYPE_MAP = {
    "auto": None,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    label_mask: torch.Tensor
    target_ids: torch.Tensor
    num_labeled_tokens: torch.Tensor


@dataclass
class Metrics:
    num_samples: int = 0
    num_labeled_tokens: int = 0
    num_correct_tokens: int = 0
    num_exact_match: int = 0

    @property
    def token_accuracy(self) -> float:
        if self.num_labeled_tokens == 0:
            return 0.0
        return self.num_correct_tokens / self.num_labeled_tokens

    @property
    def exact_match_accuracy(self) -> float:
        if self.num_samples == 0:
            return 0.0
        return self.num_exact_match / self.num_samples


def normalize_task_name(task: str) -> str:
    return "reverse_copy" if task == "reversal" else task


def infer_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def infer_dtype(dtype_arg: str, device: torch.device) -> Optional[torch.dtype]:
    if dtype_arg != "auto":
        return DTYPE_MAP[dtype_arg]
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def resolve_split_dataset(dataset_name: str, split: str) -> Dataset:
    if os.path.isdir(dataset_name):
        split_file = os.path.join(dataset_name, f"{split}.csv")
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Could not find split file for local dataset: {split_file}"
            )
        return load_dataset("csv", data_files={split: split_file}, split=split)

    if os.path.isfile(dataset_name):
        return load_dataset("csv", data_files={split: dataset_name}, split=split)

    return load_dataset(dataset_name, split=split)


def batch_iter(dataset: Dataset, batch_size: int, max_samples: Optional[int]) -> Iterable[List[Dict[str, str]]]:
    limit = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for start in range(0, limit, batch_size):
        stop = min(start + batch_size, limit)
        yield [dataset[idx] for idx in range(start, stop)]


def tokenize_aligned_batch(batch_rows: Sequence[Dict[str, str]], tokenizer, device: torch.device) -> Batch:
    x_texts = [str(row["x"]) for row in batch_rows]
    y_texts = [str(row["y"]) for row in batch_rows]

    x_encoding = tokenizer(
        x_texts,
        add_special_tokens=False,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    y_encoding = tokenizer(
        y_texts,
        add_special_tokens=False,
        padding=True,
        return_attention_mask=False,
        return_tensors="pt",
    )

    input_ids = x_encoding["input_ids"]
    attention_mask = x_encoding["attention_mask"]
    target_ids = y_encoding["input_ids"]

    raw_label_mask: List[List[bool]] = []
    for row_idx, y_text in enumerate(y_texts):
        raw_tokens = y_text.split()
        if len(raw_tokens) != int(attention_mask[row_idx].sum().item()):
            raise ValueError(
                "Token alignment mismatch between raw y text and tokenized x/y lengths. "
                f"Sample index in batch: {row_idx}, raw y tokens={len(raw_tokens)}, "
                f"tokenized length={int(attention_mask[row_idx].sum().item())}."
            )
        raw_label_mask.append([token != "_" for token in raw_tokens])

    max_len = input_ids.size(1)
    label_mask = torch.zeros((len(batch_rows), max_len), dtype=torch.bool)
    for row_idx, mask in enumerate(raw_label_mask):
        label_mask[row_idx, : len(mask)] = torch.tensor(mask, dtype=torch.bool)

    # Causal LM logits at position t predict token t+1, so labels must be shifted.
    shifted_input_ids = input_ids[:, 1:]
    shifted_label_mask = label_mask[:, 1:]
    shifted_attention_mask = attention_mask[:, :-1]
    shifted_target_ids = target_ids[:, 1:]
    num_labeled_tokens = shifted_label_mask.sum(dim=1)

    return Batch(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        label_mask=shifted_label_mask.to(device),
        target_ids=shifted_target_ids.to(device),
        num_labeled_tokens=num_labeled_tokens.to(device),
    )


@torch.inference_mode()
def evaluate_dataset(
    model,
    tokenizer,
    dataset: Dataset,
    task: str,
    batch_size: int,
    max_samples: Optional[int],
    device: torch.device,
) -> Tuple[Metrics, Dict[str, str]]:
    metrics = Metrics()
    task = normalize_task_name(task)

    progress_total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    progress = tqdm(total=progress_total, desc=f"Evaluating {task}", unit="sample")

    for rows in batch_iter(dataset, batch_size=batch_size, max_samples=max_samples):
        batch = tokenize_aligned_batch(rows, tokenizer=tokenizer, device=device)

        outputs = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        logits = outputs.logits[:, :-1, :]
        predictions = logits.argmax(dim=-1)

        correct_mask = (predictions == batch.target_ids) & batch.label_mask
        correct_per_sample = correct_mask.sum(dim=1)
        exact_per_sample = correct_per_sample == batch.num_labeled_tokens

        metrics.num_samples += len(rows)
        metrics.num_labeled_tokens += int(batch.num_labeled_tokens.sum().item())
        metrics.num_correct_tokens += int(correct_mask.sum().item())
        metrics.num_exact_match += int(exact_per_sample.sum().item())

        progress.update(len(rows))

    progress.close()

    primary_metric_name = primary_metric_for_task(task)
    summary = {
        "task": task,
        "primary_metric": primary_metric_name,
        "primary_accuracy": format_metric(primary_metric_value(metrics, task)),
        "token_accuracy": format_metric(metrics.token_accuracy),
        "exact_match_accuracy": format_metric(metrics.exact_match_accuracy),
    }
    return metrics, summary


def primary_metric_for_task(task: str) -> str:
    if task in SINGLE_TARGET_TASKS:
        return "accuracy"
    if task in SEQUENCE_EXACT_MATCH_TASKS:
        return "sequence_exact_match"
    if task in QUERY_EXACT_MATCH_TASKS:
        return "query_exact_match"
    if task in SPARSE_TOKEN_TASKS:
        return "labeled_token_accuracy"
    return "exact_match_accuracy"


def primary_metric_value(metrics: Metrics, task: str) -> float:
    if task in SINGLE_TARGET_TASKS:
        return metrics.token_accuracy
    if task in SEQUENCE_EXACT_MATCH_TASKS:
        return metrics.exact_match_accuracy
    if task in QUERY_EXACT_MATCH_TASKS:
        return metrics.exact_match_accuracy
    if task in SPARSE_TOKEN_TASKS:
        return metrics.token_accuracy
    return metrics.exact_match_accuracy


def format_metric(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def append_results_csv(
    csv_path: str,
    args: argparse.Namespace,
    metrics: Metrics,
    summary: Dict[str, str],
) -> None:
    parent_dir = os.path.dirname(os.path.abspath(csv_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    fieldnames = [
        "model_name",
        "task",
        "dataset_name",
        "split",
        "batch_size",
        "max_samples",
        "device",
        "dtype",
        "revision",
        "num_samples",
        "num_labeled_tokens",
        "primary_metric",
        "primary_accuracy",
        "token_accuracy",
        "exact_match_accuracy",
    ]
    row = {
        "model_name": args.model_name,
        "task": summary["task"],
        "dataset_name": args.dataset_name,
        "split": args.split,
        "batch_size": args.batch_size,
        "max_samples": "" if args.max_samples is None else args.max_samples,
        "device": str(args.device),
        "dtype": args.dtype,
        "revision": "" if args.revision is None else args.revision,
        "num_samples": metrics.num_samples,
        "num_labeled_tokens": metrics.num_labeled_tokens,
        "primary_metric": summary["primary_metric"],
        "primary_accuracy": summary["primary_accuracy"],
        "token_accuracy": summary["token_accuracy"],
        "exact_match_accuracy": summary["exact_match_accuracy"],
    }

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a Hugging Face causal LM on a tasklets dataset split."
    )
    parser.add_argument("--model-name", type=str, required=True,
                        help="Model name or local path loadable via AutoModelForCausalLM.")
    parser.add_argument("--task", type=str, required=True, choices=TASK_CHOICES,
                        help="Task name matching the conventions in tasklets/data_generator.py.")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Dataset repo id or local dataset path. Local directories should contain split CSVs.")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to evaluate. Defaults to test.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of examples per forward pass.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional cap on the number of evaluated samples.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--dtype", type=str, default="auto", choices=list(DTYPE_MAP.keys()),
                        help="Torch dtype for model loading.")
    parser.add_argument("--revision", type=str, default=None,
                        help="Optional model or dataset revision.")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Pass trust_remote_code=True to Hugging Face loaders.")
    parser.add_argument("--results-csv", type=str, default=None,
                        help="Optional CSV file path for saving one evaluation result row per run.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    task = normalize_task_name(args.task)
    device = infer_device(args.device)
    dtype = infer_dtype(args.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        # revision=args.revision,
        # trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    dataset = resolve_split_dataset(args.dataset_name, split=args.split)
    if not isinstance(dataset, Dataset):
        raise TypeError(
            "Expected a map-style datasets.Dataset for evaluation. "
            "If you are loading a dataset dict, pass a concrete split with --split."
        )

    missing_columns = {"x", "y"} - set(dataset.column_names)
    if missing_columns:
        raise ValueError(
            f"Dataset split '{args.split}' is missing required columns: {sorted(missing_columns)}"
        )

    metrics, summary = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        task=task,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=device,
    )

    print(f"task: {summary['task']}")
    print(f"split: {args.split}")
    print(f"samples: {metrics.num_samples}")
    print(f"labeled_tokens: {metrics.num_labeled_tokens}")
    print(f"primary_metric: {summary['primary_metric']}")
    print(f"primary_accuracy: {summary['primary_accuracy']}")
    print(f"token_accuracy: {summary['token_accuracy']}")
    print(f"exact_match_accuracy: {summary['exact_match_accuracy']}")

    if args.results_csv is not None:
        append_results_csv(
            csv_path=args.results_csv,
            args=args,
            metrics=metrics,
            summary=summary,
        )
        print(f"results_csv: {args.results_csv}")


if __name__ == "__main__":
    main()
