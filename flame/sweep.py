import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from copy import deepcopy

import wandb

from flame.train import init_logger


def _extract_arg_value(args: list[str], key: str) -> str | None:
    flag = f"--{key}"
    for idx, arg in enumerate(args):
        if arg == flag:
            return args[idx + 1] if idx + 1 < len(args) else None
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


def _parse_wrapper_args(argv: list[str]) -> tuple[str, list[str]]:
    remaining_args: list[str] = []
    sweep_config_path: str | None = None
    idx = 0

    while idx < len(argv):
        arg = argv[idx]
        if arg == "--sweep-config":
            idx += 1
            if idx >= len(argv):
                raise ValueError("--sweep-config requires a path")
            sweep_config_path = argv[idx]
        elif arg.startswith("--sweep-config="):
            sweep_config_path = arg.split("=", 1)[1]
        else:
            remaining_args.append(arg)
        idx += 1

    if not sweep_config_path:
        raise ValueError("Missing required --sweep-config argument")

    return sweep_config_path, remaining_args


def _load_sweep_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Sweep config JSON must contain a top-level object")

    return payload


def _flatten_mapping(mapping: Mapping, prefix: str = "") -> dict[str, object]:
    flattened: dict[str, object] = {}
    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_mapping(value, prefix=full_key))
        else:
            flattened[full_key] = value
    return flattened


def _format_cli_override(key: str, value: object) -> list[str]:
    if "." not in key:
        raise ValueError(
            f"Sweep parameter '{key}' must use the same dotted format as train.py args"
        )

    if isinstance(value, bool):
        return [f"--{key}"] if value else []

    if isinstance(value, list):
        return [f"--{key}", ",".join(str(item) for item in value)]

    return [f"--{key}", str(value)]


def _resolve_torchrun() -> str:
    configured = os.environ.get("FLAME_TORCHRUN")
    if configured:
        return configured

    python_path = os.environ.get("FLAME_PYTHON")
    if python_path:
        candidate = os.path.join(os.path.dirname(python_path), "torchrun")
        if os.path.exists(candidate):
            return candidate

    discovered = shutil.which("torchrun")
    if discovered:
        return discovered

    raise RuntimeError("Could not find torchrun. Set FLAME_TORCHRUN or FLAME_PYTHON.")


def _build_torchrun_command(base_train_args: list[str], sweep_overrides: Mapping[str, object]) -> list[str]:
    nnodes = os.environ.get("NNODE", "1")
    ngpu = os.environ.get("NGPU", "1")
    log_rank = os.environ.get("LOG_RANK", "0")
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "0")

    dump_folder = _extract_arg_value(base_train_args, "job.dump_folder")
    command = [
        _resolve_torchrun(),
        f"--nnodes={nnodes}",
        f"--nproc_per_node={ngpu}",
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint",
        f"{master_addr}:{master_port}",
        "--local-ranks-filter",
        str(log_rank),
        "--role",
        "rank",
        "--tee",
        "3",
    ]
    if dump_folder:
        command.extend(["--log-dir", os.path.join(dump_folder, "logs")])
    command.extend(["-m", "flame.train", *base_train_args])

    for key, value in sweep_overrides.items():
        command.extend(_format_cli_override(key, value))

    if "--metrics.enable_wandb" not in base_train_args:
        command.append("--metrics.enable_wandb")

    if "--checkpoint.disable_checkpoint" not in base_train_args:
        command.append("--checkpoint.disable_checkpoint")

    return command


def _build_trial_runner(base_train_args: list[str]):
    base_train_args = list(base_train_args)

    def _run_trial():
        run = wandb.init()
        sweep_overrides = _flatten_mapping(wandb.config.as_dict())
        command = _build_torchrun_command(base_train_args, sweep_overrides)

        child_env = os.environ.copy()
        child_env.setdefault("WANDB_RESUME", "allow")
        child_env["WANDB_RUN_ID"] = run.id
        if run.name:
            child_env["WANDB_NAME"] = run.name
        sweep_id = getattr(run, "sweep_id", None) or os.environ.get("WANDB_SWEEP_ID")
        if sweep_id:
            child_env["WANDB_SWEEP_ID"] = sweep_id

        # Hand the active sweep run identity to the training subprocess.
        wandb.finish()
        subprocess.run(command, check=True, env=child_env)

    return _run_trial


def main() -> None:
    init_logger()
    sweep_config_path, base_train_args = _parse_wrapper_args(sys.argv[1:])
    sweep_config = _load_sweep_config(sweep_config_path)

    sweep_id = wandb.sweep(sweep=deepcopy(sweep_config), project="fla-sweeps")
    wandb.agent(sweep_id, function=_build_trial_runner(base_train_args))


if __name__ == "__main__":
    main()
