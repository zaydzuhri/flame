#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfFolder, snapshot_download


def resolve_subdir(
    step: int | None, subdir: str | None, checkpoint_subdir: str | None
) -> tuple[str | None, str | None]:
    if step is not None and subdir is not None:
        raise ValueError("Provide only one of --step or --subdir.")
    if subdir is not None:
        local_subdir = subdir
    elif step is not None:
        local_subdir = f"step-{step}"
    else:
        local_subdir = None

    if checkpoint_subdir is None:
        return local_subdir, local_subdir
    if local_subdir is None:
        return checkpoint_subdir, None
    return f"{checkpoint_subdir.rstrip('/')}/{local_subdir}", local_subdir


def copy_contents(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def relocate_subdir(checkpoint_dir: Path, remote_subdir: str, local_subdir: str | None) -> None:
    source_dir = checkpoint_dir / remote_subdir
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Remote subdir not found in snapshot: {remote_subdir}"
        )
    dest_dir = checkpoint_dir if local_subdir is None else checkpoint_dir / local_subdir
    if source_dir.is_dir():
        shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
        shutil.rmtree(source_dir)
    else:
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir, dest_dir)
        source_dir.unlink()
    for parent in source_dir.parents:
        if parent == checkpoint_dir:
            break
        try:
            parent.rmdir()
        except OSError:
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a DCP checkpoint folder from Hugging Face Hub."
    )
    parser.add_argument("--repo-id", required=True, help="HF repo id.")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "Local run folder for the downloaded checkpoint data "
            "(defaults to output-root/checkpoint/step-<n>)."
        ),
    )
    parser.add_argument("--step", type=int, help="Step number to download.")
    parser.add_argument(
        "--subdir",
        type=str,
        help="Explicit subdirectory in the repo (e.g. step-10000).",
    )
    parser.add_argument(
        "--checkpoint-subdir",
        type=str,
        help=(
            "Optional repo subdirectory that contains the DCP step folder "
            "(e.g. checkpoint or runs/run-A/checkpoint)."
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Git revision, branch, or tag.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Extra allow patterns (repeatable) for snapshot_download.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF token override (defaults to cached token).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token or HfFolder.get_token()

    checkpoint_dir = args.output_root / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    remote_subdir, local_subdir = resolve_subdir(
        args.step, args.subdir, args.checkpoint_subdir
    )
    allow_patterns = None
    if remote_subdir is not None:
        allow_patterns = [f"{remote_subdir}/*"]
    if args.allow_pattern:
        if allow_patterns is None:
            allow_patterns = list(args.allow_pattern)
        else:
            allow_patterns.extend(args.allow_pattern)

    if args.checkpoint_subdir is None:
        snapshot_download(
            repo_id=args.repo_id,
            token=token,
            local_dir=checkpoint_dir,
            revision=args.revision,
            allow_patterns=allow_patterns,
        )
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            snapshot_download(
                repo_id=args.repo_id,
                token=token,
                local_dir=tmp_path,
                revision=args.revision,
                allow_patterns=allow_patterns,
            )

            copy_contents(tmp_path, checkpoint_dir)
            if remote_subdir is not None:
                relocate_subdir(checkpoint_dir, remote_subdir, local_subdir)

    if local_subdir is not None:
        print(f"Downloaded to {checkpoint_dir / local_subdir}")
    else:
        print(f"Downloaded repository snapshot to {checkpoint_dir}")


if __name__ == "__main__":
    main()
