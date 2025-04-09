import os
import glob
import re
import shutil
from torchtitan.tools.logging import logger


def cleanup_local_checkpoints(checkpoint_dir: str, keep_latest_k: int):
    """Removes older checkpoint directories locally, keeping only the latest k for both DCP and HF formats."""
    if keep_latest_k <= 0:
        return # Keep all checkpoints

    logger.info(f"Cleaning up local checkpoints in {checkpoint_dir}, keeping latest {keep_latest_k}")

    # Cleanup DCP checkpoints (step-*)
    dcp_checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "step-*")),
        key=lambda x: int(re.search(r"step-(\d+)", os.path.basename(x)).group(1)) if re.search(r"step-(\d+)", os.path.basename(x)) and not x.endswith("-hf") else -1,
        reverse=True
    )
    # Filter out HF format directories
    dcp_checkpoints = [d for d in dcp_checkpoints if not d.endswith("-hf")]

    if len(dcp_checkpoints) > keep_latest_k:
        checkpoints_to_delete = dcp_checkpoints[keep_latest_k:]
        logger.info(f"Deleting {len(checkpoints_to_delete)} old DCP checkpoints: {[os.path.basename(c) for c in checkpoints_to_delete]}")
        for ckpt_path in checkpoints_to_delete:
            if os.path.isdir(ckpt_path): # Ensure it's a directory
                 try:
                     shutil.rmtree(ckpt_path)
                 except OSError as e:
                     logger.error(f"Error removing directory {ckpt_path}: {e}")


    # Cleanup HF checkpoints (step-*-hf)
    hf_checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "step-*-hf")),
         key=lambda x: int(re.search(r"step-(\d+)-hf", os.path.basename(x)).group(1)) if re.search(r"step-(\d+)-hf", os.path.basename(x)) else -1,
        reverse=True
    )

    if len(hf_checkpoints) > keep_latest_k:
        checkpoints_to_delete = hf_checkpoints[keep_latest_k:]
        logger.info(f"Deleting {len(checkpoints_to_delete)} old HF checkpoints: {[os.path.basename(c) for c in checkpoints_to_delete]}")
        for ckpt_path in checkpoints_to_delete:
             if os.path.isdir(ckpt_path): # Ensure it's a directory
                 try:
                     shutil.rmtree(ckpt_path)
                 except OSError as e:
                     logger.error(f"Error removing directory {ckpt_path}: {e}")
