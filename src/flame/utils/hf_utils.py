import os
import re
from huggingface_hub import HfApi, HfFolder, logging as hf_logging, create_repo
from torchtitan.tools.logging import logger 

def upload_checkpoint_to_hf(
    local_path: str,         
    step: int,
    hf_repo_id_for_run: str,
    hf_keep_latest_k: int,
    upload_format: str        
):
    """Uploads a checkpoint directory to HF Hub and manages retention."""
    if not os.path.isdir(local_path):
        logger.error(f"Local path for upload does not exist or is not a directory: {local_path}")
        return

    api = HfApi()
    token = HfFolder.get_token() 
    if not token:
        logger.warning("Hugging Face Hub token not found. Skipping upload. Login via `huggingface-cli login` or set HF_TOKEN.")
        return

    # --- Ensure the specific repository for this run exists ---
    try:
        logger.info(f"Ensuring repository {hf_repo_id_for_run} exists...")
        # Use create_repo which handles creation only if it doesn't exist
        create_repo(repo_id=hf_repo_id_for_run, token=token, repo_type="model", exist_ok=True)
        logger.info(f"Repository {hf_repo_id_for_run} ensured.")
    except Exception as e:
        logger.error(f"Failed to create or ensure repository {hf_repo_id_for_run}: {e}", exc_info=True)
        return # Stop if repo interaction fails

    commit_message = f"Upload {upload_format.upper()} checkpoint step {step}"
    path_in_repo = f"step-{step}"

    logger.info(f"Uploading {local_path} to {hf_repo_id_for_run}/{path_in_repo} on Hugging Face Hub...")
    try:
        api.upload_folder(
            folder_path=local_path,
            path_in_repo=path_in_repo,
            repo_id=hf_repo_id_for_run,
            repo_type="model", 
            commit_message=commit_message,
            token=token,
        )
        logger.info(f"Successfully uploaded step {step} to {hf_repo_id_for_run}.")
    except Exception as e:
        logger.error(f"Failed to upload checkpoint step {step} to {hf_repo_id_for_run}: {e}", exc_info=True)
    if hf_keep_latest_k > 0:
        logger.info(f"Cleaning up old checkpoints on {hf_repo_id_for_run}, keeping latest {hf_keep_latest_k}")
        try:
            repo_files = api.list_repo_tree(hf_repo_id_for_run, repo_type="model", token=token, recursive=False)
            step_folders = [
                item.path for item in repo_files
                if item.path.startswith("step-") and item.path[5:].isdigit()
            ]

            step_folders.sort(key=lambda x: int(x.split('-')[1]), reverse=True)

            if len(step_folders) > hf_keep_latest_k:
                folders_to_delete = step_folders[hf_keep_latest_k:]
                logger.info(f"Found {len(step_folders)} checkpoints on Hub. Deleting {len(folders_to_delete)} older ones: {folders_to_delete}")
                for folder in folders_to_delete:
                    # Deleting requires repo_id, path_in_repo, and token
                    api.delete_folder(
                        repo_id=hf_repo_id_for_run,
                        path_in_repo=folder,
                        repo_type="model",
                        commit_message=f"Delete old checkpoint {folder}",
                        token=token
                    )
                logger.info("Hub cleanup complete.")
            else:
                logger.info("No old checkpoints found on Hub to delete.")
        except Exception as e:
            logger.error(f"Error during Hub checkpoint cleanup for {hf_repo_id_for_run}: {e}", exc_info=True)