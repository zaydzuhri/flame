import sys

sys.path.append(".")

import argparse
import os
from huggingface_hub import HfApi, HfFolder, snapshot_download
from flame.utils import convert_dcp_to_hf

def download_dcp_model(repo_id: str, folder_in_repo: str):
    repo_name = repo_id.split("/")[-1]
    current_dir = os.getcwd()
    if "evaluation_lm_eval" in current_dir:
        current_dir = os.path.dirname(current_dir)

    dcp_folder = os.path.join(
        current_dir,
        "evaluation_lm_eval", 
        "models", 
        repo_name,
        "checkpoint"
    )
    os.makedirs(dcp_folder, exist_ok=True)

    # check if the content of the folder has already been downloaded
    if os.path.exists(os.path.join(dcp_folder, folder_in_repo)):
        print(f"The content of {folder_in_repo} has already been downloaded.")
        return

    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[folder_in_repo.rstrip("/")+"/"],
        local_dir=dcp_folder,
    )

def main():
    parser = argparse.ArgumentParser(description="Download a DCP model from Hugging Face Hub.")
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="The repository ID on Hugging Face Hub.",
    )
    parser.add_argument(
        "--folder_in_repo",
        type=str,
        required=True,
        help="The folder in the repository to download.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="The path to the config file in the repository.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="The path to the tokenizer file in the repository.",
    )
    args = parser.parse_args()
    
    download_dcp_model(args.repo_id, args.folder_in_repo)

    current_dir = os.getcwd()
    if "evaluation_lm_eval" in current_dir:
        current_dir = os.path.dirname(current_dir)

    dcp_folder = os.path.join(
        current_dir,
        "evaluation_lm_eval",
        "models",
        args.repo_id.split("/")[-1],
        "."
    )

    step = int(args.folder_in_repo.split("-")[-1])

    convert_dcp_to_hf.save_pretrained(
        path=dcp_folder,
        step=step,
        config=args.config_path,
        tokenizer=args.tokenizer_path,
    )

if __name__ == "__main__":
    main()

