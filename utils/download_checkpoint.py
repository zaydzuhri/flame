import argparse
import os
from huggingface_hub import HfApi, HfFolder, snapshot_download

def main(args):
    api = HfApi()
    token = HfFolder.get_token()
    experiment_checkpoint_folder = os.path.join(args.experiment_checkpoint_folder, "checkpoint")
    os.makedirs(
        experiment_checkpoint_folder,
        exist_ok=True
    )

    snapshot_download(
        repo_id=args.repo_id, 
        token=token,
        local_dir=experiment_checkpoint_folder,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a checkpoint from Hugging Face Hub.")
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="The repository ID on Hugging Face Hub.",
    )
    parser.add_argument(
        "--experiment_checkpoint_folder",
        type=str,
        required=True,
        help="The local directory to save the downloaded checkpoint.",
    )
    args = parser.parse_args()
    main(args)