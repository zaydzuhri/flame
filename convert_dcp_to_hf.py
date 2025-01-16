# -*- coding: utf-8 -*-

import argparse
import os
import tempfile

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import fla  # noqa


def save_pretrained(
    checkpoint: str,
    path: str,
    config: str,
    tokenizer: str
):
    print(f"Loading the config from {config}")
    config = AutoConfig.from_pretrained(config, trust_remote_code=True)

    print(f"Saving the config to {path}")
    config.save_pretrained(path)
    print(f"Loading the tokenizer from {tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    print(f"Saving the tokenizer to {path}")
    tokenizer.save_pretrained(path)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
        print(f"Saving the distributed checkpoint to {checkpoint_path}")
        dcp_to_torch_save(checkpoint, checkpoint_path)

        print(f"Initializing the model from config\n{config}")
        model = AutoModelForCausalLM.from_config(config)
        print(model)
        print("Loading state dict from the checkpoint")
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model'])
        print(f"Saving the model to {path}")
        model.save_pretrained(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    args = parser.parse_args()
    save_pretrained(args.checkpoint, args.path, args.config, args.tokenizer)
