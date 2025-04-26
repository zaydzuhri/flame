# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
from typing import Any, Dict, List

from transformers import AutoTokenizer, PreTrainedTokenizer

from flame.data import build_dataset
from torchtitan.tools.logging import init_logger, logger


def tokenize(
    examples: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, List[List[int]]]:
    if 'text' in examples:
        input_ids = tokenizer(examples['text'])['input_ids']
    elif 'content' in examples:
        input_ids = tokenizer(examples['content'])['input_ids']
    else:
        raise ValueError(f'No "text" or "content" field found in examples:\n{examples}')
    return {'input_ids': input_ids}


if __name__ == '__main__':
    init_logger()
    parser = argparse.ArgumentParser(description='Preprocess the dataset.')
    parser.add_argument(
        '--dataset',
        default='HuggingFaceFW/fineweb-edu',
        help='Dataset to use, with comma separated values',
    )
    parser.add_argument(
        '--dataset_name',
        default='sample-100BT',
        help='The name of the dataset config, with comma separated values if provided',
    )
    parser.add_argument(
        '--dataset_split',
        default='train',
        help='Dataset split to use, with comma separated values if provided',
    )
    parser.add_argument(
        '--data_dir',
        default=None,
        help='Data dirs to use, with comma separated values if provided',
    )
    parser.add_argument(
        '--data_files',
        default=None,
        help='Data files to use, with comma separated values if provided',
    )
    parser.add_argument(
        '--data_probs',
        default=None,
        help='Data sampling probabilities, with comma separated values if provided',
    )
    parser.add_argument(
        '--streaming',
        action='store_true',
        help='Whether to use streaming mode',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=32,
        help='Number of workers to use for preprocessing',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for preprocessing',
    )
    parser.add_argument(
        '--path',
        default='data',
        help='Path to save the preprocessed dataset',
    )
    parser.add_argument(
        '--tokenizer',
        default='fla-hub/transformer-1.3B-100B',
        help='Tokenizer to use',
    )
    args = parser.parse_args()

    logger.info(f'Loading tokenizer {args.tokenizer}')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    logger.info(f'{tokenizer}')
    logger.info(f'Loading dataset {args.dataset} {args.dataset_name} {args.dataset_split}')
    dataset = build_dataset(
        dataset=args.dataset,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        data_dir=args.data_dir,
        data_files=args.data_files,
        data_probs=args.data_probs,
        streaming=args.streaming,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    logger.info('Tokenizing dataset')
    tokenized_dataset = dataset.map(tokenize, batched=True, fn_kwargs={'tokenizer': tokenizer})
    logger.info(f'{tokenized_dataset}')
    logger.info(f'Saving tokenized dataset to {args.path}')
    tokenized_dataset.save_to_disk(args.path)
