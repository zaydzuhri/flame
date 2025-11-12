import itertools

import torch
from datasets import IterableDataset

from flame.data import FinetuneIterableDataset, build_dataloader


def make_iterable_dataset():
    texts = [
        "short prompt",
        "a significantly longer piece of text used for testing finetune dataloader behaviour",
        "tiny",
    ]

    def generator():
        for text in texts:
            yield {"text": text}

    return IterableDataset.from_generator(generator)


class DummyTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    model_max_length = 16

    def __call__(self, texts, return_attention_mask=False, truncation=False, max_length=None):
        if isinstance(texts, str):
            texts = [texts]
        max_length = max_length or self.model_max_length
        outputs = []
        for text in texts:
            tokens = [self.bos_token_id]
            tokens.extend([(ord(ch) % 50) + 3 for ch in text])
            tokens.append(self.eos_token_id)
            if truncation and len(tokens) > max_length:
                tokens = tokens[:max_length]
            outputs.append(tokens)
        return {"input_ids": outputs}

    def pad(self, encoded_inputs, return_tensors=None, return_attention_mask=False):
        lengths = [len(entry["input_ids"]) for entry in encoded_inputs]
        max_len = max(lengths)
        padded, masks = [], []
        for entry in encoded_inputs:
            ids = list(entry["input_ids"])
            pad_len = max_len - len(ids)
            padded.append(ids + [self.pad_token_id] * pad_len)
            masks.append([1] * len(ids) + [0] * pad_len)

        batch = {"input_ids": torch.tensor(padded, dtype=torch.long)}
        if return_attention_mask:
            batch["attention_mask"] = torch.tensor(masks, dtype=torch.long)
        return batch


def test_finetune_iterable_dataset_preserves_examples():
    dataset = make_iterable_dataset()
    tokenizer = DummyTokenizer()
    finetune_dataset = FinetuneIterableDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        context_len=tokenizer.model_max_length,
        rank=0,
        world_size=1,
    )

    samples = list(itertools.islice(iter(finetune_dataset), 2))
    assert all("input_ids" in sample for sample in samples)
    assert samples[0]["input_ids"].shape[0] <= tokenizer.model_max_length
    assert samples[1]["input_ids"].shape[0] <= tokenizer.model_max_length


def test_build_dataloader_finetune_mode_batches_and_pads():
    dataset = make_iterable_dataset()
    tokenizer = DummyTokenizer()
    dataloader = build_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        rank=0,
        world_size=1,
        batch_size=2,
        seq_len=8,
        context_len=tokenizer.model_max_length,
        varlen=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        snapshot_every_n_steps=None,
        dataset_mode="finetune",
    )

    batch = next(iter(dataloader))
    assert batch["input_ids"].shape[0] == 2
    assert "attention_mask" in batch
    # attention_mask should reflect true lengths
    true_lengths = batch["attention_mask"].sum(dim=1)
    assert true_lengths[0] != true_lengths[1]
