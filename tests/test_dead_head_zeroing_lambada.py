import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.ablations import dead_head_zeroing_lambada as dhzl


class DummyTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        tokens = [tok for tok in text.split(" ") if tok]
        return {"input_ids": [len(tok) for tok in tokens]}


def test_split_prompt_target_basic() -> None:
    split = dhzl.split_prompt_target("the quick brown fox")
    assert split == ("the quick brown", "fox")


def test_prepare_lambada_sample_returns_input_and_target() -> None:
    tokenizer = DummyTokenizer()
    prepared = dhzl.prepare_lambada_sample(
        tokenizer=tokenizer,
        text="the quick brown fox",
        max_length=16,
    )
    assert prepared is not None
    input_ids, target_ids = prepared
    assert input_ids.tolist() == [3, 5, 5]
    assert target_ids.tolist() == [3]


def test_prepare_lambada_sample_respects_max_length() -> None:
    tokenizer = DummyTokenizer()
    prepared = dhzl.prepare_lambada_sample(
        tokenizer=tokenizer,
        text="the quick brown fox",
        max_length=3,
    )
    assert prepared is None


def test_validate_args_auto_mask_default_allows_no_json() -> None:
    args = argparse.Namespace(
        skip_baseline=False,
        skip_masked=False,
        max_length=32,
        auto_mask_dead_threshold=0.95,
        auto_mask_eps=1e-6,
        mask_json=None,
        step=None,
        auto_mask=True,
    )
    dhzl.validate_args(args)


def test_validate_args_step_requires_json() -> None:
    args = argparse.Namespace(
        skip_baseline=False,
        skip_masked=False,
        max_length=32,
        auto_mask_dead_threshold=0.95,
        auto_mask_eps=1e-6,
        mask_json=None,
        step=10000,
        auto_mask=True,
    )
    with pytest.raises(ValueError, match="--step is only valid when --mask-json is provided"):
        dhzl.validate_args(args)


def test_validate_args_missing_mask_json_has_hint() -> None:
    args = argparse.Namespace(
        skip_baseline=False,
        skip_masked=False,
        max_length=32,
        auto_mask_dead_threshold=0.95,
        auto_mask_eps=1e-6,
        mask_json=Path("/tmp/does-not-exist-mask.json"),
        step=None,
        auto_mask=True,
    )
    with pytest.raises(ValueError, match="Omit --mask-json to use default auto-mask derivation"):
        dhzl.validate_args(args)


def test_infer_step_from_model_name(tmp_path) -> None:
    payload = {"dead_heads_by_step": {"10000": [[0, 0]], "20000": [[0, 1]]}}
    path = tmp_path / "dead.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    inferred = dhzl.infer_step_for_model(
        path,
        "analysis/attention_sink/hf_models/softpick-1B-4096-step-20000",
        explicit_step=None,
    )
    assert inferred == 20000
