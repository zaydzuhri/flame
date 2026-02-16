import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.ablations import dead_head_zeroing_ppl as dhz


def test_get_dead_head_count_for_step(tmp_path) -> None:
    payload = {"dead_heads_by_step": {"10000": [[0, 1], [1, 0], [1, 1]]}}
    path = tmp_path / "dead.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert dhz.get_dead_head_count(path, 10000) == 3


def test_validate_args_requires_mask_fields_when_masked() -> None:
    args = argparse.Namespace(
        skip_baseline=False,
        skip_masked=False,
        padding="longest",
        batch_size=2,
        streaming=False,
        mask_json=None,
        step=None,
        auto_mask=False,
        auto_mask_dead_threshold=0.95,
        auto_mask_eps=1e-6,
    )
    with pytest.raises(ValueError, match="Masked run requires either --mask-json or --auto-mask"):
        dhz.validate_args(args)


def test_validate_args_streaming_two_pass_allowed() -> None:
    mask_path = Path(__file__)
    args = argparse.Namespace(
        skip_baseline=False,
        skip_masked=False,
        padding="longest",
        batch_size=2,
        streaming=True,
        mask_json=mask_path,
        step=10000,
        auto_mask=True,
        auto_mask_dead_threshold=0.95,
        auto_mask_eps=1e-6,
    )
    dhz.validate_args(args)


def test_validate_args_auto_mask_default_allows_no_json() -> None:
    args = argparse.Namespace(
        skip_baseline=False,
        skip_masked=False,
        padding="longest",
        batch_size=2,
        streaming=False,
        mask_json=None,
        step=None,
        auto_mask=True,
        auto_mask_dead_threshold=0.95,
        auto_mask_eps=1e-6,
    )
    dhz.validate_args(args)


def test_validate_args_step_requires_json() -> None:
    args = argparse.Namespace(
        skip_baseline=False,
        skip_masked=False,
        padding="longest",
        batch_size=2,
        streaming=False,
        mask_json=None,
        step=10000,
        auto_mask=True,
        auto_mask_dead_threshold=0.95,
        auto_mask_eps=1e-6,
    )
    with pytest.raises(ValueError, match="--step is only valid when --mask-json is provided"):
        dhz.validate_args(args)


def test_validate_args_mask_json_missing_shows_auto_mask_hint() -> None:
    args = argparse.Namespace(
        skip_baseline=False,
        skip_masked=False,
        padding="longest",
        batch_size=2,
        streaming=False,
        mask_json=Path("/tmp/does-not-exist-mask.json"),
        step=None,
        auto_mask=True,
        auto_mask_dead_threshold=0.95,
        auto_mask_eps=1e-6,
    )
    with pytest.raises(ValueError, match="Omit --mask-json to use default auto-mask derivation"):
        dhz.validate_args(args)


def test_infer_step_for_single_step_mask(tmp_path) -> None:
    payload = {"dead_heads_by_step": {"10000": [[0, 0]]}}
    path = tmp_path / "dead.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    inferred = dhz.infer_step_for_model(path, "any/model/path", explicit_step=None)
    assert inferred == 10000


def test_infer_step_from_model_name(tmp_path) -> None:
    payload = {"dead_heads_by_step": {"10000": [[0, 0]], "20000": [[0, 1]]}}
    path = tmp_path / "dead.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    inferred = dhz.infer_step_for_model(
        path,
        "analysis/attention_sink/hf_models/softpick-1B-4096-step-20000",
        explicit_step=None,
    )
    assert inferred == 20000
