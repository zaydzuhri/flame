import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "3rdparty/flash-linear-attention"))
sys.path.insert(0, str(ROOT / "3rdparty/lm-evaluation-harness"))

HARNESS_PATH = ROOT / "3rdparty/flash-linear-attention/evals/harness.py"
SPEC = importlib.util.spec_from_file_location("fla_eval_harness", HARNESS_PATH)
assert SPEC is not None and SPEC.loader is not None
harness = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = harness
SPEC.loader.exec_module(harness)


def test_as_bool_accepts_common_values() -> None:
    assert harness._as_bool(True, "x") is True
    assert harness._as_bool(False, "x") is False
    assert harness._as_bool("true", "x") is True
    assert harness._as_bool("0", "x") is False


def test_as_bool_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        harness._as_bool("maybe", "x")


def test_infer_step_from_model_ref(tmp_path: Path) -> None:
    payload = {"dead_heads_by_step": {"10000": [[0, 0]], "20000": [[0, 1]]}}
    mask_path = tmp_path / "dead.json"
    mask_path.write_text(json.dumps(payload), encoding="utf-8")
    step = harness._infer_step_for_model_ref(
        mask_json=mask_path,
        model_ref="analysis/attention_sink/hf_models/softpick-1B-4096-step-20000",
        explicit_step=None,
    )
    assert step == 20000


def test_masks_from_off_rates_threshold() -> None:
    masks = harness._masks_from_off_rates(
        off_rates=[[0.1, 0.96], [0.2, 0.3, 0.99]],
        dead_threshold=0.95,
    )
    assert [mask.tolist() for mask in masks] == [
        [False, True],
        [False, False, True],
    ]
