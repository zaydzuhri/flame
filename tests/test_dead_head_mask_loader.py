import json

import pytest

from fla.models.dead_head_mask import load_dead_head_masks_by_layer


def test_load_dead_head_masks_by_layer_from_dead_heads_by_step(tmp_path) -> None:
    payload = {
        "dead_heads_by_step": {
            "10000": [[0, 1], [1, 0]],
        }
    }
    path = tmp_path / "dead.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    masks = load_dead_head_masks_by_layer(
        mask_path=path,
        step=10000,
        num_heads_by_layer=[2, 2],
    )

    assert masks[0].tolist() == [False, True]
    assert masks[1].tolist() == [True, False]


def test_load_dead_head_masks_by_layer_missing_step_fails(tmp_path) -> None:
    payload = {"dead_heads_by_step": {"10000": [[0, 0]]}}
    path = tmp_path / "dead.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Step 20000 not found"):
        load_dead_head_masks_by_layer(
            mask_path=path,
            step=20000,
            num_heads_by_layer=[2],
        )


def test_load_dead_head_masks_by_layer_invalid_index_fails(tmp_path) -> None:
    payload = {"dead_heads_by_step": {"10000": [[0, 3]]}}
    path = tmp_path / "dead.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Head index 3 out of range"):
        load_dead_head_masks_by_layer(
            mask_path=path,
            step=10000,
            num_heads_by_layer=[2],
        )


def test_load_dead_head_masks_by_layer_falls_back_to_off_rates(tmp_path) -> None:
    payload = {
        "off_rates": {
            "10000": [
                [0.1, 0.96],
                [0.2, 0.99],
            ]
        },
        "dead_threshold": 0.95,
    }
    path = tmp_path / "dead.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    masks = load_dead_head_masks_by_layer(
        mask_path=path,
        step=10000,
        num_heads_by_layer=[2, 2],
    )
    assert masks[0].tolist() == [False, True]
    assert masks[1].tolist() == [False, True]
