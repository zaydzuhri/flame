import argparse

import torch

from analysis.ablations import head_dead_analysis as hda


def test_compute_off_counts_without_mask() -> None:
    head_output = torch.tensor(
        [
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 5e-7]],
            ]
        ]
    )
    off_counts, valid_tokens = hda.compute_off_counts(head_output, None, eps=1e-6)
    assert valid_tokens == 2
    assert off_counts.tolist() == [1.0, 2.0]


def test_compute_off_counts_with_mask() -> None:
    head_output = torch.tensor(
        [
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 5e-7]],
            ]
        ]
    )
    attention_mask = torch.tensor([[1, 0]], dtype=torch.int64)
    off_counts, valid_tokens = hda.compute_off_counts(
        head_output, attention_mask, eps=1e-6
    )
    assert valid_tokens == 1
    assert off_counts.tolist() == [1.0, 1.0]


def test_accumulator_updates_from_flat_output() -> None:
    head_output = torch.tensor(
        [
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 5e-7]],
            ]
        ]
    )
    flat = head_output.view(1, 2, 4)
    accumulator = hda.HeadOffRateAccumulator(eps=1e-6, num_heads_by_layer=[2])
    accumulator.set_attention_mask(torch.tensor([[1, 1]], dtype=torch.int64))
    accumulator.update_from_o_proj(layer_idx=0, num_heads=2, flat_head_output=flat)
    rates = accumulator.off_rates()
    assert rates == [[0.5, 1.0]]


def test_compute_death_summary() -> None:
    steps = [10000, 20000, 30000]
    off_rates_by_step = [
        [[0.0, 1.0]],
        [[0.96, 1.0]],
        [[0.0, 1.0]],
    ]
    summary = hda.compute_death_summary(
        steps,
        off_rates_by_step,
        dead_threshold=0.95,
        mostly_dead_threshold=0.75,
    )
    assert summary["death_step"] == [[20000, 10000]]
    assert summary["revived"] == [[True, False]]
    assert summary["dead_heads_by_step"] == {
        "10000": [[0, 1]],
        "20000": [[0, 0], [0, 1]],
        "30000": [[0, 1]],
    }
    assert summary["dead_never_revive_heads"] == [[0, 1]]
    assert summary["dead_revive_heads"] == [[0, 0]]
    assert summary["lifetime_summary"]["never_dead"] == 0
    assert summary["lifetime_summary"]["dead_never_revive"] == 1
    assert summary["lifetime_summary"]["dead_revive"] == 1


def test_parse_steps() -> None:
    args = argparse.Namespace(
        steps="20000,10000,10000",
        step_start=None,
        step_end=None,
        step_stride=10000,
    )
    assert hda.parse_steps(args) == [10000, 20000]
