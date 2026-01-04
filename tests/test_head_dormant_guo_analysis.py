import torch

from analysis.ablations import head_dormant_analysis_guo as hda


def test_compute_sink_stats_counts() -> None:
    attentions = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.95, 0.05]],
                [[0.5, 0.5], [0.99, 0.01]],
            ]
        ]
    )
    sink_sum, entropy_sum, dormant_count, valid_tokens = hda.compute_sink_stats(
        attentions,
        attention_mask=None,
        sink_index=0,
        entropy_eps=1e-12,
        sink_dominance_threshold=0.9,
        entropy_threshold=None,
    )
    assert valid_tokens == 2
    assert torch.allclose(sink_sum, torch.tensor([1.95, 1.49], dtype=sink_sum.dtype))
    assert dormant_count.tolist() == [2.0, 1.0]
    assert entropy_sum.numel() == 2


def test_dormant_summary_from_rates() -> None:
    steps = [10000, 20000]
    dormant_rates = [
        [[1.0, 0.5]],
        [[0.96, 0.2]],
    ]
    summary = hda.compute_dormant_summary(
        steps,
        dormant_rates,
        dormant_threshold=0.95,
        mostly_dormant_threshold=0.75,
    )
    assert summary["dormant_head_count"] == {"10000": 1, "20000": 1}
    assert summary["mostly_dormant_head_count"] == {"10000": 1, "20000": 1}
    assert summary["dormant_heads_by_step"]["10000"] == [[0, 0]]
