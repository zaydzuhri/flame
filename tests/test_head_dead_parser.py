from analysis.ablations import parse_head_dead_analysis as parser


def test_compute_summary_from_dead_heads_by_step() -> None:
    data = {
        "steps": [10000, 20000],
        "lifetime_summary": {"total_heads": 4},
        "dead_heads_by_step": {
            "10000": [[0, 1], [1, 0]],
            "20000": [[0, 1]],
        },
        "mostly_dead_heads_by_step": {
            "10000": [[0, 1], [1, 0], [1, 1]],
            "20000": [[0, 1]],
        },
    }
    summary = parser.compute_summary(data)
    assert summary["dead_union_count"] == 2
    assert summary["persistent_dead_heads"] == [(0, 1)]
    assert summary["per_step"]["10000"]["dead_count"] == 2
    assert summary["per_step"]["20000"]["revived"] == [(1, 0)]


def test_compute_summary_fallback_from_off_rates() -> None:
    data = {
        "steps": [10000],
        "off_rates": {"10000": [[0.0, 0.96], [0.97, 0.2]]},
        "dead_threshold": 0.95,
        "mostly_dead_threshold": 0.75,
        "lifetime_summary": {"total_heads": 4},
    }
    summary = parser.compute_summary(data)
    assert summary["dead_heads_by_step"]["10000"] == [(0, 1), (1, 0)]
    assert summary["mostly_dead_heads_by_step"]["10000"] == [(0, 1), (1, 0)]


def test_render_text_has_key_fields() -> None:
    data = {
        "steps": [10000],
        "lifetime_summary": {"total_heads": 2},
        "dead_heads_by_step": {"10000": [[0, 0]]},
        "mostly_dead_heads_by_step": {"10000": [[0, 0]]},
    }
    summary = parser.compute_summary(data)
    output = parser.render_text(summary, data, parser.argparse.Namespace(
        show_heads="none",
        max_heads=50,
        format="text",
        input=None,
    ))
    assert "Total heads: 2" in output
    assert "Step 10000" in output
