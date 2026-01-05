from analysis.ablations import scan_dormant_thresholds as scan


def test_compute_threshold_summary_counts() -> None:
    data = {
        "steps": [10000, 20000],
        "dormant_rate": {
            "10000": [[0.96, 0.2], [0.5]],
            "20000": [[0.97, 0.1], [0.8]],
        },
        "lifetime_summary": {"total_heads": 3},
    }
    summary = scan.compute_threshold_summary(data, [0.95, 0.75])
    per_095 = summary["per_threshold"]["0.95"]
    assert per_095["dormant_head_count"] == {"10000": 1, "20000": 1}
    assert per_095["union_count"] == 1
    assert per_095["persistent_count"] == 1
    per_075 = summary["per_threshold"]["0.75"]
    assert per_075["dormant_head_count"] == {"10000": 1, "20000": 2}
    assert per_075["union_count"] == 2
    assert per_075["persistent_count"] == 1
