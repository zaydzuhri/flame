from analysis.ablations import scan_sink_dominance_thresholds as scan


def test_parse_thresholds_default_list() -> None:
    thresholds = scan.parse_thresholds("")
    assert thresholds[0] == 0.1
    assert thresholds[-1] == 0.95
    assert 0.9 in thresholds


def test_format_threshold() -> None:
    assert scan.format_threshold(0.1) == "0p10"
    assert scan.format_threshold(0.95) == "0p95"
