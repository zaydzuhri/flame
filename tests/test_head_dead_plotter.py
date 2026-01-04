import numpy as np

from analysis.ablations import plot_head_dead_analysis as plotter


def test_off_rate_matrix_and_layer_sizes() -> None:
    data = {
        "steps": [10000, 20000],
        "off_rates": {
            "10000": [[0.0, 0.5], [0.2]],
            "20000": [[0.1, 0.6], [0.3]],
        },
    }
    steps = plotter.sorted_steps(data)
    matrix, layer_sizes = plotter.off_rate_matrix(data, steps)
    assert layer_sizes == [2, 1]
    assert matrix.shape == (3, 2)
    assert np.allclose(matrix[:, 0], [0.0, 0.5, 0.2])


def test_persistent_dead_heads() -> None:
    heads_by_step = {
        10000: {(0, 1), (1, 0)},
        20000: {(0, 1)},
    }
    persistent = plotter.persistent_dead_heads(heads_by_step)
    assert persistent == {(0, 1)}


def test_dead_counts_and_layer_counts() -> None:
    data = {
        "steps": [10000],
        "dead_head_count": {"10000": 2},
        "mostly_dead_head_count": {"10000": 3},
        "dead_heads_by_step": {"10000": [[0, 0], [1, 2]]},
    }
    steps = plotter.sorted_steps(data)
    dead_counts, mostly_dead_counts = plotter.dead_counts_per_step(data, steps)
    assert dead_counts == [2]
    assert mostly_dead_counts == [3]
    heads_by_step = plotter.dead_heads_by_step(data, steps)
    counts = plotter.layer_dead_counts_at_step(heads_by_step, 10000)
    assert counts == {0: 1, 1: 1}


def test_build_persistent_dead_grid_bins() -> None:
    persistent = {(0, 0), (1, 3), (2, 4), (3, 7)}
    grid, head_labels, layer_labels, vmax = plotter.build_persistent_dead_grid(
        persistent,
        num_layers=4,
        num_heads=8,
        head_bin_size=4,
        layer_bin_size=2,
    )
    assert grid.shape == (2, 2)
    assert head_labels == ["0-3", "4-7"]
    assert layer_labels == ["0-1", "2-3"]
    assert vmax == 8
    assert grid[0, 0] == 2
    assert grid[1, 1] == 2
