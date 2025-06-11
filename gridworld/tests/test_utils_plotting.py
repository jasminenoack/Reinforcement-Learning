import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pytest

from gridworld.utils import (
    UP,
    DOWN,
    render_heatmap,
    render_directional_heatmap_for_q_table,
    line_plot,
)


class TestRenderHeatmap:
    def test_returns_filename_and_draws_arrow(self, tmp_path, monkeypatch):
        arrow_calls = []

        def fake_arrow(*args, **kwargs):
            arrow_calls.append(args)

        monkeypatch.setattr(plt, "arrow", fake_arrow)

        q_table = {(0, 0): {UP: 1.0, DOWN: 0.0, "left": 0.0, "right": 0.0}}

        result = render_heatmap(
            visit_counts={(0, 0): 1},
            rows=1,
            cols=1,
            folder=str(tmp_path),
            q_table=q_table,
            show=False,
        )

        assert result == "gridworld_visit_count_heatmap.png"
        saved_file = tmp_path / result
        assert saved_file.exists()
        assert len(arrow_calls) == 1
        assert arrow_calls[0][:4] == pytest.approx((0.5, 0.5, 0.0, -1.0))


class TestRenderDirectionalHeatmap:
    def test_returns_path_and_draws_arrow(self, tmp_path, monkeypatch):
        arrow_calls = []

        def fake_arrow(*args, **kwargs):
            arrow_calls.append(args)

        monkeypatch.setattr(plt, "arrow", fake_arrow)

        q_table = {(0, 0): {UP: 1.0, DOWN: 0.0, "left": 0.0, "right": 0.0}}

        result = render_directional_heatmap_for_q_table(
            visit_counts={(0, 0): 1},
            rows=1,
            cols=1,
            q_table=q_table,
            folder=str(tmp_path),
        )

        expected = tmp_path / "gridworld_favorite_direction_heatmap.png"
        assert result == str(expected)
        assert expected.exists()
        assert len(arrow_calls) == 1
        assert arrow_calls[0][:4] == pytest.approx((0.0, 0.0, 0.0, -1.0))


class TestLinePlot:
    def test_saves_plot(self, tmp_path):
        line_plot(
            [0, 1],
            {"series": [1.0, 2.0]},
            title="Example",
            x_label="epoch",
            folder=str(tmp_path),
        )

        assert (tmp_path / "example.png").exists()
