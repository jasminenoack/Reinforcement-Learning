import matplotlib.pyplot as plt
import numpy as np

from gridworld.utils import render_directional_heatmap_for_q_table, render_heatmap


class TestRenderHeatmap:
    def test_render_heatmap_runs(self, mocker):
        mocker.patch.object(plt, "imshow", autospec=True)
        mocker.patch.object(plt, "colorbar", autospec=True)
        mocker.patch.object(plt, "title", autospec=True)
        mocker.patch.object(plt, "show", autospec=True)
        mocker.patch.object(plt, "pause", autospec=True)
        mocker.patch.object(plt, "close", autospec=True)
        mocker.patch("os.mkdir", autospec=True)
        mocker.patch.object(plt, "savefig", autospec=True)
        visit_counts = {
            (0, 0): 1,
            (0, 1): 2,
            (1, 0): 3,
            (1, 1): 4,
        }
        rows = 2
        cols = 2
        render_heatmap(
            visit_counts=visit_counts,
            rows=rows,
            cols=cols,
        )
        args, kwargs = plt.imshow.call_args
        assert np.array_equal(args[0], np.array([[1, 2], [3, 4]]))
        assert kwargs["cmap"] == "hot"
        assert kwargs["interpolation"] == "nearest"
        plt.colorbar.assert_called_once_with(label="Visit Count")
        plt.title.assert_called_once_with("Gridworld Visit Count Heatmap")
        plt.show.assert_called_once()
        plt.pause.assert_called_once()
        plt.close.assert_called_once()
        plt.savefig.assert_called_once()


class TestRenderDirectionalHeatmapForQTable:
    def test_render_directional_heatmap_for_q_table_runs(self, mocker):
        mocker.patch.object(plt, "imshow", autospec=True)
        mocker.patch.object(plt, "colorbar", autospec=True)
        mocker.patch.object(plt, "title", autospec=True)
        mocker.patch.object(plt, "show", autospec=True)
        mocker.patch.object(plt, "pause", autospec=True)
        mocker.patch.object(plt, "close", autospec=True)
        mocker.patch("os.mkdir", autospec=True)
        mocker.patch.object(plt, "savefig", autospec=True)
        visit_counts = {
            (0, 0): 1,
            (0, 1): 2,
            (1, 0): 3,
            (1, 1): 4,
        }
        rows = 2
        cols = 2
        q_table = {
            (0, 0): {"UP": 1.0, "DOWN": 0.0, "LEFT": 0.0, "RIGHT": 0.0},
            (0, 1): {"DOWN": 2.0, "UP": 0.0, "LEFT": 0.0, "RIGHT": 0.0},
            (1, 0): {"LEFT": 3.0, "UP": 0.0, "DOWN": 0.0, "RIGHT": 0.0},
            (1, 1): {"RIGHT": 4.0, "UP": 0.0, "DOWN": 0.0, "LEFT": 0.0},
        }
        render_directional_heatmap_for_q_table(
            visit_counts=visit_counts,
            rows=rows,
            cols=cols,
            q_table=q_table,
        )
        args, kwargs = plt.imshow.call_args
        assert np.array_equal(args[0], np.array([[1, 2], [3, 4]]))
        assert kwargs["cmap"] == "hot"
        assert kwargs["interpolation"] == "nearest"
        plt.colorbar.assert_called_once_with(label="Favorite Direction")
        plt.title.assert_called_once_with("Gridworld Favorite Direction Heatmap")
        plt.show.assert_called_once()
        plt.pause.assert_called_once()
        plt.close.assert_called_once()
        plt.savefig.assert_called_once()
