import matplotlib.pyplot as plt
import numpy as np

from gridworld.utils import render_heatmap


class TestRenderHeatmap:
    def test_render_heatmap_runs(self, mocker):
        mocker.patch.object(plt, "imshow", autospec=True)
        mocker.patch.object(plt, "colorbar", autospec=True)
        mocker.patch.object(plt, "title", autospec=True)
        mocker.patch.object(plt, "show", autospec=True)
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
