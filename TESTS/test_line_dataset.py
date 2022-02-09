from unittest import TestCase
import matplotlib.pyplot as plt
import utils
import line_dataset
import numpy as np


class TestLineDataset(TestCase):
    def test_sample_lines(self):
        rng = np.random.default_rng(0)
        ld = line_dataset.LineDataset(rng=rng)
        images, labels = ld.sample_lines(9)
        grid = utils.make_grid(images)
        grid.transpose_(0, 1).transpose_(1, 2)
        plt.imshow(grid)
        plt.show()
