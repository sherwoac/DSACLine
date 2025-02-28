import tqdm
import numpy as np
import math
import time
import cv2

from skimage.draw import line, line_aa, disk, set_color, circle_perimeter_aa
from skimage.util import random_noise


class LineDataset:
    """
    Generator of line segment images.

    Images will have 1 random line segment each, filled with noise and distractor circles.
    Class also offers functionality for drawing line parameters, hypotheses and point predictions.
    """
    _red = (1, 0, 0)
    _green = (0, 1, 0)
    _blue = (0, 0, 1)
    _dark = (0.2, 0.2, 0.2)
    _light = (0.7, 0.7, 0.7)
    _min_circles = 2
    _max_circles = 3
    _maxSlope = 10  # restrict the maximum slope of generated lines for stability
    _minLength = 20  # restrict the minimum length of line segments

    def __init__(self,
                 imgW: int = 64,
                 imgH: int = 64,
                 margin: int = -2,
                 bg_clr: float = 0.5,
                 max_sample_size: int = 1024,
                 rng: np.random._generator.Generator = None):
        """
        Constructor.

        imgW -- image width (default 64)
        imgH -- image height (default 64)
        margin -- lines segments are sampled within this margin, negative value means that a line segment can start or end outside the image (default -5)
        bg_clr -- background intensity (default 0.5)
        """
        if rng is not None:
            self._rng = rng
        else:
            self._rng = np.random.default_rng()

        self.imgW = imgW
        self.imgH = imgH
        self.margin = margin
        self.bg_clr = bg_clr

        # make the random numbers once
        # do common mass random numbers up front
        self.line_colours = self._rng.random((max_sample_size * 10, 3))
        self.lX = np.arange(self.margin, self.imgW - self.margin + 1)
        self.lY = np.arange(self.margin, self.imgH - self.margin + 1)
        self.mean_radius = int(0.5 * .9 * self.imgW)

    @staticmethod
    def draw_line(data, lX1, lY1, lX2, lY2, clr):
        """
        Draw a line with the given color and opacity.

        data -- image to draw to
        lX1 -- x value of line segment start point
        lY1 -- y value of line segment start point
        lX2 -- x value of line segment end point
        lY2 -- y value of line segment end point
        clr -- line color, triple of values
        alpha -- opacity (default 1.0)
        """
        _ = cv2.line(data, (lX1, lY1), (lX2, lY2), clr, 1, lineType=cv2.LINE_AA)

    def draw_hyps(self, labels, scores, data=None):
        '''
        Draw a set of line hypothesis for a batch of images.

        labels -- line parameters, array shape (NxMx2) where
            N is the number of images in the batch
            M is the number of hypotheses per image
            2 is the number of line parameters (intercept, slope)
        scores -- hypotheses scores, array shape (NxM), see above, higher score will be drawn with higher opacity
        data -- batch of images to draw to, if empty a new batch wil be created according to the shape of labels

        '''

        n = labels.shape[0]  # number of images
        m = labels.shape[1]  # number of hypotheses

        if data is None:  # create new batch of images
            data = np.zeros((n, self.imgH, self.imgW, 3), dtype=np.float32)
            data.fill(self.bg_clr)

        clr = LineDataset._blue

        for i in range(n):
            for j in range(m):
                lY1 = int(labels[i, j, 0] * self.imgH)
                lY2 = int(labels[i, j, 1] * self.imgW + labels[i, j, 0] * self.imgH)
                self.draw_line(data[i], 0, lY1, self.imgW, lY2, clr, scores[i, j])

        return data

    def draw_models(self, labels, data=None, correct=None):
        '''
        Draw lines for a batch of images.

        labels -- line parameters, array shape (Nx2) where
            N is the number of images in the batch
            2 is the number of line parameters (intercept, slope)
        data -- batch of images to draw to, if empty a new batch wil be created according to the shape of labels
            and lines will be green, lines will be blue otherwise
        correct -- array of shape (N) indicating whether a line estimate is correct
        '''

        n = labels.shape[0]
        if data is None:
            data = np.zeros((n, self.imgH, self.imgW, 3), dtype=np.float32)
            data.fill(self.bg_clr)
            clr = LineDataset._green
        else:
            clr = LineDataset._blue

        for i in range(n):
            lY1 = int(labels[i, 0] * self.imgH)
            lY2 = int(labels[i, 1] * self.imgW + labels[i, 0] * self.imgH)
            self.draw_line(data[i], 0, lY1, self.imgW, lY2, clr)

            if correct is not None:
                # draw border green if estiamte is correct, red otherwise
                if correct[i]:
                    borderclr = LineDataset._green
                else:
                    borderclr = LineDataset._red

                set_color(data[i], line(0, 0, 0, self.imgW - 1), borderclr)
                set_color(data[i], line(0, 0, self.imgH - 1, 0), borderclr)
                set_color(data[i], line(self.imgH - 1, 0, self.imgH - 1, self.imgW - 1), borderclr)
                set_color(data[i], line(0, self.imgW - 1, self.imgH - 1, self.imgW - 1), borderclr)

        return data

    def draw_points(self, points, data, inliers=None):
        """
        Draw 2D points for a batch of images.

        points -- 2D points, array shape (Nx2xM) where
            N is the number of images in the batch
            2 is the number of point dimensions (x, y)
            M is the number of points
        data -- batch of images to draw to
        inliers -- soft inlier score for each point,
            if given and score < 0.5 point will be drawn green, red otherwise
        """
        n = points.shape[0]  # number of images
        m = points.shape[2]  # number of points
        for i in range(n):
            for j in range(m):
                clr = LineDataset._dark  # draw predicted points as dark circles
                if inliers is not None and inliers[i, j] > 0.5:
                    clr = LineDataset._light  # draw inliers as light circles

                r = int(points[i, 0, j] * self.imgH)
                c = int(points[i, 1, j] * self.imgW)
                rr, cc = disk((r, c), 2)
                set_color(data[i], (rr, cc), clr)

        return data

    @staticmethod
    def draw_circle(image, centre_x, centre_y, radius, colour):
        _ = cv2.circle(image, (centre_x, centre_y), radius, colour, 1, lineType=cv2.LINE_AA)

    def sample_lines(self, n: int, progress: bool = False):
        """'''
        Create new input images of random line segments and distractors along with ground truth parameters.

        n -- number of images to create
        """
        data = np.zeros((n, self.imgH, self.imgW, 3), dtype=np.float32)
        data.fill(self.bg_clr)
        labels = np.ndarray((n, 2, 1, 1), dtype=np.float32)
        number_of_circles = self._rng.integers(LineDataset._min_circles, LineDataset._max_circles, n)
        circle_radii = self._rng.integers(int(0.1 * self.imgW), int(self.imgW), (n, LineDataset._max_circles))
        circle_centres_x = self._rng.integers(-1 * self.mean_radius, self.imgW + self.mean_radius + 1, (n, LineDataset._max_circles))
        circle_centres_y = self._rng.integers(-1 * self.mean_radius, self.imgH + self.mean_radius + 1, (n, LineDataset._max_circles))
        circle_colours = self._rng.choice(self.line_colours, size=(n, LineDataset._max_circles), axis=0)
        line_colours = self._rng.choice(self.line_colours, size=n, axis=0)
        slopes = np.tan(self._rng.uniform(-1. * math.atan(LineDataset._maxSlope), math.atan(LineDataset._maxSlope), n))
        lengths = self._rng.uniform(LineDataset._minLength, self.imgH + self.imgW)
        mid_points_x = self._rng.choice(self.lX, size=n)
        mid_points_y = self._rng.choice(self.lY, size=n)
        offsets_x = lengths * np.sqrt(1. / (4. + 4. * slopes * slopes))
        offsets_y = offsets_x * slopes
        line_point_1_x = (mid_points_x - offsets_x).astype(int)
        line_point_2_x = (mid_points_x + offsets_x).astype(int)
        line_point_1_y = (mid_points_y - offsets_y).astype(int)
        line_point_2_y = (mid_points_y + offsets_y).astype(int)
        intercepts = line_point_1_y - slopes * line_point_1_x
        labels[:, 0, 0, 0] = intercepts / self.imgH
        labels[:, 1, 0, 0] = slopes
        gauss = self._rng.normal(loc=0., scale=0.1, size=(n, self.imgH, self.imgW, 3)).astype(np.float16)
        for i in tqdm.tqdm(range(n), disable=not progress):  # for each image
            # create a random number of distractor circles
            for c in range(number_of_circles[i]):
                self.draw_circle(data[i], circle_centres_x[i, c], circle_centres_y[i, c], circle_radii[i, c], circle_colours[i, c])

            self.draw_line(data[i], line_point_1_x[i], line_point_1_y[i], line_point_2_x[i], line_point_2_y[i], line_colours[i])

        data += data * gauss  # random noise in a 1er
        return data, labels
