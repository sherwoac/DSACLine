import unittest

import torch

import line_loss_aio
import line_loss
import line_area_loss

class LineLossTestCase(unittest.TestCase):
    _pts = torch.tensor([[0.0633, 0.0000],
                         [0.3220, 1.0000]])

    _slopes = torch.tensor(3.8658).repeat((2))
    _intercepts = torch.tensor(-0.2448).repeat((2))

    _image_size = 64

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.line_loss_aio = line_loss_aio.LineLossAio(LineLossTestCase._image_size)
        self.line_loss = line_loss.LineLoss(LineLossTestCase._image_size)
        self.line_area_loss = line_area_loss.LineLossArea(LineLossTestCase._image_size)

    def test_get_unit_square_intercepts(self):
        # bottom left top right
        delta = .0001
        test_slope = torch.tensor([1. - delta])
        test_intercept = torch.tensor([0. - delta])
        test_pts = torch.tensor([[0., 1.], [0., 1.]])
        pts = self.line_loss_aio._get_unit_square_intercepts(test_slope, test_intercept)[0]
        assert torch.allclose(test_pts, pts, atol=delta * 2), f"unequal points: {pts} != {test_pts}"

        test_slope = torch.tensor([1. + delta])
        test_intercept = torch.tensor([0. - delta])
        test_pts = torch.tensor([[0., 1.], [0., 1.]])
        pts = self.line_loss_aio._get_unit_square_intercepts(test_slope, test_intercept)[0]
        assert torch.allclose(test_pts, pts, atol=delta * 2), f"unequal points: {pts} != {test_pts}"

        test_slope = torch.tensor([1. - delta])
        test_intercept = torch.tensor([0. + delta])
        test_pts = torch.tensor([[0., 1.], [0., 1.]])
        pts = self.line_loss_aio._get_unit_square_intercepts(test_slope, test_intercept)[0]
        assert torch.allclose(test_pts, pts, atol=delta * 2), f"unequal points: {pts} != {test_pts}"

        test_slope = torch.tensor([1. + delta])
        test_intercept = torch.tensor([0. + delta])
        test_pts = torch.tensor([[0., 1.], [0., 1.]])
        pts = self.line_loss_aio._get_unit_square_intercepts(test_slope, test_intercept)[0]
        assert torch.allclose(test_pts, pts, atol=delta * 2), f"unequal points: {pts} != {test_pts}"

        # top right bottom left
        test_slope = torch.tensor([-1. + delta])
        test_intercept = torch.tensor([1. + delta])
        test_pts = torch.tensor([[0., 1.], [1., 0.]])
        pts = self.line_loss_aio._get_unit_square_intercepts(test_slope, test_intercept)[0]
        assert torch.allclose(test_pts, pts, atol=delta * 3), f"unequal max points: {pts} != {test_pts}"

        test_slope = torch.tensor([-1. - delta])
        test_intercept = torch.tensor([1. + delta])
        test_pts = torch.tensor([[0., 1.], [1., 0.]])
        pts = self.line_loss_aio._get_unit_square_intercepts(test_slope, test_intercept)[0]
        assert torch.allclose(test_pts, pts, atol=delta * 3), f"unequal max points: {pts} != {test_pts}"

        test_slope = torch.tensor([-1. + delta])
        test_intercept = torch.tensor([1. - delta])
        test_pts = torch.tensor([[0., 1.], [1., 0.]])
        pts = self.line_loss_aio._get_unit_square_intercepts(test_slope, test_intercept)[0]
        assert torch.allclose(test_pts, pts, atol=delta * 3), f"unequal max points: {pts} != {test_pts}"

        test_slope = torch.tensor([-1. - delta])
        test_intercept = torch.tensor([1. - delta])
        test_pts = torch.tensor([[0., 1.], [1., 0.]])
        pts = self.line_loss_aio._get_unit_square_intercepts(test_slope, test_intercept)[0]
        assert torch.allclose(test_pts, pts, atol=delta * 3), f"unequal max points: {pts} != {test_pts}"

        test_slope = torch.tensor(2.2111).reshape((1, 1))
        test_intercept = torch.tensor(-0.6713).reshape((1, 1))
        test_pts = torch.tensor([[[0.7559, 0.3036], [1.0000, 0.0000]]])
        pts = self.line_loss._get_unit_square_intercepts(test_slope, test_intercept)
        assert torch.allclose(test_pts, pts, rtol=.0001), f"unequal max points: {pts} != {test_pts}"

    def test_line_loss(self):
        est = torch.tensor([[-1.0516,  2.1709]])
        gt = torch.tensor([[ 1.0764, -0.8889]])
        loss = self.line_loss(est.squeeze(), gt.squeeze())
        loss_aio = self.line_loss_aio.get_line_loss(est, gt)
        # these should be equal
        # assert torch.allclose(loss, loss_aio)

        est = torch.tensor([[-1.0516,  2.1709]]).repeat((LineLossTestCase._image_size, 1))
        gt = torch.tensor([[ 1.0764, -0.8889]]).repeat((LineLossTestCase._image_size, 1))
        # need to fix the batches x pts case
        loss = self.line_loss_aio.get_line_loss(est, gt) / LineLossTestCase._image_size

        assert torch.allclose(loss, torch.tensor(90.0952))

    def test_intercept_cases(self):
        # nan over a +/- range
        est = (torch.randn((10000, 2)) - 0.5) * 10.
        aio_points = self.line_loss_aio._get_unit_square_intercepts(est[:, 1], est[:, 0])
        assert not aio_points.isnan().any(), f"aio_points isnan: {aio_points.isnan().any()}"

        # test with some lines
        delta = 0.0001
        est = torch.tensor([[0., 1.]]) + delta
        aio_points = self.line_loss_aio._get_unit_square_intercepts(est[:, 1], est[:, 0])
        original_points = self.line_loss._LineLoss__get_max_points(est[0, 1], est[0, 0])
        assert torch.allclose(aio_points, original_points)

        est = torch.tensor([[100., -.10]])
        aio_points = self.line_loss_aio._get_unit_square_intercepts(est[:, 1], est[:, 0])
        original_points = self.line_loss._LineLoss__get_max_points(est[0, 1], est[0, 0])
        assert torch.allclose(aio_points, original_points)

        est = torch.tensor([[1.1, 1000]])
        aio_points = self.line_loss_aio._get_unit_square_intercepts(est[:, 1], est[:, 0])
        original_points = self.line_loss._LineLoss__get_max_points(est[0, 1], est[0, 0])
        assert torch.allclose(aio_points, original_points)


        est = torch.tensor([[0., 1.]]) + delta
        aio_points = self.line_loss_aio._get_unit_square_intercepts(est[:, 1], est[:, 0])
        original_points = self.line_loss._LineLoss__get_max_points(est[0, 1], est[0, 0])
        assert torch.allclose(aio_points, original_points)

        est = torch.tensor([[0., 1.]]) + delta
        aio_points = self.line_loss_aio._get_unit_square_intercepts(est[:, 1], est[:, 0])
        original_points = self.line_loss._LineLoss__get_max_points(est[0, 1], est[0, 0])
        assert torch.allclose(aio_points, original_points)

    def test_line_loss(self):
        delta = 0.0001
        est = torch.tensor([[0., 0.]]).repeat((2, 1)) + delta
        gt = torch.tensor([[1., 0.]]).repeat((2, 1)) - delta
        loss_aio = self.line_loss_aio.get_loss(est, gt)

        loss = 0.
        for batch in range(gt.size(0)):
            loss += self.line_loss.get_loss(est[batch].squeeze(), gt[batch].squeeze())

        self.assertAlmostEqual(loss, loss_aio.sum())

        # weird out of image behaviour
        est = torch.tensor([[100., -.10]])
        gt = torch.tensor([[100., -.10]])
        loss_aio = self.line_loss_aio.get_loss(est, gt)
        loss = self.line_loss.get_loss(est.squeeze(), gt.squeeze())

        est = torch.tensor([[10., 10.]])
        gt = torch.tensor([[10., 10.]])
        loss = self.line_loss.get_loss(est.squeeze(), gt.squeeze())
        est = torch.tensor([[-10., -10.]])
        gt = torch.tensor([[-10., -10.]])
        loss = self.line_loss.get_loss(est.squeeze(), gt.squeeze())



    def test_line_area_loss(self):
        est = torch.tensor([[0., 1.]])
        gt = torch.tensor([[1., -1.]])
        area_squared = self.line_area_loss.get_loss(est, gt)
        self.assertAlmostEqual(area_squared.item(), self._image_size / 3., places=5)



if __name__ == '__main__':
    unittest.main()
