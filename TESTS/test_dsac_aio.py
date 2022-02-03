import unittest

import torch

import dsac
import dsac_aio
import line_loss
import line_loss_aio


class DsacTestCase(unittest.TestCase):
    _point_predictions = torch.tensor([[[ 0.2214,  0.1800,  0.0999,  0.1541,  0.1390,  0.1065,  0.0897,
                                          0.0430,  0.3143,  0.1930,  0.2961,  0.1954,  0.1673,  0.0914,
                                          0.1752,  0.2629,  0.3830,  0.3741,  0.4154,  0.0940,  0.3337,
                                          0.3990,  0.5090,  0.2976,  0.2966,  0.6357,  0.3941,  0.5523,
                                          0.3780,  0.5803,  0.5738,  0.4483,  0.6507,  0.6206,  0.5632,
                                          0.8081,  0.7899,  0.6812,  0.6270,  0.5804,  0.6277,  0.8352,
                                          0.9079,  0.7520,  0.8007,  0.5419,  0.7307,  0.7162,  0.8348,
                                          0.9453,  1.0816,  0.9904,  0.8308,  0.9352,  0.8241,  0.8187,
                                          1.0029,  0.8708,  0.9513,  0.9445,  0.8290,  0.9289,  1.0556,
                                          0.9182],
                                        [ 0.0394,  0.0664,  0.2769,  0.3876,  0.4415,  0.6941,  0.6667,
                                          0.9332,  0.0113,  0.0592,  0.2321,  0.4292,  0.5762,  0.6607,
                                          0.6865,  0.7999, -0.0556,  0.2331,  0.3529,  0.3067,  0.4873,
                                          0.6457,  0.7214,  0.8791,  0.0499,  0.0640,  0.1850,  0.2463,
                                          0.4348,  0.6301,  0.7408,  0.8968, -0.0852,  0.0695,  0.4456,
                                          0.4551,  0.5218,  0.5873,  0.7680,  0.8454,  0.0459,  0.1959,
                                          0.1779,  0.2343,  0.4835,  0.6589,  0.6666,  0.8597,  0.0630,
                                          -0.0465,  0.0769,  0.5364,  0.5061,  0.5354,  0.7694,  0.7731,
                                          0.0131,  0.1576,  0.3364,  0.3239,  0.4353,  0.6124,  0.6961,
                                          0.8364]]])
    _labels = torch.tensor([[[[-0.0877]],
                             [[ 3.1538]]]])

    _exp_loss_result = torch.tensor(62.3388)
    _top_loss_result = torch.tensor(25.2508)

    _slope = torch.tensor(2.1234)
    _intercept = torch.tensor(-0.7880)
    _points_x = torch.tensor([ 0.0597,  0.1829,  0.2722,  0.6091,  0.4831,  0.6739,  0.7426,  0.9503,
                               -0.0087,  0.0391,  0.1209,  0.3860,  0.3571,  0.6611,  0.6991,  0.8302,
                               -0.0225,  0.0714,  0.3123,  0.3606,  0.3823,  0.6101,  0.7975,  0.8516,
                               0.1384,  0.1822,  0.2737,  0.3727,  0.3651,  0.7228,  0.8136,  0.9589,
                               0.1144,  0.0512,  0.1797,  0.4066,  0.3779,  0.6392,  0.8607,  0.9635,
                               -0.1710,  0.0915,  0.1454,  0.2500,  0.4996,  0.6582,  0.7933,  0.9549,
                               0.0295,  0.1157,  0.2576,  0.3977,  0.4471,  0.5728,  0.7970,  0.9350,
                               0.0565,  0.1139,  0.1719,  0.3228,  0.5531,  0.5768,  0.7088,  0.9071])

    _points_y = torch.tensor([0.0560, 0.1761, 0.0716, 0.0778, 0.2379, 0.0803, 0.0585, 0.0912, 0.2260,
                              0.1929, 0.1878, 0.0422, 0.2850, 0.2394, 0.2641, 0.2776, 0.4066, 0.3047,
                              0.4278, 0.2939, 0.2803, 0.0998, 0.3693, 0.3858, 0.4621, 0.3063, 0.3519,
                              0.3583, 0.5108, 0.4289, 0.6132, 0.5455, 0.5558, 0.5095, 0.6768, 0.5218,
                              0.6347, 0.5175, 0.7117, 0.5952, 0.6491, 0.6954, 0.6653, 0.8112, 0.6805,
                              0.6184, 0.7313, 0.8033, 0.8294, 0.8710, 0.8201, 0.9814, 0.8463, 0.7332,
                              0.9044, 0.8584, 0.9803, 1.0420, 1.1162, 0.9322, 0.9455, 0.8842, 1.0407,
                              0.9599])

    _score = torch.tensor(5.0986)
    _inliers = torch.tensor([0.0000e+00, 0.0000e+00, 9.1267e-04, 1.7881e-06, 9.9328e-01, 0.0000e+00,
                             0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 9.8954e-01,
                             2.2262e-04, 1.6093e-05, 1.4305e-06, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                             0.0000e+00, 2.0915e-04, 2.6529e-03, 4.2915e-06, 0.0000e+00, 0.0000e+00,
                             0.0000e+00, 0.0000e+00, 0.0000e+00, 4.0174e-05, 0.0000e+00, 1.9461e-04,
                             1.3554e-04, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 8.3447e-07,
                             0.0000e+00, 9.4235e-01, 1.2708e-04, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                             0.0000e+00, 0.0000e+00, 4.2915e-06, 9.9030e-01, 1.1525e-01, 1.3113e-06,
                             0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 3.3832e-04,
                             9.9329e-01, 7.9274e-05, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                             0.0000e+00, 8.3447e-07, 1.5247e-04, 6.9531e-02])

    _repeat_batches = 16
    _repeat_hypothesis = 32


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # one rng
        self.rng = torch.Generator()
        self.rng.manual_seed(0)

        self.loss_function = line_loss.LineLoss(64)
        self.loss_function_aio = line_loss_aio.LineLossAio(64)
        self.dsac_instance = dsac.DSAC(64, 0.05, 100, 0.5, self.loss_function, random_generator=self.rng)
        self.dsac_aio_instance = dsac_aio.DsacAio(64, 0.05, 100, 0.5, self.loss_function_aio, random_generator=self.rng)

    def test_losses(self):
        _exp_loss, _top_loss = self.dsac_instance.calculate_loss(DsacTestCase._point_predictions, DsacTestCase._labels)

        # test original result
        self.assertAlmostEqual(_exp_loss.item(),
                               DsacTestCase._exp_loss_result.item(),
                               places=5,
                               msg="exp_loss changed from canned result")
        self.assertAlmostEqual(DsacTestCase._top_loss_result.item(),
                               _top_loss.item(),
                               places=4,
                               msg="_top_loss changed from canned result")

    def test_losses(self):
        _points = DsacTestCase._point_predictions
        _exp_loss, _top_loss = self.dsac_instance.calculate_loss(DsacTestCase._point_predictions, DsacTestCase._labels)
        _exp_loss_aio, _top_loss_aio = self.dsac_aio_instance.calculate_loss(_points.repeat(self._repeat_batches, 1, 1),
                                                                     DsacTestCase._labels.repeat(self._repeat_batches, 1, 1, 1))
        print(_exp_loss, _top_loss)

    def test_aio_losses(self):
        _points = DsacTestCase._point_predictions
        # _exp_loss, _top_loss = self.dsac_instance.calculate_loss(DsacTestCase._point_predictions, DsacTestCase._labels)
        fake_points = torch.tensor([[[0., 1.], [0., 1.]]])
        fake_labels = torch.tensor([[0.], [1.]]).reshape((1, 2, 1, 1))

        saved_hyps = self.dsac_aio_instance.hyps
        self.dsac_aio_instance.hyps = 1
        _exp_loss_aio, _top_loss_aio = self.dsac_aio_instance.calculate_loss(fake_points,
                                                                             fake_labels)
        self.dsac_aio_instance.hyps = saved_hyps

        saved_hyps = self.dsac_instance.hyps
        self.dsac_instance.hyps = 1
        _exp_loss, _top_loss = self.dsac_instance.calculate_loss(fake_points,
                                                                 fake_labels)
        self.dsac_instance.hyps = saved_hyps

        print(_exp_loss_aio, _exp_loss)

    def test_original_loss(self):
        fake_points = torch.tensor([[[0., 1.], [0., 1.]]])
        fake_labels = torch.tensor([[0.], [1.]]).reshape((1, 2, 1, 1))
        saved_hyps = self.dsac_instance.hyps
        self.dsac_instance.hyps = 1
        _exp_loss, _top_loss = self.dsac_instance.calculate_loss(fake_points,
                                                                fake_labels)
        self.dsac_instance.hyps = saved_hyps

    def test_sample_hypothesis(self):
        _test_slope = torch.tensor([-0.2246])
        _test_intercept = torch.tensor([0.9093])
        slope, intercept, valid = self.dsac_instance._sample_hyp(DsacTestCase._point_predictions[0, 1],
                                                                 DsacTestCase._point_predictions[0, 0])

        self.assertTrue(valid, f'invalid returned from self.dsac_instance._sample_hyp')

        self.assertAlmostEqual(slope.item(),
                               _test_slope.item(),
                               places=4,
                               msg=f'slope: {slope}, _test_slope.item(): {_test_slope.item()}')

        self.assertAlmostEqual(intercept.item(),
                               _test_intercept.item(),
                               places=4,
                               msg=f'intercept: {intercept}, _test_intercept: {_test_intercept}')

        x_points_repeated_b_h = DsacTestCase._point_predictions[:, 1].repeat((DsacTestCase._repeat_batches, 1))
        y_points_repeated_b_h = DsacTestCase._point_predictions[:, 0].repeat((DsacTestCase._repeat_batches, 1))
        slope, intercept = self.dsac_aio_instance._sample_hyp(x_points_repeated_b_h, y_points_repeated_b_h)

        self.assertEqual(
            slope.size(),
            (DsacTestCase._repeat_batches, DsacTestCase._point_predictions.size(-1)),
            msg= f'size incorrect: slope: {slope.size()}, '
                 f'_test: {(DsacTestCase._repeat_batches, DsacTestCase._point_predictions.size(-1))}')

        self.assertEqual(
            intercept.size(),
            (DsacTestCase._repeat_batches, DsacTestCase._point_predictions.size(-1)),
            msg=f'size incorrect: slope: {intercept.size()}, '
                f'_test: {(DsacTestCase._repeat_batches, DsacTestCase._point_predictions.size(-1))}')

    def test_distance_hypothesis(self):
        score, inliers = self.dsac_instance._soft_inlier_count(DsacTestCase._slope,
                                                               DsacTestCase._intercept,
                                                               DsacTestCase._points_x,
                                                               DsacTestCase._points_y)

        self.assertAlmostEqual(
            DsacTestCase._score.item(),
            score.item(),
            places=4,
            msg=f'DsacTestCase._score + score unequal: {DsacTestCase._score.item()} + {score.item()}')
        assert torch.allclose(DsacTestCase._inliers, inliers, rtol=0.001), \
            f'dsac_instance._inliers + inliers unequal: {DsacTestCase._inliers} + {inliers}'

        scores, inliersa = self.dsac_aio_instance._soft_inlier_count(
            DsacTestCase._slope.repeat(2),
            DsacTestCase._intercept.repeat(2),
            DsacTestCase._points_x,
            DsacTestCase._points_y)

        self.assertAlmostEqual(
            scores[0].item(),
            scores[1].item(),
            places=4,
            msg=f'dsac_aio_instance._score[0] + dsac_aio_instance._score[1]: '
                f'{scores[0].item()} + {scores[1].item()}')

        self.assertAlmostEqual(
            scores[0].item(),
            DsacTestCase._score.item(),
            places=4,
            msg=f'dsac_aio_instance._score[0] + DsacTestCase._score: '
                f'{scores[0].item()} + {DsacTestCase._score.item()}')

        assert torch.allclose(inliersa, inliers)

        # now for batches x hypothesis x points
        num_batches = 10
        num_hypotheses = 4
        num_points = DsacTestCase._points_x.size()[0]

        scores_b_h, inliers_b_h_p = self.dsac_aio_instance._soft_inlier_count(
            DsacTestCase._slope.repeat((num_batches, num_hypotheses)),
            DsacTestCase._intercept.repeat((num_batches, num_hypotheses)),
            DsacTestCase._points_x.repeat((num_batches, 1)),
            DsacTestCase._points_y.repeat((num_batches, 1)))

        assert scores_b_h.size() == torch.Size((num_batches, num_hypotheses)), f"scores_b_h.size: {scores_b_h.size()} {torch.Size((num_batches, num_hypotheses))}"
        assert inliers_b_h_p.size() == torch.Size((num_batches, num_hypotheses, num_points)), f"inliers_b_h_p.size: {inliers_b_h_p.size()} {(num_batches, num_hypotheses, num_points)}"
        assert (scores_b_h == scores_b_h[0, 0]).all().item(), f'scores_b_h unequal: {scores_b_h}'
        assert (inliers_b_h_p == inliers_b_h_p[0, 0]).all().item(), f'scores_b_h unequal: {scores_b_h}'

        self.assertAlmostEqual(
            scores_b_h[0, 0].item(),
            DsacTestCase._score.item(),
            places=4,
            msg=f'scores_b_h[0, 0] + DsacTestCase._score: '
                f'{scores_b_h[0, 0].item()} + {DsacTestCase._score.item()}')

        assert torch.allclose(inliers_b_h_p[0, 0], DsacTestCase._inliers, atol=0.0001)

    def test_refine_hypothesis(self):

        x = torch.tensor([0.1085, 0.2242, 0.3244, 0.4015, 0.5896, 0.6062, 0.7719, 0.9013, 0.0842,
                          0.1589, 0.3319, 0.4596, 0.5471, 0.7442, 0.8406, 0.8813, 0.1131, 0.1624,
                          0.3421, 0.4423, 0.4547, 0.7798, 0.9401, 0.8223, 0.0788, 0.2216, 0.2175,
                          0.4476, 0.5700, 0.6283, 0.8036, 0.9616, 0.0358, 0.1733, 0.3602, 0.4845,
                          0.5861, 0.6745, 0.8171, 0.9908, 0.0822, 0.2495, 0.3654, 0.4227, 0.6133,
                          0.6618, 0.7073, 0.9520, 0.0226, 0.1776, 0.3185, 0.3877, 0.5472, 0.6695,
                          0.8422, 0.9302, 0.0606, 0.1605, 0.2856, 0.4396, 0.6163, 0.6603, 0.7671,
                          0.9157])

        y = torch.tensor([0.0861, 0.1257, 0.0515, 0.1268, 0.0770, 0.0997, 0.0592, 0.0958, 0.2231,
                          0.2381, 0.2704, 0.2795, 0.2634, 0.2071, 0.2360, 0.1441, 0.3529, 0.4061,
                          0.3052, 0.2759, 0.3166, 0.3056, 0.3228, 0.3671, 0.4645, 0.4836, 0.5504,
                          0.4749, 0.4456, 0.5506, 0.4487, 0.3892, 0.6053, 0.6116, 0.5717, 0.6192,
                          0.6300, 0.6466, 0.6653, 0.6228, 0.7433, 0.7244, 0.7168, 0.7845, 0.7109,
                          0.7292, 0.8752, 0.7742, 0.8426, 0.8965, 0.8667, 0.8021, 0.8961, 0.8720,
                          0.9262, 0.9114, 0.9545, 1.0058, 0.9443, 0.9712, 0.9435, 0.9914, 0.9653,
                          0.9858])

        weights = torch.tensor([0.0000e+00, 2.0504e-05, 4.9069e-03, 9.9331e-01, 8.2016e-05, 6.5565e-06,
                                0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0848e-05, 9.8987e-01, 7.6473e-04,
                                7.1526e-07, 0.0000e+00, 0.0000e+00, 0.0000e+00, 3.6836e-05, 3.2816e-02,
                                8.8804e-01, 4.1499e-03, 2.1392e-04, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                                2.9665e-04, 9.8973e-01, 8.6490e-01, 2.3842e-07, 0.0000e+00, 0.0000e+00,
                                0.0000e+00, 0.0000e+00, 4.1988e-03, 9.5099e-01, 7.6294e-06, 0.0000e+00,
                                0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 9.9331e-01, 1.2577e-04,
                                0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                                9.8623e-01, 2.6941e-05, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                                0.0000e+00, 0.0000e+00, 5.7524e-02, 8.3447e-07, 0.0000e+00, 0.0000e+00,
                                0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00])

        # now for batches x hypothesis x points
        num_batches = 10
        num_hypotheses = 4
        num_points = DsacTestCase._points_x.size()[0]

        slope, intercept = self.dsac_instance._refine_hyp(x, y, weights)
        slope_aio, intercept_aio = self.dsac_aio_instance._refine_hyp(
            x.repeat((num_batches, 1)),
            y.repeat((num_batches, 1)),
            weights.repeat((num_batches, num_hypotheses, 1)))

        self.assertAlmostEqual(slope, slope_aio[0, 0])
        self.assertAlmostEqual(intercept, intercept_aio[0, 0])


if __name__ == '__main__':
    unittest.main()
