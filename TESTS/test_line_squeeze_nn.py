import unittest
import torch
import line_squeeze
import line_squeeze_fire
import line_simple_nn
import line_dataset
import line_nn

import main
import utils
import dsac_aio
from line_loss_aio import LineLossAio


class MyTestCase(unittest.TestCase):
    def test_network(self, point_nn):
        opt = main.parser.parse_args()
        original_loss_function = LineLossAio(image_size=opt.imagesize)
        dsac = dsac_aio.DsacAio(opt.hypotheses, opt.inlierthreshold, opt.inlierbeta, opt.inlieralpha, original_loss_function)
        dataset = line_dataset.LineDataset(opt.imagesize, opt.imagesize, max_sample_size=opt.batchsize)
        val_images, val_labels = dataset.sample_lines(opt.valsize)
        val_inputs, val_labels = utils.prepare_data(opt, val_images, val_labels)
        if not opt.cpu:
            point_nn = point_nn.cuda()

        point_nn.train()
        opt_point_nn = torch.optim.Adam(point_nn.parameters(), lr=opt.learningrate)
        opt_point_nn.zero_grad()
        point_prediction = point_nn(val_inputs)
        exp_loss, top_loss = dsac.calculate_loss(point_prediction, val_labels.cuda())
        exp_loss.backward()		# calculate gradients (pytorch autograd)

        opt_point_nn.step()

        self.assertEqual(True, True)

    def test_LineSqueezeNN(self):
        opt = main.parser.parse_args()
        point_nn = line_squeeze.LineSqueezeNN(receptive_field=opt.receptivefield, image_size=opt.imagesize)
        self.test_network(point_nn)
        self.assertEqual(True, True)  # ran end to end

    def test_LineNN(self):
        opt = main.parser.parse_args()
        point_nn = line_nn.LineNN(opt.capacity, receptive_field=opt.receptivefield, image_size=opt.imagesize)
        self.test_network(point_nn)
        self.assertEqual(True, True)  # ran end to end

    def test_LineSqueezeFireNN(self):
        opt = main.parser.parse_args()
        point_nn = line_squeeze_fire.LineSqueezeFireNN(receptive_field=opt.receptivefield, image_size=opt.imagesize)
        self.test_network(point_nn)

    def test_LineSimpleNN(self):
        def sample_line(val_labels, keypoints=64, image_size=64):
            batch_size = val_labels.size()[0]
            samples_x = torch.rand(size=(batch_size, keypoints)) * image_size
            samples_y = val_labels[:, 1, 0] * samples_x + val_labels[:, 0, 0]
            return torch.stack((samples_x, samples_y), dim=-1)

        point_nn = line_simple_nn.LineSimpleNN()
        opt = main.parser.parse_args()
        original_loss_function = LineLossAio(image_size=opt.imagesize)
        dsac = dsac_aio.DsacAio(opt.hypotheses, opt.inlierthreshold, opt.inlierbeta, opt.inlieralpha, original_loss_function)
        dataset = line_dataset.LineDataset(opt.imagesize, opt.imagesize, max_sample_size=opt.batchsize)
        val_images, val_labels = dataset.sample_lines(opt.valsize)
        val_inputs, val_labels = utils.prepare_data(opt, val_images, val_labels)
        if not opt.cpu:
            point_nn = point_nn.cuda()

        point_nn.train()
        opt_point_nn = torch.optim.Adam(point_nn.parameters(), lr=opt.learningrate)
        opt_point_nn.zero_grad()
        keypoints = {'boxes': None}
        keypoints['keypoints'] = sample_line(val_labels)
        point_prediction = point_nn(val_inputs, keypoints)
        exp_loss, top_loss = dsac.calculate_loss(point_prediction, val_labels.cuda())
        exp_loss.backward()		# calculate gradients (pytorch autograd)

        opt_point_nn.step()

        self.test_network(point_nn)



if __name__ == '__main__':
    unittest.main()
