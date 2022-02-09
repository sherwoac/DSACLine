import unittest
import torch
import line_squeeze
import line_squeeze_fire
import line_dataset
import main
import utils
import dsac_aio
from line_loss_aio import LineLossAio


class MyTestCase(unittest.TestCase):
    def test_forward(self):
        opt = main.parser.parse_args()
        original_loss_function = LineLossAio(image_size=opt.imagesize)
        dsac = dsac_aio.DsacAio(opt.hypotheses, opt.inlierthreshold, opt.inlierbeta, opt.inlieralpha, original_loss_function)
        point_nn = line_squeeze.LineSqueezeNN(receptive_field=opt.receptivefield, image_size=opt.imagesize)
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

        self.assertEqual(True, True)  # ran end to end

    def test_forward_fire(self):
        opt = main.parser.parse_args()
        original_loss_function = LineLossAio(image_size=opt.imagesize)
        dsac = dsac_aio.DsacAio(opt.hypotheses, opt.inlierthreshold, opt.inlierbeta, opt.inlieralpha, original_loss_function)
        point_nn = line_squeeze_fire.LineSqueezeFireNN(receptive_field=opt.receptivefield, image_size=opt.imagesize)
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

        self.assertEqual(True, True)  # ran end to end

if __name__ == '__main__':
    unittest.main()
