import os
import warnings
import time
import sys

import torch
import torchvision
import imageio

import matplotlib.pyplot as plt


# local
import utils
from dsac_aio import DsacAio
from line_dataset import LineDataset
from line_nn import LineNN
from line_area_loss import LineLossArea
from line_loss import LineLoss
from line_loss_aio import LineLossAio


def validation(opt):
    # reload model
    sid = utils.get_sid(opt)
    weights_filename = os.path.join(opt.output_dir, f'weights_pointnn_{sid}.net')
    assert os.path.isfile(weights_filename), f'need: {weights_filename} for validation'
    point_nn = LineNN(opt.capacity, opt.receptivefield)
    point_nn.load(weights_filename)

    # setup the training process
    dataset = LineDataset(opt.imagesize, opt.imagesize)

    val_images, val_labels = dataset.sample_lines(opt.valsize)
    val_inputs, val_labels = utils.prepare_data(opt, val_images, val_labels)

    # loss_function = LineLossArea(opt.imagesize)
    loss_function = LineLossAio(image_size=opt.imagesize)
    dsac = DsacAio(opt.hypotheses, opt.inlierthreshold, opt.inlierbeta, opt.inlieralpha, loss_function)

    if not opt.cpu:
        point_nn = point_nn.cuda()
        val_labels = val_labels.cuda()

    # infer
    val_prediction = point_nn(val_inputs)
    _, _ = dsac.calculate_loss(val_prediction, val_labels)
    val_correct = dsac.est_losses < opt.valthresh
    dsac_val_est = dsac.est_parameters.cpu()
    points = val_prediction.cpu()

    # plot output
    viz_dsac = dataset.draw_models(val_labels)
    viz_dsac = dataset.draw_points(points, viz_dsac, dsac.batch_inliers)
    viz_dsac = dataset.draw_models(dsac_val_est, viz_dsac, val_correct)

    viz_inputs = utils.make_grid(val_images)
    viz_dsac = utils.make_grid(viz_dsac)

    viz = torch.cat((viz_inputs, viz_dsac, viz_dsac), 2)
    # viz = torch.cat((viz_inputs, viz_dsac, viz_direct), 2)
    viz.transpose_(0, 1).transpose_(1, 2)
    viz = viz.numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outfolder = 'images_' + sid
        os.makedirs(outfolder, exist_ok=True)
        filename = f'{opt.output_dir}/{outfolder}/validation_output_{sid}.png'
        print(f'saving: {filename}')
        imageio.imsave(filename, viz)


