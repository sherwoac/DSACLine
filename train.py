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




def batch_loss(loss_function, prediction, labels):
    # calculate the loss for each image in the batch

    losses = torch.zeros(labels.size(0))

    for b in range(0, labels.size(0)):
        losses[b] = loss_function.get_loss(prediction[b], labels[b])

    return losses


def train(opt):

    if opt.set_detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        print(f'torch.autograd.set_detect_anomaly: {True}')

    if len(opt.session) > 0:
        opt.session = '_' + opt.session

    sid = utils.get_sid(opt)
    weights_filename = os.path.join(opt.output_dir, f'weights_pointnn_{sid}.net')

    # setup the training process
    dataset = LineDataset(opt.imagesize, opt.imagesize)

    # loss_function = LineLossArea(opt.imagesize)
    original_loss_function = LineLossAio(image_size=opt.imagesize)
    dsac = DsacAio(opt.hypotheses, opt.inlierthreshold, opt.inlierbeta, opt.inlieralpha, original_loss_function)

    # we train two CNNs in parallel
    # 1) a CNN that predicts points and is trained with DSAC -> PointNN (good idea)
    point_nn = LineNN(opt.capacity, opt.receptivefield)
    if opt.reload:
        print(f'reloading model weights from: {weights_filename}')
        point_nn.load(weights_filename)

    if not opt.cpu:
        point_nn = point_nn.cuda()

    point_nn.train()
    opt_point_nn = torch.optim.Adam(point_nn.parameters(), lr=opt.learningrate)
    # lrs_point_nn = torch.optim.lr_scheduler.StepLR(opt_point_nn, opt.lrstep, gamma=0.5)

    # 2) a CNN that predicts the line parameters directly -> DirectNN (bad idea)
    # direct_nn = LineNN(opt.capacity, 0, True)
    # if not opt.cpu:
    #     direct_nn = direct_nn.cuda()

    # direct_nn.train()
    # opt_direct_nn = torch.optim.Adam(direct_nn.parameters(), lr=opt.learningrate)
    # lrs_direct_nn = torch.optim.lr_scheduler.StepLR(opt_direct_nn, opt.lrstep, gamma=0.5)

    # keep track of training progress
    train_log = open(opt.output_dir + 'log_'+sid+'.txt', 'w', 1)

    # generate validation data (for consistent vizualisation only)
    val_images, val_labels = dataset.sample_lines(opt.valsize)
    val_inputs, val_labels = utils.prepare_data(opt, val_images, val_labels)

    # start training
    for iteration in range(opt.trainiterations+1):
        start_time = time.time()

        # reset gradient buffer
        opt_point_nn.zero_grad()
        # opt_direct_nn.zero_grad()

        # generate training data
        input_images, labels = dataset.sample_lines(opt.batchsize)
        inputs, labels = utils.prepare_data(opt, input_images, labels)

        # point nn forward pass
        point_prediction = point_nn(inputs)

        # robust line fitting with DSAC
        exp_loss, top_loss = dsac.calculate_loss(point_prediction, labels.cuda())
        # exp_loss.register_hook(lambda grad: print(f'grad: {grad}'))
        exp_loss.backward()		# calculate gradients (pytorch autograd)

        assert not exp_loss.isnan().any(), f"exp_loss is nan: {exp_loss}"
        # update parameters
        # if iteration >= opt.lrstepoffset:
        #     lrs_point_nn.step()		# update learning rate schedule
        opt_point_nn.step()
        direct_loss = 0.
        # # also train direct nn
        # direct_prediction = direct_nn(inputs)
        # direct_loss = batch_loss(original_loss_function, direct_prediction, labels.cuda()).mean()
        #
        # direct_loss.backward()			# calculate gradients (pytorch autograd)
        # opt_direct_nn.step()			# update parameters

        # if iteration >= opt.lrstepoffset:
        #     lrs_direct_nn.step()		# update learning rate schedule

        # wrap up
        end_time = time.time()-start_time
        print('Iteration: %6d, DSAC Expected Loss: %2.2f, DSAC Top Loss: %2.2f, Direct Loss: %2.2f, Time: %.2fs'
              % (iteration, exp_loss, top_loss, direct_loss, end_time), flush=True)

        train_log.write('%d %f %f %f\n' % (iteration, exp_loss, top_loss, direct_loss))

        # del exp_loss, top_loss, direct_loss

        # store prediction vizualization and nn weights (each couple of iterations)
        if iteration % int(opt.storeinterval * 32 / opt.batchsize) == 0:

            point_nn.eval()
            # direct_nn.eval()

            # DSAC validation prediction

            # for some reason val_inputs is much smaller than the training batch.
            val_prediction = point_nn(val_inputs)
            if val_prediction.isnan().any():
                val_prediction[val_prediction.isnan()] = 0.
                val_exp, val_loss = 0., 0.
                points = torch.zeros_like(val_prediction)
                # direct_val_est = direct_val_est.cpu().numpy()
            else:
                val_exp, val_loss = dsac.calculate_loss(val_prediction, val_labels.cuda())
                val_correct = dsac.est_losses < opt.valthresh
                dsac_val_est = dsac.est_parameters.cpu()
                points = val_prediction.cpu()

            # direct nn validation prediction
            # direct_val_est = direct_nn(val_inputs)
            # direct_val_loss = batch_loss(original_loss_function, direct_val_est, val_labels)
            # direct_val_correct = direct_val_loss < opt.valthresh
            #
            # direct_val_est = direct_val_est.cpu().numpy()


            # draw DSAC estimates
            viz_dsac = dataset.draw_models(val_labels)
            viz_dsac = dataset.draw_points(points, viz_dsac, dsac.batch_inliers)
            viz_dsac = dataset.draw_models(dsac_val_est, viz_dsac, val_correct)

            # draw direct estimates
            # viz_direct = dataset.draw_models(val_labels)
            # viz_direct = dataset.draw_models(direct_val_est, viz_direct, direct_val_correct)
            viz_inputs = utils.make_grid(val_images)
            viz_dsac = utils.make_grid(viz_dsac)
            # viz_direct = make_grid(viz_direct)

            viz = torch.cat((viz_inputs, viz_dsac, viz_dsac), 2)
            # viz = torch.cat((viz_inputs, viz_dsac, viz_direct), 2)
            viz.transpose_(0, 1).transpose_(1, 2)
            viz = viz.numpy()

            # store image (and ignore warning about loss of precision)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outfolder = 'images_' + sid
                os.makedirs(outfolder, exist_ok=True)
                filename = f'{opt.output_dir}/{outfolder}/prediction_{sid}_{iteration:06d}.png'
                print(f'saving: {filename}')
                imageio.imsave(filename, viz)

            # store model weights
            torch.save(point_nn.state_dict(), weights_filename)
            # torch.save(direct_nn.state_dict(), opt.outdir + 'weights_directnn_' + sid + '.net')

            print('Storing snapshot. Validation loss: %2.2f' % val_loss, flush=True)

            del val_exp, val_loss
            # del direct_val_loss

            point_nn.train()
            # direct_nn.train()

    print('Done without errors.')
    train_log.close()
