import torch
import torchvision


def prepare_data(opt, inputs, labels):
    # convert from numpy images to normalized torch arrays

    inputs = torch.tensor(inputs, requires_grad=True)
    labels = torch.tensor(labels, requires_grad=True)

    if not opt.cpu:
        inputs = inputs.cuda()

    inputs.transpose_(1, 3).transpose_(2, 3)
    inputs = inputs - 0.5 # normalization

    return inputs, labels


def make_grid(batch):
    batch = torch.from_numpy(batch)
    batch.transpose_(1, 3).transpose_(2, 3)
    return torchvision.utils.make_grid(batch, nrow=3,normalize=False)


def get_sid(opt):
    return 'rf%d_c%d_h%d_t%.2f%s' % (opt.receptivefield, opt.capacity, opt.hypotheses, opt.inlierthreshold, opt.session)