import torch
import torch.nn as nn
import torchvision


class LineSqueezeFireNN(nn.Module):
    def __init__(self, receptive_field, image_size: int = 64):
        super(LineSqueezeFireNN, self).__init__()
        self.patch_size = receptive_field / image_size
        self.cnn_model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            torchvision.models.squeezenet.Fire(64, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(128, image_size * 2, kernel_size=7),
            nn.Sigmoid())

    def forward(self, input_images):
        normalized_output = self.cnn_model(input_images).view(input_images.size()[0], 2, 8, 8)
        patch_offset = 1. / input_images.size(2)
        x = normalized_output * self.patch_size - 0.5 * self.patch_size + 0.5 * patch_offset

        for col in range(x.size(3)):
            x[:, 1, :, col] = x[:, 1, :, col] + col * patch_offset

        for row in range(x.size(2)):
            x[:, 0, row, :] = x[:, 0, row, :] + row * patch_offset

        return x.view(input_images.size()[0], 2, -1)

