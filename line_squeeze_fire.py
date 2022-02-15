import torch
import torch.nn as nn
import torchvision


class LineSqueezeFireNN(nn.Module):
    def __init__(self, receptive_field, image_size: int = 64):
        super(LineSqueezeFireNN, self).__init__()
        self.patch_size = receptive_field / image_size
        self.cnn_model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, ceil_mode=True),
            torchvision.models.squeezenet.Fire(8, 4, 4, 4),
            nn.Linear(64, 128),
            nn.Sigmoid())

    def forward(self, input_images):
        normalized_output = self.cnn_model(input_images)
        patch_offset = 1. / input_images.size(2)
        x = normalized_output * self.patch_size - 0.5 * self.patch_size + 0.5 * patch_offset

        for col in range(x.size(3)):
            x[:, 1, :, col] = x[:, 1, :, col] + col * patch_offset

        for row in range(x.size(2)):
            x[:, 0, row, :] = x[:, 0, row, :] + row * patch_offset
        print(x.shape)
        return x.view(input_images.size()[0], 2, -1)

