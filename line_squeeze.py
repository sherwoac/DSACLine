import torch
import torch.nn as nn
import torchvision


class LineSqueezeNN(nn.Module):
    def __init__(self, receptive_field, image_size: int = 64):
        super(LineSqueezeNN, self).__init__()
        self.patch_size = receptive_field / image_size
        self.cnn_model = torchvision.models.squeezenet1_1(pretrained=False, num_classes=image_size * 2)
        self.cnn_model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, self.cnn_model.num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
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

