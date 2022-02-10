import torch
import torchvision


class LineSimpleNN(torch.nn.Module):
    def __init__(self, keypoints=64):
        super(LineSimpleNN, self).__init__()
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(num_keypoints=keypoints, pretrained_backbone=False)

    def forward(self, x, points):
        ans = self.model(x, points)
        return ans['keypoints'].view(x.size(0), 2, -1)