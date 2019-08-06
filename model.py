import torch
import torch.nn as nn
import torchvision.models as models

class Resnet(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        model = models.resnet18(pretrained=False)
        model = list(model.children())[:-1]
        model.append(nn.Conv2d(512, out_size, 1))
        self.net = nn.Sequential(*model)

    def forward(self, image):
        return self.net(image).squeeze(-1).squeeze(-1)

