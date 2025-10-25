import torch
import torch.nn as nn
from torchvision.models import resnet50

# Segmentation Model (i.e., U-Net) Architecture
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        resnet = resnet50(pretrained=True) # Load pre-trained ResNet50
        self.encoder = nn.Sequential(  
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4)
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, num_classes, kernel_size=1))
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
