import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models

class AlexNet(nn.Module):
    def __init__(self, image_size, out_classes):
        super().__init__()
        #self.l1_size = image_size
        self.out_classes = out_classes

        # usual would be kernel size 11, stride 4, padding 0, but we use kernel size 3 instead for CIFAR10 and MNIST
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        #self.l1_size = (self.l1_size - 3 + 2 * 1) // 1 + 1

        self.pooling1 = nn.MaxPool2d(kernel_size=3, stride=2)
        #self.l1_size = (self.l1_size - 3 + 2 * 0) // 2 + 1

        self.lrm1 = nn.LocalResponseNorm(5, 1e-4, 0.75, 2)

        self.conv2 = nn.Conv2d(64, 192, 5, padding=2) # add padding to preserve dimensions according to diagram.
        #self.l1_size = (self.l1_size - 5 + 2 * 2) // 1 + 1

        self.pooling2 = nn.MaxPool2d(kernel_size=3, stride=2)
        #self.l1_size = (self.l1_size - 3 + 2 * 0) // 2 + 1

        self.lrm2 = nn.LocalResponseNorm(5, 1e-4, 0.75, 2)

        self.conv3 = nn.Conv2d(192, 384, 3, padding=1)
        #self.l1_size = (self.l1_size - 3 + 2 * 1) // 1 + 1

        self.conv4 = nn.Conv2d(384, 256, 3, padding=1)
        #self.l1_size = (self.l1_size - 3 + 2 * 1) // 1 + 1

        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        #self.l1_size = (self.l1_size - 3 + 2 * 1) // 1 + 1

        self.pooling3 = nn.MaxPool2d(kernel_size=3, stride=2)
        #self.l1_size = (self.l1_size - 3 + 2 * 0) // 2 + 1

        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.d1 = nn.Dropout()
        self.l1 = nn.Linear(3 * 3 * 256, 4096)
        self.d2 = nn.Dropout()
        self.l2 = nn.Linear(4096, 4096)
        self.l3 = nn.Linear(4096, self.out_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x, True)
        x = self.pooling1(x)
        x = self.lrm1(x)

        x = self.conv2(x)
        x = F.relu(x, True)
        x = self.pooling2(x)
        x = self.lrm2(x)

        x = self.conv3(x)
        x = F.relu(x, inplace=True)

        x = self.conv4(x)
        x = F.relu(x, inplace=True)

        x = self.conv5(x)
        x = F.relu(x, inplace=True)
        x = self.pooling3(x)

        #x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.d1(x)
        x = self.l1(x)
        x = F.relu(x, inplace=True)

        x = self.d2(x)
        x = self.l2(x)
        x = F.relu(x, inplace=True)

        x = self.l3(x)
        return x

def choose_model(original_model_version: bool):
    if not original_model_version:
        return AlexNet(32, 10)
    return models.AlexNet(10)

if __name__ == "__main__":
    # testing code.
    image_size = 32
    model = AlexNet(image_size, 10)
    x = torch.rand(16, 3, image_size, image_size)
    print(model(x).shape)