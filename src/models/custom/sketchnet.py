import torch.nn as nn
import torch.nn.functional as F

class SketchNet(nn.Module):
    # I don't remember the paper, I don't think it was the 2016 sketchnet
    def __init__(self, channel_in=3, nb_class=10):
        super(SketchNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(channel_in, 64, kernel_size=(15,15), stride=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),

            nn.Conv2d(64, 128, kernel_size=(5,5), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),

            nn.Conv2d(128, 256, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, nb_class),
        )

        

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), 256 * 7 * 7)
        x = self.classifier(x)
        return x


class SketchNet2(nn.Module):
    def __init__(self, channel_in, nb_class):
        super(SketchNet2, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(channel_in, 64, kernel_size=(15,15), stride=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),

            nn.Conv2d(64, 128, kernel_size=(5,5), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),

            nn.Conv2d(128, 256, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),

            nn.Conv2d(256, 512, kernel_size=(7,7), stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(),

            nn.Conv2d(512, 512, kernel_size=(1,1), stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(),

            nn.Conv2d(512, 250, kernel_size=(1,1), stride=1, padding=0),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(250, nb_class)
        )

        for layer in self.modules() :
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)


    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), 250)   
        x = self.classifier(x)
        return x
