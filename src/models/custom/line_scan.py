import torch.nn as nn
import torch

# I want to make a model that would act like a line scanner, might finish it someday 
class Line_scan_features(nn.Module):
    def __init__(self, input_depth):
        super(Line_scan_features, self).__init__()

        self.horizontal_scan = nn.Sequential(
            nn.Conv2d(input_depth, 32, kernel_size = (7,2), stride=(7,1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(input_depth, 32, kernel_size = (7,2), stride=(7,1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(input_depth, 32, kernel_size = (7,2), stride=(7,1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(input_depth, 32, kernel_size = (7,2), stride=(7,1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(input_depth, 32, kernel_size = (7,2), stride=(7,1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )