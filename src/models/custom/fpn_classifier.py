 
import torch.nn as nn
import torch


class fpnFeatureExtractor(nn.Module):
    def __init__(self, input_depth=3):
        super(fpnFeatureExtractor, self).__init__()
        
        self.stage_1 = nn.Sequential(
            nn.Conv2d(input_depth, 64, kernel_size = (5,5), stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size = (5,5), stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size = (5,5), stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.stage_1_reduction = nn.AdaptiveAvgPool2d((7,7))
        
        self.stage_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size = (3,3), stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size = (3,3), stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size = (3,3), stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.stage_2_reduction = nn.AdaptiveAvgPool2d((7,7))
        
        self.stage_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = (3,3), stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size = (3,3), stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size = (3,3), stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.stage_3_reduction = nn.AdaptiveAvgPool2d((7,7))

        self.final_stage = nn.Sequential(
            nn.Conv2d(896, 1024, kernel_size = 1, stride=1),
            nn.ReLU(),
        )

        
    def forward(self, x):
        x = self.stage_1(x)
        x1 = self.stage_1_reduction(x)

        x = self.stage_2(x)
        x2 = self.stage_2_reduction(x)

        x = self.stage_3(x)
        x = self.stage_3_reduction(x)
        
        x = torch.cat((x1, x2, x), 1)
        
        x = self.final_stage(x)

        return x
    
    
class fpnClassifier(nn.Module):
    def __init__(self, input_depth=3, nb_class=10):
        super(fpnClassifier, self).__init__()
        
        self.feature_extractor = fpnFeatureExtractor(input_depth)
        
        self.classifier = nn.Sequential(
            nn.Linear(1024*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, nb_class),  
        )
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 1024* 7* 7)
        x = self.classifier(x)
        
        return(x)