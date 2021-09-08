from torchvision import models
import torch
from .custom import *

# Models included with torchvision
torchvision_models_dict = {
    'alexnet' : models.alexnet,
    'densenet121' : models.densenet121,
    'densenet161' : models.densenet161,
    'densenet169' : models.densenet169,
    'densenet201' : models.densenet201,
    'googlenet' : models.googlenet,
    'inception_v3' : models.inception_v3,
    'resnet18' : models.resnet18,
    'resnet34' : models.resnet34,
    'resnet50' : models.resnet50,
    'resnet101' : models.resnet101,
    'resnet152' : models.resnet152,
    'resnext50_32x4d' : models.resnext50_32x4d,
    'resnext101_32x8d' : models.resnext101_32x8d,
    'wide_resnet50_2': models.wide_resnet50_2,
    'wide_resnet101_2' : models.wide_resnet101_2,
    'vgg11' : models.vgg11,
    'vgg11_bn' : models.vgg11_bn,
    'vgg13' : models .vgg13,
    'vgg13_bn' : models.vgg13_bn,
    'vgg16' : models.vgg16,
    'vgg16_bn' : models.vgg16_bn,
    'vgg19_bn' : models.vgg19_bn,
    'vgg19' : models.vgg19,
    'squeezenet1_0' : models.squeezenet1_0,
    'squeezenet1_1' : models.squeezenet1_1,
    'mobilenet_v2' : models.mobilenet_v2,
    'mobilenet_v3_large' : models.mobilenet_v3_large,
    'mobilenet_v3_small' : models.mobilenet_v3_small,
    'mnasnet0_5' : models.mnasnet0_5,
    'mnasnet0_75' : models.mnasnet0_75,
    'mnasnet1_0' : models.mnasnet1_0,
    'mnasnet1_3' : models.mnasnet1_3,
    'shufflenet_v2_x0_5' : models.shufflenet_v2_x0_5,
    'shufflenet_v2_x1_0' : models.shufflenet_v2_x1_0,
    'shufflenet_v2_x1_5' : models.shufflenet_v2_x1_5,
    'shufflenet_v2_x2_0' : models.shufflenet_v2_x2_0,
    }

# Custom models written
custom_models_dict = {
    'fpnClassifier' : fpnClassifier,
    'sketchnet': SketchNet,
    'sketchnet2' : SketchNet2
}

try: 
    all_models_dict = torchvision_models_dict | custom_models_dict
except:
    print("Python 3.9 is here, update maybe, or you know, don't, i'm not your supervisor")
    all_models_dict = {**torchvision_models_dict, **custom_models_dict}

def get_model(model_name: str, pretrained: bool, n_class: int):
    """ Return a model if it is avaible."""
    assert (model_name in all_models_dict)

    model = all_models_dict[model_name](pretrained=pretrained)

    # Replace the last layer to match the number of class you have
    # some model have a classifier section, other might just have a liner layer
    try : 
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features=in_features, out_features=n_class)
    except AttributeError:
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features=in_features, out_features=n_class)

    return model