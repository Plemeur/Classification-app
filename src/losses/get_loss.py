from torch import nn 
from .custom import *

torch_losses_dict = {
    'binary_crossentropy' : nn.BCELoss,
    'categorical_crossentropy' : nn.CrossEntropyLoss, 
}

custom_losses_dict = {
    'label_smoothing' : LabelSmoothing,
    'focal_loss' : FocalLoss
}

try:
    all_losses_dict = torch_losses_dict | custom_losses_dict
except:
    all_losses_dict = {**torch_losses_dict, **custom_losses_dict}

def get_loss(loss_name:str, loss_parameters:dict):
    assert (loss_name in all_losses_dict)
    return all_losses_dict[loss_name](**loss_parameters)