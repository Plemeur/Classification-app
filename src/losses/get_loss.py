from torch import nn 

torch_losses_dict = {
    'binary_crossentropy' : nn.BCELoss,
    'categorical_crossentropy' : nn.CrossEntropyLoss, 
}

custom_losses_dict = {}

try:
    all_losses_dict = torch_losses_dict | custom_losses_dict
except:
    all_losses_dict = {**torch_losses_dict, **custom_losses_dict}

def get_loss(loss_name):
    assert (loss_name in all_losses_dict)
    return torch_losses_dict[loss_name]()