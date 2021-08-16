from torch import optim
from torch import nn

torch_optim_dict = {
    'Adadelta' : optim.Adadelta,
    'Adagrad' : optim.Adagrad, 
    'Adam' : optim.Adam, 
    'AdamW' : optim.AdamW, 
    'SparseAdam' : optim.SparseAdam, 
    'Adamax'  : optim.Adamax, 
    'ASGD' : optim.ASGD, 
    'SGD' : optim.SGD, 
    'Rprop' : optim.Rprop, 
    'RMSprop' : optim.RMSprop,  
    'LBFGS' : optim.LBFGS
}

custom_optim_dict = {}

try:
    all_optim_dict = torch_optim_dict | custom_optim_dict
except:
    all_optim_dict = {**torch_optim_dict, **custom_optim_dict}

def get_optimizer(optimizer_name:str, optimizer_parameters:dict, model:nn.Module):
    assert optimizer_name in all_optim_dict
    optimizer = all_optim_dict[optimizer_name](params = model.parameters(), **optimizer_parameters)
    return optimizer
