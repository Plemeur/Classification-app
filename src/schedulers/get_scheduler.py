from torch.optim import lr_scheduler
from torch.optim import Optimizer

torch_scheduler_dict = {
    'LambdaLR' : lr_scheduler.LambdaLR, 
    'StepLR' : lr_scheduler.StepLR, 
    'MultiStepLR' : lr_scheduler.MultiStepLR, 
    'ExponentialLR' : lr_scheduler.ExponentialLR, 
    'CosineAnnealingLR' : lr_scheduler.CosineAnnealingLR, 
    'ReduceLROnPlateau' : lr_scheduler.ReduceLROnPlateau, 
    'CyclicLR' : lr_scheduler.CyclicLR, 
    'CosineAnnealingWarmRestarts' : lr_scheduler.CosineAnnealingWarmRestarts, 
    'OneCycleLR' : lr_scheduler.OneCycleLR
}

custom_scheduler_dict = {}

try:
    all_scheduler_dict = torch_scheduler_dict | custom_scheduler_dict
except:
    all_scheduler_dict = {**torch_scheduler_dict, **custom_scheduler_dict}

def get_scheduler(scheduler_name:str, scheduler_parameters:dict, optimizer:Optimizer):
    assert scheduler_name in all_scheduler_dict

    scheduler = all_scheduler_dict[scheduler_name](optimizer, **scheduler_parameters)
    return scheduler