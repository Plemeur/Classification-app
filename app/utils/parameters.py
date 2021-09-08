import json
import os
from torchvision.transforms.functional import _interpolation_modes_from_int

from src.losses.get_loss import get_loss
from src.models.get_model import get_model
from src.optimizers.get_optimizer import get_optimizer
from src.transforms.get_transforms import get_transforms
from src.schedulers.get_scheduler import get_scheduler
from src.trainer.trainer import Trainer
from src.evaluator.evaluator import Evaluator 



inter2int = {}
for i in range(6):
    inter2int[str(_interpolation_modes_from_int(i))] = i


def parse_parameters(request):
    parameters = json.loads(request)
    loss_parameters = {}
    optimizer_parameters = {}
    scheduler_parameters = {}
    training_transforms = {}
    validation_transforms = {}
    training_transform_list = []
    validation_transform_list = []
    pop_list = []
    
    for key, value in parameters.items():
        # Deal with numeric types
        try:
            value = int(value)
        except ValueError:  
            try:
                value = float(value)
            except:
                # Deal with booleans and None
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.lower() == 'none':
                    value = None
                else:
                    try: 
                        for b in ['(', ')', '[', ']']:
                            value = value.replace(b,'')
                        try: 
                            value = [int(x) for x in value.split(',')]
                        except:
                            value = [float(x) for x in value.split(',')]
                    except :
                        try : 
                            value = inter2int[value]
                        except :
                            pass

        parameters[key] = value

        # Create parameters dictionaries
        if "loss_" in key and "name" not in key:
            loss_parameters[key[5:]] = value
            pop_list.append(key)
        
        if "optimizer_" in key and "name" not in key:
            optimizer_parameters[key[10:]] = value
            pop_list.append(key)

        if "scheduler_" in key and "name" not in key:
            scheduler_parameters[key[10:]] = value
            pop_list.append(key)


        # TODO : Allow for more than 10 tranforms, probably need to use regex
        if "transform" in key and value is not None:
            try:
                n = int(key[-1])
                if "val" in key:
                    validation_transforms[value] = {}
                    validation_transform_list.append(value)
                else:
                    training_transforms[value] = {}
                    training_transform_list.append(value)
            except:
                if "val" in key:
                    n = int(key[13])
                    p = key[15:]
                    t = validation_transform_list[n]
                    validation_transforms[t][p] = value
                else:
                    n = int(key[9])
                    p = key[11:]
                    t = training_transform_list[n]
                    training_transforms[t][p] = value

            pop_list.append(key)

    for key in pop_list:
        parameters.pop(key)

    parameters['loss_parameters'] = loss_parameters
    parameters['optimizer_parameters'] = optimizer_parameters
    parameters['scheduler_parameters'] = scheduler_parameters
    parameters['training_transforms'] = training_transforms
    parameters['validation_transforms'] = validation_transforms
    parameters['pretrained'] = parameters.get('pretrained', False)

    return parameters

def validate_parameters(parameters):
    errors = []

    if not os.path.exists(parameters['dataset_path']):
        errors.append('dataset_path')

    try: 
        # Load the model
        model = get_model(
            model_name = parameters['model_name'],
            pretrained = parameters['pretrained'],
            n_class = 2
            )
    except TypeError:
        errors.append('pretrained')
        # Get dummy model to test optim
        model = get_model(
            model_name = 'alexnet',
            pretrained = False,
            n_class = 2
            )

    try: 
        # Load the optimizer and loss function
        optimizer = get_optimizer(
            optimizer_name=parameters['optimizer_name'],
            optimizer_parameters=parameters['optimizer_parameters'],
            model = model
        )
    except TypeError as e:
        errors.append('optimizer_parameters')
        # get dummy optim to test scheduler
        optimizer = get_optimizer(
            optimizer_name='SGD',
            model = model
        )
    
    try: 
        loss = get_loss(loss_name=parameters['loss_name'],
                        loss_parameters=parameters['loss_parameters'])

    except TypeError as e:
        print(e)
        errors.append('loss_parameters')

    try:
        scheduler = get_scheduler(
            scheduler_name=parameters['scheduler_name'],
            scheduler_parameters=parameters['scheduler_parameters'], 
            optimizer = optimizer
        )

    except TypeError as e:
        errors.append('scheduler_parameters')

    return errors