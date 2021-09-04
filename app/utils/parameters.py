import json
from torchvision.transforms.functional import _interpolation_modes_from_int

inter2int = {}
for i in range(6):
    inter2int[str(_interpolation_modes_from_int(i))] = i


def parse_parameters(request):
    parameters = json.loads(request)
    print(json.dumps(parameters, indent=4))
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


        # TODO : Allow for more than 10 tranforms
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
    
    print(json.dumps(parameters, indent=4))

    return parameters