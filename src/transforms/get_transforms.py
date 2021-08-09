from ast import Dict
from torchvision import transforms

# Transforms available in torchvision. (exlude AutoAugment)
torchvision_transforms_dict = {
    'ToTensor': transforms.ToTensor,
    'PILToTensor' : transforms.PILToTensor,
    'ConvertImageDtype' : transforms.ConvertImageDtype,
    'ToPILImage' : transforms.ToPILImage,
    'Normalize' : transforms.Normalize,
    'Resize' : transforms.Resize,
    'Scale' : transforms.Scale,
    'CenterCrop' : transforms.CenterCrop,
    'Pad' : transforms.Pad,
    'Lambda' : transforms.Lambda,
    'RandomApply' : transforms.RandomApply,
    'RandomChoice' : transforms.RandomChoice,
    'RandomOrder' : transforms.RandomOrder,
    'RandomCrop' : transforms.RandomCrop,
    'RandomHorizontalFlip' : transforms.RandomHorizontalFlip,
    'RandomVerticalFlip' : transforms.RandomVerticalFlip,
    'RandomResizedCrop' : transforms.RandomResizedCrop,
    'RandomSizedCrop' : transforms.RandomSizedCrop,
    'FiveCrop' : transforms.FiveCrop,
    'TenCrop' : transforms.TenCrop,
    'LinearTransformation' : transforms.LinearTransformation,
    'ColorJitter' : transforms.ColorJitter,
    'RandomRotation' : transforms.RandomRotation,
    'RandomAffine' : transforms.RandomAffine,
    'Grayscale' : transforms.Grayscale,
    'RandomGrayscale' : transforms.Grayscale,
    'RandomPerspective' : transforms.RandomPerspective,
    'RandomErasing' : transforms.RandomErasing,
    'GaussianBlur' : transforms.GaussianBlur,
    'InterpolationMode' : transforms.InterpolationMode,
    'RandomInvert' : transforms.RandomInvert,
    'RandomPosterize' : transforms.RandomPosterize,
    'RandomSolarize' : transforms.RandomSolarize,
    'RandomAdjustSharpness' : transforms.RandomAdjustSharpness,
    'RandomAutocontrast' : transforms.RandomAutocontrast,
    'RandomEqualize' : transforms.RandomEqualize,
    }

# Custom transforms
custom_transforms_dict = {}

try: 
    all_transforms_dict = torchvision_transforms_dict | custom_transforms_dict
except:
    all_transforms_dict = {**torchvision_transforms_dict, **custom_transforms_dict}

def get_transforms(training_transform_dict: dict, validation_transform_dict: dict):
    """ Helper to get the transformation for the training."""
    training_transform_list = []
    validation_transform_list = []

    for transform, parameters in training_transform_dict.items():
        assert (transform in all_transforms_dict)
        training_transform_list.append(all_transforms_dict[transform](**parameters))

    for transform, parameters in validation_transform_dict.items():
        assert (transform in all_transforms_dict)
        validation_transform_list.append(all_transforms_dict[transform](**parameters))

    return transforms.Compose(training_transform_list), transforms.Compose(validation_transform_list)