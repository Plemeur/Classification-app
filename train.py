import json
import argparse
from unittest import main

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataloader

from src.losses.get_loss import get_loss
from src.models.get_model import get_model
#from src.optimizers.get_optimizer import get_optimizer
from src.transforms.get_transforms import get_transforms

parser = argparse.ArgumentParser(description='Train a model using a parameter file')
parser.add_argument('parameters', metavar='p', type=str,
                    help='Path to the parameter file')
args = parser.parse_args()


if __name__ == '__main__':
    # Read the parameter file.
    with open(args.parameters, 'r') as f:
        parameters = json.load(f)

    # Create the transform items
    train_transform, val_transform = get_transforms(
        parameters['training_transforms'], 
        parameters['validation_transforms']
        )

    # Get the dataloader
    train_data = ImageFolder(
        f"{parameters['dataset_path']}/train", 
        transform = train_transform
        )
    val_data = ImageFolder(
        f"{parameters['dataset_path']}/val", 
        transform = val_transform
        )

    train_loader = DataLoader(
        train_data,
        shuffle = True,
        batch_size=parameters['batch_size'],
        num_workers=parameters['num_workers'],
        pin_memory=True
        )
    val_loader = DataLoader(
        val_data, 
        shuffle = False,
        batch_size=parameters['batch_size'],
        num_workers=parameters['num_workers'],
        pin_memory=True
        )

    print(parameters)
    
