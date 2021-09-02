import json
import argparse

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from src.losses.get_loss import get_loss
from src.models.get_model import get_model
from src.optimizers.get_optimizer import get_optimizer
from src.transforms.get_transforms import get_transforms
from src.schedulers.get_scheduler import get_scheduler
from src.trainer.trainer import Trainer
from src.evaluator.evaluator import Evaluator 

parser = argparse.ArgumentParser(description='Train a model using a parameter file')
parser.add_argument('parameters', metavar='p', type=str,
                    help='Path to the parameter file')
args = parser.parse_args()


def train(parameters:dict):

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

    # Load the model
    model = get_model(
        model_name = parameters['model_name'],
        pretrained = parameters['pretrained'],
        n_class = len(val_data.classes)
    )

    # Load the optimizer and loss function
    optimizer = get_optimizer(
        optimizer_name=parameters['optimizer_name'],
        optimizer_parameters=parameters['optimizer_parameters'],
        model = model
    )

    loss = get_loss(loss_name=parameters['loss_name'])

    scheduler = get_scheduler(
        scheduler_name=parameters['scheduler_name'],
        scheduler_parameters=parameters['scheduler_parameters'], 
        optimizer = optimizer
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = Trainer(
        model = model,
        optimizer = optimizer, 
        scheduler= scheduler,
        loss_function= loss,
        trainloader = train_loader,
        valloader = val_loader,
        device=device,
        parameters = parameters
    )

    #trainer.run_training()

    if parameters.get('evaluate', False):
        try:
            test_data = ImageFolder(
                f"{parameters['dataset_path']}/test", 
                transform = val_transform
                )
            assert len(test_data) > 0
        except:
            print("No test dataset, evaluating on validation dataset")
            test_data = ImageFolder(
                f"{parameters['dataset_path']}/val", 
                transform = val_transform
                )

        test_loader = DataLoader(
            test_data, 
            shuffle = False,
            batch_size=parameters['batch_size'],
            num_workers=parameters['num_workers'],
            pin_memory=True
            )

        evaluator = Evaluator(
            model = model,
            testloader = test_loader, 
            device = device,
            parameters=parameters,
            writer = trainer.writer
            )
        
        evaluator.run_evaluation()


if __name__=='__main__':
    # Read the parameter file.
    with open(args.parameters, 'r') as f:
        parameters = json.load(f)

    train(parameters)