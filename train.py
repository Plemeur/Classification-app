import json
import argparse
from unittest import main

from src.losses.get_loss import get_loss
from src.models.get_model import get_model
#from src.optimizers.get_optimizer import get_optimizer
from src.transforms.get_transforms import get_transforms

parser = argparse.ArgumentParser(description='Train a model using a parameter file')
parser.add_argument('parameters', metavar='p', type=str,
                    help='Path to the parameter file')
args = parser.parse_args()


if __name__ == '__main__':
    with open(args.parameters, 'r') as f:
        parameters = f.read()

    print(parameters)
    
