## About this repository

This is a base code for classification project using pytorch, I keep it more or less vanilla because I feel like package such as Fastai or Pytorch Lightning are too complicated for the task I am trying to accomplish here. If you are working with multiple GPUs or maybe with TPUs, you might want to check them out rather than being here. 

## Code structure 
```
├───notebooks # Contains notebook for development mainly, some quick tests
├───parameters # Contains the parameters file for the experiments
├───src # Contains all the core code 
│   ├───models # (will) Contain custom models and helper function related to models
│   ├───optimizer # 
│   ├───transforms
│   ├───scheduler #
│   ├───losses #
│   ├───trainer # Only contains the trainer class, that handle the training steps.
└───tests
```

## How to use 
Usage is made to be simple, you just need to write the parameters that you want to use in a json file, and run the following command

```
python train.py <path-to-your-parameters-json>
```

You cna check the existing `parameters.json` file to see how it is done, and also refer to the `get_XXX.py` files to see available methods, as well as the pytorch documentation to see the parameters each function takes.

## How to improve 
I made this code to be (in my opinion) easily upgradable. 
To add any missing function, models, sechduler etc...  One needs to write the method in a new python file and import it in the corresponding `get_XXX.py` file, then add it to the `custom_XXX_dict`. If the pytorch format is properly respected, it should integrate simply with the rest of the code and run without any issues.

## TODOs
- [ ] Add a validation section 
    - [ ] Give sample of images that are wrongly classified
    - [ ] Give metrics graphs 
- [ ] Write comments
- [ ] Add tests
- [ ] Make a simple frontend
    - [ ] Create a parameters parser
    - [ ] make a link to tensorboard ?
