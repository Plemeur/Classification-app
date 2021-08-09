## About this repository

This is a base code for classification project using pytorch, I keep it more or less vanilla because I feel like package such as Fastai or Pytorch Lightning are too complicated for the task I am trying to accomplish here. If you are working with multiple GPUs or maybe with TPUs, you might want to check them out rather than being here. 

## Code structure 
```
├───notebooks # Contains notebook for development mainly, some quick tests
├───parameters # Contains the parameters file for the experiments
├───src # Contains all the core code 
│   ├───models # (will) Contain custom models and helper function related to models
│   ├───trainer # 
│   └───transforms
└───tests
```

## TODOs

- [ ] Write the trainer class
- [ ] Add the optimizer selection
- [ ] Add the scheduler selection
- [ ] Add a validation section 
    - [ ] Give sample of images that are wrongly classified
    - [ ] Give metrics graphs 
 