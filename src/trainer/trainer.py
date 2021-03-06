import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Trainer():
    def __init__(self, model = None, optimizer = None, scheduler = None, loss_function = None,
                trainloader = None, valloader = None, log_interval= 1000, 
                start_epoch = 0, last_epoch = 10, device = 'cpu', parameters=None):

        # Pytorch items
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loader = trainloader
        self.val_loader = valloader
        self.loss_function = loss_function
        self.log_interval = parameters.get('log_interval', len(self.train_loader)*0.2)
        self.epoch = parameters.get('first_epoch', 0)
        self.last_epoch = parameters.get('last_epoch', 10)
        self.best_accuracy = 0

        self.saving_path = parameters.get('saving_path', parameters['model_name'])
        self.writer = SummaryWriter(self.saving_path)
        self.device = device

    def train(self):
        self.model.train()
        train_loss = 0.0
        train_acc = 0.0
        running_loss = 0.0
        running_acc = 0.0
        for batch_idx, (data, target) in tqdm(enumerate(self.train_loader)):
            data = data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            results = self.model(data)

            loss = self.loss_function(results, target)
            loss.backward()
            train_loss += loss.item()
            running_loss += loss.item()

            _, predicted = torch.max(results, 1)
            c = (predicted == target).sum()
            running_acc += c
            train_acc += c

            self.optimizer.step()
            if batch_idx % self.log_interval == (self.log_interval-1):   
                print(f'[Epoch: {self.epoch}, batch: {batch_idx}/{len(self.train_loader)} ]\t'
                      f'Loss: {running_loss/self.log_interval:.4f}\t'
                      f'Accuracy: {100*running_acc/(self.train_loader.batch_size*self.log_interval):.4f}')

                running_loss = 0.0
                running_acc = 0.0

        return train_loss, train_acc


    def test(self):
        self.model.eval()
        test_loss = 0.0
        test_acc = 0.0
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(self.val_loader)):
                data = data.to(self.device)
                target = target.to(self.device)
                results = self.model(data)
                test_loss += self.loss_function(results, target).item()

                _, predicted = torch.max(results, 1)
                c = (predicted == target).sum()
                test_acc += c
                    

        test_loss /= len(self.val_loader)
        test_acc = 100 * test_acc / len(self.val_loader.dataset)
        return test_loss, test_acc


    def train_one_epoch(self):
        train_loss, train_acc = self.train()
        test_loss, test_acc= self.test()

        print(f'====> Epoch: {self.epoch} Average loss: {train_loss / len(self.train_loader):.4f}')

        print(f'====> Test set loss: {test_loss:.4f}, \t Accuracy: {test_acc:.4f}')

        self.writer.add_scalar('Loss/train', train_loss, self.epoch)
        self.writer.add_scalar('Loss/test', test_loss, self.epoch)
        self.writer.add_scalar('Accuracy/train', train_acc, self.epoch)
        self.writer.add_scalar('Accuracy/test', test_acc, self.epoch)

        if test_acc > self.best_accuracy :
            torch.save(self.model.state_dict(), f'{self.saving_path}/model.pt')
            self.best_accuracy = test_acc
        
        self.epoch += 1

    def run_training(self):

        if self.epoch != 0:
            for _ in range(self.epoch):
                self.scheduler.step()

        while self.epoch != self.last_epoch:
            self.train_one_epoch()

        print(f'Finished Training, Best Accuracy : {self.best_accuracy:.3f}')