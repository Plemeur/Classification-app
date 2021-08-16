import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Trainer():
    def __init__(self, model = None, optimizer = None, scheduler = None, loss_function = None,
                trainloader = None, testloader = None, log_interval= 1000, 
                start_epoch = 0, last_epoch = 10, device = 'cpu', model_name='model'):

        # Config dictionnary
        self.model_config = model
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler

        # Pytorch items
        self.model = None
        self.optimizer = None
        self.scheduler = None

        # To be changed.
        self.train_loader = trainloader
        self.test_loader = testloader
        self.loss_function = loss_function
        self.log_interval = log_interval
        self.epoch = start_epoch
        self.last_epoch = last_epoch
        self.best_accuracy = 0
        self.model_name = model_name

        self.writer = SummaryWriter()
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

            print(results)
            loss = self.loss_function(results, target)
            loss.backward()
            train_loss += loss.item()
            running_loss += loss.item()

            _, predicted = torch.max(results, 1)
            c = (predicted == target).sum()
            running_acc += c
            train_acc += c

            self.optimizer.step()
            if batch_idx % self.log_interval == (self.log_interval-1):    # print every 2000 mini-batches
                print(f'[Epoch: {self.epoch}, batch: {batch_idx}/{len(self.train_loader)} ]\
                    Loss: {running_loss/self.log_interval}\t \
                    Accuracy: {100*running_acc/(len(data)*self.log_interval)}')

                running_loss = 0.0
                running_acc = 0.0

        return train_loss, train_acc


    def test(self):
        self.model.eval()
        test_loss = 0.0
        test_acc = 0.0
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(self.test_loader)):
                data = data.to(self.device)
                target = target.to(self.device)
                results = self.model(data)
                test_loss += self.loss_function(results, target).item()

                _, predicted = torch.max(results, 1)
                c = (predicted == target).sum()
                test_acc += c
                    

        test_loss /= len(self.test_loader)
        test_acc = 100 * test_acc / len(self.test_loader.dataset)
        return test_loss, test_acc


    def train_one_epoch(self):
        train_loss, train_acc = self.train()
        test_loss, test_acc= self.test()

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            self.epoch, train_loss / len(self.train_loader)))

        print('====> Test set loss: {:.4f}, \t Accuracy: {}'.format(test_loss, test_acc))

        self.writer.add_scalar('Loss/train', train_loss, self.epoch)
        self.writer.add_scalar('Loss/test', test_loss, self.epoch)
        self.writer.add_scalar('Accuracy/train', train_acc, self.epoch)
        self.writer.add_scalar('Accuracy/test', test_acc, self.epoch)

        if test_acc > self.best_accuracy :
            torch.save(self.model.state_dict(), self.model_name)
        
        self.epoch += 1

    def run_training(self):

        if self.epoch != 0:
            for _ in range(self.epoch):
                self.scheduler.step()

        while self.epoch != self.last_epoch:
            self.train_one_epoch()    