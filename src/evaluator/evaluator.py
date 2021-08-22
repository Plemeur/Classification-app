import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

class Evaluator():
    def __init__(self, model = None, testloader = None, device = 'cpu', parameters=None, writer=None):

        self.parameters=parameters
        self.saving_path = parameters.get('saving_path', parameters['model_name'])


        self.writer = writer
        self.device = device

        self.model = self._load_best_model(model)
        self.test_loader = testloader
        self.classes = testloader.dataset.classes 

    def _load_best_model(self, model):
        model.load_state_dict(torch.load(f'{self.saving_path}/model.pt'))

        return model

    def get_predictions(self):
        self.model.eval()
        n_cls = len(self.classes)
        confusion_matrix = np.zeros(shape=(n_cls, n_cls), dtype=int)
        predictions = np.zeros(shape=(len(self.test_loader.dataset), 3), dtype=object)

        with torch.no_grad():
            j=0
            for data, target in tqdm(self.test_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                results = self.model(data)
                _, predicted = torch.max(results, 1)
                
                for i, (label, prediction) in enumerate(zip(target, predicted)):
                    confusion_matrix[label, prediction] += 1
                    predictions[i+j] = [
                        self.test_loader.dataset.imgs[i+j][0],
                        label.item(),
                        prediction.item()
                        ]
                j+=self.test_loader.batch_size

        
        return confusion_matrix, predictions 

    def save_confusion_matrix(self, confusion_matrix):
        n_classes = len(self.classes)
        im_ = plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

        text_ = np.empty_like(confusion_matrix, dtype=object)

        # print text with appropriate color depending on background
        thresh = (confusion_matrix.max() + confusion_matrix.min()) / 2.0

        for i in range(n_classes):
            for j in range(n_classes):
                color = cmap_max if confusion_matrix[i, j] < thresh else cmap_min

                
                text_cm = format(confusion_matrix[i, j], '.2g')
                if confusion_matrix.dtype.kind != 'f':
                    text_d = format(confusion_matrix[i, j], 'd')
                    if len(text_d) < len(text_cm):
                        text_cm = text_d

                text_[i, j] = plt.text(
                    j, i, text_cm,
                    ha="center", va="center",
                    color=color)

        plt.colorbar()
        plt.xticks(np.arange(n_classes, step=1, dtype=int),
                   labels = self.classes,
                   rotation=45,
                   ha = 'right',
                   rotation_mode='anchor')
        plt.yticks(np.arange(n_classes), labels = self.classes)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.ylim((n_classes - 0.5, -0.5))
        plt.tight_layout()

        plt.savefig(f'{self.saving_path}/confusion_matrix')
        np.save(f'{self.saving_path}/confusion_matrix.npy', confusion_matrix)

    def save_predictions_as_csv(self, predictions):
        df = pd.DataFrame(data = predictions, columns=['Filename', 'Ground Truth', 'Prediction'])
        df.to_csv(f'{self.saving_path}/predictions.csv', index=False)


    def run_evaluation(self):
        confusion_matrix, predictions = self.get_predictions()
        self.save_confusion_matrix(confusion_matrix)
        self.save_predictions_as_csv(predictions)