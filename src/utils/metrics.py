from typing import List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from tabulate import tabulate
from torch import nn
from torch.utils.data import DataLoader


class Statistics:
    def __init__(self, model: nn.Module, loader: DataLoader, device='cuda:0'):
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.loader = loader
        self.dataset = loader.dataset
        self.tokenizer = self.dataset.tokenizer
        self.record2idx = self.dataset.record2idx
        self.index2label = self.dataset.index2label
        self.record_ids, self.true_labels, self.predicted_labels = self._compute_true_and_predicted_labels()
        self.confusion_matrix = self._compute_confusion_matrix()
        self.fault_ids = self._get_fault_record_ids()

    @torch.no_grad()
    def _compute_true_and_predicted_labels(self) -> Tuple[List[int], List[int], List[int]]:
        record_ids = []
        predictions = []
        true_labels = []
        for batch_ids, inputs, labels in self.loader:
            inputs = inputs.to(self.device)
            batch_pred = self.model(inputs).cpu().argmax(1)
            record_ids.extend(batch_ids.tolist())
            predictions.extend(batch_pred.tolist())
            true_labels.extend(labels.cpu().tolist())

        return record_ids, true_labels, predictions

    def _compute_confusion_matrix(self):
        confusion_matrix = np.zeros((len(self.index2label), len(self.index2label)))
        for true_label, pred_label in zip(self.true_labels, self.predicted_labels):
            confusion_matrix[true_label][pred_label] += 1

        return confusion_matrix

    def _get_fault_record_ids(self):
        faults = set()
        for record_id, true_label, pred_label in zip(self.record_ids, self.true_labels, self.predicted_labels):
            if true_label != pred_label:
                faults.add(record_id)

        return list(sorted(list(faults)))

    def get_classification_report(self):
        return classification_report(self.true_labels, self.predicted_labels, target_names=self.index2label)

    @staticmethod
    def _get_rgb_cell_map(float_cell_map: np.ndarray):
        assert len(float_cell_map.shape) == 2, "Confusion matrix has more than two dimensions"
        cmap = plt.cm.get_cmap('Greens')
        max_values = np.sum(float_cell_map, axis=1)
        rgb_cell_map = [[cmap(cell / max_value) for cell in row] for row, max_value in zip(float_cell_map, max_values)]
        return rgb_cell_map

    def plot_confusion_matrix(self):
        """
        Plot the confusion matrix
        """
        fig, ax = plt.subplots()
        fig.set_figwidth(25)
        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ax.set_title('y: true/x: predicted labels')
        ax.table(self.confusion_matrix.tolist(), rowLabels=self.index2label, colLabels=self.index2label,
                 cellColours=self._get_rgb_cell_map(self.confusion_matrix), loc='center')
        fig.tight_layout()
        plt.show()

    @torch.no_grad()
    def print_random_failed_predictions(self, num_samples: int = 5):
        record_ids, sentences, label_ids, pred_ids = [], [], [], []
        internal_ids = np.random.choice(np.arange(0, len(self.fault_ids)), num_samples, replace=False)
        for i in internal_ids:
            record_id = self.fault_ids[i]
            idx = self.record2idx[record_id]
            _, token_ids, label_id = self.dataset[idx]
            pred_id = self.model(token_ids.unsqueeze(0))[0].argmax()
            record_ids.append(record_id)
            sentences.append(self.tokenizer.decode(token_ids.tolist()))
            label_ids.append(label_id.item())
            pred_ids.append(pred_id.item())

        print('Wrongly predicted examples:')
        print(tabulate([['Record id:', 'Sentence:', 'True label:', 'Pred labels:']] +
                       [[record_id, sentence, self.index2label[t_index], self.index2label[p_index]]
                        for record_id, sentence, t_index, p_index in zip(record_ids, sentences, label_ids, pred_ids)],
                       tablefmt='orgtbl'))
