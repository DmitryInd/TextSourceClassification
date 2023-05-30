from typing import Tuple, List

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


class CSVWriter:
    def __init__(self, model: nn.Module, dataloader: DataLoader, index2label, device="cuda:0"):
        self.model = model.to(device)
        self.model.eval()
        self.loader = dataloader
        self.index2label = index2label
        self.device = device
        self.record_ids, self.predictions = self._compute_true_and_predicted_labels()

    @torch.no_grad()
    def _compute_true_and_predicted_labels(self) -> Tuple[List[int], List[int]]:
        record_ids = []
        predictions = []
        for batch_ids, inputs in self.loader:
            inputs = inputs.to(self.device)
            batch_pred = self.model(inputs).cpu().argmax(1)
            record_ids.extend(batch_ids.tolist())
            predictions.extend(batch_pred.tolist())

        predictions = [self.index2label[idx] for idx in predictions]
        return record_ids, predictions

    def write(self, path_to_file: str):
        df = pd.DataFrame()
        df['Id'] = self.record_ids
        df['Class'] = self.predictions
        df.to_csv(path_to_file, index=False)
