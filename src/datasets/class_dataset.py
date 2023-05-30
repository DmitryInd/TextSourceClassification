from pathlib import Path
from typing import Tuple, List

import pandas as pd
import torch
from torch.utils.data import Dataset

from datasets.tokenization import SentencePieceTokenizer


class CSVDataset(Dataset):
    def __init__(self, path_to_file: str, is_uncased=False, without_labels=False,
                 pretrained_tokenizer: str = None, max_length=100, device="cuda:0"):
        """
        :param path_to_file: путь к директории с xml файлами, содержащими размеченные данные
        :param is_uncased: приводить все символы к нижнему регистру или нет
        :param without_labels: есть ли в наборе данных верные предсказания
        :param pretrained_tokenizer: путь к сохранённым параметрам токенизатора
        :param max_length: максимальное количество токенов в примере
        :param device: устройство, на котором будет исполняться запрос
        """
        path = Path(path_to_file)
        assert path.exists(), "The specified folder doesn't exist"
        # Start of reading files
        self.without_labels = without_labels
        self.is_uncased = is_uncased
        self._record_ids, tokenized_source_list, tokenized_target_list = self._read_csv(path_to_file,
                                                                                        is_uncased,
                                                                                        without_labels)
        # Setting up entity labels
        self.index2label = sorted(set(tokenized_target_list))
        self.label2index = {label: i for i, label in enumerate(self.index2label)}
        self._tokenized_target_list = [self.label2index[label] for label in tokenized_target_list]
        # Data tokenization
        self.tokenizer = SentencePieceTokenizer(tokenized_source_list,
                                                True,
                                                max_sent_len=max_length,
                                                pretrained_name=pretrained_tokenizer)
        self._tokenized_source_list = [self.tokenizer(s) for s in tokenized_source_list]
        self.record2idx = {record_id: i for i, record_id in enumerate(self._record_ids)}
        self._record_ids = torch.tensor(self._record_ids, device=device)
        self.device = device

    @staticmethod
    def _read_csv(path_to_file: str, is_uncased: bool, without_labels: bool) -> Tuple[List[int], List[str], List[str]]:
        df = pd.read_csv(path_to_file)
        id_batch = pd.to_numeric(df['Id']).to_list()
        source_batch = df['Text'].to_list()
        target_batch = []
        if not without_labels:
            target_batch = df['Class'].to_list()
        if is_uncased:
            source_batch = [s.lower() for s in source_batch]
        return id_batch, source_batch, target_batch

    def __len__(self):
        return len(self._tokenized_source_list)

    def __getitem__(self, idx):
        source_token_ids = torch.tensor(self._tokenized_source_list[idx], device=self.device)
        if not self.without_labels:
            target_label_id = torch.tensor(self._tokenized_target_list[idx], device=self.device)
            return self._record_ids[idx], source_token_ids, target_label_id
        else:
            return self._record_ids[idx], source_token_ids
