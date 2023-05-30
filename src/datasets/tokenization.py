from typing import List

import sentencepiece as spm
from pathlib import Path
from transformers import AutoTokenizer, DebertaV2Tokenizer


class SentencePieceTokenizer:
    def __init__(self, sentence_list: List[str], pad_flag: bool,
                 max_sent_len: int = None, pretrained_name: str = None):
        """
        :param sentence_list: список предложений для обучения, разбитых на размеченные части
        :param pad_flag: нужно ли приводить последовательности токенов к одной длине
        :param max_sent_len: максимальная допустимая длина предложений в токенах (+2)
        :param pretrained_name: путь до сохранённых параметров или название токенизатора
        """
        # Initialisation
        self.max_sent_len = max_sent_len
        self.pad_flag = pad_flag
        if pretrained_name is None:
            self._tokenizer = self._train(sentence_list)
        else:
            self._tokenizer = self._load(pretrained_name)
        # Preparing dictionaries mapping tokens and ids
        self.word2index = self._tokenizer.get_vocab()
        self.index2word = {w_id: word for word, w_id in self.word2index.items()}
        if self.pad_flag and self.max_sent_len is None:
            self.max_sent_len = self._get_max_length_in_tokens(sentence_list)

    def __call__(self, sentence: str, force_padding: bool = None):
        """
        :param sentence: токенизируемое предложение
        :param force_padding: форсирующий флаг о приведении предложений к одной длине
        :return: последовательность id текстовых токенов
        """
        padding = self.pad_flag if force_padding is None else force_padding
        if self._downloaded:
            token_id_list = self._tokenizer.encode(sentence, padding=False, truncation=False)[1:-1]
        else:
            token_id_list = self._tokenizer.encode(sentence).ids
        token_id_list = [self.word2index[self.sos_token]] + token_id_list + [self.word2index[self.eos_token]]
        if padding and self.max_sent_len is not None:
            if len(token_id_list) > self.max_sent_len + 2:
                token_id_list = token_id_list[:self.max_sent_len + 2]
            elif len(token_id_list) < self.max_sent_len + 2:
                pad_len = self.max_sent_len + 2 - len(token_id_list)
                token_id_list.extend([self.word2index[self.pad_token]] * pad_len)

        return token_id_list

    def decode(self, token_id_list: List[int]) -> str:
        """
        :param token_id_list: последовательность id текстовых токенов (для одного предложения)
        :return: предложение
        """
        return self._tokenizer.decode(token_id_list, skip_special_tokens=True)

    def _get_max_length_in_tokens(self, sentence_list: List[str]) -> int:
        max_length = 0
        for sentence in sentence_list:
            max_length = max(max_length, len(self(sentence, False)))
        return max_length

    def _train(self, sentence_list: List[str]) -> DebertaV2Tokenizer:
        # Pretrained flag
        self._downloaded = False
        # Special tokens
        self.unknown_token = "[UNK]"
        self.sos_token = "[CLS]"
        self.eos_token = "[SEP]"
        self.pad_token = "[PAD]"
        # Save training sentences to file
        if not Path("./data/spm/train/").exists():
            Path("./data/spm/train/").mkdir(parents=True)
        with open("./data/spm/train/sentences.txt", "w", encoding="utf-8") as file:
            for sentence in sentence_list:
                file.write(sentence)
        # Training
        if not Path("./data/spm/saved/").exists():
            Path("./data/spm/saved/").mkdir(parents=True)
        spm.SentencePieceTrainer.Train(
            input="./data/spm/train/sentences.txt",
            model_prefix='./data/spm/saved/spModel',
            vocab_size=10000,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece=self.pad_token,
            unk_piece=self.unknown_token,
            bos_piece=self.sos_token,
            eos_piece=self.eos_token,
            model_type='bpe'
        )
        # Initialization
        self._tokenizer = DebertaV2Tokenizer(
            vocab_file="./data/spm/saved/spModel.model",
            pad_token=self.pad_token,
            unk_token=self.unknown_token,
            bos_token=self.sos_token,
            eos_token=self.eos_token
        )
        return self._tokenizer

    def _load(self, pretrained_name: str) -> AutoTokenizer:
        # Pretrained flag
        self._downloaded = True
        # Download
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        # Special tokens
        self.unknown_token = self._tokenizer.unk_token
        self.sos_token = self._tokenizer.cls_token
        self.eos_token = self._tokenizer.sep_token
        self.pad_token = self._tokenizer.pad_token
        return self._tokenizer
