from pathlib import Path
from typing import List

from tokenizers import Tokenizer
from tokenizers import decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import DebertaV2Tokenizer, PreTrainedTokenizer


class BPETokenizer:
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
        if pretrained_name is None or Path(pretrained_name).exists():
            self._tokenizer = self._train(sentence_list, pretrained_name)
        else:
            self._tokenizer = self._load(pretrained_name)
        if self.pad_flag and self.max_sent_len is None:
            self.max_sent_len = self._get_max_length_in_tokens(sentence_list)
        # Preparing dictionaries mapping tokens and ids
        self.word2index = self._tokenizer.get_vocab()
        self.index2word = {w_id: word for word, w_id in self.word2index.items()}

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
        token_id_list = [self.word2index[self.sos_token]] + token_id_list
        if padding and (self.max_sent_len is not None) and (len(token_id_list) > self.max_sent_len + 1):
            token_id_list = self._truncate(token_id_list)
            if len(token_id_list) < self.max_sent_len + 1:
                pad_len = self.max_sent_len + 1 - len(token_id_list)
                token_id_list.extend([self.word2index[self.pad_token]] * pad_len)

        token_id_list.append(self.word2index[self.eos_token])
        return token_id_list

    def decode(self, token_id_list: List[int]):
        """
        :param token_id_list: последовательность id текстовых токенов (для одного предложения)
        :return: последовательный список слов из предложения
        """
        return self._tokenizer.decode(token_id_list, skip_special_tokens=True).split()

    def _truncate(self, token_id_list: List[int]):
        """
        Возвращает последовательность токенов, обрезанную до максимального размера без дробления слова в конце
        """
        for i, token_id in enumerate(reversed(token_id_list)):
            if self.index2word[token_id][-4:] != "</w>" and len(token_id_list) - i <= self.max_sent_len + 1:
                token_id_list = token_id_list[:-i]
                break

        return token_id_list

    def _get_max_length_in_tokens(self, sentence_list: List[str]) -> int:
        max_length = 0
        for sentence in sentence_list:
            max_length = max(max_length, len(self(sentence, False)))
        return max_length

    def _train(self, sentence_list: List[str], path_to_pretrained: str = None) -> Tokenizer:
        # Pretrained flag
        self._downloaded = False
        # Special tokens
        self.unknown_token = "[UNK]"
        self.sos_token = "[CLS]"
        self.eos_token = "[SEP]"
        self.pad_token = "[PAD]"
        # Initialization
        if path_to_pretrained is None:
            self._tokenizer = Tokenizer(BPE(unk_token=self.unknown_token))
        else:
            self._tokenizer = Tokenizer(BPE.from_file(path_to_pretrained))
        self._tokenizer.pre_tokenizer = Whitespace()
        self._tokenizer.decoder = decoders.BPEDecoder()
        # Training
        if path_to_pretrained is None:
            trainer = BpeTrainer(
                special_tokens=[self.unknown_token, self.sos_token, self.eos_token, self.pad_token],
                end_of_word_suffix="</w>"
            )
            self._tokenizer.train_from_iterator(sentence_list, trainer)
        return self._tokenizer

    def _load(self, pretrained_name: str) -> PreTrainedTokenizer:
        # Pretrained flag
        self._downloaded = True
        # Download
        self._tokenizer = DebertaV2Tokenizer.from_pretrained(pretrained_name)
        # Special tokens
        self.unknown_token = self._tokenizer.unk_token
        self.sos_token = self._tokenizer.cls_token
        self.eos_token = self._tokenizer.sep_token
        self.pad_token = self._tokenizer.pad_token
        return self._tokenizer
