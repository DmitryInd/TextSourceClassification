import torch
import pytorch_lightning as pl
from torch import nn
from torchmetrics import Accuracy, Recall
from transformers import DebertaV2Model, RobertaModel


class PretrainedDeRoBertaClass(pl.LightningModule):
    def __init__(self, pretrained_name: str, model_type: str, encoder_vocab_size: int, num_classes: int,
                 lr: float, total_steps: int, div_factor: int, human_index: int, is_pooling=True):
        super().__init__()
        self.save_hyperparameters()
        if model_type == "deberta":
            self.model = DebertaV2Model.from_pretrained(pretrained_name)
        elif model_type == "roberta":
            self.model = RobertaModel.from_pretrained(pretrained_name)
        else:
            raise ValueError("Only 'deberta' and 'roberta' types are supported")
        self.activation = nn.ReLU()
        self.pooling = nn.AdaptiveAvgPool1d(1) if is_pooling else lambda x: x
        self.head = nn.Linear(self.model.config.hidden_size, num_classes)
        # Expanding or reducing the space of the encoder embeddings
        self.model.resize_token_embeddings(encoder_vocab_size)
        # Parameters of optimization
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.lr = lr
        self.div_factor = div_factor
        self.total_steps = total_steps
        # Metrics
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.recall = Recall(task="multiclass", num_classes=num_classes, ignore_index=human_index)

    def forward(self, x):
        x = self.model(x).last_hidden_state.transpose(-1, -2)  # B, L, H -> B, H, L
        x = self.pooling(x)[..., 0]  # -> B, H
        x = self.activation(x)
        x = self.head(x)  # B, C
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            total_steps=self.total_steps,
            max_lr=self.lr,
            pct_start=0.1,
            anneal_strategy='linear',
            final_div_factor=self.div_factor
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        _, x, y = batch
        predictions = self(x)  # B, C
        loss = self.criterion(predictions, y)
        hard_pred = torch.argmax(predictions, dim=-1)
        acc = self.acc(hard_pred, y)
        recall = self.recall(hard_pred, y)
        self.log('train_loss', loss.item(), on_step=False, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, logger=True)
        self.log('train_recall', recall, on_step=False, on_epoch=True, logger=True)
        self.step += 1
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, x, y = batch
        predictions = self(x)  # B, C
        loss = self.criterion(predictions, y)
        hard_pred = torch.argmax(predictions, dim=-1)
        acc = self.acc(hard_pred, y)
        recall = self.recall(hard_pred, y)
        self.log('val_loss', loss.item(), on_step=False, on_epoch=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, logger=True)
        self.log('val_recall', recall, on_step=False, on_epoch=True, logger=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        _, x, y = batch
        hard_pred = torch.argmax(self(x), dim=-1)
        acc = self.acc(hard_pred, y)
        recall = self.recall(hard_pred, y)
        self.log('test_acc', acc, on_step=False, on_epoch=True, logger=True)
        self.log('test_recall', recall, on_step=False, on_epoch=True, logger=True)
