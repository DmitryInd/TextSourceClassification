import torch
import pytorch_lightning as pl
from torch import nn
from torchmetrics import Recall, Precision, F1Score
from transformers import DebertaModel


class PretrainedDeBertaNER(pl.LightningModule):
    def __init__(self, pretrained_name: str, encoder_vocab_size: int, num_classes: int,
                 lr: float, total_steps: int, div_factor: int, human_index: int, is_pooling=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = DebertaModel.from_pretrained(pretrained_name)
        self.pooling = nn.AdaptiveAvgPool1d(1) if is_pooling else lambda x: x[..., 0]
        self.activation = nn.ReLU()
        self.head = nn.Linear(self.model.config.hidden_size, num_classes)
        # Expanding or reducing the space of the encoder embeddings
        self.model.resize_token_embeddings(encoder_vocab_size)
        # Parameters of optimization
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.lr = lr
        self.div_factor = div_factor
        self.total_steps = total_steps
        # Metrics
        self.recall = Recall(task="multiclass", num_classes=num_classes, ignore_index=human_index)
        self.precision = Precision(task="multiclass", num_classes=num_classes)
        self.f1_score = F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.model(x).last_hidden_state.transpose(-1, -2)  # B, L, H -> B, H, L
        x = self.pooling(x)  # -> B, H
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
        x, y = batch
        predictions = self(x)  # B, C
        loss = self.criterion(predictions, y)
        hard_pred = torch.argmax(predictions, dim=-1)
        recall = self.recall(hard_pred, y)
        precision = self.precision(hard_pred, y)
        f1 = self.f1_score(hard_pred, y)
        self.log('train_loss', loss.item(), on_step=False, on_epoch=True, logger=True)
        self.log('train_recall', recall, on_step=False, on_epoch=True, logger=True)
        self.log('train_precision', precision, on_step=False, on_epoch=True, logger=True)
        self.log('train_f1', f1, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)  # B, C
        loss = self.criterion(predictions, y)
        hard_pred = torch.argmax(predictions, dim=-1)
        recall = self.recall(hard_pred, y)
        precision = self.precision(hard_pred, y)
        f1 = self.f1_score(hard_pred, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recall', recall, on_step=False, on_epoch=True, logger=True)
        self.log('val_precision', precision, on_step=False, on_epoch=True, logger=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        hard_pred = torch.argmax(self(x), dim=-1)
        recall = self.recall(hard_pred, y)
        precision = self.precision(hard_pred, y)
        f1 = self.f1_score(hard_pred, y)
        self.log('test_recall', recall, on_step=False, on_epoch=True, logger=True)
        self.log('test_precision', precision, on_step=False, on_epoch=True, logger=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True, logger=True)