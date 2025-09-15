import torch
from torch import nn

from fl_manager.components.models.lightning.nets._base_multiclass_classification_module import (
    BaseMultiClassClassificationModule,
)


class BaseMNISTNet(BaseMultiClassClassificationModule):
    def __init__(
        self,
        num_classes: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__(
            num_classes=num_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        self.loss = nn.CrossEntropyLoss()
        self._setup_metrics()

    def _eval_step(self, batch, phase='val'):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        y_hat = logits.argmax(dim=1)

        if phase == 'val':
            self.val_metrics.update(y_hat, y)
            self.val_confusion_matrix.update(y_hat, y)
        elif phase == 'test':
            self.test_metrics.update(y_hat, y)
            self.test_confusion_matrix.update(y_hat, y)

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.train_metrics.update(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        if batch_idx % 100 == 0:
            self.log_dict(self.train_metrics, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._eval_step(batch, phase='val')
        self.log(
            'val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(self.val_metrics, on_epoch=True, prog_bar=False)

    def test_step(self, batch, batch_idx):
        loss = self._eval_step(batch, phase='test')
        self.log(
            'test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(self.test_metrics, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams['learning_rate'],
            weight_decay=self.hparams['weight_decay'],
        )
