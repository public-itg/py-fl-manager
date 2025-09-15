import torch
from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from fl_manager.components.models.lightning.nets._base_multiclass_classification_module import (
    BaseMultiClassClassificationModule,
)


class TinyBertClassifierModel(BaseMultiClassClassificationModule):
    def __init__(self, num_classes: int, learning_rate=2e-5, weight_decay=0):
        super().__init__(
            num_classes=num_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'huawei-noah/TinyBERT_General_4L_312D', num_labels=self._num_classes
        )
        self._setup_metrics()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def _eval_step(self, batch, phase='val'):
        input_tokens, labels = batch
        outputs: SequenceClassifierOutput = self.forward(
            input_ids=input_tokens['input_ids'],
            attention_mask=input_tokens['attention_mask'],
            labels=labels,
        )
        assert outputs.loss is not None
        assert outputs.logits is not None
        y_hat = outputs.logits.argmax(dim=1)

        if phase == 'val':
            self.val_metrics.update(y_hat, labels)
            self.val_confusion_matrix.update(y_hat, labels)
        elif phase == 'test':
            self.test_metrics.update(y_hat, labels)
            self.test_confusion_matrix.update(y_hat, labels)

        return outputs.loss

    def training_step(self, batch, batch_idx):
        input_tokens, labels = batch
        outputs: SequenceClassifierOutput = self.forward(
            input_ids=input_tokens['input_ids'],
            attention_mask=input_tokens['attention_mask'],
            labels=labels,
        )
        assert outputs.loss is not None
        self.train_metrics.update(outputs.logits, labels)
        self.log(
            'train_loss',
            outputs.loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        if batch_idx % 100 == 0:
            self.log_dict(self.train_metrics, on_step=True, on_epoch=False)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        loss = self._eval_step(batch, phase='val')
        self.log(
            'val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(self.val_metrics, on_epoch=True, prog_bar=False)

    def test_step(self, batch, batch_idx):
        loss = self._eval_step(batch, phase='test')
        self.log(
            'test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(self.test_metrics, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams['learning_rate'],
            weight_decay=self.hparams['weight_decay'],
        )
