from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import MetricCollection, Accuracy
from torchmetrics.classification import MulticlassConfusionMatrix

from fl_manager.components.models.lightning.utils.metrics_utils import MetricUtils


class BaseMultiClassClassificationModule(LightningModule):
    def __init__(
        self, num_classes: int, learning_rate: float = 1e-3, weight_decay: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['num_classes'])
        self._num_classes = num_classes
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._setup_metrics()

    def _setup_metrics(self):
        metrics = MetricCollection(
            {'acc': Accuracy(task='multiclass', num_classes=self._num_classes)}
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        self.val_confusion_matrix = MulticlassConfusionMatrix(
            num_classes=self._num_classes
        )
        self.test_confusion_matrix = MulticlassConfusionMatrix(
            num_classes=self._num_classes
        )

    def _log_confusion_matrix(self, confusion_matrix: MulticlassConfusionMatrix, phase):
        cm = confusion_matrix.compute().detach()
        scores = MetricUtils.compute_scores(cm)
        for score, value in scores.items():
            self.log(f'{phase}_{score}', value.mean(), on_epoch=True, prog_bar=False)
            v: float
            for i, v in enumerate(value):
                self.log(f'{phase}_{score}_{i}', v, on_epoch=True, prog_bar=False)
        fig, _ = confusion_matrix.plot()
        if isinstance(self.logger, TensorBoardLogger) and hasattr(
            self.logger, 'experiment'
        ):
            self.logger.experiment.add_figure(
                f'Confusion Matrix ({phase})', fig, global_step=self.current_epoch
            )
        confusion_matrix.reset()

    def on_validation_epoch_end(self):
        self._log_confusion_matrix(self.val_confusion_matrix, 'val')

    def on_test_epoch_end(self):
        self._log_confusion_matrix(self.test_confusion_matrix, 'test')
