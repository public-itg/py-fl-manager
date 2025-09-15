import logging
from typing import Optional, Any

from lightning import seed_everything, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from fl_manager.core.constants import FL_MANAGER_HOME
from fl_manager.core.schemas.dataset import GenericDataset
from fl_manager.lightning.components.datamodule import LightningDataModuleRegistry
from fl_manager.lightning.components.datamodule.base_lightning_datamodule import (
    BaseLightningDataModule,
)

logger = logging.getLogger(__name__)


def datamodule(
    dataset: GenericDataset,
    datamodule_name: str,
    datamodule_kwargs: dict,
    seed: Optional[int] = 42,
) -> LightningDataModule:
    seed_everything(seed, workers=True)
    _datamodule: BaseLightningDataModule = LightningDataModuleRegistry.create(
        datamodule_name, **datamodule_kwargs
    )
    _datamodule.configure_dataloaders_from_datasets(
        train_dataset=dataset.train, val_dataset=dataset.val, test_dataset=dataset.test
    )
    return _datamodule


# sf-hamilton(1.88.0) check_input_type validation is False when type if Generic, use Any for now
def model(fl_model: Any) -> LightningModule:
    _model = fl_model.get_model()
    assert isinstance(_model, LightningModule)
    return _model


def train(
    fl_train_id: str,
    datamodule: LightningDataModule,  # noqa
    model: LightningModule,  # noqa
    trainer_kwargs: dict,
    seed: Optional[int] = 42,
    deterministic: Optional[bool] = True,
) -> None:
    """Performs standard ML training (i.e. non-federated)."""
    seed_everything(seed, workers=True)
    trainer = Trainer(
        deterministic=deterministic,
        **(
            trainer_kwargs
            | {
                'logger': TensorBoardLogger(
                    save_dir=FL_MANAGER_HOME / 'logs', name=fl_train_id
                )
            }
        ),
    )
    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule)
