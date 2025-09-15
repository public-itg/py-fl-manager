from lightning import LightningDataModule

from fl_manager.lightning.dataflows.base_lightning_trainer import datamodule


def test_datamodule(torch_dataset_alpha):
    _datamodule = datamodule(
        dataset=torch_dataset_alpha, datamodule_name='default', datamodule_kwargs={}
    )
    assert isinstance(_datamodule, LightningDataModule)
    assert _datamodule.train_dataloader() is not None
    assert _datamodule.val_dataloader() is not None
    assert _datamodule.test_dataloader() is not None
