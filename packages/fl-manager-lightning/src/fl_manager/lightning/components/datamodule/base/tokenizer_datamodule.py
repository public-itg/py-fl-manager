from typing import Optional

import torch
from transformers import AutoTokenizer

from fl_manager.lightning.components.datamodule import LightningDataModuleRegistry
from fl_manager.lightning.components.datamodule.base_lightning_datamodule import (
    BaseLightningDataModule,
)


@LightningDataModuleRegistry.register(name='tokenizer')
class TokenizerDataModule(BaseLightningDataModule):
    def __init__(
        self,
        tokenizer_name: str,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        super().__init__(batch_size=batch_size, num_workers=num_workers)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._max_length = max_length
        self._padding = padding
        self._truncation = truncation

    @property
    def dataloader_kwargs(self) -> dict:
        return {'collate_fn': self._collate_fn}

    def _collate_fn(self, batch):
        texts, labels = zip(*batch, strict=True)
        input_tokens = self._tokenizer(
            texts,
            padding=self._padding,
            truncation=self._truncation,
            max_length=self._max_length,
            return_tensors='pt',
        )
        labels = torch.tensor(labels)
        return input_tokens, labels
