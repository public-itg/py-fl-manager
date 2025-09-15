import logging
from typing import TYPE_CHECKING, List, Optional, cast

import numpy as np

from fl_manager.core.components.distributors import (
    DatasetDistributor,
    DatasetDistributorRegistry,
)

if TYPE_CHECKING:
    from pandas import DataFrame

logger = logging.getLogger(__name__)


@DatasetDistributorRegistry.register(name='dirichlet')
class DirichletDatasetDistributor(DatasetDistributor):
    """
    Inspired by: FlowerDatasets (0.5.0) - DirichletPartitioner
    """

    def __init__(
        self,
        num_clients: int,
        target_col: str,
        alpha: float | List[float],
        min_distribution_size: int = 0,
        max_tries: int = 10,
        balancing: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        with_server: bool = True,
        global_test_size: Optional[float] = None,
    ):
        super().__init__(num_clients, seed, with_server, global_test_size)
        self._target_col = target_col
        self._alpha = alpha if isinstance(alpha, list) else ([alpha] * num_clients)
        assert len(self._alpha) == num_clients, (
            'when passing alpha as list, the length must be equal to num_clients.'
        )
        assert all([_alpha > 0 for _alpha in self._alpha]), (
            'alpha(s) must be greater than 0.'
        )
        assert min_distribution_size >= 0, (
            'min_distribution_size must be greater or equal than 0.'
        )
        self._min_distribution_size = min_distribution_size
        self._max_tries = max_tries
        self._balancing = balancing
        self._shuffle = shuffle

    def apply_seed(self):
        np.random.seed(self._seed)

    def _distribute(self, df: 'DataFrame') -> List['DataFrame']:
        assert self._target_col in df.columns, (
            'components does not contain target column.'
        )
        targets = cast(np.ndarray, df[self._target_col].values)
        unique_targets = np.unique(targets)
        _avg_size = len(targets) // self._num_clients
        client_data_indices = None
        for _try_number in range(self._max_tries):
            candidate_data_indices = self._generate_dirichlet_distribution(
                targets, unique_targets, _avg_size
            )
            sample_sizes = [len(indices) for indices in candidate_data_indices.values()]
            min_sample_size_on_client = min(sample_sizes)
            if min_sample_size_on_client >= self._min_distribution_size:
                client_data_indices = candidate_data_indices
                break
            alpha_not_met = [
                self._alpha[i]
                for i, s in enumerate(sample_sizes)
                if s == min_sample_size_on_client
            ]
            logger.warning(
                f'The min_distribution_size, {min_sample_size_on_client}, was not met '
                f'for alphas: {alpha_not_met} on try {_try_number}. It is recommended to '
                f'adjust the values of "alpha" or "min_distribution_size"."'
            )
        if client_data_indices is None:
            raise RuntimeError(
                f'Could not generate dirichlet partition after {self._max_tries} attempts. '
                f'Check "alpha" and "min_distribution_size".'
            )
        _distribution = [
            df.iloc[split].sample(frac=1).reset_index(drop=True)
            if self._shuffle
            else df.iloc[split].reset_index(drop=True)
            for split in client_data_indices.values()
        ]
        return _distribution

    def _generate_dirichlet_distribution(
        self, targets: np.ndarray, unique_targets: np.ndarray, avg_size: int
    ) -> dict:
        _data_indices = {i: [] for i in range(self._num_clients)}
        self._rng.shuffle(unique_targets)
        for t in unique_targets:
            target_indices = np.nonzero(targets == t)[0]
            self._rng.shuffle(target_indices)
            num_samples = len(target_indices)

            proportions = self._rng.dirichlet(self._alpha)
            if self._balancing:
                proportions = np.array(
                    [
                        p * (len(curr) < avg_size)
                        for p, curr in zip(
                            proportions, _data_indices.values(), strict=True
                        )
                    ]
                )
                proportions = proportions / proportions.sum()
            valid_mask = proportions > 0
            proportions = proportions * num_samples
            adjusted_proportions = np.floor(proportions).astype(int)

            # Adjust to match exact count
            diff = num_samples - adjusted_proportions.sum()
            if diff > 0:
                diff_ind = np.argsort(
                    proportions[valid_mask] - adjusted_proportions[valid_mask]
                )[-diff:]
            elif diff < 0:
                diff_ind = np.argsort(
                    adjusted_proportions[valid_mask] - proportions[valid_mask]
                )[: abs(diff)]
            else:
                diff_ind = None  # Already exact

            if diff_ind is not None:
                adjusted_proportions[diff_ind] += np.sign(diff)

            # Assign samples to clients
            split_indices = np.split(
                target_indices, np.cumsum(adjusted_proportions)[:-1]
            )
            for i, indices in enumerate(split_indices):
                _data_indices[i].extend(indices.tolist())
        return _data_indices
