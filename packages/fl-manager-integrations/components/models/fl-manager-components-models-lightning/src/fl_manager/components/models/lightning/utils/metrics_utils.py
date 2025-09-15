import numpy as np
import torch
from numpy.typing import NDArray


class MetricUtils:
    @staticmethod
    def compute_scores(cm: torch.Tensor) -> dict[str, NDArray[np.float_]]:
        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp  # Sum columns (predicted) and subtract TP
        fn = cm.sum(dim=1) - tp  # Sum rows (true) and subtract TP

        precision = tp / (tp + fp + 1e-9)  # Add epsilon to avoid division by zero
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

        return {
            'precision': precision.detach().cpu().numpy(),
            'recall': recall.detach().cpu().numpy(),
            'f1': f1.detach().cpu().numpy(),
        }
