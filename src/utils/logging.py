from typing import Callable, Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import (
    binary_accuracy,
    binary_f1_score,
    binary_precision,
    binary_recall,
)


def calculate_epoch_metric(
    metric_records: list[float], batch_sizes: list[int]
) -> float:
    return np.sum(np.multiply(metric_records, batch_sizes)) / np.sum(batch_sizes)


def log_epoch_metrics(
    writer: SummaryWriter,
    epoch: int,
    metrics: dict[str, list[float]],
    batch_sizes: list[int],
    dataset: Literal["val"] | Literal["train"],
) -> None:
    for m_name, metrics_records in metrics.items():
        epoch_metric = calculate_epoch_metric(metrics_records, batch_sizes)
        writer.add_scalar(f"{m_name}/{dataset}", epoch_metric, epoch)


def calculate_batch_metrics(
    xb: Tensor,
    yb: Tensor,
    loss_func: Callable[[Tensor, Tensor], Tensor],
    metrics: dict[str, list[float]],
    batch_sizes: list[int],
    probabilities: Tensor,
) -> None:
    batch_sizes.append(len(xb))
    batch_predictions = torch.round(probabilities)

    metrics["loss"].append(loss_func(batch_predictions, yb).item())
    # does not work with tensors of type int8
    metrics["accuracy"].append(binary_accuracy(batch_predictions, yb).item())

    batch_predictions, yb = (
        batch_predictions.to(torch.int8),
        yb.to(torch.int8),
    )

    for metric, f in zip(
        ["precision", "recall", "f1"],
        [binary_precision, binary_recall, binary_f1_score],
    ):
        metrics[metric].append(f(batch_predictions, yb).item())
