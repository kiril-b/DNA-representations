from typing import Callable, Literal

import numpy as np
import torch
import torch.optim as optim
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import (
    binary_accuracy,
    binary_f1_score,
    binary_precision,
    binary_recall,
)
from tqdm import tqdm

torch.manual_seed(23)


def fit(
    epochs: int,
    model: nn.Module,
    loss_func: Callable[[Tensor, Tensor], Tensor],
    opt: optim.Optimizer,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    writer: SummaryWriter,
    device: torch.device,
    start_epoch_idx: int = 0,
) -> None:
    for epoch in range(epochs):
        epoch = epoch + start_epoch_idx
        print(f"Epoch: {epoch}")
        # --- TRAIN ---
        model.train()

        losses: list[float] = []
        batch_sizes: list[int] = []
        for xb, yb in tqdm(train_dl, colour="red"):
            xb: Tensor = xb.to(device)  # type: ignore [no-redef]
            yb: Tensor = yb.to(device)  # type: ignore [no-redef]
            batch_size = len(xb)

            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

            losses.append(loss.item())
            batch_sizes.append(batch_size)

        _log_epoch_metrics(
            writer=writer,
            epoch=epoch,
            metrics={"loss": losses},
            batch_sizes=batch_sizes,
            dataset="train",
        )

        # --- EVALUATION ---
        model.eval()
        with torch.no_grad():
            metrics: dict[str, list[float]] = {
                k: [] for k in ["loss", "accuracy", "precision", "recall", "f1"]
            }
            batch_sizes: list[int] = []  # type: ignore [no-redef]

            for xb, yb in tqdm(valid_dl, colour="green"):
                xb: Tensor = xb.to(device)  # type: ignore [no-redef]
                yb: Tensor = yb.to(device)  # type: ignore [no-redef]

                probabilities = model(xb)

                _calculate_batch_metrics(
                    xb=xb,
                    yb=yb,
                    loss_func=loss_func,
                    metrics=metrics,
                    batch_sizes=batch_sizes,
                    probabilities=probabilities,
                )

            _log_epoch_metrics(
                writer=writer,
                epoch=epoch,
                metrics=metrics,
                batch_sizes=batch_sizes,
                dataset="val",
            )


def _calculate_epoch_metric(
    metric_records: list[float], batch_sizes: list[int]
) -> float:
    return np.sum(np.multiply(metric_records, batch_sizes)) / np.sum(batch_sizes)


def _log_epoch_metrics(
    writer: SummaryWriter,
    epoch: int,
    metrics: dict[str, list[float]],
    batch_sizes: list[int],
    dataset: Literal["val"] | Literal["train"],
) -> None:
    for m_name, metrics_records in metrics.items():
        epoch_metric = _calculate_epoch_metric(metrics_records, batch_sizes)
        writer.add_scalar(f"{m_name}/{dataset}", epoch_metric, epoch)


def _calculate_batch_metrics(
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
