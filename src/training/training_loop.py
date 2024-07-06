from pathlib import Path
from typing import Callable

import torch
import torch.optim as optim
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.logging import (
    calculate_batch_metrics,
    log_epoch_metrics,
)


def fit(
    epochs: int,
    model: nn.Module,
    loss_func: Callable[[Tensor, Tensor], Tensor],
    opt: optim.Optimizer,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    writer: SummaryWriter,
    device: torch.device,
    start_epoch_idx: int,
    checkpoint_path: Path,
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

        log_epoch_metrics(
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

                calculate_batch_metrics(
                    xb=xb,
                    yb=yb,
                    loss_func=loss_func,
                    metrics=metrics,
                    batch_sizes=batch_sizes,
                    probabilities=probabilities,
                )

            log_epoch_metrics(
                writer=writer,
                epoch=epoch,
                metrics=metrics,
                batch_sizes=batch_sizes,
                dataset="val",
            )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
            },
            checkpoint_path,
        )
