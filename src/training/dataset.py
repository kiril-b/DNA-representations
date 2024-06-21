import pathlib
from typing import Callable

import numpy as np
import pandas as pd
import torch
from Bio.Seq import Seq
from torch.utils.data import DataLoader, Dataset, random_split

from src.preprocessing.scaling import min_max_scale_globally
from src.preprocessing.sequence_transformations import DNA


class SequenceDataset(Dataset):
    def __init__(
        self,
        data_path: pathlib.Path,
        transformation: Callable[[str | Seq], np.ndarray],
    ) -> None:
        # the dataset fits in ram
        data = pd.read_csv(data_path)

        embedded_sequences = data["sequences"].map(transformation).to_numpy()
        scaled_embedded_sequences = min_max_scale_globally(embedded_sequences)

        self.sequences = torch.stack(
            tensors=tuple(
                scaled_embedded_sequences.map(
                    lambda x: torch.tensor(x, requires_grad=False, dtype=torch.float32)
                )
            )
        )

        self.classes = torch.stack(
            tuple(
                data["classes"].map(
                    lambda x: torch.tensor(x, requires_grad=False, dtype=torch.float32)
                )
            )
        )

        del data

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.classes[idx]


def load_data(
    batch_size: int,
    training_set_size_percentage: float,
    dataset_path: pathlib.Path,
    seq_transformation: Callable[[DNA], np.ndarray],
) -> tuple[DataLoader, DataLoader]:
    dataset = SequenceDataset(dataset_path, seq_transformation)
    train_size = int(training_set_size_percentage * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=(batch_size * 2), shuffle=False)
    return train_loader, val_loader
