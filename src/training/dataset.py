import pathlib
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing.scaling import min_max_scale_globally


class SequenceDataset(Dataset):
    def __init__(
        self, data_path: pathlib.Path, transformation: Callable[[Sequence], np.ndarray]
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
