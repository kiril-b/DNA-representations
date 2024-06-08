import pathlib

import pandas as pd
import torch
from sequence_transformations import Transformation
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, data_path: pathlib.Path, transformation: Transformation) -> None:
        data = pd.read_csv(data_path)

        self.sequences = torch.stack(
            tensors=tuple(
                data["sequences"]
                .map(transformation.transform)
                .map(
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
