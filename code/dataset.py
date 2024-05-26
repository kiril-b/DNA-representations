import pathlib

import torch
import pandas as pd
from sequence_transformations import Transformation
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, data_path: pathlib.Path, transformation: Transformation) -> None:
        data = pd.read_csv(data_path)
        self.sequences = data["sequences"]
        self.classes = data["classes"]
        self.transform = transformation.transform
        del data

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(
            self.transform(self.sequences[idx]),
            requires_grad=False,
            dtype=torch.float32,
        ), torch.tensor(self.classes[idx], dtype=torch.float32)
