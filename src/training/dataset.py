import pathlib

import torch
from torch.utils.data import DataLoader, Dataset, random_split



class SequenceDataset(Dataset):
    def __init__(
        self,
        sequences_path: pathlib.Path,
        classes_path: pathlib.Path,
    ) -> None:
        self.sequences: torch.Tensor = torch.load(sequences_path)
        self.classes: torch.Tensor = torch.load(classes_path)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.classes[idx]


def load_data(
    sequences_path: pathlib.Path,
    classes_path: pathlib.Path,
    batch_size: int,
    training_set_size_percentage: float,
) -> tuple[DataLoader, DataLoader]:
    dataset = SequenceDataset(sequences_path, classes_path)

    train_size = int(training_set_size_percentage * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=(batch_size * 2), shuffle=False)

    return train_loader, val_loader
