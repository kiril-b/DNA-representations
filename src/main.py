import logging
from enum import StrEnum
from pathlib import Path

import numpy as np
import torch
import typer
from Bio.Seq import Seq
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.models.cnn import DNAClassifierCNN
from src.models.logistic_regression import LogisticRegression
from src.models.lstm import LSTMClassifier
from src.models.transformer import TransformerEncoderClassifier
from src.preprocessing.sequence_transformations import (
    TransformationHuffman,
    TransformationImageGrayscale,
    TransformationRefined,
)
from src.training.dataset import SequenceDataset

torch.manual_seed(23)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

app = typer.Typer()


class ModelType(StrEnum):
    cnn = "cnn"
    lstm = "lstm"
    transformer = "transformer"
    logistic_regression = "logistic_regression"


class DnaRepresentation(StrEnum):
    refined = "refined"
    huffman = "huffman"
    grayscale = "grayscale"


def transform_sequence_huffman(seq: Seq | str) -> np.ndarray:
    return TransformationHuffman(seq).transform(seq)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@app.command()
def start_training(
    model_type: ModelType,
    dna_representation: DnaRepresentation,
    num_epochs: int,
    learning_rate: float,
    batch_size: int = 64,
    training_set_size_percentage: float = 0.8,
    dataset_path: Path = Path("data/classification/data.csv"),
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Device set to: {str(device)}")

    match dna_representation:
        case DnaRepresentation.refined:
            seq_transformation = TransformationRefined().transform
        case DnaRepresentation.huffman:
            seq_transformation = transform_sequence_huffman
        case DnaRepresentation.grayscale:
            seq_transformation = TransformationImageGrayscale().transform

    model: nn.Module
    match model_type:
        case ModelType.logistic_regression:
            model = LogisticRegression(sequence_len=500).to(device)
        case ModelType.lstm:
            model = LSTMClassifier(
                input_dim=1, hidden_dim=1, output_dim=1, num_layers=1
            )
            pass
        case ModelType.transformer:
            model = TransformerEncoderClassifier(
                input_dim=2,
                d_model=2,
                nhead=1,
                num_layers=1,
                max_seq_length=500,
                dim_dense=256,
                device=device,
            ).to(device)
        case ModelType.cnn:
            model = DNAClassifierCNN().to(device)

    logger.info(f"Number of trainable parameters: {count_trainable_parameters(model)}")

    dataset = SequenceDataset(dataset_path, seq_transformation)
    train_size = int(training_set_size_percentage * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=(batch_size * 2), shuffle=False)


if __name__ == "__main__":
    app()
