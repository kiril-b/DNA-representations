from typing import Callable

import numpy as np
import torch
from torch import nn

from src.data_models.models import (
    DnaRepresentation,
    ModelType,
)
from src.models.cnn import DNAClassifierCNN
from src.models.logistic_regression import LogisticRegression
from src.models.lstm import LSTMClassifier
from src.models.transformer import TransformerEncoderClassifier
from src.preprocessing.sequence_transformations import (
    DNA,
    TransformationImageGrayscale,
    TransformationRefined,
)
from src.utils.misc import transform_sequence_huffman


def get_model(model_type: ModelType, device: torch.device) -> nn.Module:
    match model_type:
        case ModelType.logistic_regression:
            return LogisticRegression(sequence_len=500).to(device)
        case ModelType.lstm:
            return LSTMClassifier(
                input_dim=1, hidden_dim=1, output_dim=1, num_layers=1
            ).to(device)

        case ModelType.transformer:
            return TransformerEncoderClassifier(
                input_dim=2,
                d_model=2,
                nhead=1,
                num_layers=1,
                max_seq_length=500,
                dim_dense=256,
                device=device,
            ).to(device)
        case ModelType.cnn:
            return DNAClassifierCNN().to(device)


def get_transformation_function(
    dna_representation: DnaRepresentation,
) -> Callable[[DNA], np.ndarray]:
    match dna_representation:
        case DnaRepresentation.refined:
            seq_transformation = TransformationRefined().transform
        case DnaRepresentation.huffman:
            seq_transformation = transform_sequence_huffman
        case DnaRepresentation.grayscale:
            seq_transformation = TransformationImageGrayscale().transform
    return seq_transformation
