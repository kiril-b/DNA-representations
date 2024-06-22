from typing import Callable, assert_never

import numpy as np
import torch
from torch import nn

from src.data_models.models import (
    DnaRepresentation,
    ModelType,
)
from src.models.cnn import CnnClassifier
from src.models.logistic_regression import LogisticRegression
from src.models.lstm import LstmClassifier
from src.models.transformer import TransformerEncoderClassifier
from src.preprocessing.sequence_transformations import (
    DNA,
    TransformationImageGrayscale,
    TransformationRefined,
    TransformationRudimentary,
)
from src.utils.misc import transform_sequence_huffman


def get_model(
    model_type: ModelType, sequence_len: int, device: torch.device
) -> nn.Module:
    match model_type:
        case ModelType.logistic_regression:
            return LogisticRegression(sequence_len=sequence_len).to(device)
        case ModelType.lstm:
            return LstmClassifier().to(device)
        case ModelType.transformer:
            return TransformerEncoderClassifier(
                max_seq_length=sequence_len, device=device
            ).to(device)
        case ModelType.cnn:
            return CnnClassifier().to(device)


def get_transformation_function(
    dna_representation: DnaRepresentation,
) -> Callable[[DNA], np.ndarray]:
    match dna_representation:
        case DnaRepresentation.refined:
            return TransformationRefined().transform
        case DnaRepresentation.huffman:
            return transform_sequence_huffman
        case DnaRepresentation.grayscale:
            return TransformationImageGrayscale().transform
        case DnaRepresentation.rudimentary:
            return TransformationRudimentary().transform
        case _:
            assert_never(dna_representation)
