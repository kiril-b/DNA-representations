from pathlib import Path

import numpy as np
from torch import nn
from torchinfo import summary

from src.data_models.models import DnaRepresentation, ModelType
from src.preprocessing.sequence_transformations import (
    DNA,
    TransformationHuffman,
)

TENSOR_REFINED_FILE_NAME = "tensor_refined.pt"
TENSOR_HUFFMAN_FILE_NAME = "tensor_huffman.pt"
TENSOR_GRAYSCALE_FILE_NAME = "tensor_grayscale.pt"


def transform_sequence_huffman(seq: DNA) -> np.ndarray:
    return TransformationHuffman(seq).transform(seq)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_architecture_str(
    model: nn.Module, model_type: ModelType, batch_size: int, sequence_len: int
) -> str:
    input_size = (
        (batch_size, sequence_len)
        if model_type == ModelType.cnn
        else (batch_size, sequence_len, 2)
    )
    return str(summary(model=model, input_size=input_size, verbose=0)).replace("=", "")


def get_sequences_path(data_path: Path, dna_representation: DnaRepresentation) -> Path:
    match dna_representation:
        case DnaRepresentation.refined:
            return data_path / Path(TENSOR_REFINED_FILE_NAME)
        case DnaRepresentation.huffman:
            return data_path / Path(TENSOR_HUFFMAN_FILE_NAME)
        case DnaRepresentation.grayscale:
            return data_path / Path(TENSOR_GRAYSCALE_FILE_NAME)
        case _:
            raise ValueError(
                f"Preprocessed data for {dna_representation} representation does not exist"
            )
