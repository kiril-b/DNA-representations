import numpy as np
from torch import nn
from torchinfo import summary

from src.data_models.models import ModelType
from src.preprocessing.sequence_transformations import (
    DNA,
    TransformationHuffman,
)


def transform_sequence_huffman(seq: DNA) -> np.ndarray:
    return TransformationHuffman(seq).transform(seq)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_architecture_str(
    model: nn.Module, model_type: ModelType, batch_size: int, sequence_len=500
) -> str:
    input_size = (
        (batch_size, sequence_len, 1)
        if model_type == ModelType.cnn
        else (batch_size, sequence_len, 2)
    )
    return str(summary(model=model, input_size=input_size, verbose=0)).replace("=", "")
