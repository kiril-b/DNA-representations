import numpy as np
from torch import nn
from torchinfo import summary

from src.preprocessing.sequence_transformations import (
    DNA,
    TransformationHuffman,
)


def transform_sequence_huffman(seq: DNA) -> np.ndarray:
    return TransformationHuffman(seq).transform(seq)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_architecture_str(model: nn.Module) -> str:
    return str(summary(model, input_size=(64, 500, 2), verbose=0)).replace("=", "")
