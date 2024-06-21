import numpy as np
from torch import nn

from src.preprocessing.sequence_transformations import (
    DNA,
    TransformationHuffman,
)


def transform_sequence_huffman(seq: DNA) -> np.ndarray:
    return TransformationHuffman(seq).transform(seq)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
