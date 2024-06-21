import math
from abc import ABC, abstractmethod
from typing import override

import numpy as np
from Bio.Seq import Seq

from src.utils.huffman_encoding import encode, huffman_code

DNA = Seq | str


class Transformation(ABC):
    @abstractmethod
    def transform(self, x: DNA) -> np.ndarray: ...


class TransformationRudimentary(Transformation):
    def __init__(self) -> None:
        self._mapping = {
            "A": (0, -1),
            "G": (1, 0),
            "C": (-1, 0),
            "T": (0, 1),
        }

    @override
    def transform(self, seq: DNA) -> np.ndarray:
        mapped_seq = np.array([np.array(self._mapping[s]) for s in seq])
        return np.cumsum(np.array(mapped_seq), axis=0)


class TransformationRefined(Transformation):
    def __init__(self) -> None:
        self._mapping = {
            "A": (1 / 2, -math.sqrt(3) / 2),
            "G": (math.sqrt(3) / 2, -1 / 2),
            "C": (math.sqrt(3) / 2, 1 / 2),
            "T": (1 / 2, math.sqrt(3) / 2),
        }

    @override
    def transform(self, seq: DNA) -> np.ndarray:
        mapped_seq = np.array([np.array(self._mapping[s]) for s in seq])
        return np.cumsum(np.array(mapped_seq), axis=0)


class TransformationHuffman(Transformation):
    def __init__(self, huffman_code_string: str | Seq) -> None:
        self._code = huffman_code(str(huffman_code_string))
        self._mapping = {"0": (1, -1), "1": (1, 1)}

    @override
    def transform(self, seq: DNA) -> np.ndarray:
        encoded_x = encode(str(seq), self._code)
        mapped_seq = np.array([np.array(self._mapping[s]) for s in list(encoded_x)])
        return np.cumsum(np.array(mapped_seq), axis=0)


class TransformationImageGrayscale(Transformation):
    def __init__(self) -> None:
        self._mapping = {
            "A": 0,
            "G": 0.7,
            "C": 0.3,
            "T": 1,
        }

    @override
    def transform(self, seq: DNA) -> np.ndarray:
        return np.array([self._mapping[s] for s in seq])