import math
from abc import ABC, abstractmethod

import numpy as np
from Bio.Seq import Seq
from huffman_encoding import encode, huffman_code
from typing import TypeVar, Generic

T = TypeVar("T")

class Transformation(Generic[T], ABC):
    @abstractmethod
    def transform(self, x: Seq) -> T: ...


class TransformationRudimentary(Transformation[list[np.ndarray]]):
    def __init__(self) -> None:
        self._mapping = {
            "A": (0, -1),
            "G": (1, 0),
            "C": (-1, 0),
            "T": (0, 1),
        }

    def transform(self, seq: Seq) -> list[np.ndarray]:
        return [np.array(self._mapping[s]) for s in seq]


class TransformationRefined(Transformation[list[np.ndarray]]):
    def __init__(self) -> None:
        self._mapping = {
            "A": (1 / 2, -math.sqrt(3) / 2),
            "G": (math.sqrt(3) / 2, -1 / 2),
            "C": (math.sqrt(3) / 2, 1 / 2),
            "T": (1 / 2, math.sqrt(3) / 2),
        }

    def transform(self, seq: Seq) -> list[np.ndarray]:
        return [np.array(self._mapping[s]) for s in seq]


class TransformationHuffman(Transformation[list[np.ndarray]]):
    def __init__(self, huffman_code_string: str) -> None:
        self._code = huffman_code(huffman_code_string)
        self._mapping = {"0": (1, -1), "1": (1, 1)}

    def transform(self, seq: Seq) -> list[np.ndarray]:
        encoded_x = encode(str(seq), self._code)
        return [np.array(self._mapping[s]) for s in list(encoded_x)]


class TransformationImageGrayscale(Transformation[np.ndarray]):
    def __init__(self) -> None:
        self._mapping = {
            "A": 0,
            "G": 0.7,
            "C": 0.3,
            "T": 1,
        }

    def transform(self, seq: Seq) -> np.ndarray:
        return np.array([self._mapping[s] for s in seq])