import abc
from enum import StrEnum
from typing import Any, assert_never

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from matplotlib.axes import Axes

from src.sequence_transformations import (
    Transformation,
    TransformationHuffman,
    TransformationImageGrayscale,
    TransformationRefined,
    TransformationRudimentary,
)


class NucleotideMappingMethod(StrEnum):
    RUDIMENTARY = "RUDIMENTARY"
    REFINED = "REFINED"
    HUFFMAN = "HUFFMAN"
    IMAGE_GRAYSCALE = "IMAGE_GRAYSCALE"


class SequenceRepresentation:
    @abc.abstractmethod
    def _get_representation(self) -> np.ndarray: ...

    @abc.abstractmethod
    def plot_representation(
        self, color: str | None = None, ax: Axes | None = None, **kwargs: Any
    ) -> Axes: ...


class NumericSequenceRepresentation(SequenceRepresentation):
    def __init__(
        self,
        seq: Seq,
        transformation: Transformation,
    ) -> None:
        self._transformation = transformation
        self._seq = seq
        self._representation = self._get_representation()

    def _get_representation(self) -> np.ndarray:
        return self._transformation.transform(self._seq)

    def plot_representation(
        self, color: str | None = None, ax: Axes | None = None, **kwargs: Any
    ) -> Axes:
        df = pd.DataFrame(self._representation, columns=["x", "y"])

        if ax is None:
            ax = plt.gca()

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.plot(
            df["x"],
            df["y"],
            linestyle="-",
            marker="o",
            color=color,
            **kwargs,
        )

        return ax

    @property
    def representation(self) -> np.ndarray:
        return self._representation

    @property
    def seq(self) -> Seq:
        return self._seq


class ImageSequenceRepresentation(SequenceRepresentation):
    def __init__(
        self,
        seq: Seq,
        transformation: Transformation,
    ) -> None:
        self._transformation = transformation
        self._seq = seq
        self._representation = self._get_representation()

    def _get_representation(self) -> np.ndarray:
        row = self._transformation.transform(self._seq)
        return np.tile(row, (len(row) // 5, 1))

    def plot_representation(
        self, color: str | None = None, ax: Axes | None = None, **kwargs: Any
    ) -> Axes:
        if ax is None:
            ax = plt.gca()

        ax.imshow(self._representation, cmap="gray")
        ax.axis("off")

        return ax


class SequenceRepresentationFactory:
    @staticmethod
    def create(
        seq: Seq,
        mapping_key: NucleotideMappingMethod,
        huffman_code_string: str | None = None,
    ) -> SequenceRepresentation:
        if not all(nucleotide in {"A", "C", "T", "G"} for nucleotide in seq):
            raise ValueError("Invalid sequence")

        match mapping_key:
            case NucleotideMappingMethod.RUDIMENTARY:
                return NumericSequenceRepresentation(
                    seq,
                    TransformationRudimentary(),
                )

            case NucleotideMappingMethod.REFINED:
                return NumericSequenceRepresentation(seq, TransformationRefined())

            case NucleotideMappingMethod.HUFFMAN:
                if huffman_code_string is None:
                    raise ValueError(
                        "The code string must be provided in order to use HUFFMAN nucleotide mapping method."
                    )

                return NumericSequenceRepresentation(
                    seq,
                    TransformationHuffman(huffman_code_string),
                )

            case NucleotideMappingMethod.IMAGE_GRAYSCALE:
                return ImageSequenceRepresentation(seq, TransformationImageGrayscale())

            case _:
                assert_never(mapping_key)
