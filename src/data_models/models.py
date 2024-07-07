from enum import StrEnum


class ModelType(StrEnum):
    cnn = "cnn"
    lstm = "lstm"
    transformer = "transformer"
    logistic_regression = "logistic_regression"


class DnaRepresentation(StrEnum):
    rudimentary = "rudimentary"
    refined = "refined"
    huffman = "huffman"
    grayscale = "grayscale"
