from enum import StrEnum

from pydantic import BaseModel


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


class TrainingState(BaseModel):
    start_epoch_idx: int = 0
    run_id: int = 0


class ModelTrainingStates(BaseModel):
    lr: TrainingState = TrainingState()
    lstm: TrainingState = TrainingState()
    transformer: TrainingState = TrainingState()
    cnn: TrainingState = TrainingState()
