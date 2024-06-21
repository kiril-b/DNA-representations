import json
from pathlib import Path

from src.data_models.models import ModelTrainingStates

TRAINING_STATES_PATH = "src/logs/training_states.json"

training_states_path = Path(TRAINING_STATES_PATH)


def load_training_states() -> ModelTrainingStates:
    with training_states_path.open(mode="r") as json_file:
        return ModelTrainingStates.model_validate(json.load(json_file))


def dump_training_states(training_states: ModelTrainingStates) -> None:
    with training_states_path.open(mode="w") as json_file:
        json.dump(training_states.model_dump(), json_file, indent=4)


def init_training_states() -> None:
    if not training_states_path.exists():
        dump_training_states(ModelTrainingStates())
