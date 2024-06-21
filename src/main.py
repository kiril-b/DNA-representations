import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import typer
from torch.utils.tensorboard import SummaryWriter

from src.data_models.models import DnaRepresentation, ModelType
from src.training.dataset import load_data
from src.training.training_loop import fit
from src.utils.factories import get_model, get_transformation_function
from src.utils.misc import count_trainable_parameters, model_architecture_str

torch.manual_seed(23)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

app = typer.Typer()


DATASET_PATH = "data/classification/data.csv"
TENSORBOARD_LOGS_PATH = "src/logs/tensorboard"
TRAINING_CHECKPOINTS_PATH = "src/logs/checkpoints"


@app.command()
def start_training(
    model_type: ModelType,
    dna_representation: DnaRepresentation,
    num_epochs: int,
    learning_rate: float,
    reset_training: bool,
    start_epoch_idx: int,
    checkpoint_id: int,
    batch_size: int = 64,
    training_set_size_percentage: float = 0.8,
    dataset_path: Path = Path(DATASET_PATH),
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Device set to: {str(device)}")

    seq_transformation = get_transformation_function(dna_representation)
    model: nn.Module = get_model(model_type, device)
    logger.info(f"Number of trainable parameters: {count_trainable_parameters(model)}")

    opt = optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_path = Path(
        f"{TRAINING_CHECKPOINTS_PATH}/{model_type}/{dna_representation}/checkopoint_{checkpoint_id}.pt"
    )

    if not checkpoint_path.parent.is_dir():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists() and not reset_training:
        logger.info("Loading the pretrained model from disk...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        opt = optim.Adam(model.parameters())
        opt.load_state_dict(checkpoint["optimizer_state_dict"])

    train_loader, val_loader = load_data(
        batch_size, training_set_size_percentage, dataset_path, seq_transformation
    )

    writer = SummaryWriter(
        f"{TENSORBOARD_LOGS_PATH}/{model_type}/{dna_representation}/log_{checkpoint_id}"
    )
    writer.add_text(
        tag="model_architecture",
        text_string=f"```{model_architecture_str(model)}```",
    )

    fit(
        epochs=num_epochs,
        model=model,
        loss_func=F.binary_cross_entropy,
        opt=opt,
        train_dl=train_loader,
        valid_dl=val_loader,
        writer=writer,
        device=device,
        start_epoch_idx=start_epoch_idx,
    )
    writer.flush()
    writer.close()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
        },
        checkpoint_path,
    )


if __name__ == "__main__":
    app()
