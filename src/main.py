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
    model_type: ModelType = typer.Option(help="The model architecture"),
    dna_representation: DnaRepresentation = typer.Option(
        help="The way of encoding the DNA sequences"
    ),
    num_epochs: int = typer.Option(help="Number of training epochs"),
    learning_rate: float = typer.Option(help="Learning rate for the optimizer"),
    start_epoch_idx: int = typer.Option(
        help="The index of the initial epoch, if continuing training (useful for logging in tensorboard)"
    ),
    checkpoint_id: int = typer.Option(
        help="The ID of the checkpoint for the current config defined by the model type and dna representation"
    ),
    batch_size: int = typer.Option(
        default=64,
        help="The number of tensors that get loaded on the device (cpu or gpu) for inference/backprop",
    ),
    reset_training: bool = typer.Option(
        default=False,
        help="Whether to continue training the model defined by it's model type, dna representation and checkpoint id",
    ),
    training_set_size_percentage: float = typer.Option(
        default=0.8, help="The size of the training set"
    ),
    dataset_path: Path = typer.Option(
        default=Path(DATASET_PATH),
        help="The path of the preprocessed dataset containing the DNA sequences as strings",
    ),
) -> None:
    logger.info(reset_training)
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
        text_string=f"```{model_architecture_str(model=model, model_type=model_type, batch_size=batch_size)}```",
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
