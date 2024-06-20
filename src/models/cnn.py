import torch.nn as nn
from torch import Tensor


class DNAClassifierCNN(nn.Module):
    def __init__(self) -> None:
        super(DNAClassifierCNN, self).__init__()

        layers: list[nn.Module] = []  # conv layers, activations and pooling
        in_channels = 1  # the images are 1D
        for out_channels in [4, 3, 2]:
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.conv_block = nn.Sequential(*layers)

        # dense  layers
        self.fc_layers = nn.Sequential(
            nn.Linear(2 * 62 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(2)  # (B, T, 1)
        x = x.repeat(1, 1, 100)  # (B, T, 100)
        x = x.unsqueeze(1)  # (B, 1, T, 100)
        x = self.conv_block(x)
        x = x.view(-1, 2 * 62 * 12)
        x = self.fc_layers(x)
        return x.squeeze()
