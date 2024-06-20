import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LSTMClassifier(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim, dtype=torch.float32
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim, dtype=torch.float32
        ).to(x.device)

        x, _ = self.lstm(x, (h0, c0))  # (B, T, H)

        # get the last time step's output
        x = x[:, -1, :]  # (B, H)
        x = self.linear(x)
        x = F.sigmoid(x)
        x = x.squeeze()

        return x
