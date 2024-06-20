import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TransformerEncoderClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dim_dense: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        max_seq_length: int,
        device: torch.device,
    ) -> None:
        super(TransformerEncoderClassifier, self).__init__()

        # project input_dim to d_model
        self.input_projection = nn.Linear(input_dim, d_model)

        # precompute fixed positional encodings
        self.positional_encoding = self.generate_positional_encoding(
            max_seq_length, d_model
        ).to(device)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_dense,
            batch_first=True,
            norm_first=True,  # TODO: research
            activation=F.gelu,
            device=device,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
            norm=nn.LayerNorm(d_model),
        )

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(d_model, 1)

    def generate_positional_encoding(self, max_len: int, d_model: int) -> Tensor:
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        return pe

    def forward(self, x: Tensor) -> Tensor:
        _, T, _ = x.size()  # (B, T, input_dim)
        x = self.input_projection(x)  # (B, T, d_model)
        x = x + self.positional_encoding[:, :T, :]  # (B, T, d_model)
        x = self.transformer_encoder(x)  # (B, T, d_model)
        x = x.permute(0, 2, 1)  # (B, d_model, T)
        x = self.pooling(x)  # (B, d_model, 1)
        x = x.squeeze(-1)  # (B, d_model)
        x = self.linear(x)  # (B, 1)
        x = F.sigmoid(x)  # (B, 1)
        x = x.squeeze()  # (B)
        return x
