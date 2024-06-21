import torch.nn.functional as F
from torch import Tensor, nn


class LogisticRegression(nn.Module):
    def __init__(self, sequence_len: int) -> None:
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features=sequence_len * 2, out_features=1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)  # (B, T * C)
        x = F.sigmoid(self.linear(x))  # (B, 1)
        x = x.squeeze()  # B
        return x
