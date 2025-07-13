import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class FeedForward(torch.nn.Module):
    """
    Position-wise FeedForward neural network used in Transformer models.

    This module applies two linear transformations with a ReLU activation and dropout
    in between, as used in the original "Attention Is All You Need" paper.

    Attributes:
        fc1 (nn.Linear): The first linear layer that expands the input dimension to the hidden dimension.
        fc2 (nn.Linear): The second linear layer that projects the hidden representation back to the original dimension.
        mid_dropout (nn.Dropout): Dropout applied after the ReLU activation.
        out_dropout (nn.Dropout): Dropout applied after the second linear layer.
    """


    def __init__(self, d_model: int, hidden_dim: int = 2048, dropout: float = 0.1):
        """Initializes the FeedForward network.

        Args:
            d_model (int): The input and output feature dimension.
            hidden_dim (int, optional): The hidden layer dimension. Default is 2048.
            dropout (float, optional): The dropout probability. Default is 0.1.
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.mid_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Xavier initialization
        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the FeedForward network.

        The input tensor is passed through a linear layer, followed by ReLU activation,
        dropout, and a final linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.mid_dropout(x)
        x = self.fc2(x)
        x = self.out_dropout(x)
        return x