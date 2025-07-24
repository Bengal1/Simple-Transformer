import torch
import torch.nn as nn


class NormLayer(torch.nn.Module):
    """
    Implements layer normalization used in the Transformer.

    This normalization technique stabilizes the training process by normalizing
    inputs across the last dimension and scaling them with learnable parameters.

    Attributes:
        gamma (nn.Parameter): Learnable scale parameter initialized to ones.
        beta (nn.Parameter): Learnable shift parameter initialized to zeros.
        epsilon (float): A small value added to variance for numerical stability.
    """

    def __init__(self,
                 d_model: int,
                 epsilon: float = 1e-5):
        """Initializes the layer normalization module.

        Args:
            d_model (int): The dimension of the input tensor.
            epsilon (float, optional): A small value added to variance for
                                    numerical stability. Default is 1e-6.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies layer normalization to the input tensor.

        Normalizes the input across the last dimension and applies learnable
        scaling (`gamma`) and shifting (`beta`).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as the input.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        std = torch.sqrt(var + self.epsilon)
        normalized = (x - mean) / std
        return self.gamma * normalized + self.beta