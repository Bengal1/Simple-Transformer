import torch


class PositionalEncoding(torch.nn.Module):
    """
    Adds positional encoding to the input tensor.

    The positional encoding follows the formula from "Attention Is All You Need"
    and helps the Transformer model retain positional information.

    Attributes:
        d_model (int): The embedding dimension of the model.
        n (int): The base for the sinusoidal encoding.
    """

    def __init__(self,
                 d_model: int,
                 n: int = 10000):
        """Initializes the positional encoding module.

        Args:
            d_model (int): The embedding dimension of the model.
            n (int, optional): The base for the sinusoidal encoding. Default is 10000.
        """
        super().__init__()
        self.d_model = d_model
        self.n = n

    def _create_positional_encoding(self,
                                    seq_len: int,
                                    device: torch.device) -> torch.Tensor:
        """Creates the positional encoding matrix.

        The encoding is based on sinusoidal functions that encode relative
        position information for each token.

        Args:
            seq_len (int): Length of the sequence for positional encoding.
            device (torch.device): Device where the tensor should be allocated.

        Returns:
            torch.Tensor: The positional encoding matrix of shape (seq_len, d_model).
        """
        pos_encoding = torch.zeros(seq_len, self.d_model, device=device)
        k_pos = torch.arange(seq_len, device=device).unsqueeze(dim=1).float()
        _2i = torch.arange(0, self.d_model, step=2, device=device).float()

        pos_encoding[:, 0::2] = torch.sin(k_pos / self.n ** (_2i / self.d_model))
        pos_encoding[:, 1::2] = torch.cos(k_pos / self.n ** (_2i / self.d_model))

        return pos_encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Tensor with added positional encoding,
            of shape (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.shape
        pos_encoding = self._create_positional_encoding(seq_len, x.device)

        return x + pos_encoding.unsqueeze(0)  # Broadcast across batch
