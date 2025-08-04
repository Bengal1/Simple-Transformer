import torch


class PositionalEncoding(torch.nn.Module):
    """
    Adds positional encoding to the input tensor.

    The positional encoding follows the formula from "Attention Is All You Need"
    and helps the Transformer model retain positional information.

    Attributes:
        d_model (int): The embedding dimension of the model.
        encoding_scalar (int): The base for the sinusoidal encoding (often 10000).
        _positional_encoding_matrix (torch.Tensor): Buffer to store the pre-calculated
                                                    positional encoding matrix.
    """

    def __init__(self,
                 d_model: int,
                 encoding_scalar: int = 10000):
        """Initializes the positional encoding module.

        Args:
            d_model (int): The embedding dimension of the model.
            encoding_scalar (int, optional): The base for the sinusoidal encoding.
                                            Default is 10000.
        """
        super().__init__()
        self.d_model = d_model
        self.encoding_scalar = encoding_scalar
        # Register The Positional Encoding Matrix as a buffer
        self.register_buffer('_positional_encoding_matrix',
                             torch.empty(0, d_model), persistent=False)

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

        pos_encoding[:, 0::2] = torch.sin(
            k_pos / self.encoding_scalar ** (_2i / self.d_model))
        pos_encoding[:, 1::2] = torch.cos(
            k_pos / self.encoding_scalar ** (_2i / self.d_model))

        return pos_encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape
                            (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: Tensor with added positional encoding of shape
                        (batch_size, sequence_length, d_model).
        """
        batch_size, sequence_length, _ = x.shape

        # If PE matrix is shorter than the required sequence_length, re-generate it
        if sequence_length > self._positional_encoding_matrix.shape[0]:
            self._positional_encoding_matrix = self._create_positional_encoding(
                sequence_length, x.device)

        # Slice the PE matrix to the required sequence length
        current_positional_encoding = self._positional_encoding_matrix[
                                      :sequence_length, :]

        # Broadcast across batch
        return x + current_positional_encoding.unsqueeze(0)
