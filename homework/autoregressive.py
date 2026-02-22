import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_latent = d_latent
        # convert to imbeddings
        self.token_embedding = torch.nn.Embedding(n_tokens, d_latent)
        
        # Positional embeddings for up to 600 tokens 600 comes from 20*30=600
        self.position_embedding = torch.nn.Embedding(600, d_latent)
        
        #Transformer Decoder...we use Encoder because we only need self attention and not cross attention
        self.transformer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=8,
            dim_feedforward=512, # 128 * 4; d_ff is 2-4 times d_model
            dropout=0.1, 
            activation='relu',
            batch_first=True)
        
        self.output_layer = torch.nn.Linear(d_latent, n_tokens)

        # raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Flatten input from (B, h, w) to (B, seq_len)
        B, h, w = x.shape
        seq_len = h * w
        x_flat = x.view(B, seq_len)  # (B, seq_len)
        
        # Create embeddings
        z = self.token_embedding(x_flat)  # (B, seq_len, d_latent)

        # Create positional embeddings
        positions = torch.arange(seq_len, device=x.device)
        z = z + self.position_embedding(positions).unsqueeze(0)  # (1, seq_len, d_latent)

        # Create mask
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)  # (seq_len, seq_len)

        # Shift inputs by 1
        start_token = torch.zeros(B, 1, self.d_latent, device=x.device) 
        z_shifted = torch.cat([start_token, z[:, :-1, :]], dim=1)  # (B, seq_len, d_latent)

        # Apply transformer
        z_out = self.transformer(z_shifted, src_mask=causal_mask)

        logits = self.output_layer(z_out)  # (B, seq_len, n_tokens)
        
        # Reshape back to (B, h, w, n_tokens)
        logits = logits.view(B, h, w, self.n_tokens)
        return logits, {}

        # raise NotImplementedError()

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        self.eval()
        seq_len = h * w
        current = torch.zeros(B, h, w, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for i in range(seq_len):
                logits, _ = self.forward(current)  # (B, h, w, n_tokens)
                # Flatten to get position i
                logits_flat = logits.view(B, seq_len, self.n_tokens)
                next_token_logits = logits_flat[:, i, :]  # (B, n_tokens)
                
                # Sample from the distribution instead of argmax
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
                
                # Update position i in the flattened view
                current_flat = current.view(B, seq_len)
                current_flat[:, i] = next_token

        return current
        # raise NotImplementedError()
