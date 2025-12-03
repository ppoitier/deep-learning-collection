import torch
from torch import nn
import torch.nn.functional as F


class MaskGITTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1024,  # Size of your VQ-VAE codebook
        hidden_dim: int = 512,  # Paper uses 768 for ImageNet, 512 is good for Galaxy10
        n_layers: int = 12,  # Paper uses 24, 12 is sufficient here
        n_heads: int = 8,
        max_seq_len: int = 256,  # 16x16 grid from your VQ-VAE
        dropout: float = 0.1,
    ):
        super().__init__()

        # We add 1 to vocab size. Indices 0-1023 are VQ codes. Index 1024 is [MASK].
        self.mask_token_id = vocab_size
        self.token_emb = nn.Embedding(vocab_size + 1, hidden_dim)

        # [cite_start]2. Learnable Positional Embeddings [cite: 138]
        # Standard for VQGANs. Initialized small.
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # 3. The Backbone (Encoder Only)
        # norm_first=True (Pre-LN) is generally more stable for deep transformers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # 4. The Prediction Head
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head_bias = nn.Parameter(torch.zeros(vocab_size))

        # Weight tying (optional but recommended in papers like BERT/MaskGIT)
        # It ties the embedding weights to the output layer weights.
        # self.head.weight = self.token_emb.weight[:-1]  # Exclude the [MASK] token weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T) LongTensor containing indices (0..1023) and mask tokens (1024)
        """
        # 1. Embed
        # (B, T) -> (B, T, D)
        x = self.token_emb(x)

        # 2. Add Positional Encoding
        x = x + self.pos_emb

        # 3. Transformer Pass
        # No attention mask needed! The model is allowed to see ALL provided tokens.
        x = self.transformer(x)

        # 4. Project to Vocab
        x = self.ln_f(x)
        logits = F.linear(x, weight=self.token_emb.weight[:-1], bias=self.head_bias)

        return logits


if __name__ == "__main__":
    _model = MaskGITTransformer()

    B, T = 16, 256
    _x = torch.randint(low=0, high=1025, size=(B, T))
    _y = _model(_x)
    print(_y.shape)
