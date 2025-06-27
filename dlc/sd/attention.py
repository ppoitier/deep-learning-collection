import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        assert d_embed % n_heads == 0, "d_embed must be divisible by n_heads"

    def forward(self, x: Tensor, causal_mask=False) -> Tensor:
        # x: (N, L, D)
        input_shape = x.shape
        batch_size, length, d_embed = input_shape

        inter_shape = (batch_size, length, self.n_heads, self.d_head)

        # (N, L, D) -> 3x(N, L, D)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (N, L, D) -> (N, H, L, D/H)
        q = q.view(inter_shape).transpose(1, 2)
        k = k.view(inter_shape).transpose(1, 2)
        v = v.view(inter_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(diagonal=1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        # (N, H, L, L) @ (N, H, L, D/H) -> (N, H, L, D/H)
        output = weight @ v
        output = output.transpose(1, 2)
        # (N, L, H, D/H)
        output = output.reshape(input_shape)
        output = self.out_proj(output)
        # (N, L, D)
        return output
