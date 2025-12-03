import torch
from torchmetrics import Metric
import math


class CodebookStats(Metric):
    """
    Computes statistics about the Codebook usage over an epoch:
    1. Perplexity: The 'effective' number of codebook entries used.
    2. Entropy (Normalized): How uniform the distribution is (0.0 to 1.0).
    3. Usage: The percentage of codes that were used at least once.
    """

    # Set to True if you want to sync counts across GPUs in DDP
    full_state_update: bool = False

    def __init__(self, n_embeddings: int):
        super().__init__()
        self.n_embeddings = n_embeddings

        # We accumulate the counts of every token index seen
        self.add_state(
            "token_counts",
            default=torch.zeros(n_embeddings, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(self, indices: torch.Tensor):
        # indices: (B, H, W) or (B, T)
        # bincount is highly vectorized and fast
        batch_counts = torch.bincount(
            indices.flatten(), minlength=self.n_embeddings
        ).float()
        self.token_counts += batch_counts

    def compute(self):
        # 1. Probabilities
        probs = self.token_counts / self.token_counts.sum()

        # 2. Perplexity = exp(-sum(p * log(p)))
        # Add epsilon to prevent log(0)
        log_probs = torch.log(probs + 1e-10)
        entropy_sum = -torch.sum(probs * log_probs)
        perplexity = torch.exp(entropy_sum)

        # 3. Normalized Entropy (0 to 1)
        # Max entropy for K codes is log(K)
        max_entropy = math.log(self.n_embeddings)
        entropy_norm = entropy_sum / max_entropy

        # 4. Usage (Percentage of "alive" codes)
        # Count how many bins have > 0 hits
        n_used = (self.token_counts > 0).float().sum()
        usage_ratio = n_used / self.n_embeddings

        return {
            "perplexity": perplexity,
            "entropy_norm": entropy_norm,
            "usage_ratio": usage_ratio,
        }
