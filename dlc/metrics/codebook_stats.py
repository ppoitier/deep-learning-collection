import torch
from torchmetrics import Metric


class CodebookStats(Metric):
    def __init__(self, n_embeddings: int):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.add_state("used_indices", default=torch.zeros(n_embeddings, dtype=torch.bool), dist_reduce_fx="max")
        self.add_state("perplexity_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_steps", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, indices: torch.Tensor):
        # indices shape: (B, H, W) or (B, T)
        flat_indices = indices.flatten()

        # 1. Track unique usage over the epoch
        unique_in_batch = torch.unique(flat_indices)
        self.used_indices[unique_in_batch] = True

        # 2. Calculate batch perplexity (instantaneous usage)
        # Count frequency of each code in the batch
        counts = torch.bincount(flat_indices, minlength=self.n_embeddings).float()
        probs = counts / counts.sum()
        # Perplexity = exp(-sum(p * log(p)))
        # We add 1e-10 to avoid log(0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        perplexity = torch.exp(entropy)

        self.perplexity_sum += perplexity
        self.total_steps += 1

    def compute(self):
        # Returns (Unique Codes Used % , Average Perplexity)
        usage_ratio = self.used_indices.float().mean()
        avg_perplexity = self.perplexity_sum / self.total_steps
        return usage_ratio, avg_perplexity
