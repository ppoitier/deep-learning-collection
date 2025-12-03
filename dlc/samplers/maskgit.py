import torch
import torch.nn.functional as F
import math

class MaskGITSampler:
    def __init__(
            self,
            transformer: torch.nn.Module,
            vq_vae: torch.nn.Module,
            mask_token_id: int = 1024,
            device: str = "cuda"
    ):
        self.transformer = transformer
        self.vq_vae = vq_vae
        self.mask_token_id = mask_token_id
        self.device = device

    def gamma_func(self, t: float) -> float:
        """Cosine schedule from the paper[cite: 129, 213]."""
        return math.cos(t * math.pi / 2)

    @torch.no_grad()
    def sample(
            self,
            n_samples: int = 4,
            seq_len: int = 256, # 16x16 grid
            steps: int = 8,     # The 'T' parameter. Paper suggests 8-12[cite: 219].
            temperature: float = 1.0, # Annealed during sampling
            starting_mask: torch.Tensor = None
    ):
        self.transformer.eval()
        self.vq_vae.eval()

        # 1. Initialize Blank Canvas
        # Shape: (B, T) populated with [MASK]
        if starting_mask is None:
            input_ids = torch.full((n_samples, seq_len), self.mask_token_id, device=self.device)
        else:
            input_ids = starting_mask # For Inpainting/Editing tasks

        # We keep track of which tokens are currently masked
        unknown_mask = (input_ids == self.mask_token_id)

        # 2. Iterative Loop
        for step in range(steps):
            # Calculate progress (0.0 to 1.0)
            ratio = (step + 1) / steps

            # --- A. Predict ---
            # Forward pass: predict probabilities for ALL tokens
            logits = self.transformer(input_ids)

            # --- B. Sample (with Temperature Annealing) ---
            # The paper suggests annealing temperature to 0.
            # We add small noise to prevent mode collapse early on.
            current_temp = temperature * (1 - ratio) + 1e-6

            probs = F.softmax(logits / current_temp, dim=-1)

            # We sample from the distribution to get potential candidates
            # Alternatively, you can use argmax() for strict "most likely"
            pred_ids = torch.distributions.Categorical(probs).sample()

            # --- C. Confidence Scoring ---
            # Get the probability (confidence) of the chosen token
            # shape: (B, T)
            confidence = probs.gather(1, pred_ids.unsqueeze(-1)).squeeze(-1)

            # Crucial: If a token was ALREADY revealed in a previous step,
            # we force its confidence to Infinity (or 1.0 + epsilon) so it is NOT re-masked.
            # We only want to update the currently 'unknown' positions.
            # [cite: 102] "For the unmasked position... we simply set its confidence score to 1.0"
            known_mask = ~unknown_mask
            confidence[known_mask] = 100.0

            # Update our canvas with the new predictions
            # (We overwrite everything, but will re-mask the low-confidence ones below)
            # Note: In practice, we usually only overwrite the 'unknown' parts to be safe,
            # but MaskGIT technically re-predicts everything.
            # Let's trust the 'confidence' logic to keep the old ones.
            input_ids = torch.where(known_mask, input_ids, pred_ids)

            # --- D. Mask Schedule ---
            # Calculate how many tokens should remain masked at this step
            # standard cosine schedule: starts at 1.0, goes to 0.0
            mask_ratio = self.gamma_func(ratio)

            # How many tokens to mask? n = ceil(gamma * L)
            n_mask = math.ceil(mask_ratio * seq_len)

            # If we are at the last step, we mask 0 tokens.
            if n_mask == 0:
                break

            # --- E. Re-masking (The "Critic") ---
            # We want to keep the tokens with HIGH confidence.
            # We want to re-mask the tokens with LOW confidence.

            # We find the threshold confidence value for the bottom n_mask tokens
            # "topk" with largest=False gives us the smallest values
            cutoff_values, _ = torch.topk(confidence, k=n_mask, dim=1, largest=False)
            # The largest of the small values is our threshold
            threshold = cutoff_values[:, -1].unsqueeze(1)

            # Create new mask: True where confidence < threshold
            # We treat < threshold as "uncertain, try again next time"
            unknown_mask = confidence < threshold

            # Apply the mask token back to the input
            input_ids[unknown_mask] = self.mask_token_id

        # 3. Decode to Pixels (using VQ-VAE Decoder)
        # Note: input_ids is (B, T). VQ-VAE Quantizer usually expects (B, H, W).
        # We need to look up the embeddings manually.

        with torch.no_grad():
            # Get codebook vectors
            # (B, T) -> (B, T, Embed_Dim)
            z_q = self.vq_vae.quantizer.embeddings(input_ids)

            # Reshape back to image grid (B, H, W, C) -> permute to (B, C, H, W)
            # Assuming 16x16 grid
            H_grid = int(math.sqrt(seq_len))
            z_q = z_q.view(n_samples, H_grid, H_grid, -1).permute(0, 3, 1, 2)

            # Decoder
            imgs = self.vq_vae.decoder(z_q)

        return imgs