import torch
import torch.nn as nn


class DDPM(nn.Module):
    """Noise-prediction network for a denoising diffusion probabilistic model.

    Given a noisy image and its timestep, the network predicts the noise that was
    added — not the clean image. That is what the reverse process in
    `inference.py` subtracts, and what `train.py` supervises against.

    On the timestep conditioning: the embedding used to be added to the *input*,
    as `self.network(x + t_embed)`. A (B, 1, 64, 64) image plus a (B, 64, 1, 1)
    embedding broadcasts to (B, 64, 64, 64), which the first convolution — built
    for one input channel — rejects. The model could not run at all. The
    embedding is now added after the input convolution, where the tensor already
    has `hidden_dim` channels and the shapes line up.
    """

    def __init__(self, channels=1, hidden_dim=64, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        self.hidden_dim = hidden_dim

        self.conv_in = nn.Conv2d(channels, hidden_dim, 3, padding=1)
        self.conv_mid1 = nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1)
        self.conv_mid2 = nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1)
        self.conv_out = nn.Conv2d(hidden_dim, channels, 3, padding=1)
        self.act = nn.ReLU()

        self.time_embedding = nn.Embedding(timesteps, hidden_dim)

    def forward(self, x, t):
        t_embed = self.time_embedding(t).view(-1, self.hidden_dim, 1, 1)

        h = self.act(self.conv_in(x))
        h = h + t_embed                      # (B, hidden, H, W) + (B, hidden, 1, 1)
        h = self.act(self.conv_mid1(h))
        h = self.act(self.conv_mid2(h))
        return self.conv_out(h)


def get_noise_schedule(timesteps=1000):
    """Linear beta schedule, plus the alphas the sampler needs."""
    beta_start, beta_end = 1e-4, 0.02
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod
