import torch
import torch.nn as nn

class DDPM(nn.Module):
    def __init__(self, channels=1, hidden_dim=64, timesteps=1000):
        super(DDPM, self).__init__()
        self.timesteps = timesteps
        self.network = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, channels, 3, padding=1)
        )
        self.time_embedding = nn.Embedding(timesteps, hidden_dim)

    def forward(self, x, t):
        t_embed = self.time_embedding(t).view(-1, self.time_embedding.embedding_dim, 1, 1)
        x = self.network(x + t_embed)
        return x

def get_noise_schedule(timesteps=1000):
    beta_start, beta_end = 1e-4, 0.02
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod