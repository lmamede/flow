import torch
import torch.nn as nn
import math


class VectorField(nn.Module):
    """
    Parametriza dx/dt = f(x, t) usando rede neural.
    Args:
        features: dimensão dos dados
        hidden_dims: lista com dimensões das camadas ocultas
        time_embed_dim: dimensão do embedding temporal
    """
    def __init__(self, features, hidden_dims=[64, 64], time_embed_dim=16):
        super().__init__()
        self.features = features
        self.time_embed_dim = time_embed_dim

        layers = []
        input_dim = features + time_embed_dim

        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h

        final = nn.Linear(input_dim, features)
        nn.init.normal_(final.weight, mean=0.0, std=0.01)
        nn.init.zeros_(final.bias)

        layers.append(final)
        self.net = nn.Sequential(*layers)

    def time_embedding(self, t):
        """
        Sinusoidal time embedding.
        Args:
            t: (batch,) ou escalar
        Returns:
            (batch, time_embed_dim)
        """
        if not torch.is_tensor(t):
            t = torch.tensor(t)

        t = t.float().unsqueeze(-1)

        half_dim = self.time_embed_dim // 2

        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        ).to(t.device)

        args = t * frequencies

        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.time_embed_dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))

        return emb

    def forward(self, t, x):
        """
        Calcula f(x, t).
        Args:
            t: tempo (escalar ou (batch,))
            x: estado (batch, features)
        Returns:
            dx_dt: (batch, features)
        """
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=x.dtype, device=x.device)

        if t.ndim == 0:
            t = t.repeat(x.shape[0])

        t_emb = self.time_embedding(t)  

        h = torch.cat([x, t_emb], dim=-1)

        return self.net(h)
