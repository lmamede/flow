import torch
import torch.nn as nn

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
        # TODO: Implementar
        # Dicas:
        # 1. Time embedding: usar sinusoidal encoding
        # 2. Network: MLP simples ou com skip connections
        # 3. Inicialização: última camada com pesos pequenos (σ=0.01)
        
    def time_embedding(self, t):
        """
        Sinusoidal time embedding.
        Args:
            t: (batch,) ou escalar
        Returns:
            embedded: (batch, time_embed_dim)
        """
        # TODO: implementar
        # t_emb[2i] = sin(t / 10000^(2i/d))
        # t_emb[2i+1] = cos(t / 10000^(2i/d))
        pass

    def forward(self, t, x):
        """
        Calcula f(x, t).
        Args:
            t: tempo (escalar ou (batch,))
            x: estado (batch, features)
        Returns:
            dx_dt: (batch, features)
        """
        # TODO: implementar
        # 1. Expandir t para batch se necessário
        # 2. Time embedding
        # 3. Concatenar [x, t_emb]
        # 4. Passar pela rede
        pass