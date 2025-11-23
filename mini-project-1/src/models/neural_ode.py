from torchdiffeq import odeint
import torch
import torch.nn as nn

class NeuralODE(nn.Module):
    """
    Neural ODE: integra dx/dt = f(x,t) de t=0 até t=1.
    """
    def __init__(self, vector_field, solver='dopri5', rtol=1e-3, atol=1e-4):
        super().__init__()
        self.vf = vector_field
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

    def forward(self, x0, t_span=None):
        """
        Integra ODE de t=0 até t=1.
        Args:
            x0: estado inicial (batch, features)
            t_span: tempos para avaliar (default: [0, 1])
        Returns:
            x_t: trajetória (len(t_span), batch, features)
        """
        if t_span is None:
            t_span = torch.tensor([0., 1.]).to(x0)

        # TODO: usar odeint do torchdiffeq
        # Dica: odeint(self.vf, x0, t_span, method=self.solver, ...)
        pass