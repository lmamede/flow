from torchdiffeq import odeint_adjoint
import torch
import torch.nn as nn

class CNF(nn.Module):
    """
    Continuous Normalizing Flow com trace exato.
    """
    def __init__(self, vector_field, base_dist=None):
        super().__init__()
        self.vf = vector_field
        if base_dist is None:
            # Prior: N(0, I)
            features = vector_field.features
            self.base_dist = torch.distributions.MultivariateNormal(
                torch.zeros(features),
                torch.eye(features)
            )
        else:
            self.base_dist = base_dist


    def _augmented_dynamics(self, t, state):
        """
        Augmented ODE: integra [x, log_det] simultaneamente.
        dx/dt = f(x, t)
        d(log_det)/dt = trace(∂f/∂x)
        Args:
            t: tempo escalar
            state: (batch, features + 1) # [x, log_det]
        Returns:
            d_state: (batch, features + 1)
        """
        batch_size = state.shape[0]
        x = state[:, :-1] # (batch, features)
        
        # Habilitar gradientes para x
        x = x.requires_grad_(True)
        
        # Compute vector field
        dx_dt = self.vf(t, x) # (batch, features)
        
        # Compute trace do Jacobiano
        trace = divergence_exact(lambda x: self.vf(t, x), x) # (batch,)
        
        # d(log_det)/dt = -trace (note o sinal!)
        dlogdet_dt = -trace.unsqueeze(-1) # (batch, 1)

        return torch.cat([dx_dt, dlogdet_dt], dim=-1)

    def forward(self, x):
        """
        Forward: x → z (usado para sampling)
        Integra de t=0 para t=1.
        """
        batch_size = x.shape[0]
        
        # Estado inicial: [x, 0]
        log_det_init = torch.zeros(batch_size, 1).to(x)
        state_0 = torch.cat([x, log_det_init], dim=-1)
        
        # Integrar
        t_span = torch.tensor([0., 1.]).to(x)
        state_1 = odeint_adjoint(
            self._augmented_dynamics,
            state_0,
            t_span,
            method='dopri5',
            rtol=1e-3,
        atol=1e-4
        )[-1] # Pegar apenas t=1
        
        z = state_1[:, :-1]
        delta_log_det = state_1[:, -1]

        return z, delta_log_det
    
    def log_prob(self, x):
        """
        Calcula log p(x) usando change of variables.
        """
        # Forward pass: x → z
        z, delta_log_det = self.forward(x)
        
        # log p(z) do prior
        log_pz = self.base_dist.log_prob(z)
        
        # log p(x) = log p(z) + log |det J|
        log_px = log_pz + delta_log_det

        return log_px
    
    def sample(self, n_samples):
        """
        Sampling: z ~ p(z) → x = φ^{-1}(z)
        Integra de t=1 para t=0 (reverso).
        """
        # Sample do prior
        z = self.base_dist.sample((n_samples,))
        
        # Integrar de t=1 para t=0
        # Apenas precisamos de x, não de log_det durante sampling
        t_span = torch.tensor([1., 0.]).to(z)

        # Usar ODE sem augmentation (mais rápido)
        x = odeint(
            self.vf,
            z,
            t_span,
            method='dopri5',
            rtol=1e-3,
            atol=1e-4
        )[-1]

        return x



def divergence_exact(f, x):
    """
    Calcula trace(∂f/∂x) exatamente usando autograd.
    ATENÇÃO: Custo O(d2) - só viável para dimensão baixa!
    Args:
        f: função R^d -> R^d
        x: input (batch, d)
        Returns:
        trace: (batch,)
    """
    batch_size, dim = x.shape

    # TODO: Implementar
    # Estratégia:
    # 1. Para cada dimensão i:
    # - Compute ∂f_i/∂x_i usando torch.autograd.grad
    # 2. Somar todas as derivadas diagonais

    # Pseudo-código:
    # trace = 0
    # for i in range(dim):
    # # Compute ∂f[i]/∂x[i]
    # df_i = autograd.grad(f[:, i].sum(), x, create_graph=True)[0]
    # trace += df_i[:, i]
    # return trace
    pass