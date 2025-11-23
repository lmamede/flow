import torch
import torch.nn as nn

class FFJORD(nn.Module):
    """
    Free-Form Jacobian of Reversible Dynamics.
    CNF escalável usando Hutchinson estimator.
    """
    def __init__(self, vector_field, base_dist=None, n_trace_samples=1, noise='rademacher'):
        super().__init__()
        self.vf = vector_field
        self.n_trace_samples = n_trace_samples
        self.noise = noise

        if base_dist is None:
            features = vector_field.features
            self.base_dist = torch.distributions.MultivariateNormal(
                torch.zeros(features),
                torch.eye(features)
            )
        else:
            self.base_dist = base_dist

    def _augmented_dynamics(self, t, state):
        """
        Augmented dynamics com trace estimation.
        """
        batch_size = state.shape[0]
        x = state[:, :-1]
        x = x.requires_grad_(True)
        # Vector field
        dx_dt = self.vf(t, x)

        # Trace estimation
        trace = hutchinson_trace_estimator(
        lambda x: self.vf(t, x),
        x,
        n_samples=self.n_trace_samples,
        noise=self.noise
        )
        dlogdet_dt = -trace.unsqueeze(-1)

        return torch.cat([dx_dt, dlogdet_dt], dim=-1)
    
    # forward, log_prob, sample: similares ao CNF
    # (copiar implementação, mudando apenas _augmented_dynamics)

    def hutchinson_trace_estimator(f, x, n_samples=1, noise='gaussian'):
        """
        Estima trace(∂f/∂x) usando Hutchinson's trick.
        Custo: O(d) por sample - escalável!
        Args:
            f: função R^d -> R^d
            x: input (batch, d)
            n_samples: número de samples para estimativa
            noise: 'gaussian' ou 'rademacher'
        Returns:
            trace_estimate: (batch,)
        """
        batch_size, dim = x.shape
        
        trace_estimates = []
        
        for _ in range(n_samples):
            # Sample ε
            if noise == 'gaussian':
                epsilon = torch.randn_like(x) # N(0, I)
            elif noise == 'rademacher':
                epsilon = torch.randint(0, 2, x.shape).float() * 2 - 1 # {-1, +1}
            else:
                raise ValueError(f"Unknown noise type: {noise}")
            
            # Compute f(x)
            with torch.enable_grad():
                x_req = x.requires_grad_(True)
                f_x = f(x_req)
                
                # Compute ε^T (∂f/∂x) via vector-Jacobian product
                # Equivalente a ε^T A onde A = ∂f/∂x
                # torch.autograd.grad computa exatamente isso!
                vjp = torch.autograd.grad(
                    f_x,
                    x_req,
                    grad_outputs=epsilon,
                    create_graph=True,
                    retain_graph=True
                )[0]

                # trace ≈ ε^T v onde v = (∂f/∂x)^T ε
                trace_est = (epsilon * vjp).sum(dim=-1) # (batch,)
            
            trace_estimates.append(trace_est)

        # Média sobre samples
        return torch.stack(trace_estimates).mean(dim=0)