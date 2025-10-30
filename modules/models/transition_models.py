import math
import torch
import torch.nn as nn

class TransitionModel(nn.Module):
    ''''
    A stochastic transition model with mean given by an MLP and 
    noise covariance given by directly parameterized Cholesky factors.
    '''
    def __init__(
        self,
        state_dim: int,
        num_layers: int,
        hidden_dim: int,
        scale: float,
        initial_variance: float
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            *[
                layer for _ in range(num_layers - 1) 
                for layer in (nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            ],
            nn.Linear(hidden_dim, state_dim)
        )
        self.scale = float(scale)

        initial_std = math.sqrt(initial_variance)
        self.L = nn.Parameter(torch.eye(state_dim) * initial_std)

    def get_covariance(self):
        return self.L @ self.L.T

    def forward(
        self,
        state: torch.Tensor,
        deterministic = False,
        broadcast_covariance = True
    ):
        delta = self.model(state)
        next_state = state + torch.tanh(delta) * self.scale
        if deterministic: 
            return next_state
        else:
            covariance = self.get_covariance()
            if broadcast_covariance and len(next_state.shape) == 2:
                batch_size, state_dim = next_state.shape
                covariance = covariance.expand(batch_size, state_dim, state_dim)
            return next_state, covariance