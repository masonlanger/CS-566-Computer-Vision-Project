import torch

class ObservationModel(torch.nn.Module):
    def __init__(self, covariance=None):
        super().__init__()
        self.covariance = covariance

    def compute_covariance(self, state, broadcast_covariance):
        covariance = self.covariance
        if broadcast_covariance and len(state.shape) == 2:
            batch, _ = state.shape
            covariance = self.covariance.expand(batch, self.dim, self.dim)
        return covariance

    def forward(
        self, 
        state,
        broadcast_covariance=True
    ):
        x, y, x_vel, y_vel = state.unbind(dim=-1)
        observation = torch.stack([x, y], dim=-1).to(torch.float32)
        return observation, self.compute_covariance(state, broadcast_covariance)
