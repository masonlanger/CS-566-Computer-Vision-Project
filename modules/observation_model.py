import torch

class Observation(torch.nn.Module):
    def forward(self, state):
        x, y, x_vel, y_vel = state.unbind(dim=-1)
        return torch.stack([x, y], dim=-1).to(torch.float32)