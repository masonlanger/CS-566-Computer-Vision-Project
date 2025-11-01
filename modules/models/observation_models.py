import torch
import cv2

# class ObservationModel(torch.nn.Module):
#     def __init__(self, covariance=None):
#         super().__init__()
#         self.covariance = covariance

#     def compute_covariance(self, state, broadcast_covariance):
#         covariance = self.covariance
#         if broadcast_covariance and len(state.shape) == 2:
#             batch, _ = state.shape
#             covariance = self.covariance.expand(batch, self.dim, self.dim)
#         return covariance

#     def forward(
#         self, 
#         state,
#         broadcast_covariance=True
#     ):
#         x, y, x_vel, y_vel = state.unbind(dim=-1)
#         observation = torch.stack([x, y], dim=-1).to(torch.float32)
#         return observation, self.compute_covariance(state, broadcast_covariance)

class CameraObservationModel(torch.nn.Module):
    dim = 2
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
        state: torch.Tensor,
        H: torch.Tensor,
        broadcast_covariance=True
    ):
        state = state.to(dtype=torch.float32)
        H = H.to(dtype=state.dtype)

        # turn world (x,y) into homogeneous coordinates
        world_xy1 = torch.cat([
            state[..., :2], 
            torch.ones_like(state[..., :1])
        ], dim=-1)
        # apply homography
        image_xyz = world_xy1 @ H.T
        # homogeneous -> euclidean
        w = image_xyz[..., 2:3]
        image_xy = image_xyz[..., :2] / w # torch.clamp(w, min=1e-8)
        covariance = self.compute_covariance(state, broadcast_covariance)
        return image_xy, covariance