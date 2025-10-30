from dataclasses import dataclass, field
import torch

@dataclass
class TrackPosteriors:
    id: int 
    initial_state: torch.Tensor
    initial_state_noise: torch.Tensor

    # the element at index t corresponds to the index of 
    # the observation that this track was associated with at time t
    associations: list | torch.Tensor
    
    particles: list | torch.Tensor = field(default_factory=list)
    pre_resample_particles: list | torch.Tensor = field(default_factory=list)
    weights: list | torch.Tensor = field(default_factory=list)
    m_f: list | torch.Tensor = field(default_factory=list)
    P_f: list | torch.Tensor = field(default_factory=list)
    m_s: torch.Tensor | None = None
    P_s: torch.Tensor | None = None
    smoothed_trajectories: torch.Tensor = None
