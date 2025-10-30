import torch
from typing import Tuple
import math

from .state_estimators import WorldTracker, TrackPosteriors
from .math import logpdf_gaussian, logpdf_student

class BatchEM:
    '''
    EM over a batch of video sequences.
    '''
    state_dim = 4
    obs_dim = 2
    def __init__(
        self,
        transition: torch.nn.Module,
        observation: torch.nn.Module,
        world_tracker: WorldTracker,
        optimizer,
        device = 'cpu'
    ):
        self.transition = transition
        self.observation = observation
        self.world_tracker = world_tracker
        self.optimizer = optimizer
        self.device = device

    @torch.inference_mode()
    def e_step(self, batch_observations: list) -> list[list[TrackPosteriors]]:
        all_tracks = []
        for observations in batch_observations:
            tracks = self.world_tracker.filter(observations)
            self.world_tracker.process_tracks(tracks)
            tracks = self.world_tracker.smooth(tracks)
            all_tracks.append(tracks)
        return all_tracks
    
    def _mc_exp_log_likelihood(
        self, 
        observations: torch.Tensor,
        actions: torch.Tensor,
        samples: torch.Tensor
    ) -> torch.Tensor:
        # TODO
        ...

    def _loss(
        self, 
        batch_observations: list, 
        track_posteriors: list[list[TrackPosteriors]]
    ) -> Tuple[torch.Tensor, float]:
        loss = torch.tensor(0.0)
        for i, observations in enumerate(batch_observations):
            # TODO 
            ...
            
        return loss

    def m_step(
        self, 
        batch_observations: list, 
        track_posteriors: list[list[TrackPosteriors]]
    ) -> float | None:
        if self.optimizer is None: return None
        self.optimizer.zero_grad(set_to_none=True)
        loss = self._loss(
            batch_observations, 
            track_posteriors
        )
        loss.backward()
        self.optimizer.step()
        return loss.item()
