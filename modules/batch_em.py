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
        transition_model: torch.nn.Module,
        observation_model: torch.nn.Module,
        world_tracker: WorldTracker,
        optimizer,
        device = 'cpu'
    ):
        self.transition_model = transition_model
        self.observation = observation_model
        self.world_tracker = world_tracker
        self.optimizer = optimizer
        self.device = device

    @torch.inference_mode()
    def e_step(self, detections: list, homographies: list) -> list[list[TrackPosteriors]]:
        assert len(detections) == len(homographies)
        N = len(detections)
        posteriors = []
        for i in range(N):
            tracks = self.world_tracker.filter(detections[i], homographies[i])
            self.world_tracker.process_tracks(tracks)
            tracks = self.world_tracker.smooth(tracks)
            posteriors.append(tracks)
        return posteriors
    
    def _mc_exp_log_likelihood(
        self, 
        detections: torch.Tensor,
        homographies: torch.Tensor,
        posteriors: TrackPosteriors
    ) -> torch.Tensor:
        
        samples = posteriors.smoothed_trajectories.permute(1, 0, 2).contiguous()
        N, T, _ = samples.shape
        m_0 = posteriors.initial_state
        P_0 = posteriors.initial_state_noise
        initial_state_residuals = samples[:, 0, :] - m_0
        initial_state_term = logpdf_gaussian(initial_state_residuals, P_0)

        observation_term = torch.zeros(N)
        for t in range(T):
            H = homographies[t]
            expanded_observation = detections[t].unsqueeze(0).expand(N, -1)
            # disabling gradients because assuming observation model is not learned for now
            with torch.no_grad():
                pred_observations, R = self.observation_model(
                    samples[:, t, :], 
                    H,
                    broadcast_covariance = False
                )

            observation_residuals = expanded_observation - pred_observations
            observation_term += logpdf_student(observation_residuals, R)

        transition_term = torch.zeros(N)
        for t in range(1, T):
            pred_states, Q = self.transition_model(
                samples[:, t-1, :],
                broadcast_covariance = False
            )
            transition_residuals = samples[:, t, :] - pred_states
            transition_term += logpdf_gaussian(transition_residuals, Q)
        return (initial_state_term + observation_term + transition_term).mean()


    def _loss(
        self, 
        detections: list, 
        homographies: list,
        posteriors: list[list[TrackPosteriors]]
    ) -> Tuple[torch.Tensor, float]:
        assert len(detections) == len(homographies) == len(posteriors)
        # num. videos in batch
        N = len(detections)
        loss = torch.tensor(0.0)
        # looping through each video
        for n in range(N):
            # total tracks in i-th video
            video_detections = detections[n]
            video_homographies = homographies[n]
            video_posteriors = posteriors[n]
            M = len(video_posteriors)
            # looping through each track
            for m in range(M): 
                Q = self._mc_exp_log_likelihood(
                    video_detections, 
                    video_homographies, 
                    video_posteriors[m]
                )
                loss -= Q
        
        return loss

    def m_step(
        self, 
        detections: list, 
        homographies: list,
        posteriors: list
    ) -> float | None:
        if self.optimizer is None: return None
        self.optimizer.zero_grad(set_to_none=True)
        loss = self._loss(
            detections, 
            homographies,
            posteriors
        )
        loss.backward()
        self.optimizer.step()
        return loss.item()
