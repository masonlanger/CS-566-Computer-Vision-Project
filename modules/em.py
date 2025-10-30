import torch
from typing import Tuple
import math

from .state_estimators import WorldTracker, TrackPosteriors
from .math import logpdf_gaussian, logpdf_student

class BatchEM:
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
        N, T, _ = samples.shape
        m_0, P_0 = self.initial_state()
        initial_state_residuals = wrap_angles(samples[:, 0, :] - m_0, self.state_angles)
        initial_state_term = self._logpdf_gaussian(initial_state_residuals, P_0)

        observation_term = torch.zeros(N)
        for t in range(T):
            action = torch.zeros(self.action_dim, dtype=torch.float32) if t == 0 else actions[t-1]
            expanded_observation = observations[t].unsqueeze(0).expand(N, -1)
            pred_observations, R = self.observation(
                samples[:, t, :], 
                action.unsqueeze(0).expand(N, -1),
                broadcast_covariance = False
            )
            # Assuming observation model is not learned so detaching gradient here.
            pred_observations.detach()
            R.detach()
            observation_residuals = wrap_angles(
                expanded_observation - pred_observations, 
                self.obs_angles
            )
            observation_term += self._logpdf_student(observation_residuals, R)

        transition_term = torch.zeros(N)
        for t in range(1, T):
            action, next_states = actions[t-1], samples[:, t, :]
            pred_states, Q = self.transition(
                samples[:, t-1, :], 
                action.unsqueeze(0).expand(N, -1),
                broadcast_covariance = False
            )
            transition_residuals = wrap_angles(
                next_states - pred_states, 
                self.state_angles
            )
            transition_term += self._logpdf_gaussian(transition_residuals, Q)
        return (initial_state_term + observation_term + transition_term).mean()

    def _loss(
        self, 
        batch_observations: list, 
        track_posteriors: list[list[TrackPosteriors]]
    ) -> Tuple[torch.Tensor, float]:
        loss = torch.tensor(0.0)
        for i, observations in enumerate(batch_observations):
            
            observations, actions = trajectory.observations, trajectory.actions
            # reshape to (num_samples, T, ...)
            samples = smoother_data[i].smoothed_trajectories.permute(1, 0, 2).contiguous()
            Q = self._mc_exp_log_likelihood(
                observations, 
                actions, 
                samples
            )
            loss -= Q
            
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
