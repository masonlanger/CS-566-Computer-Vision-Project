from dataclasses import dataclass
import torch
import rich
import sys
from typing import Tuple
import math

from ..math import logpdf_gaussian, particles_to_gaussian
from .track_posteriors import TrackPosteriors

class TrackSmoother:
    '''
    A per-track backward simulation particle smoother.
    '''
    def __init__(
        self, 
        state_dim: int,
        transition_model: torch.nn.Module,
        num_trajectories: int,
        device = 'cpu'
    ):
        self.state_dim = state_dim
        self.transition_model = transition_model
        self.num_trajectories = num_trajectories
        self.device = device

    @torch.inference_mode()
    def smooth(
        self,
        track: TrackPosteriors
    ):
        pre_resample_particles = track.pre_resample_particles
        weights = track.weights
        T, num_particles, _= pre_resample_particles.shape

        # initialize
        smoothed_trajectories = torch.empty(T, self.num_trajectories, self.state_dim)
        m_s = torch.empty(T, self.state_dim)
        P_s = torch.empty(T, self.state_dim, self.state_dim)

        # sample last state for each backward trajectory
        last_indices = torch.multinomial(weights[T-1], self.num_trajectories, replacement=True)
        smoothed_trajectories[T-1] = pre_resample_particles[T-1][last_indices]
        m_s[T-1], P_s[T-1] = particles_to_gaussian(smoothed_trajectories[T-1])
        
        # backward simulation
        for t in range(T-2, -1, -1):
            current_particles = pre_resample_particles[t]
            next_particles = smoothed_trajectories[t+1]
            # precompute pred. particles for all current particles
            pred_particles, Q = self.transition_model(
                current_particles,
                broadcast_covariance=False
            )

            # for each backward trajectory
            for j in range(self.num_trajectories):
                next_state = next_particles[j]
                residuals = next_state.unsqueeze(0) - pred_particles
                log_trans = logpdf_gaussian(residuals, Q)
                log_weights = torch.log(weights[t] + 1e-20)
                logits = log_weights + log_trans
                idx = torch.distributions.Categorical(logits=logits).sample()
                smoothed_trajectories[t, j] = current_particles[idx]
            
            m_s[t], P_s[t] = particles_to_gaussian(smoothed_trajectories[t])

        track.m_s = m_s
        track.P_s = P_s
        track.smoothed_trajectories = smoothed_trajectories