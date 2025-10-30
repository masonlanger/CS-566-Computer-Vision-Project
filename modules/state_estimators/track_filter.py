from dataclasses import dataclass, field
import torch
import rich
import math
import numpy as np
import sys
import multiprocessing as mp
from typing import Tuple

from ..math import logpdf_student, matrix_sqrt
    
class TrackFilter:
    '''
    A per-track bootstrap particle filter.
    Weights particles with a Student-t distribution for robustness.
    '''
    obs_dim = 2
    def __init__(
        self, 
        state_dim: int,
        transition_model: torch.nn.Module,
        observation_model: torch.nn.Module,
        num_particles: int,
        prediction_noise: float,
        nu: int,
        ess_scale: float,
        device = 'cpu'
    ):
        self.state_dim = state_dim
        self.transition_model = transition_model
        self.observation_model = observation_model
        # filter hyperparameters
        self.num_particles = num_particles
        self.prediction_noise = prediction_noise
        self.nu = nu
        self.ess_threshold = ess_scale * num_particles
        self.device = device

    def generate_particles(self, m: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        L = matrix_sqrt(P)
        noise = torch.randn(self.num_particles, self.state_dim, device=self.device)
        particles = m + noise @ L.T
        return particles

    def predict(
        self, 
        particles: torch.Tensor
    ) -> torch.Tensor: 
        pred_particles, Q = self.transition_model(
            particles,
            broadcast_covariance = False
        )
        L = matrix_sqrt(Q)
        pred_particles += torch.randn_like(pred_particles) @ L.T
        if self.prediction_noise > 0.0:
            prediction_noise = self.prediction_noise * torch.randn_like(pred_particles)
            pred_particles += prediction_noise
        return pred_particles
    

    def compute_log_likelihoods(
        self, 
        particles: torch.Tensor, 
        observation: torch.Tensor, 
        H: torch.Tensor
    ):
        pred_observations, R = self.observation_model(particles, H, broadcast_covariance=False)
        residuals = observation.unsqueeze(0) - pred_observations
        log_likelihoods = logpdf_student(residuals, R, self.nu)
        return log_likelihoods


    def update(
        self, 
        particles: torch.Tensor, 
        observation: torch.Tensor,
        H: torch.Tensor
    ) -> torch.Tensor:
        '''
        Computes particle weights w.r.t. the associated observation.
        '''
        log_likelihoods = self.compute_log_likelihoods(particles, observation, H)
        norm_log_weights = log_likelihoods - torch.logsumexp(log_likelihoods, dim=0) 
        weights = torch.exp(norm_log_weights)    
        return weights

    def uniform_weights(self) -> torch.Tensor:
        return torch.full(
            (self.num_particles,),
            1.0 / self.num_particles,
            device = self.device,
            dtype = torch.float32
        )
    
    def resample(
        self, 
        particles: torch.Tensor, 
        weights: torch.Tensor
    ) -> torch.Tensor:
        ess = 1.0 / (torch.sum(weights ** 2) + 1e-12)
        if ess < self.ess_threshold * self.num_particles:
            idx = torch.multinomial(weights, num_samples=self.num_particles, replacement=True)
        else:
            # dont resample
            idx = torch.arange(self.num_particles)
        return particles[idx]