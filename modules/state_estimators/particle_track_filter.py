from dataclasses import dataclass, field
import torch
import rich
import math
import numpy as np
import sys
import multiprocessing as mp
from typing import Tuple

def matrix_sqrt_psd(M: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(M)
    evals = torch.clamp(evals, min=eps)
    return evecs @ torch.diag_embed(evals.sqrt())
    
class ParticleTrackFilter:
    '''
    A bootstrap particle filter that uses a Student's t-distribution.
    '''
    def __init__(
        self, 
        obs_dim: int,
        state_dim: int,
        transition, 
        transition_noise,
        observation,
        observation_noise,
        num_particles: int,
        prediction_noise: float,
        nu: int,
        ess_threshold: float,
        device = 'cpu'
    ):
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.transition = transition
        self.observation = observation
        self.transition_noise = transition_noise
        self.observation_noise = observation_noise

        # hyperparameters
        self.num_particles = num_particles
        self.prediction_noise = prediction_noise
        self.nu = nu
        self.ess_threshold = ess_threshold
        self.device = device

    def generate_particles(self, m: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        L = matrix_sqrt_psd(P)
        noise = torch.randn(self.num_particles, self.state_dim, device=self.device)
        particles = m + noise @ L.T
        return particles

    def predict(self, particles) -> torch.Tensor: 
        pred_particles = self.transition(particles)
        L = matrix_sqrt_psd(self.transition_noise)
        pred_particles += torch.randn_like(pred_particles) @ L.T
        if self.prediction_noise > 0.0:
            prediction_noise = self.prediction_noise * torch.randn_like(pred_particles)
            pred_particles += prediction_noise
        return pred_particles
    
    def _per_particle_log_likelihoods(
        self, 
        true_observation: torch.Tensor,
        pred_observations: torch.Tensor
    ) -> torch.Tensor:
        '''
        Computes the log-likelihood of the true observation under each of the predicted observations.
        '''
        diff = true_observation[None, :] - pred_observations
        return self._student_t_logpdf(diff, self.observation_noise)
    
    def _log_likelihood(self, particles: torch.Tensor, true_observation: torch.Tensor) -> float:
        '''
        Computes the log-likelihood of an observation under the predicted belief of a track.
        Note: The belief is represented by the set of particles so we marginalize over all particles.
        '''
        pred_observations = self.observation(particles)
        log_likelihoods = self._per_particle_log_likelihoods(true_observation, pred_observations)
        loglik = torch.logsumexp(log_likelihoods, dim=0) - np.log(max(pred_observations.shape[0], 1))
        return float(loglik)

    def _student_t_logpdf(
        self, 
        diff: torch.Tensor, 
        Sigma: torch.Tensor
    ) -> torch.Tensor:
        '''
        Computes the log-PDF of a multivariate Student-t distribution at residual 'diff' with scale 'Sigma'.
        Note: I'm using Student-t instead of Gaussian because its more resilient to outliers which becomes an issue when learning the transition model.
        '''
        dim = diff.shape[-1]
        jitter = 1e-6 * torch.eye(dim, dtype=torch.float32)
        L = torch.linalg.cholesky(Sigma + jitter)

        if diff.dim() == 1:
            sol = torch.cholesky_solve(diff.unsqueeze(-1), L)
            q = (diff.unsqueeze(-1) * sol).sum()
        else:
            sol = torch.cholesky_solve(diff.transpose(0, 1), L).transpose(0, 1)
            q = (diff * sol).sum(dim=-1)

        log_det = 2.0 * torch.log(torch.diag(L)).sum()
        logC = (
            torch.lgamma(torch.tensor((self.nu + dim) / 2.0, dtype=torch.float32))
            - torch.lgamma(torch.tensor(self.nu / 2.0, dtype=torch.float32))
            - (dim / 2.0) * torch.log(torch.tensor(self.nu * np.pi, dtype=torch.float32))
            - 0.5 * log_det
        )
        log_kernel = -0.5 * (self.nu + dim) * torch.log1p(q / self.nu)
        return logC + log_kernel

    def update(
        self, 
        particles: torch.Tensor, 
        observation: torch.Tensor
    ):
        '''
        Computes particle weights w.r.t. the associated obsevation.
        '''
        pred_observations = self.observation(particles)
        logp = self._per_particle_log_likelihoods(observation, pred_observations)
        logw_post = logp - np.log(max(self.num_particles, 1))
        logw_post = logw_post - torch.logsumexp(logw_post, dim=0)
        w_post = torch.exp(logw_post)
        return w_post

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