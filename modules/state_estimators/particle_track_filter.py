from dataclasses import dataclass, field
import torch
import rich
import math
import numpy as np
import sys
import multiprocessing as mp
from typing import Tuple
    
class ParticleTrackFilter:
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
        ess_threshold: float,
        device = 'cpu'
    ):
        self.state_dim = state_dim
        self.transition_model = transition_model
        self.observation_model = observation_model
        # filter hyperparameters
        self.num_particles = num_particles
        self.prediction_noise = prediction_noise
        self.nu = nu
        self.ess_threshold = ess_threshold
        self.device = device
    
    @staticmethod
    def _matrix_sqrt(M: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        evals, evecs = torch.linalg.eigh(M)
        evals = torch.clamp(evals, min=eps)
        return evecs @ torch.diag_embed(evals.sqrt()) @ evecs.transpose(-2, -1)
    
    @staticmethod
    def estimate_gaussian_density(
        particles: torch.Tensor,
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Estimates Gaussian density given particles and corresponding weights.
        '''
        weighted_particles = particles * weights.view(-1, 1)
        m = weighted_particles.sum(dim=0)
        residuals = particles - m
        P = torch.sum(weights.view(-1, 1, 1) * (residuals.unsqueeze(-1) * residuals.unsqueeze(-2)), dim=0)
        return m, P

    def generate_particles(self, m: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        L = self._matrix_sqrt(P)
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
        L = self._matrix_sqrt(Q)
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
        pred_observations = self.observation_model(particles)
        log_likelihoods = self._per_particle_log_likelihoods(true_observation, pred_observations)
        total_log_likelihood = torch.logsumexp(log_likelihoods, dim=0) - np.log(max(pred_observations.shape[0], 1))
        return float(total_log_likelihood)

    def _student_t_logpdf(
        self, 
        residuals: torch.Tensor, 
        covariances: torch.Tensor, 
        eps: float = 1e-9
    ) -> torch.Tensor:
        '''
        Args:
            residuals: (batch_size, dim)
            covariance: (dim, dim) or (batch_size, dim, dim)
        '''
        nu = torch.as_tensor(self.nu, dtype=torch.float32)
        batch_size, dim = residuals.shape
        if covariances.ndim == 2:
            covariances = covariances.unsqueeze(0).expand(batch_size, dim, dim)

        # add jitter for stability
        # covariances = 0.5 * (covariances + covariances.transpose(-1, -2)) + eps * torch.eye(dim)
        sign, logabsdet = torch.linalg.slogdet(covariances)
        sol = torch.linalg.solve(covariances, residuals.unsqueeze(-1)).squeeze(-1)
        maha = (sol * residuals).sum(dim=-1)
        const = (
            torch.lgamma((nu + dim) / 2) - torch.lgamma(nu / 2)
            - 0.5 * dim * torch.log(nu * torch.tensor(math.pi))
            - 0.5 * logabsdet
        )
        logpdf = const - 0.5 * (nu + dim) * torch.log1p(maha / nu)
        return logpdf

    def update(
        self, 
        particles: torch.Tensor, 
        observation: torch.Tensor
    ) -> torch.Tensor:
        '''
        Computes particle weights w.r.t. the associated obsevation.
        '''
        pred_observations = self.observation_model(particles)
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