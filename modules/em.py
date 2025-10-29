import torch
from typing import Tuple
import math

from modules import (
    Trajectory,
    BPF, BPF_DATA, BSPS, BSPS_DATA,
    wrap_angles
)

from .em import EM

class SSM_EM(EM):
    filter: BPF
    smoother: BSPS

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        state_angles: list,
        obs_angles: list,
        #
        initial_state: torch.nn.Module,
        transition: torch.nn.Module,
        observation: torch.nn.Module,
        #
        filter: BPF, 
        smoother: BSPS,
        optimizer,
        device = 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_angles = state_angles
        self.obs_angles = obs_angles
        self.initial_state = initial_state
        self.transition = transition
        self.observation = observation
        self.filter = filter
        self.smoother = smoother
        self.optimizer = optimizer
        self.device = device

    @torch.inference_mode()
    def e_step(self, trajectories: list) -> Tuple[list, list]:
        batch_filter_data, batch_smoother_data = [], []
        mean, covariance = self.initial_state()
        for trajectory in trajectories:
            filter_data = self.filter.filter(
                mean.clone(),
                covariance.clone(),
                trajectory.actions,
                trajectory.observations
            )
            smoother_data = self.smoother.smooth(filter_data, trajectory.actions) 
            batch_filter_data.append(filter_data)
            batch_smoother_data.append(smoother_data)
        return batch_filter_data, batch_smoother_data
    
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
        trajectories: list, 
        filter_data: list,
        smoother_data: list
    ) -> Tuple[torch.Tensor, float]:
        loss = torch.tensor(0.0)

        for i, trajectory in enumerate(trajectories):
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
    
    def _logpdf_gaussian(
        self, 
        residuals: torch.Tensor,  
        covariances: torch.Tensor
    ) -> torch.Tensor:
        '''
        Args:
            residuals: (batch_size, dim)
            covariance: (dim, dim) or (batch_size, dim, dim)
        '''
        batch_size, dim = residuals.shape
        const = dim * torch.log(torch.tensor(2.0 * math.pi))
        if covariances.ndim == 2:
            covariances = covariances.unsqueeze(0).expand(batch_size, dim, dim)
        
        sign, logabsdet = torch.linalg.slogdet(covariances)
        sol = torch.linalg.solve(covariances, residuals.unsqueeze(-1)).squeeze(-1)
        maha = (sol * residuals).sum(dim=-1)
        logpdf = -0.5 * (const + logabsdet + maha)
        return logpdf
    
    def _logpdf_student(
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
        nu = torch.as_tensor(self.filter.dof, dtype=torch.float32)
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

    def m_step(
        self, 
        trajectories: list[Trajectory], 
        filter_data: list,
        smoother_data: list,
        threshold: float | None = None
    ) -> float | None:
        if self.optimizer is None: return None
        self.optimizer.zero_grad(set_to_none=True)
        loss = self._loss(
            trajectories, 
            filter_data, 
            smoother_data
        )

        if threshold is not None and loss.item() > threshold:
            return None

        loss.backward()
        self.optimizer.step()
        return loss.item()
