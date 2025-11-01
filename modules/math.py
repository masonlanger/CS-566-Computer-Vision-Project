import torch
import math
from typing import Tuple

def apply_homography(
    points: torch.Tensor, 
    H: torch.Tensor
) -> torch.Tensor:
    ones = torch.ones(*points.shape[:-1], 1, dtype=points.dtype, device=points.device)
    homo = torch.cat([points, ones], dim=-1)
    projections = homo @ H.T
    projections = projections[..., :2] / projections[..., 2:]
    return projections

def logpdf_student(
    residuals: torch.Tensor, 
    covariances: torch.Tensor, 
    nu: int,
    eps: float = 1e-9
) -> torch.Tensor:
    '''
    Args:
        residuals: (batch_size, dim)
        covariance: (dim, dim) or (batch_size, dim, dim)
    '''
    nu = torch.as_tensor(nu, dtype=torch.float32)
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

def logpdf_gaussian(
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

def matrix_sqrt(M: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(M)
    evals = torch.clamp(evals, min=eps)
    return evecs @ torch.diag_embed(evals.sqrt()) @ evecs.transpose(-2, -1)

def to_gaussian(
    particles: torch.Tensor,
    weights: torch.Tensor | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Estimates Gaussian density given particles and corresponding weights.
    If weights are not provided, just uses particles.
    '''
    N, _ = particles.shape
    if weights is None:
        m = particles.mean(dim=0)
        residuals = particles - m
        P = (residuals.T @ residuals) / (N - 1)
    else:
        weighted_particles = particles * weights.view(-1, 1)
        m = weighted_particles.sum(dim=0)
        residuals = particles - m
        P = torch.sum(weights.view(-1, 1, 1) * (residuals.unsqueeze(-1) * residuals.unsqueeze(-2)), dim=0)
        
    return m, P