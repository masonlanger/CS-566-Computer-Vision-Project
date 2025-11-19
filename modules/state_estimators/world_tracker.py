from dataclasses import dataclass, field
import torch
from scipy.optimize import linear_sum_assignment
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

from ..math import particles_to_gaussian, apply_homography
from ..logger import Logger
from .track_filter import TrackFilter
from .track_smoother import TrackSmoother
from .track_posteriors import TrackPosteriors

class WorldTracker:
    '''
    This performs state inference at the video level.
    It manages the per-track filters/smoothers and performs data association.
    '''
    obs_dim = 2
    def __init__(
        self, 
        initial_state_noise: float,
        track_filter: TrackFilter, 
        track_smoother: TrackSmoother
    ):
        self.initial_state_noise = initial_state_noise
        self.track_filter = track_filter
        self.track_smoother = track_smoother
        # data association hyperparameters
        self.chi2_gate = 9.21
        self.large_cost = 1e6

    def _build_cost_matrix(
        self, 
        tracks: list,
        observation: torch.Tensor, 
        H: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        '''
        Builds the Hungarian cost matrix. 
        Invalid associations are set to self.large_cost.
        '''
        M = len(tracks)
        N = observation.shape[0]
        C = torch.full((M, N), self.large_cost, dtype=torch.float32)
        for m, track in enumerate(tracks):
            for n, detection in enumerate(observation):
                if mask is None or mask[m, n]:
                    log_likelihoods, pred_observations = self.track_filter.per_particle_log_likelihoods(
                        track.pre_resample_particles[-1], 
                        detection,
                        H
                    )
                    log_likelihood = torch.logsumexp(log_likelihoods, dim=0) \
                                   - math.log(log_likelihoods.numel())
                    C[m, n] = -float(log_likelihood)
        return C

    def _get_associations(
        self, 
        C: torch.Tensor
    ) -> dict:
        '''
        Solves min-cost assignment with the Hungarian algorithm.
        Returns a list of (track, observation) tuples indicating associations.
        Note: If K != M, Hungarian will still return min(#rows, #cols) matches.
        We'll treat unassigned tracks/detections explicitly after.
        '''
        if C.numel() == 0: return {}
        row_idx, col_idx = linear_sum_assignment(C.cpu().numpy())
        associations = {}
        for r, c in zip(row_idx, col_idx):
            if C[r, c] >= self.large_cost * 0.5:
                # treat as unassigned if it's essentially forbidden
                continue
            associations[int(r)] = int(c)
        return associations
        
    def _initial_birth(self, observation: torch.Tensor, H: torch.Tensor, t: int):
        '''
        Initializes all tracks using the initial observations.
        '''
        new_tracks = []
        for i, detection in enumerate(observation):

            # reshaped_points = np.array([detection]).reshape(-1, 1, 2).astype(np.float32)
            # world_xy = cv2.perspectiveTransform(reshaped_points, np.linalg.inv(H.numpy()))
            # world_xy.reshape(-1, 2).astype(np.float32)
            # world_xy = torch.tensor(world_xy).squeeze()

            world_xy = apply_homography(detection, torch.inverse(H))
            track = TrackPosteriors(
                id = len(new_tracks), 
                initial_state = torch.tensor(
                    [world_xy[0], world_xy[1], 0.0, 0.0], 
                    dtype=torch.float32
                ),
                initial_state_noise = self.initial_state_noise.clone(),
                birth_step = t + 1
                # on birth, the track can be automatically associated with 
                # the observation that 'birthed' it
                # associations = [i]
            )
            new_tracks.append(track)
            break

        return new_tracks
    
    @torch.inference_mode()
    def filter(self, detections: list, homographies: list) -> list:
        T = len(detections) 
        tracks = []

        for t in range(T):
            observation = detections[t]
            H = homographies[t]
            # num. of detections
            N = observation.shape[0]

            if N > 0 and len(tracks) == 0:
                new_tracks = self._initial_birth(observation, H, t)
                tracks.extend(new_tracks)
                Logger.debug(f't={t}: Created {len(new_tracks)} initial tracks.')
                # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                # for track in tracks:
                #     m = track.initial_state[:2]
                #     P = track.initial_state_noise[:2, :2]
                #     plot_gaussian_2d(ax, m, P)
                # ax.grid(True, alpha=0.2)
                # ax.axis('equal')
                # Logger.save_fig(fig, f'test')
                continue

            for track in tracks:
                if track.particles:
                    prev_particles = track.particles[-1]
                else:
                    # if track.particles is empty this means this is the
                    # first prediction step for this particular track 
                    # so we generate particles from its initial state distribution
                    prev_particles = self.track_filter.generate_particles(
                        track.initial_state, 
                        track.initial_state_noise
                    )

                pred_particles = self.track_filter.predict(prev_particles)
                track.pre_resample_particles.append(pred_particles)

            C = self._build_cost_matrix(
                tracks, 
                observation, 
                H,
                # not gating associations for now
                mask = None 
            )
            associations = self._get_associations(C)
            # breakpoint()

            associated_tracks = associations.keys()
            associated_observations = associations.values()
            unassociated_observations = [
                i for i in range(N) 
                if i not in associated_observations
            ]

            for track in tracks:
                associated = track.id in associated_tracks
                pre_resample_particles = track.pre_resample_particles[-1]
                if associated:
                    obs_idx = associations[track.id]
                    detection = observation[obs_idx]
                    weights, pred_observations = self.track_filter.update(
                        pre_resample_particles,
                        detection,
                        H
                    )
                    particles = self.track_filter.resample(
                        pre_resample_particles,
                        weights
                    )
                    association = obs_idx
                else:
                    # do not resample if theres no association
                    particles = pre_resample_particles.clone()
                    weights = self.track_filter.uniform_weights()
                    association = -1

                m, P = particles_to_gaussian(pre_resample_particles, weights)
                track.m_f.append(m)
                track.P_f.append(P)
                track.particles.append(particles)
                track.weights.append(weights)
                track.associations.append(association)

                # if association != -1 and track.confirmation_step is None:
                #     track.confirmation_step = t
            
            for i in unassociated_observations:
                detection = observation[i]
                world_xy = apply_homography(detection, torch.linalg.inv(H))
                track = TrackPosteriors(
                    id = len(tracks), 
                    initial_state = torch.tensor(
                        [world_xy[0], world_xy[1], 0.0, 0.0], 
                        dtype=torch.float32
                    ),
                    initial_state_noise = self.initial_state_noise.clone(),
                    birth_step = t + 1
                    # on birth, the track can be automatically associated with 
                    # the observation that 'birthed' it
                    # associations = [i]
                )
                tracks.append(track)

            if unassociated_observations:
                Logger.debug(f't={t}: Created {len(unassociated_observations)} new tracks.')

        return tracks

    def process_tracks(self, tracks: list[TrackPosteriors]):
        '''
        Turn track lists into tensors and verify shapes are correct.
        '''
        for track in tracks:
            track.particles = torch.stack(track.particles).to(torch.float32)
            track.associations = torch.tensor(track.associations).to(torch.int8)
            track.pre_resample_particles = torch.stack(track.pre_resample_particles).to(torch.float32)
            track.weights = torch.stack(track.weights).to(torch.float32)
            track.m_f = torch.stack(track.m_f).to(torch.float32)
            track.P_f = torch.stack(track.P_f).to(torch.float32)
            assert (
                track.associations.shape[0]
                == track.particles.shape[0]
                == track.pre_resample_particles.shape[0]
                == track.weights.shape[0]
                == track.m_f.shape[0]
                == track.P_f.shape[0]
            )

        return tracks

    @torch.inference_mode()
    def smooth(self, tracks: list) -> list: 
        for track in tracks:
            # smoother modifies in-place
            self.track_smoother.smooth(track)
        return tracks
        