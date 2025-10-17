from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import torch
import supervision as sv
from scipy.optimize import linear_sum_assignment

from supervision.detection.core import Detections
from modules.view_transformer import ViewTransformer
import copy

from modules.particle_filter import ParticleFilter

@dataclass
class Track:
    id: int
    particles: torch.Tensor # (N, Dx)
    associated: bool

    def mean(self):
        return self.particles.mean(dim=0)

    def cov(self):
        # Uses unbiased; not important here since we only use it diagnostically
        return torch.cov(self.particles.T)
    
@dataclass
class History:
    tracks: List[List[Track]] = field(default_factory=list)
    observations: List[np.ndarray] = field(default_factory=list)
    associations: List[List[Tuple[int, int]]] = field(default_factory=list)

class HeuristicTransition(torch.nn.Module):
    def __init__(self, dt = 1.0):
        super().__init__()
        self.dt = dt

    # constant-velocity for now
    def forward(self, state):
        x, y, x_vel, y_vel = state.unbind(dim=-1)
        next_x = x + x_vel * self.dt
        next_y = y + y_vel * self.dt
        return torch.stack([next_x, next_y, x_vel, y_vel], dim=-1).to(torch.float32)
    
class Observation(torch.nn.Module):
    def forward(self, state):
        x, y, x_vel, y_vel = state.unbind(dim=-1)
        return torch.stack([x, y], dim=-1).to(torch.float32)

class WorldTrack:

    def __init__(self, config):
        self.config = config

        self.obs_dim = 2
        self.state_dim = 4 

        self.initial_state_noise = torch.eye(self.state_dim) * 100.0
        self.observation_noise = torch.eye(self.obs_dim) * 0.1  # R
        self.transition_noise = torch.eye(self.state_dim) * 100.0

        self.transition = HeuristicTransition()
        self.observation = Observation()

        # this is a per-track filter
        self.filter = ParticleFilter(
            obs_dim = self.obs_dim,
            state_dim = self.state_dim,
            transition = self.transition, 
            transition_noise = self.transition_noise,
            observation = self.observation,
            observation_noise = self.observation_noise,
            num_particles = 100,
            prediction_noise = 0.0,
            nu = 5,
            ess_threshold = 0.5
        )

        self.tracks: List[Track] = []
        self.history = History()

        # data association hyperparameters
        self.chi2_gate = 9.21 # 95% ~ 5.99, 99% ~ 9.21, 99.7% ~ 11.83
        self.large_cost = 1e6 # cost for invalid association
        self.image_belief_size = 25

    def reset(self):
        self.history = History()
    
    def _gate_associations(self, observations: torch.Tensor) -> torch.Tensor:
        '''
        Computes a boolean mask that indicates plausible track <-> observation associations.
        Simplifies association step by making unrealistic associations impossible.
        '''
        K, M = len(self.tracks), observations.shape[0]
        if K == 0 or M == 0:
            return torch.zeros((K, M), dtype=torch.bool)

        valid = torch.zeros((K, M), dtype=torch.bool)
        R = self.filter.observation_noise

        for k, track in enumerate(self.tracks):
            # predict the observation corresponding to each particle
            pred_observations = self.filter.observation(track.particles)
            mu = pred_observations.mean(dim=0)
            S = torch.cov(pred_observations.T) + R

            # precompute inverse
            jitter = 1e-6 * torch.eye(self.filter.obs_dim, dtype=torch.float32)
            S_inv = torch.linalg.inv(S + jitter)

            for m, observation in enumerate(observations):
                diff = observation - mu
                d2 = float(diff @ (S_inv @ diff))
                # if the predicted observation is within chi2_gate of the true observation
                # this is a plausible association
                valid[k][m] = (d2 <= self.chi2_gate)
        return valid

    def _build_cost_matrix(self, observations: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        '''
        Builds the Hungarian cost matrix. 
        Invalid associations are set to self.large_cost.
        '''
        K = len(self.tracks)
        M = observations.shape[0]
        C = torch.full((K, M), self.large_cost, dtype=torch.float32)
        for k, track in enumerate(self.tracks):
            for m, observation in enumerate(observations):
                if mask is None or mask[k, m]:
                    log_likelihood = self.filter._log_likelihood(track.particles, observation)
                    C[k, m] = -float(log_likelihood)
        return C

    def _get_associations(self, C: torch.Tensor) -> List[Tuple[int, int]]:
        '''
        Solves min-cost assignment with the Hungarian algorithm.
        Returns a list of (track, observation) tuples indicating associations.
         
        Note: If K != M, Hungarian will still return min(#rows, #cols) matches.
        We'll treat unassigned tracks/detections explicitly after.
        '''
        if C.numel() == 0: return []
        row_idx, col_idx = linear_sum_assignment(C.cpu().numpy())
        pairs = []
        for r, c in zip(row_idx, col_idx):
            if C[r, c] >= self.large_cost * 0.5:
                # treat as unassigned if it's essentially forbidden
                continue
            pairs.append((int(r), int(c)))
        return pairs

    def _initial_birth(self, observations: torch.Tensor):
        '''
        Initializes all tracks using the initial observations.
        '''
        for observation in observations:
            mean = torch.tensor([observation[0], observation[1], 0.0, 0.0], dtype=torch.float32)
            cov = self.initial_state_noise.clone()
            particles = self.filter.generate_particles(mean, cov)  # expected (N, Dx)
            track = Track(id=len(self.tracks), particles=particles, associated=False)
            self.tracks.append(track)

    def update_with_detections(self, keypoints: sv.KeyPoints, detections: Detections) -> Detections:

        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
        transformer = ViewTransformer(
            source=keypoints.xy[0][mask].astype(np.float32),
            target=np.array(self.config.vertices)[mask].astype(np.float32)
        )
        xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        observations = torch.tensor(transformer.transform_points(points=xy), dtype=torch.float32)

        self.history.observations.append(observations)

        # num. detections/observations
        n = observations.shape[0]

        # if we have no tracks yet but have observations, birth them
        if n > 0 and len(self.tracks) == 0:
            self._initial_birth(observations)
            # don't gate initial beliefs
            C = self._build_cost_matrix(observations)
            associations = self._get_associations(C)
            self.history.associations.append(associations)
            detections.tracker_id = np.empty((n,), dtype=int)
            for (k, m) in associations:
                track = self.tracks[k]
                track.associated = True
                detections.tracker_id[m] = self.tracks[k].id

            self.history.tracks.append(copy.deepcopy(self.tracks))
            return detections

        # predict for all tracks
        for track in self.tracks:
            track.particles = self.filter.predict(track.particles)

        # gate predictions
        # valid_mask = self._gate_associations(observations)
        valid_mask = None

        # data association
        C = self._build_cost_matrix(observations, mask=valid_mask)  # (K, M)
        associations = self._get_associations(C)
        self.history.associations.append(associations)

        # update tracks that are associated with an observation
        for (k, m) in associations:
            track, observation = self.tracks[k], observations[m]
            track.associated = True
            weights = self.filter.update(track.particles, observation)
            track.particles = self.filter.resample(track.particles, weights)
        # tracks that are not associated just carry prediction forward


        # for all associations, set detection.tracker_id to the track id
        detections.tracker_id = np.full((n,), -1, dtype=int)
        for (k, m) in associations:
            detections.tracker_id[m] = self.tracks[k].id

        # if an observation is not associated with any track, birth a new track
        unassociated_observations = [
            i for i in range(n) 
            if i not in set(o for _, o in associations)
        ]
        for i in unassociated_observations:
            mean = torch.tensor([observations[i, 0], observations[i, 1], 0.0, 0.0], dtype=torch.float32)
            cov = self.initial_state_noise.clone()
            particles = self.filter.generate_particles(mean, cov)
            track = Track(id=len(self.tracks), particles=particles, associated=False)
            detections.tracker_id[i] = track.id
            self.tracks.append(track)

        # for tracks that are not associated with any observation
        # this means they were not detected
        # so add fake detection by computing world belief -> image space
        unassociated_tracks = [
            i for i in range(len(self.tracks)) 
            if i not in set(t for t, _ in associations)
        ]

        if len(unassociated_tracks) > 0:
            global_xy = np.array([self.tracks[i].mean().numpy()[:2] for i in unassociated_tracks])
            image_xy = transformer.inverse_transform_points(points=global_xy)
            new_detections = sv.Detections(
                xyxy=np.array([[
                    u - self.image_belief_size/2, 
                    v - self.image_belief_size/2, 
                    u + self.image_belief_size/2, 
                    v + self.image_belief_size/2
                ] for (u, v) in image_xy]),
                confidence=np.array(
                    [1.0] * len(unassociated_tracks)
                ),
                class_id=np.array(
                    [2] * len(unassociated_tracks)
                ),
                tracker_id=np.array(
                    [self.tracks[i].id for i in unassociated_tracks]
                , dtype=int)
            )
            # remove extra data to enable merge
            detections.data, detections.metadata = {}, {}
            detections = sv.Detections.merge([detections, new_detections])
        
        self.history.tracks.append(copy.deepcopy(self.tracks))
        return detections
