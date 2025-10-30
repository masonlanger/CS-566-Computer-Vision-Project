from dataclasses import dataclass, field
import torch
from scipy.optimize import linear_sum_assignment

from ..math import to_gaussian
from .particle_track_filter import ParticleTrackFilter
from .particle_track_smoother import ParticleTrackSmoother

@dataclass
class TrackPosteriors:
    id: int 
    initial_state: torch.Tensor
    initial_state_noise: torch.Tensor

    # the element at index t corresponds to the index of 
    # the observation that this track was associated with at time t
    associations: list | torch.Tensor
    
    particles: list | torch.Tensor = field(default_factory=list)
    pre_resample_particles: list | torch.Tensor = field(default_factory=list)
    weights: list | torch.Tensor = field(default_factory=list)
    m_f: list | torch.Tensor = field(default_factory=list)
    P_f: list | torch.Tensor = field(default_factory=list)
    m_s: torch.Tensor | None = None
    P_s: torch.Tensor | None = None
    smoothed_trajectories: torch.Tensor = None

class WorldTracker:
    '''
    A per-video filter/smoother.
    '''
    obs_dim = 2
    def __init__(
        self, 
        initial_state_noise: float,
        track_filter: ParticleTrackFilter, 
        track_smoother: ParticleTrackSmoother
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
                    log_likelihood = self.track_filter._log_likelihood(
                        track.pre_resample_particles[-1], 
                        detection
                    )
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
        if C.numel() == 0: return []
        row_idx, col_idx = linear_sum_assignment(C.cpu().numpy())
        associations = {}
        for r, c in zip(row_idx, col_idx):
            if C[r, c] >= self.large_cost * 0.5:
                # treat as unassigned if it's essentially forbidden
                continue
            associations[int(r)] = int(c)
        return associations
        
    def _initial_birth(self, observation: torch.Tensor, T: int):
        '''
        Initializes all tracks using the initial observations.
        '''
        new_tracks = []
        for i, detection in enumerate(observation):
            mean = torch.tensor(
                [detection[0], detection[1], 0.0, 0.0], 
                dtype=torch.float32
            )
            cov = self.initial_state_noise.clone()
            particles = self.track_filter.generate_particles(mean, cov)  # expected (N, Dx)
            track = TrackPosteriors(
                id = len(new_tracks), 
                initial_state = mean,
                initial_state_noise = cov,
                # on birth, the track can be automatically associated with 
                # the observation that 'birthed' it
                associations = [i]
            )
            new_tracks.append(track)
    
    @torch.inference_mode()
    def filter(self, observations: list) -> list:
        T = len(observations)
        tracks = []

        for t in range(T):
            observation = observations[t]
            # num. of detections
            N = observation.shape[0]

            if N > 0 and len(tracks) == 0:
                new_tracks = self._initial_birth(observation, T)
                tracks.extend(new_tracks)
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

            valid_mask = None
            C = self._build_cost_matrix(tracks, observations, mask=valid_mask)
            associations = self._get_associations(C)

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
                    weights = self.track_filter.update(
                        pre_resample_particles,
                        detection
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

                m, P = to_gaussian(pre_resample_particles, weights)
                track.m_f.append(m)
                track.P_f.append(P)
                track.particles.append(particles)
                track.weights.append(weights)
                track.associations.append(association)

            for i in unassociated_observations:
                mean = torch.tensor(
                    [observation[i, 0], observation[i, 1], 0.0, 0.0], 
                    dtype=torch.float32
                )
                cov = self.initial_state_noise.clone()
                particles = self.track_filter.generate_particles(mean, cov)
                track = TrackPosteriors(
                    id = len(tracks), 
                    initial_state = mean,
                    initial_state_noise = cov,
                    # on birth, the track can be automatically associated with 
                    # the observation that 'birthed' it
                    associations = [i],
                    particles = [particles]
                )

        return tracks

    def process_tracks(self, tracks: list[TrackPosteriors]):
        '''
        Turn track lists into tensors and verify shapes are correct.
        '''
        for track in tracks:
            track.particles = torch.stack(track.particles, dtype=torch.float32)
            track.associations = torch.stack(track.associations, dtype=torch.int8)
            track.pre_resample_particles = torch.stack(track.pre_resample_particles, dtype=torch.float32)
            track.weights = torch.stack(track.weights, dtype=torch.float32)
            track.m_f = torch.stack(track.m_f, dtype=torch.float32)
            track.P_f = torch.stack(track.P_s, dtype=torch.float32)
            assert (
                track.associations.shape[0] - 1
                == track.particles.shape[0]
                == track.pre_resample_particles.shape[0]
                == track.weights.shape[0]
                == track.m_f.shape[0]
                == track.P_f.shape[0]
            )

        return tracks

    @torch.inference_mode()
    def smooth(self, tracks: list): 
        for track in tracks:
            # smoother modifies in-place
            self.track_smoother(track)
        return tracks
        