from dataclasses import dataclass
import torch
from scipy.optimize import linear_sum_assignment

from .particle_track_filter import ParticleTrackFilter

@dataclass
class TrackPosteriors:
    id: int 
    initial_state: torch.Tensor
    initial_state_noise: torch.Tensor
    m: list[torch.Tensor]
    P: list[torch.Tensor]
    pred_particles: list[torch.Tensor]
    particles: list[torch.Tensor]
    # the element at index t corresponds to the index of 
    # the observation that this track was associated with at time t
    associations: list[int]

class WorldFilter:
    obs_dim = 2
    def __init__(
        self, 
        initial_state_noise: float,
        track_filter: ParticleTrackFilter, 
    ):
        self.initial_state_noise = initial_state_noise
        self.track_filter = track_filter

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
                        track.pred_particles[-1], 
                        detection
                    )
                    C[m, n] = -float(log_likelihood)
        return C

    def _get_associations(
        self, 
        C: torch.Tensor
    ) -> list:
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
                particles = [particles], 
                m = [],
                P = [],
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
                track.pred_particles.append(
                    self.track_filter.predict(track.particles[-1])
                )

            valid_mask = None
            C = self._build_cost_matrix(tracks, observations, mask=valid_mask)
            associations = self._get_associations(C)
            for (track_idx, obs_idx) in associations:
                track, detection = tracks[track_idx], observation[obs_idx]
                weights = self.track_filter.update(
                    track.pred_particles[-1], 
                    detection
                )
                particles = self.track_filter.resample(
                    track.pred_particles[-1], 
                    weights
                )
                track.particles.append(particles)
                track.associations.append(obs_idx)

            
            associated_observations = {o for _, o in associations}
            unassociated_observations = [
                i for i in range(N) 
                if i not in associated_observations
            ]

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
                    particles = [particles], 
                    m = [],
                    P = [],
                    # on birth, the track can be automatically associated with 
                    # the observation that 'birthed' it
                    associations = [i]
                )

        return tracks

            



            



