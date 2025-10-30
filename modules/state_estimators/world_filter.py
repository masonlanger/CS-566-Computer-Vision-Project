from dataclasses import dataclass
import torch

from .particle_track_filter import ParticleTrackFilter

@dataclass
class TrackTrajectory:
    id: int 
    initial_state: torch.Tensor
    initial_state_noise: torch.Tensor
    m: list[torch.Tensor]
    P: list[torch.Tensor]
    particles: list[torch.Tensor]
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
        
    def _initial_birth(self, observation: torch.Tensor, T: int):
        '''
        Initializes all tracks using the initial observations.
        '''
        new_tracks = []
        for player in observation:
            mean = torch.tensor([player[0], player[1], 0.0, 0.0], dtype=torch.float32)
            cov = self.initial_state_noise.clone()
            particles = self.track_filter.generate_particles(mean, cov)  # expected (N, Dx)
            track = TrackTrajectory(
                id = len(new_tracks), 
                initial_state = mean,
                initial_state_noise = cov,
                particles = [particles], 
                m = [],
                P = []
            )
            new_tracks.append(track)
    
    @torch.inference_mode()
    def filter(self, observations: list) -> list:
        T = len(observations)

        tracks = []

        for t in range(T):
            observation = observations[t]
            N = observation.shape[0]

            if N > 0 and len(tracks) == 0:
                new_tracks = self._initial_birth(observation, T)
                tracks.extend(new_tracks)


            



