from dataclasses import dataclass
import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
from sports.configs.soccer import SoccerPitchConfiguration
import supervision as sv
import torch
from tqdm import tqdm
from omegaconf import OmegaConf as OC

from procedures import Procedure, register
from modules import (
    Logger, ViewTransformer,
    TransitionModel, HeuristicTransitionModel, CameraObservationModel,
    TrackFilter, TrackSmoother, WorldTrack,
    TrackPosteriors,
    animate_world_track
)

@register('eval_world_track')
class EvalWorldTrack(Procedure):

    PLAYER_CLASS_ID = 2
    field_config = SoccerPitchConfiguration()

    # ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    #     color=sv.ColorPalette.from_hex(COLORS),
    #     thickness=2
    # )   

    WT_ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
        color=sv.Color(255, 255, 255),
        thickness=2
    )

    WT_ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
        color=sv.Color(255, 255, 255),
        text_color=sv.Color(0, 0, 0),
        text_padding=5,
        text_thickness=1,
        text_position=sv.Position.TOP_CENTER,
    )

    def __init__(self, config):
        super().__init__(config)

        self.player_detection_model = YOLO(
            './models/football-player-detection.pt'
        ).to(device=self.config.device)

        self.pitch_detection_model = YOLO(
            './models/football-pitch-detection.pt'
        ).to(device=self.config.device)

    def _initialize_world_track(self):
        config = self.config

        match config.transition_model.name:
            case 'heuristic':
                transition_noise = torch.diag(torch.tensor(
                    list(config.transition_model.variance), 
                    dtype = torch.float32
                ))
                assert transition_noise.shape == (4, 4)
                transition_model = HeuristicTransitionModel(
                    covariance = transition_noise
                )
            case _:
                raise NotImplementedError()

        # transition_model = TransitionModel(
        #     num_layers = config.transition_model.num_layers,
        #     hidden_dim = config.transition_model.hidden_dim,
        #     scale = config.transition_model.scale,
        #     initial_variance = config.transition_model.initial_variance
        # )   

        observation_noise = torch.diag(torch.tensor(
            list(config.observation_model.variance), 
            dtype = torch.float32
        ))
        assert observation_noise.shape == (2, 2)
        observation_model = CameraObservationModel(
            covariance = observation_noise
        )

        track_filter = TrackFilter(
            transition_model = transition_model,
            observation_model = observation_model,
            num_particles = config.filter.num_particles,
            prediction_noise = config.filter.prediction_noise,
            nu = config.filter.nu,
            ess_scale = config.filter.ess_scale
        )

        track_smoother = TrackSmoother(
            transition_model = transition_model,
            num_trajectories = config.smoother.num_trajectories
        )

        initial_state_noise = torch.diag(torch.tensor(
            list(config.initial_state.variance), 
            dtype = torch.float32
        ))
        assert initial_state_noise.shape == (4, 4)
        world_tracker = WorldTrack(
            initial_state_noise = initial_state_noise,
            track_filter = track_filter,
            track_smoother = track_smoother
        )

        return world_tracker

    def __call__(self):
        config = self.config

        T = config.n_frames

        detections = torch.load(
            config.data_dir + '/detections.pt', 
            weights_only=False
        )
        homographies = torch.load(
            config.data_dir + '/homographies.pt', 
            weights_only=False
        )
        projections = torch.load(
            config.data_dir + '/projections.pt', 
            weights_only=False
        )

        world_track = self._initialize_world_track()
        tracks = world_track.filter(
            detections[0][:T], 
            homographies[0][:T]
        )
        world_track.process_tracks(tracks)
        tracks = world_track.smooth(tracks)

        anim = animate_world_track(
            detections[0][:T],
            projections[0][:T],
            homographies[0][:T],
            tracks,
            show_particles = True,
            video_path = config.video_dir +'/video_1.mp4'

        )
        Logger.save_anim(anim, 'world_track.mp4')