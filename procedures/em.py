import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
from sports.configs.soccer import SoccerPitchConfiguration
import supervision as sv
import torch
from tqdm import tqdm

from procedures import Procedure
from modules import (
    Logger, 
    ViewTransformer,
    TransitionModel, CameraObservationModel,
    TrackFilter, TrackSmoother, WorldTracker
)

class EM(Procedure):

    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.field_config = SoccerPitchConfiguration()
    
    def _initialize_models(self):
        config = self.config

        transition_model = TransitionModel(
            state_dim = 4,
            num_layers = config.transition_model.num_layers,
            hidden_dim = config.transition_model.hidden_dim,
            scale = config.transition_model.scale,
            initial_variance = config.transition_model.initial_variance
        )   

        observation_model = CameraObservationModel(
            covariance = torch.eye(2, dtype=torch.float32)
                       * float(config.observation_model.variance)
        )
        
        return transition_model, observation_model

    def _initialize_algorithms(
        self,
        transition_model,
        observation_model
    ):
        config = self.config
        track_filter = TrackFilter(
            state_dim = 4,
            transition_model = transition_model,
            observation_model = observation_model,
            num_particles = config.filter.num_particles,
            prediction_noise = config.filter.prediction_noise,
            nu = config.filter.nu,
            ess_scale = config.filter.ess_scale
        )

        track_smoother = TrackSmoother(
            state_dim = 4,
            transition_model = transition_model,
            num_trajectories = config.smoother.num_trajectories
        )

        world_tracker = WorldTracker(
            initial_state_noise = torch.eye(4, dtype=torch.float32)
                                  * float(config.initial_state.variance),
            track_filter = track_filter,
            track_smoother = track_smoother
        )


    def __call__(self):
        config = self.config
        epochs = config.epochs

        detections = torch.load(config.detections_file)
        homographies = torch.load(config.homographies_file)
        assert len(detections) == len(homographies)
        transition_model, observation_model = self._initialize_models()

        for _ in range(epochs):
            
            # for each video -> for each track -> ... 
            ...