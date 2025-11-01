import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
from sports.configs.soccer import SoccerPitchConfiguration
import supervision as sv
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


from procedures import Procedure
from modules import (
    Logger, 
    ViewTransformer,
    TransitionModel, CameraObservationModel,
    TrackFilter, TrackSmoother, WorldTracker,
    BatchEM,
    animate_video, animate_state_estimation,
    apply_homography
)

class EM(Procedure):

    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.debug = config.debug
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

        parameters = list(transition_model.parameters())
        optimizer = torch.optim.Adam(parameters, lr = config.em.lr)

        em = BatchEM(
            transition_model = transition_model,
            observation_model = observation_model,
            world_tracker = world_tracker,
            optimizer = optimizer
        )

        return em

    def __call__(self):
        config = self.config
        epochs = config.epochs
        E = config.em.e_steps_per_epoch
        M = config.em.m_steps_per_e_step

        detections = torch.load(config.detections_file, weights_only=False)
        homographies = torch.load(config.homographies_file, weights_only=False)
        # projections = torch.load(config.projections_file, weights_only=False)

        assert len(detections) == len(homographies)
        transition_model, observation_model = self._initialize_models()
        em = self._initialize_algorithms(transition_model, observation_model)

        # anim = animate_video(detections[1], homographies[1])
        # Logger.save_anim(anim, 'animation.mp4')
        # return

        detections = [detections[1][:20]]
        homographies = [homographies[1][:20]]
        projections = []
        for i, _detections in enumerate(detections):
            T = len(_detections) 
            _projections = []
            for t in range(T):
                _projections.append(
                    apply_homography(_detections[t], torch.inverse(homographies[i][t]))
                )
            projections.append(_projections)

        # num. videos
        N = len(detections)

        for epoch_iter in range(epochs):
            Logger.info(f'epoch={epoch_iter}')
            for e_iter in range(E):
                Logger.info(f'e={e_iter}')
                posteriors = em.e_step(detections, homographies)

                if self.debug:
                    video_idx = 0
                    anim = animate_state_estimation(
                        detections[video_idx],
                        projections[video_idx],
                        posteriors[video_idx][0]
                    )
                    Logger.save_anim(anim, 'state_estimation.mp4')
                    return


                for m_iter in range(M):
                    Logger.info(f'm={m_iter}')
                    loss = em.m_step(detections, homographies, posteriors)
                    Logger.log_metrics({'loss': loss}).debug(f'loss={loss}')

        