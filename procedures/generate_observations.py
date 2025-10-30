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
from modules import Logger, ViewTransformer

class GenerateObservations(Procedure):

    PLAYER_CLASS_ID = 2

    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.player_detection_model = YOLO(
            './models/football-player-detection.pt'
        ).to(device=self.device)

        self.pitch_detection_model = YOLO(
            './models/football-pitch-detection.pt'
        ).to(device=self.device)

        self.field_config = SoccerPitchConfiguration()

    def _get_observations(self, video_path: str, n_frames: int) -> list:
        frames = sv.get_video_frames_generator(
            source_path=video_path, end=None,
        )
        observations = []
        for i, frame in tqdm(enumerate(frames)):
            result = self.pitch_detection_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(result)
            result = self.player_detection_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)

            player_only_mask = (detections.class_id == self.PLAYER_CLASS_ID)
            detections = detections[player_only_mask]

            valid_keypoints_mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
            transformer = ViewTransformer(
                source=keypoints.xy[0][valid_keypoints_mask].astype(np.float32),
                target=np.array(self.field_config.vertices)[valid_keypoints_mask].astype(np.float32)
            )
            xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            observation = torch.tensor(transformer.transform_points(points=xy), dtype=torch.float32)
            observations.append(observation)
        
        return observations

    def _get_all_observations(self, input_dir: str) -> list:
        all_observations = []
        for file_path in sorted(Path(input_dir).glob("*.mp4")):
            Logger.debug(f'Collecting observations from ./{str(file_path)}.')
            video_info = sv.VideoInfo.from_video_path(str(file_path))
            observations = self._get_observations(
                video_path = str(file_path), 
                n_frames = video_info.total_frames,
            )
            all_observations.append(observations)
        
        return all_observations

    def __call__(self):
        config = self.config
        observations = self._get_all_observations(config.input_dir)
        torch.save(observations, config.output_file)
        Logger.info(f"Observations saved to {config.output_file}.")