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

from procedures import Procedure, register
from modules import Logger, ViewTransformer

@register('preprocess')
class Preprocess(Procedure):

    PLAYER_CLASS_ID = 2
    field_config = SoccerPitchConfiguration()

    def __init__(self, config):
        super().__init__(config)

        self.player_detection_model = YOLO(
            './models/football-player-detection.pt'
        ).to(device=self.config.device)

        self.pitch_detection_model = YOLO(
            './models/football-pitch-detection.pt'
        ).to(device=self.config.device)

    def _extract_from_video(self, video_path: str, n_frames = None) -> list:
        frames = sv.get_video_frames_generator(
            source_path=video_path, end=n_frames,
        )
        detections, homographies, projections = [], [], []
        tracker = sv.ByteTrack(minimum_consecutive_frames=3)
        for i, frame in tqdm(enumerate(frames)):
            result = self.pitch_detection_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(result)
            result = self.player_detection_model(frame, imgsz=1280, verbose=False)[0]
            frame_detections = sv.Detections.from_ultralytics(result)
            frame_detections = tracker.update_with_detections(frame_detections) 

            player_only_mask = (frame_detections.class_id == self.PLAYER_CLASS_ID)
            player_detections = frame_detections[player_only_mask]

            mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
            source = keypoints.xy[0][mask].astype(np.float32)
            target = np.array(self.field_config.vertices)[mask].astype(np.float32)
            H, _ = cv2.findHomography(source, target)
            H_inv = np.linalg.inv(H)
            xy = player_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)


            transformer = ViewTransformer(
                source=keypoints.xy[0][mask].astype(np.float32),
                target=np.array(self.field_config.vertices)[mask].astype(np.float32)
            )
            xy = player_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            transformed_xy = transformer.transform_points(points=xy)


            detections.append(torch.tensor(xy, dtype=torch.float32))
            homographies.append(torch.tensor(H_inv, dtype=torch.float32))
            projections.append(torch.tensor(transformed_xy, dtype=torch.float32))
        
        return detections, homographies, projections

    def __call__(self):
        config = self.config
        Logger.info(f'Using device {self.config.device}.')

        detections, homographies, projections = [], [], []
        for file_path in sorted(Path(config.input_dir).glob("*.mp4")):
            Logger.debug(f'Extracting data from ./{str(file_path)}.')
            # video_info = sv.VideoInfo.from_video_path(str(file_path))
            _detections, _homographies, _projections = self._extract_from_video(
                video_path = str(file_path), 
                n_frames = None,
            )
            detections.append(_detections)
            homographies.append(_homographies)
            projections.append(_projections)

        output_file = f'{config.output_dir}/detections.pt'
        torch.save(detections, output_file)
        Logger.info(f"Detections saved to {output_file}.")

        output_file = f'{config.output_dir}/homographies.pt'
        torch.save(homographies, output_file)
        Logger.info(f"World-to-image homographies saved to {output_file}.")

        output_file = f'{config.output_dir}/projections.pt'
        torch.save(projections, output_file)
        Logger.info(f"Image-to-world projections saved to {output_file}.")