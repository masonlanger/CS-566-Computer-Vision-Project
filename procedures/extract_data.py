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

class ExtractData(Procedure):

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

    def _extract_from_video(self, video_path: str, n_frames: int) -> list:
        frames = sv.get_video_frames_generator(
            source_path=video_path, end=None,
        )
        detections, homographies = [], []
        for i, frame in tqdm(enumerate(frames)):
            result = self.pitch_detection_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(result)
            result = self.player_detection_model(frame, imgsz=1280, verbose=False)[0]
            frame_detections = sv.Detections.from_ultralytics(result)

            player_only_mask = (frame_detections.class_id == self.PLAYER_CLASS_ID)
            frame_detections = frame_detections[player_only_mask]
            valid_keypoints_mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
            
            source = keypoints.xy[0][valid_keypoints_mask].astype(np.float32)
            target = np.array(self.field_config.vertices)[valid_keypoints_mask].astype(np.float32)
            H, _ = cv2.findHomography(source, target)
            H_inv = np.linalg.inv(H)
            xy = frame_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

            detections.append(torch.tensor(xy, dtype=torch.float32))
            homographies.append(torch.tensor(H_inv, dtype=torch.float32))
        
        return detections, homographies

    def __call__(self):
        config = self.config

        Logger.info(f'Using device {self.device}.')

        batch_detections = []
        batch_homographies = []
        for file_path in sorted(Path(config.input_dir).glob("*.mp4")):
            Logger.debug(f'Extracting data from ./{str(file_path)}.')
            video_info = sv.VideoInfo.from_video_path(str(file_path))
            detections, homographies = self._extract_from_video(
                video_path = str(file_path), 
                n_frames = video_info.total_frames,
            )
            batch_detections.append(detections)
            batch_homographies.append(homographies)


        output_file = f'{config.output_dir}/detections.pt'
        torch.save(batch_detections, output_file)
        Logger.info(f"Detections saved to {output_file}.")

        output_file = f'{config.output_dir}/homographies.pt'
        torch.save(batch_homographies, output_file)
        Logger.info(f"World-to-image homographies saved to {output_file}.")