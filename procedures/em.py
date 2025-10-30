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

class EM(Procedure):

    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.field_config = SoccerPitchConfiguration()

    def __call__(self):
        config = self.config
        epochs = config.epochs

        observations = torch.load(config.data)

        for _ in range(epochs):
            
            # for each video -> for each track -> ... 
            