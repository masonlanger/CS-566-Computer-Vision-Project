import argparse
from enum import Enum
from typing import Iterator, List

import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
import copy

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

# from modules.visualize import animate_history

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'models/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'models/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'models/football-ball-detection.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)

WT_BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.Color(0,0,0),
    thickness=2
)

WT_BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.Color(0,0,0),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)

ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)

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





BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)

def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """


    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    # if goalkeepers_xy.shape[0] == 0:
    #     return np.empty((0,), dtype=int)

    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)

def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray
) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    radar = draw_pitch(config=CONFIG)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar)
    return radar

def get_frame_generator(
    input: str, 
    frames: int | None,
    device: str
) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=input, end=frames, stride=STRIDE)
    
    # # train team classifier
    # crops = []
    # for frame in tqdm(frame_generator, desc='collecting crops'):
    #     result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
    #     detections = sv.Detections.from_ultralytics(result)
    #     crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    # team_classifier = TeamClassifier(device=device)
    # team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=input, end=frames)

    byte_tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for i, frame in enumerate(frame_generator):
        print(i)
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        # only keep player detections
        mask = (detections.class_id == PLAYER_CLASS_ID)
        detections = detections[mask]

        breakpoint()

        bt_detections = byte_tracker.update_with_detections(
            copy.deepcopy(detections)
        )

        annotated_frame = frame.copy()

        # get ids
        # players = bt_detections[bt_detections.class_id == PLAYER_CLASS_ID]
        # crops = get_crops(frame, players)
        # players_team_id = team_classifier.predict(crops)

        # goalkeepers = bt_detections[bt_detections.class_id == GOALKEEPER_CLASS_ID]
        # goalkeepers_team_id = resolve_goalkeepers_team_id(
        #     players, players_team_id, goalkeepers
        # )

        # referees = bt_detections[bt_detections.class_id == REFEREE_CLASS_ID]

        # draw annotated frame
        # bt_detections = sv.Detections.merge([players, goalkeepers, referees])
        # color_lookup = np.array(
        #     players_team_id.tolist() +
        #     goalkeepers_team_id.tolist() +
        #     [REFEREE_CLASS_ID] * len(referees)
        # )
        labels = [str(tracker_id) for tracker_id in bt_detections.tracker_id]
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, 
            bt_detections
        )
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, 
            bt_detections, 
            labels
        )

        yield annotated_frame

def main(
    input: str, 
    output: str, 
    frames: int | None, 
    device: str
) -> None:
    frame_generator = get_frame_generator(
        input=input, 
        frames=frames,
        device=device
    )

    video_info = sv.VideoInfo.from_video_path(input)
    with sv.VideoSink(output, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--frames', '-f', type=int, default=None)
    parser.add_argument('--device', '-d', type=str, default='cpu')
    args = parser.parse_args()
    main(
        input=args.input,
        output=args.output,
        frames=args.frames,
        device=args.device
    )