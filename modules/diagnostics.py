import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sports.configs.soccer import SoccerPitchConfiguration
import cv2

from matplotlib.transforms import offset_copy

from .math import apply_homography, particles_to_gaussian
from .state_estimators.track_posteriors import TrackPosteriors

def animate_video(
    detections: list[torch.Tensor],
    homographies: list[torch.Tensor],
    projections: list[torch.Tensor] = None,
    interval: int = 100
):
    field = SoccerPitchConfiguration()
    T = len(detections)

    if projections is None:
        projections = []
        for t in range(T):
            projections.append(
                apply_homography(detections[t], torch.inverse(homographies[t]))
            )

    fig, (ax_image, ax_world) = plt.subplots(2, 1, figsize=(5, 6))

    sc_image = ax_image.scatter([], [], s=10, color='black')
    sc_world = ax_world.scatter([], [], s=10, color='black')

    ax_image.set_xlim(0, 1920)
    ax_image.set_ylim(0, 1080)
    ax_world.set_xlim(0, field.length)
    ax_world.set_ylim(0, field.width)
    # ax_world.set_xlim(-1000, 1000)
    # ax_world.set_ylim(-1000, 1000)
    # ax_world.set_xlim(0, 1920)
    # ax_world.set_ylim(0, 1080)

    ax_image.set_aspect("equal", adjustable="box")
    ax_world.set_aspect("equal", adjustable="box")

    ax_image.set_xlabel("x")
    ax_image.set_ylabel("y")
    ax_image.set_title("Image")

    ax_world.set_xlabel("x")
    ax_world.set_ylabel("y")
    ax_world.set_title("World")

    suptxt = fig.suptitle("t=0")

    field_landmarks = np.array(field.vertices, dtype=float)
    ax_world.scatter(field_landmarks[:, 0], field_landmarks[:, 1], color='lightgray', s=5)

    image_labels = []
    world_labels = []
    image_text_offset = offset_copy(ax_image.transData, fig=fig, x=0, y=2, units='points')
    world_text_offset = offset_copy(ax_world.transData, fig=fig, x=0, y=2, units='points')

    def init():
        sc_image.set_offsets(np.empty((0, 2)))
        sc_world.set_offsets(np.empty((0, 2)))
        suptxt.set_text("t=0")
        return (sc_image, sc_world, suptxt)

    def update(t):
        pts_image = detections[t].detach().cpu().numpy()
        pts_world = projections[t].detach().cpu().numpy()

        sc_image.set_offsets(pts_image if pts_image.size else np.empty((0, 2)))
        sc_world.set_offsets(pts_world if pts_world.size else np.empty((0, 2)))

        for label in image_labels: label.remove()
        image_labels.clear()

        for i, (x, y) in enumerate(pts_image):
            txt = ax_image.text(x, y, str(i), color="black", fontsize=6, ha='center', va='bottom', transform=image_text_offset)
            image_labels.append(txt)

        for label in world_labels: label.remove()
        world_labels.clear()

        for i, (x, y) in enumerate(pts_world):
            txt = ax_world.text(x, y, str(i), color="black", fontsize=6, ha='center', va='bottom', transform=world_text_offset)
            world_labels.append(txt)

        suptxt.set_text(f"t={t}, n={len(detections[t])}")
        return (sc_image, sc_world, *image_labels, *world_labels, suptxt)
    
    ax_image.invert_yaxis()
    ax_world.invert_yaxis()
    fig.tight_layout()
    anim = FuncAnimation(
        fig, 
        update, 
        frames = T,
        init_func = init, 
        interval = interval, 
        blit = False, 
        repeat = True
    )
    return anim

def animate_state_estimation(
    detections: list,
    projections: list,
    homographies: list,
    track: TrackPosteriors,
    T: int | None = None,
    interval = 100,
    show_particles = False,
    video_path: str | None = None
):
    field = SoccerPitchConfiguration()

    frames = None
    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
        frames = []
        ok, frame = cap.read()
        while ok:
            # cv2 gives BGR, matplotlib wants RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            ok, frame = cap.read()
        cap.release()
        if len(frames) == 0:
            raise ValueError(f"No frames read from {video_path}")

    T = len(detections) if T is None else T

    fig, (ax_image, ax_world) = plt.subplots(2, 1, figsize=(7, 8))

    ax_image.set_xlim(0, 1920)
    ax_image.set_ylim(0, 1080)
    ax_world.set_xlim(0, field.length)
    ax_world.set_ylim(0, field.width)

    ax_image.set_aspect("equal", adjustable="box")
    ax_world.set_aspect("equal", adjustable="box")

    ax_image.set_title("Image")
    ax_world.set_title("World")

    field_landmarks = np.array(field.vertices, dtype=float)
    ax_world.scatter(field_landmarks[:, 0], field_landmarks[:, 1], color='lightgray', s=5)

    _image_labels = []
    image_text_offset = offset_copy(ax_image.transData, fig=fig, x=0, y=4, units='points')

    _world_labels = []
    world_text_offset = offset_copy(ax_world.transData, fig=fig, x=0, y=4, units='points')

    _title = fig.suptitle("t=0")

    _video_frame = ax_image.imshow(frames[0], alpha=1.0, zorder=0) if frames is not None else None

    # image
    _detections = ax_image.scatter([], [], s=10, color='white')
    _associated_detection = ax_image.scatter([], [], s=20, color='red')

    # world
    _world_to_image_projection = ax_image.scatter([], [], s=10, color='blue')
    _image_to_world_projections = ax_world.scatter([], [], s=10, color='black')
    _mean = ax_world.scatter([], [], s=10, color='red')
    _particles = ax_world.scatter([], [], s=1, color='blue')
    _pre_resample_particles = ax_world.scatter([], [], s=1, color='blue', alpha=0.1)

    # track = tracks[0]

    def init():

        if _video_frame is not None:
            _video_frame.set_data(frames[0])

        # image
        _detections.set_offsets(np.empty((0, 2)))
        _associated_detection.set_offsets(np.empty((0, 2)))
        # world
        _world_to_image_projection.set_offsets(np.empty((0, 2)))
        _image_to_world_projections.set_offsets(np.empty((0, 2)))
        _mean.set_offsets(np.empty((0, 2)))
        _particles.set_offsets(np.empty((0, 2)))
        _pre_resample_particles.set_offsets(np.empty((0, 2)))
        _title.set_text("t=0")
        
        return (
            _detections,
            _associated_detection,
            _image_labels,
            _world_to_image_projection,
            _image_to_world_projections,
            _mean,
            _particles,
            _pre_resample_particles,
            _title
        )
    
    def update(t):
        # try:
        # image-space
        if _video_frame is not None and t < len(frames):
            _video_frame.set_data(frames[t])

        detections_t = detections[t].detach().cpu().numpy()
        _detections.set_offsets(detections_t)

        for label in _image_labels: label.remove()
        _image_labels.clear()
        for i, (x, y) in enumerate(detections_t):
            txt = ax_image.text(
                x, y, str(i), 
                color="white", 
                fontsize=6, 
                ha='center', 
                va='bottom', 
                transform=image_text_offset
            )
            _image_labels.append(txt)

        # world-space
        projections_t = projections[t].detach().cpu().numpy()
        _image_to_world_projections.set_offsets(projections_t)

        for label in _world_labels: label.remove()
        _world_labels.clear()
        for i, (x, y) in enumerate(projections_t):
            txt = ax_world.text(
                x, y, str(i), 
                color="black", 
                fontsize=6, 
                ha='center', 
                va='bottom', 
                transform=world_text_offset
            )
            _world_labels.append(txt)



        track_step = t - track.birth_step
        if track_step >= 0:
            association = int(track.associations[track_step])

            _associated_detection.set_offsets(detections_t[association])

            mean = track.m_f[track_step].detach().cpu().numpy()[:2]
            _mean.set_offsets(mean)

            _world_to_image_projection.set_offsets(
                apply_homography(
                    track.m_f[track_step][:2], 
                    homographies[t]
                ).detach().cpu().numpy()
            )

            if show_particles:
                particles = track.particles[track_step] \
                    .detach().cpu().numpy()[:, :2]
                pre_resample_particles = track.pre_resample_particles[track_step] \
                    .detach().cpu().numpy()[:, :2]
            
                _particles.set_offsets(particles)
                # _pre_resample_particles.set_offsets(pre_resample_particles)

        
            txt = ax_world.text(
                mean[0], mean[1],
                str(association),
                color="blue",
                fontsize=6,
                ha="center",
                va="bottom",
                transform=world_text_offset,
            )
            _world_labels.append(txt)
        _title.set_text(f"t={t}")
        # except: breakpoint()
        return (
            _detections,
            _associated_detection,
            _image_labels,
            _world_to_image_projection,
            _image_to_world_projections,
            _mean,
            _particles,
            _pre_resample_particles,
            _title
        )

    ax_image.invert_yaxis()
    ax_world.invert_yaxis()
    ax_world.grid(True, alpha=0.1)
    fig.tight_layout()
    anim = FuncAnimation(
        fig, 
        update, 
        frames = T,
        init_func = init, 
        interval = interval, 
        blit = False, 
        repeat = True
    )
    return anim