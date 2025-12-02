from matplotlib.patches import Ellipse
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

# def animate_world_track(
#     detections: list,
#     projections: list,
#     homographies: list,
#     track: TrackPosteriors,
#     T: int | None = None,
#     interval = 100,
#     show_particles = False,
#     video_path: str | None = None
# ):
#     field = SoccerPitchConfiguration()

#     frames = None
#     if video_path is not None:
#         cap = cv2.VideoCapture(video_path)
#         frames = []
#         ok, frame = cap.read()
#         while ok:
#             # cv2 gives BGR, matplotlib wants RGB
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame)
#             ok, frame = cap.read()
#         cap.release()
#         if len(frames) == 0:
#             raise ValueError(f"No frames read from {video_path}")

#     T = len(detections) if T is None else T

#     fig, (ax_image, ax_world) = plt.subplots(2, 1, figsize=(7, 8), dpi=200)

#     ax_image.set_xlim(0, 1920)
#     ax_image.set_ylim(0, 1080)
#     ax_world.set_xlim(0, field.length)
#     ax_world.set_ylim(0, field.width)

#     ax_image.set_aspect("equal", adjustable="box")
#     ax_world.set_aspect("equal", adjustable="box")

#     ax_image.set_title("Image")
#     ax_world.set_title("World")

#     field_landmarks = np.array(field.vertices, dtype=float)
#     ax_world.scatter(field_landmarks[:, 0], field_landmarks[:, 1], color='lightgray', s=5)

#     _image_labels = []
#     image_text_offset_above = offset_copy(ax_image.transData, fig=fig, x=0, y=2, units='points')
#     image_text_offset_below = offset_copy(ax_image.transData, fig=fig, x=0, y=-10, units='points')

#     _world_labels = []
#     world_text_offset_above = offset_copy(ax_world.transData, fig=fig, x=0, y=2, units='points')
#     world_text_offset_below = offset_copy(ax_world.transData, fig=fig, x=0, y=-10, units='points')

#     _title = fig.suptitle("t=0")

#     _video_frame = ax_image.imshow(frames[0], alpha=1.0, zorder=0) if frames is not None else None

#     # image
#     _detections = ax_image.scatter(
#         [], [], 
#         s=20,
#         facecolors='white',
#         edgecolors='none',
#         linewidths=1.0
#     )

#     _associated_detection = ax_image.scatter(
#         [], [],
#         s=20,
#         facecolors='blue',
#         edgecolors='none',
#         linewidths=1.0
#     )

#     # world
#     _world_to_image_projection = ax_image.scatter([], [], s=10, color='blue')
#     _image_to_world_projections = ax_world.scatter([], [], s=10, marker='x', color='lightgray')
#     _mean = ax_world.scatter([], [], s=10, color='blue')
#     _particles = ax_world.scatter([], [], s=1, color='blue')
#     _pre_resample_particles = ax_world.scatter([], [], s=1, color='blue', alpha=0.1)

#     # field lines
#     for edge in field.edges:
#         p1 = field.vertices[edge[0]-1]
#         p2 = field.vertices[edge[1]-1]
#         ax_world.plot(
#             [p1[0], p2[0]],
#             [p1[1], p2[1]],
#             color='lightgray',
#             linewidth=0.5,
#             zorder=0
#         )

#     # center circle
#     center = (field.length / 2, field.width / 2)
#     circle = plt.Circle(
#         center,     
#         field.centre_circle_radius,
#         color='lightgray',
#         fill=False,
#         linewidth=0.5,
#         zorder=0
#     )   
#     ax_world.add_artist(circle)

#     def init():

#         if _video_frame is not None:
#             _video_frame.set_data(frames[0])

#         # image
#         _detections.set_offsets(np.empty((0, 2)))
#         _associated_detection.set_offsets(np.empty((0, 2)))
#         # world
#         _world_to_image_projection.set_offsets(np.empty((0, 2)))
#         _image_to_world_projections.set_offsets(np.empty((0, 2)))
#         _mean.set_offsets(np.empty((0, 2)))
#         _particles.set_offsets(np.empty((0, 2)))
#         _pre_resample_particles.set_offsets(np.empty((0, 2)))
#         _title.set_text("t=0")
        
#         return (
#             _detections,
#             _associated_detection,
#             _image_labels,
#             _world_to_image_projection,
#             _image_to_world_projections,
#             _mean,
#             _particles,
#             _pre_resample_particles,
#             _title
#         )
    
#     def update(t):
#         # try:
#         # image-space
#         if _video_frame is not None and t < len(frames):
#             _video_frame.set_data(frames[t])

#         detections_t = detections[t].detach().cpu().numpy()
#         _detections.set_offsets(detections_t)

#         for label in _image_labels: label.remove()
#         _image_labels.clear()
#         for i, (x, y) in enumerate(detections_t):
#             txt = ax_image.text(
#                 x, y, f"[{i}]", 
#                 color="lightgray", 
#                 fontsize=6, 
#                 ha='center', 
#                 va='bottom', 
#                 transform=image_text_offset_below
#             )
#             _image_labels.append(txt)


#         # world-space
#         projections_t = projections[t].detach().cpu().numpy()
#         _image_to_world_projections.set_offsets(projections_t)

#         for label in _world_labels: label.remove()
#         _world_labels.clear()
#         for i, (x, y) in enumerate(projections_t):
#             txt = ax_world.text(
#                 x, y, f"[{i}]", 
#                 color="lightgray", 
#                 fontsize=6, 
#                 ha='center', 
#                 va='bottom', 
#                 transform=world_text_offset_below
#             )
#             _world_labels.append(txt)



#         track_step = t - track.birth_step
#         if track_step >= 0:
#             association = int(track.associations[track_step])

#             associated_detection = detections_t[association]
#             _associated_detection.set_offsets(associated_detection)
#             txt = ax_image.text(
#                 associated_detection[0], 
#                 associated_detection[1], 
#                 str(track.id), 
#                 color="blue", 
#                 fontsize=8, 
#                 ha='center', 
#                 va='bottom', 
#                 transform=image_text_offset_above
#             )
#             _image_labels.append(txt)


#             mean = track.m_s[track_step].detach().cpu().numpy()[:2]
#             _mean.set_offsets(mean)

#             # _world_to_image_projection.set_offsets(
#             #     apply_homography(
#             #         track.m_s[track_step][:2], 
#             #         homographies[t]
#             #     ).detach().cpu().numpy()
#             # )

#             if show_particles:
#                 particles = track.particles[track_step] \
#                     .detach().cpu().numpy()[:, :2]
#                 pre_resample_particles = track.pre_resample_particles[track_step] \
#                     .detach().cpu().numpy()[:, :2]
            
#                 _particles.set_offsets(particles)
#                 # _pre_resample_particles.set_offsets(pre_resample_particles)

        
#             txt = ax_world.text(
#                 mean[0], 
#                 mean[1],
#                 str(track.id),
#                 color="blue",
#                 fontsize=8,
#                 ha="center",
#                 va="bottom",
#                 transform=world_text_offset_above,
#             )
#             _world_labels.append(txt)

#         _title.set_text(f"t={t}")
#         return (
#             _detections,
#             _associated_detection,
#             _image_labels,
#             _world_to_image_projection,
#             _image_to_world_projections,
#             _mean,
#             _particles,
#             _pre_resample_particles,
#             _title
#         )

#     ax_image.invert_yaxis()
#     ax_world.invert_yaxis()
#     ax_world.grid(True, alpha=0.1)
#     fig.tight_layout()
#     anim = FuncAnimation(
#         fig, 
#         update, 
#         frames = T,
#         init_func = init, 
#         interval = interval, 
#         blit = False, 
#         repeat = True
#     )
#     return anim

def animate_world_track(
    detections: list,
    projections: list,
    homographies: list,
    tracks: TrackPosteriors,
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

    fig, (ax_image, ax_world) = plt.subplots(2, 1, figsize=(7, 8), dpi=200)

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
    image_text_offset_above = offset_copy(ax_image.transData, fig=fig, x=0, y=2, units='points')
    image_text_offset_below = offset_copy(ax_image.transData, fig=fig, x=0, y=-10, units='points')

    _world_labels = []
    world_text_offset_above = offset_copy(ax_world.transData, fig=fig, x=0, y=2, units='points')
    world_text_offset_below = offset_copy(ax_world.transData, fig=fig, x=0, y=-10, units='points')

    _title = fig.suptitle("t=0")

    _video_frame = ax_image.imshow(frames[0], alpha=1.0, zorder=0) if frames is not None else None

    # image
    _detections = ax_image.scatter(
        [], [], 
        s=20,
        facecolors='white',
        edgecolors='none',
        linewidths=1.0
    )

    _associated_detection = ax_image.scatter(
        [], [],
        s=20,
        facecolors='blue',
        edgecolors='none',
        linewidths=1.0
    )

    # world
    _world_to_image_projection = ax_image.scatter([], [], s=10, color='blue')
    _image_to_world_projections = ax_world.scatter([], [], s=10, marker='x', color='lightgray')
    _mean = ax_world.scatter([], [], s=10, color='blue')
    # _particles = ax_world.scatter([], [], s=1, color='blue')
    # _pre_resample_particles = ax_world.scatter([], [], s=1, color='blue', alpha=0.1)

    _covariances = []
    for track in tracks:
        e = Ellipse(
            xy=(0, 0),
            width=1.0,
            height=1.0,
            edgecolor="blue",
            facecolor="none",
            linewidth=1
        )
        ax_world.add_patch(e)
        _covariances.append(e)

    # field lines
    for edge in field.edges:
        p1 = field.vertices[edge[0]-1]
        p2 = field.vertices[edge[1]-1]
        ax_world.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color='lightgray',
            linewidth=0.5,
            zorder=0
        )

    # center circle
    center = (field.length / 2, field.width / 2)
    circle = plt.Circle(
        center,     
        field.centre_circle_radius,
        color='lightgray',
        fill=False,
        linewidth=0.5,
        zorder=0
    )   
    ax_world.add_artist(circle)

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
        # _particles.set_offsets(np.empty((0, 2)))
        # _pre_resample_particles.set_offsets(np.empty((0, 2)))
        _title.set_text("t=0")
        
        return (
            _detections,
            _associated_detection,
            _image_labels,
            _world_to_image_projection,
            _image_to_world_projections,
            _mean,
            # _particles,
            # _pre_resample_particles,
            _title
        )
    
    def update(t):
        # image-space
        if _video_frame is not None and t < len(frames):
            _video_frame.set_data(frames[t])

        detections_t = detections[t].detach().cpu().numpy()
        _detections.set_offsets(detections_t)

        for label in _image_labels:
            label.remove()
        _image_labels.clear()
        for i, (x, y) in enumerate(detections_t):
            txt = ax_image.text(
                x, y, f"[{i}]",
                color="lightgray",
                fontsize=6,
                ha="center",
                va="bottom",
                transform=image_text_offset_below,
            )
            _image_labels.append(txt)

        # world-space
        projections_t = projections[t].detach().cpu().numpy()
        _image_to_world_projections.set_offsets(projections_t)

        for label in _world_labels:
            label.remove()
        _world_labels.clear()
        for i, (x, y) in enumerate(projections_t):
            txt = ax_world.text(
                x, y, f"[{i}]",
                color="lightgray",
                fontsize=6,
                ha="center",
                va="bottom",
                transform=world_text_offset_below,
            )
            _world_labels.append(txt)

        # Collect per-track quantities for this frame
        assoc_dets = []
        means = []
        # all_particles = []

        for i, track in enumerate(tracks):
            track_step = t - track.birth_step
            if track_step < 0:
                continue

            association = int(track.associations[track_step])
            if association != -1: 
                associated_detection = detections_t[association]
                assoc_dets.append(associated_detection)

                txt = ax_image.text(
                    associated_detection[0],
                    associated_detection[1],
                    str(track.id),
                    color="blue",
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    transform=image_text_offset_above,
                )
                _image_labels.append(txt)

            mean = track.m_s[track_step].detach().cpu().numpy()[:2]
            means.append(mean)

            covariance = track.P_s[track_step].detach().cpu().numpy()[:2, :2]
            eigvals, eigvecs = np.linalg.eigh(covariance)
            width, height = 2 * np.sqrt(np.abs(eigvals))
            angle = np.degrees(np.arctan2(*np.flip(eigvecs[:, 0])))
            C = _covariances[i]
            C.center = (mean[0], mean[1])
            C.width = width
            C.height = height
            C.angle = angle

            # if show_particles:
            #     particles = track.particles[track_step].detach().cpu().numpy()[:, :2]
            #     all_particles.append(particles)

            txt = ax_world.text(
                mean[0],
                mean[1],
                str(track.id),
                color="blue",
                fontsize=8,
                ha="center",
                va="bottom",
                transform=world_text_offset_above,
            )
            _world_labels.append(txt)

        # Update scatter artists for all tracks at once
        if assoc_dets:
            _associated_detection.set_offsets(np.stack(assoc_dets, axis=0))
        else:
            _associated_detection.set_offsets(np.empty((0, 2)))

        if means:
            _mean.set_offsets(np.stack(means, axis=0))
        else:
            _mean.set_offsets(np.empty((0, 2)))
        
        # if show_particles:
        #     if all_particles:
        #         _particles.set_offsets(np.concatenate(all_particles, axis=0))
        #     else:
        #         _particles.set_offsets(np.empty((0, 2)))

        _title.set_text(f"t={t}")

        return (
            _detections,
            _associated_detection,
            _world_labels,
            _image_labels,
            _world_to_image_projection,
            _image_to_world_projections,
            _mean,
            # _particles,
            # _pre_resample_particles,
            *_covariances,
            _title,
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