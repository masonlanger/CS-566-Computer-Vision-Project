import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sports.configs.soccer import SoccerPitchConfiguration

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
    tracks: list[TrackPosteriors],
    T: int | None = None,
    interval = 100,
    show_particles = False
):
    field = SoccerPitchConfiguration()
    T = len(detections) if T is None else T


    fig, (ax_image, ax_world) = plt.subplots(2, 1, figsize=(5, 6))

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
    # image
    _detections = ax_image.scatter([], [], s=10, color='black')
    _associated_detections = ax_image.scatter([], [], s=20, color='red')
    # world
    _projections = ax_world.scatter([], [], s=10, color='black')
    _means = ax_world.scatter([], [], s=10, color='red')
    _particles = ax_world.scatter([], [], s=1, color='blue')
    _pre_resample_particles = ax_world.scatter([], [], s=1, color='blue', alpha=0.1)

    track = tracks[0]

    def init():
        # image
        _detections.set_offsets(np.empty((0, 2)))
        _associated_detections.set_offsets(np.empty((0, 2)))
        # world
        _projections.set_offsets(np.empty((0, 2)))
        _means.set_offsets(np.empty((0, 2)))
        _particles.set_offsets(np.empty((0, 2)))
        _pre_resample_particles.set_offsets(np.empty((0, 2)))
        _title.set_text("t=0")
        
        return (
            _detections,
            _associated_detections,
            _image_labels,
            _projections,
            _means,
            _particles,
            _pre_resample_particles,
            _title
        )
    
    def update(t):
        try:
            detections_t = detections[t].detach().cpu().numpy()
            _detections.set_offsets(detections_t)
            for label in _image_labels: label.remove()
            _image_labels.clear()
            for i, (x, y) in enumerate(detections_t):
                txt = ax_image.text(x, y, str(i), color="black", fontsize=6, ha='center', va='bottom', transform=image_text_offset)
                _image_labels.append(txt)

            associations = []
            associated_detections = []
            associated_projections = []
            means = []
            track_particles = []
            track_pre_particles = []

            projections_t = projections[t].detach().cpu().numpy()

            for track_idx, track in enumerate(tracks):
                track_age = t - track.birth_step
                if track_age < 0 or track_age >= len(track.m_f):
                    continue

                mean = track.m_f[track_age].detach().cpu().numpy()[:2]
                means.append(mean)

                # Association
                assoc = int(track.associations[track_age])
                associations.append(assoc)
                if 0 <= assoc < len(detections_t):
                    associated_detections.append(detections_t[assoc])
                    if 0 <= assoc < len(projections_t):
                        associated_projections.append(projections_t[assoc])

                if show_particles:
                    particles = (
                        track.particles[track_age].detach().cpu().numpy()[:, :2]
                    )
                    pre_resample = (
                        track.pre_resample_particles[track_age]
                        .detach()
                        .cpu()
                        .numpy()[:, :2]
                    )
                    track_particles.append(particles)
                    track_pre_particles.append(pre_resample)

            # --- set scatter data for all tracks at time t ---
            if associated_detections:
                _associated_detections.set_offsets(
                    np.stack(associated_detections, axis=0)
                )
            else:
                _associated_detections.set_offsets(np.empty((0, 2)))

            if associated_projections:
                _projections.set_offsets(np.stack(associated_projections, axis=0))
            else:
                _projections.set_offsets(np.empty((0, 2)))

            if means:
                _means.set_offsets(np.stack(means, axis=0))
            else:
                _means.set_offsets(np.empty((0, 2)))

            if show_particles and track_particles:
                _particles.set_offsets(np.concatenate(track_particles, axis=0))
                _pre_resample_particles.set_offsets(
                    np.concatenate(track_pre_particles, axis=0)
                )
            else:
                _particles.set_offsets(np.empty((0, 2)))
                _pre_resample_particles.set_offsets(np.empty((0, 2)))

            for label in _world_labels:
                label.remove()
            _world_labels.clear()

            for i, mean in enumerate(means):
                txt = ax_world.text(
                    mean[0],
                    mean[1],
                    str(associations[i]),
                    color="black",
                    fontsize=6,
                    ha="center",
                    va="bottom",
                    transform=world_text_offset,
                )
                _world_labels.append(txt)

            
            _title.set_text(f"t={t}")
        except: breakpoint()
        return (
            _detections,
            _associated_detections,
            _image_labels,
            _projections,
            _means,
            _particles,
            _pre_resample_particles,
            _title
        )


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