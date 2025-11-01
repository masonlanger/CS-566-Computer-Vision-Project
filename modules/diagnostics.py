import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sports.configs.soccer import SoccerPitchConfiguration

from matplotlib.transforms import offset_copy

from .math import apply_homography

def animate_video(
    detections: list[torch.Tensor],
    homographies: list[torch.Tensor],
    interval: int = 100
):
    field = SoccerPitchConfiguration()
    T = len(detections)
    projections = []
    for t in range(T):
        projections.append(
            apply_homography(detections[t], torch.inverse(homographies[t]))
        )

    fig, (ax_image, ax_world) = plt.subplots(1, 2, figsize=(10, 5))

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
