import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_video(detections, interval=100):
    _detections = []
    for d in detections:
        if isinstance(d, torch.Tensor):
            d = d.detach().cpu().numpy()
        _detections.append(np.asarray(d, dtype=float))

    all_pts = np.vstack([p for p in _detections if len(p)])
    xlim = (float(all_pts[:, 0].min()), float(all_pts[:, 0].max()))
    ylim = (float(all_pts[:, 1].min()), float(all_pts[:, 1].max()))

    fig, ax = plt.subplots()
    sc = ax.scatter([], [])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Points per frame")

    def init():
        sc.set_offsets(np.empty((0, 2)))
        return (sc,)

    def update(i):
        pts = _detections[i]
        # set_offsets accepts (N,2); handle empty frames
        if pts.size == 0:
            sc.set_offsets(np.empty((0, 2)))
        else:
            sc.set_offsets(pts[:, :2])
        ax.set_title(f"Frame {i} — {len(pts)} points")
        return (sc,)

    anim = FuncAnimation(
        fig, update, frames=len(_detections),
        init_func=init, interval=interval, blit=True, repeat=True
    )
    return anim
