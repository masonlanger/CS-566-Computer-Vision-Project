# Extended Matplotlib animation including observations and association lines.
#
# The expected `history` structure:
#   history.tracks[frame]        -> List[Track]
#   history.observations[frame]  -> np.ndarray of shape (M, 2) with (x, y) observations
#   history.associations[frame]  -> List[Tuple[track_id: int, obs_index: int]]
#
# `history` can be a dict with keys "tracks", "observations", "associations"
# or any object supporting attribute access (e.g., SimpleNamespace).
#
# Usage:
#   anim = animate_tracks_with_obs(history, interval=150)
#   anim = animate_tracks_with_obs(history, interval=150, save_path="/mnt/data/tracks_with_obs.mp4")
#
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import types
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from modules.world_track import History, Track


def _collect_frame_data_with_obs(history: History):
    """
    Precompute per-frame:
      - track means and ids
      - association flags derived from associations list (per frame)
      - observations (numpy arrays)
      - global bounds for plotting
    """
    tracks_seq = history.tracks
    obs_seq = history.observations
    assoc_seq = history.associations

    frames_means: List[List[Tuple[float, float]]] = []
    frames_ids: List[List[int]] = []
    frames_assoc_flags: List[List[bool]] = []  # True if track is associated in that frame
    frames_obs: List[np.ndarray] = []          # (M, 2)
    frames_pairs: List[List[Tuple[int, int]]] = []  # (track_id, obs_index) per frame

    all_x, all_y = [], []
    unique_ids = set()

    T = len(tracks_seq)
    assert len(obs_seq) == T and len(assoc_seq) == T, "history lists must have same length"

    for t in range(T):
        tracks_t: List[Track] = tracks_seq[t]
        obs_t: np.ndarray = np.asarray(obs_seq[t]) if obs_seq[t] is not None else np.zeros((0, 2), dtype=float)
        assoc_t: List[Tuple[int, int]] = assoc_seq[t] if assoc_seq[t] is not None else []

        # Build a set of associated track ids for quick lookup
        assoc_track_ids = {tid for (tid, _oi) in assoc_t}

        means_t, ids_t, flags_t = [], [], []
        for tr in tracks_t:
            m = tr.mean()
            x, y = float(m[0].detach().cpu().item()), float(m[1].detach().cpu().item())
            means_t.append((x, y))
            ids_t.append(int(tr.id))
            flags_t.append(int(tr.id) in assoc_track_ids)

            all_x.append(x)
            all_y.append(y)
            unique_ids.add(int(tr.id))

        frames_means.append(means_t)
        frames_ids.append(ids_t)
        frames_assoc_flags.append(flags_t)
        frames_obs.append(obs_t.astype(float))
        frames_pairs.append(list(assoc_t))

        # Also factor observations into bounds
        if obs_t.size > 0:
            all_x.extend(obs_t[:, 0].tolist())
            all_y.extend(obs_t[:, 1].tolist())

    # Compute loose bounds with padding
    if len(all_x) == 0:
        x_min = y_min = -1.0
        x_max = y_max = 1.0
    else:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_pad = 0.1 * (x_max - x_min if x_max > x_min else 1.0)
        y_pad = 0.1 * (y_max - y_min if y_max > y_min else 1.0)
        x_min, x_max = x_min - x_pad, x_max + x_pad
        y_min, y_max = y_min - y_pad, y_max + y_pad

    return {
        "frames_means": frames_means,
        "frames_ids": frames_ids,
        "frames_assoc_flags": frames_assoc_flags,
        "frames_obs": frames_obs,
        "frames_pairs": frames_pairs,
        "bounds": (x_min, x_max, y_min, y_max),
        "unique_ids": sorted(unique_ids),
    }


def animate_history(
    history: History,
    interval: int = 100,
    save_path: Optional[str] = None,
    show_obs: bool = True,
    obs_marker: str = "x",
    obs_size: int = 50,
):
    """
    Create a matplotlib animation showing mean (x,y) position per track over time,
    the observations, and lines connecting each associated pair (track mean -> observation).

    Parameters
    ----------
    history : dict or object
        Must provide .tracks, .observations, .associations (or dict keys).
        - tracks[t]        : List[Track]
        - observations[t]  : np.ndarray of shape (M, 2) or None
        - associations[t]  : List[(track_id, obs_index)] or None
    interval : int
        Delay between frames in milliseconds.
    save_path : Optional[str]
        If provided (ends with .mp4 or .gif), saves the animation to this path.
    show_obs : bool
        Whether to plot observations.
    obs_marker : str
        Matplotlib marker for observations, default 'x'.
    obs_size : int
        Marker size for observations.
    """
    pre = _collect_frame_data_with_obs(history)
    frames_means = pre["frames_means"]
    frames_ids = pre["frames_ids"]
    frames_assoc_flags = pre["frames_assoc_flags"]
    frames_obs = pre["frames_obs"]
    frames_pairs = pre["frames_pairs"]
    x_min, x_max, y_min, y_max = pre["bounds"]
    unique_ids = pre["unique_ids"]


    # Map track id -> consistent color
    cmap = plt.get_cmap("tab10")
    id_to_color: Dict[int, Tuple[float, float, float, float]] = {
        tid: cmap(i % 10) for i, tid in enumerate(unique_ids)
    }

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def init():
        ax.cla()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return []

    def update(frame_idx):
        ax.cla()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"frame={frame_idx + 1}/{len(frames_means)}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        means = frames_means[frame_idx]
        ids = frames_ids[frame_idx]
        assoc_flags = frames_assoc_flags[frame_idx]
        obs = frames_obs[frame_idx]
        pairs = frames_pairs[frame_idx]

        # Draw observations
        if show_obs and obs is not None and obs.size > 0:
            ax.scatter(obs[:, 0], obs[:, 1], marker='o', s=100, color='black')

        # Draw tracks (filled if associated this frame)
        id_to_mean = {}
        for (x, y), tid, is_assoc in zip(means, ids, assoc_flags):
            color = id_to_color.get(tid, (0, 0, 0, 1))
            if is_assoc:
                ax.scatter([x], [y], marker="o", s=200, edgecolors='black', facecolors='red', linewidths=1.5)
            else:
                ax.scatter([x], [y], marker="o", s=200, edgecolors='red', facecolors="red", linewidths=1.5)
            ax.text(x, y, f"{tid}", fontsize=10, ha="center", va="center", color='white')
            id_to_mean[tid] = (x, y)

        # Draw association lines (track mean -> observation)
        for (tid, oi) in pairs:
            if tid in id_to_mean and obs is not None and 0 <= oi < len(obs):
                x_m, y_m = id_to_mean[tid]
                x_o, y_o = float(obs[oi, 0]), float(obs[oi, 1])
                ax.plot([x_m, x_o], [y_m, y_o], linewidth=1.5, color='black', linestyle='dashed')

        return []

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=len(frames_means), interval=interval, blit=False
    )

    if save_path:
        ext = save_path.lower().rsplit(".", 1)[-1]
        fps = max(1, int(1000 / interval))
        if ext == "mp4":
            try:
                writer = animation.FFMpegWriter(fps=fps)
                anim.save(save_path, writer=writer)
            except Exception:
                # If ffmpeg unavailable, fallback to Pillow (GIF) by changing extension
                anim.save(save_path.rsplit(".", 1)[0] + ".gif", writer="pillow", fps=fps)
        elif ext == "gif":
            anim.save(save_path, writer="pillow", fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps)
            anim.save(save_path + ".mp4", writer=writer)

    plt.close(fig)
    return anim
