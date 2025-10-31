import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Ellipse, Wedge
from matplotlib.animation import FuncAnimation

def plot_gaussian_2d( 
    ax: Axes, 
    mean: torch.Tensor,
    covariance: torch.Tensor,
    color: str = 'black',
    show_axes: bool = False,
    show_mean: bool = True
):
    mean = mean.detach().cpu().numpy()
    covariance = covariance.detach().cpu().numpy()
    eigvals, eigvecs = np.linalg.eigh(covariance)
    width, height = 2 * np.sqrt(np.abs(eigvals))
    angle = np.degrees(np.arctan2(*np.flip(eigvecs[:, 0])))
    ellipse = Ellipse(
        xy = mean, 
        width = width, 
        height = height, 
        angle = angle,
        edgecolor = color, 
        fc = 'None', 
        lw = 1, 
        alpha = 1.0
    )
    ax.add_patch(ellipse)
    if show_mean:
        ax.scatter(mean[0], mean[1], color=color, s=10)
    if show_axes:
        try:
            major_x = eigvecs[0, 1] * np.sqrt(np.maximum(eigvals[1], 0))  # use the larger eigenvalue
            major_y = eigvecs[1, 1] * np.sqrt(np.maximum(eigvals[1], 0))
            minor_x = eigvecs[0, 0] * np.sqrt(np.maximum(eigvals[0], 0))  # use the smaller eigenvalue
            minor_y = eigvecs[1, 0] * np.sqrt(np.maximum(eigvals[0], 0))
            
            # major axis
            ax.plot(
                [mean[0] - major_x, mean[0] + major_x],
                [mean[1] - major_y, mean[1] + major_y],
                color = color,
                linewidth = 1,
                alpha = 1.0
            )
            
            # minor axis
            ax.plot(
                [mean[0] - minor_x, mean[0] + minor_x],
                [mean[1] - minor_y, mean[1] + minor_y],
                color = color,
                linewidth = 1,
                alpha = 1.0
            )
        except: ...
    return