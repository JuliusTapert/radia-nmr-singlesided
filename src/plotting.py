# plotting.py

import numpy as np
import matplotlib.pyplot as plt


def plot_contour(
    field_2d,
    axis1_vals,
    axis2_vals,
    title=None,
    xlabel=None,
    ylabel=None,
    cbar_label=None,
    levels=np.linspace(0.1,0.2,100),
    cmap="viridis",
    vmin=None,
    vmax=None
):
    """
    Generic contour plot for 2D scalar fields.

    Parameters
    ----------
    field_2d : ndarray (N2, N1)
        2D scalar field.
    axis1_vals : ndarray
        Values along horizontal axis.
    axis2_vals : ndarray
        Values along vertical axis.
    """

    fig, ax = plt.subplots(figsize=(6, 6))

    contour = ax.contourf(
        axis1_vals,
        axis2_vals,
        field_2d,
        levels=levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(contour)
    if cbar_label:
        cbar.set_label(cbar_label)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

    return fig, ax