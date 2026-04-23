import numpy as np
import plotly.graph_objects as go


def plot_contour(
    field_2d,
    axis1_vals,
    axis2_vals,
    title=None,
    xlabel=None,
    ylabel=None,
    cbar_label=None,
    ncontours=None,
    cmap="viridis",
    vmin=None,
    vmax=None
):
    """
    Generic contour plot for 2D scalar fields (Plotly version).
    """

    # Handle color limits
    zmin = vmin if vmin is not None else np.min(field_2d)
    zmax = vmax if vmax is not None else np.max(field_2d)

    fig = go.Figure()

    fig.add_trace(go.Contour(
        z=field_2d,
        x=axis1_vals,
        y=axis2_vals,
        colorscale=cmap,
        zmin=zmin,
        zmax=zmax,
        ncontours=ncontours,
        colorbar=dict(title=cbar_label),
        line=dict(width=0),
        contours=dict(
            coloring="heatmap"  # filled contours
        ),
        hovertemplate=(
            f"{xlabel}=%{{x:.2f}}<br>"
            f"{ylabel}=%{{y:.2f}}<br>"
            "Value=%{z:.3e}<extra></extra>"
        )
    ))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
    )

    fig.show()
