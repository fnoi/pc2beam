"""
Visualization utilities for point cloud data.
"""

from pathlib import Path
from typing import Union, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_point_cloud(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    instances: Optional[np.ndarray] = None,
    show_normals: bool = True,
    color_by: str = "instance",
    normal_length: float = 0.1,
    title: str = "Point Cloud Visualization",
    width: int = 1000,
    height: int = 800,
) -> go.Figure:
    """
    Create an interactive 3D visualization of point cloud data.
    
    Args:
        points: Point coordinates of shape (N, 3)
        normals: Optional normal vectors of shape (N, 3)
        instances: Optional instance labels of shape (N,)
        show_normals: Whether to show normal vectors
        color_by: How to color points ('instance', 'normal', or 'uniform')
        normal_length: Length of normal vector arrows
        title: Plot title
        width: Figure width in pixels
        height: Figure height in pixels
        
    Returns:
        Plotly figure object that can be displayed in notebook or saved to HTML
    """
    # Create figure with subplots for legend
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "scene"}]],
        subplot_titles=[title]
    )
    
    # Prepare colors
    if color_by == "instance" and instances is not None:
        # Generate color map for instances
        unique_instances = np.unique(instances)
        colors = _generate_colors(len(unique_instances))
        color_map = dict(zip(unique_instances, colors))
        
        # Add points by instance for better legend
        for instance_id in unique_instances:
            mask = instances == instance_id
            fig.add_trace(
                go.Scatter3d(
                    x=points[mask, 0],
                    y=points[mask, 1],
                    z=points[mask, 2],
                    mode="markers",
                    marker=dict(
                        size=2,
                        color=color_map[instance_id],
                        opacity=0.8
                    ),
                    name=f"Instance {instance_id}"
                )
            )
    
    elif color_by == "normal" and normals is not None:
        # Color by normal direction
        point_colors = (normals + 1) / 2  # Map [-1,1] to [0,1]
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=point_colors[:, 1],  # Use y component for coloring
                    colorscale="Viridis",
                    opacity=0.8
                ),
                name="Points"
            )
        )
    
    else:
        # Uniform coloring
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color="blue",
                    opacity=0.8
                ),
                name="Points"
            )
        )
    
    # Add normal vectors if requested and available
    if show_normals and normals is not None:
        # Sample points for normal visualization (avoid cluttering)
        n_points = len(points)
        n_samples = min(1000, n_points)
        idx = np.random.choice(n_points, n_samples, replace=False)
        
        # Create arrows for normals
        normal_end = points[idx] + normals[idx] * normal_length
        
        # Add normal vectors as lines
        for i in range(n_samples):
            fig.add_trace(
                go.Scatter3d(
                    x=[points[idx[i], 0], normal_end[i, 0]],
                    y=[points[idx[i], 1], normal_end[i, 1]],
                    z=[points[idx[i], 2], normal_end[i, 2]],
                    mode="lines",
                    line=dict(color="grey", width=2),
                    showlegend=True if i == 0 else False,
                    name="Normals" if i == 0 else None
                )
            )
    
    # Update layout
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=width,
        height=height,
        showlegend=True
    )
    
    return fig


def save_html(
    fig: go.Figure,
    path: Union[str, Path],
) -> None:
    """
    Save Plotly figure as standalone HTML file.
    
    Args:
        fig: Plotly figure to save
        path: Output path for HTML file
    """
    fig.write_html(path)


def _generate_colors(n: int) -> list:
    """Generate distinct colors for visualization."""
    import colorsys
    
    # Use HSV color space for even distribution
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + 0.3 * (i % 2)  # Alternate saturation
        value = 0.8 + 0.2 * (i % 2)  # Alternate brightness
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})')
    
    return colors 