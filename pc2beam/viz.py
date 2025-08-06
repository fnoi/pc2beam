"""
Visualization utilities for point cloud data.
"""

from pathlib import Path
from typing import Union, Optional, Dict, Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_point_cloud(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    instances: Optional[np.ndarray] = None,
    features: Optional[Dict] = None,

    mode: Literal["points", "supernormals"] = "points",
    color_by: Literal["instance", "uniform", "s1"] = "instance",
    show_vectors: bool = True,
    vector_length: float = 0.1,
    title: str = "Point Cloud Visualization",
    width: int = 1000,
    height: int = 800,
    max_points: int = 10000,
    ortho_view: bool = False,
    point_size: int = 2,
) -> go.Figure:
    """
    Unified function for point cloud visualization with various modes.
    
    Args:
        points: Point coordinates of shape (N, 3)
        normals: Optional normal vectors of shape (N, 3)
        instances: Optional instance labels of shape (N,)
        features: Optional dictionary containing features like 's1'


        mode: Visualization mode ('points', 'supernormals')
        color_by: How to color points ('instance', 'uniform', 's1')
        show_vectors: Whether to show normal/supernormal vectors
        vector_length: Length of vector arrows
        title: Plot title
        width: Figure width in pixels
        height: Figure height in pixels
        max_points: Maximum number of points to display
        ortho_view: If True, use orthographic projection
        point_size: Size of points in the visualization
        
    Returns:
        Plotly figure object
    """
    # Validate inputs based on mode
    if mode == "supernormals" and (features is None or 's1' not in features):
        raise ValueError("S1 feature required for supernormals mode")
    
    # Get total number of points
    total_points = len(points)
    
    # Sample points if there are too many
    if total_points > max_points:
        np.random.seed(42)
        sample_idx = np.random.choice(total_points, max_points, replace=False)
        points_viz = points[sample_idx]
        normals_viz = normals[sample_idx] if normals is not None else None
        instances_viz = instances[sample_idx] if instances is not None else None
        features_viz = {k: v[sample_idx] for k, v in features.items()} if features else None


        enhanced_title = f"{title} | Points: {max_points} (of {total_points})"
    else:
        points_viz = points
        normals_viz = normals
        instances_viz = instances
        features_viz = features


        enhanced_title = f"{title} | Points: {total_points}"
    
    # Create figure
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "scene"}]],
        subplot_titles=[enhanced_title]
    )
    
    # Prepare colors
    if color_by == "instance" and instances_viz is not None:
        unique_instances = np.unique(instances_viz)
        colors = _generate_colors(len(unique_instances))
        color_map = dict(zip(unique_instances, colors))
        
        # Add points by instance
        for instance_id in unique_instances:
            mask = instances_viz == instance_id
            _add_point_trace(fig, points_viz[mask], color_map[instance_id], 
                           f"Instance {instance_id}", point_size)
    elif color_by == "s1" and features_viz and 's1' in features_viz:
        s1_magnitudes = np.linalg.norm(features_viz['s1'], axis=1)
        _add_point_trace(fig, points_viz, s1_magnitudes, "Points (S1 Value)", 
                        point_size, colorscale='Viridis', colorbar_title="S1 Magnitude")

    else:
        # Uniform coloring
        _add_point_trace(fig, points_viz, "blue", "Points", point_size)
    
    # Add vectors if requested
    if show_vectors:
        if mode == "points" and normals_viz is not None:
            _add_vector_traces(fig, points_viz, normals_viz, vector_length, "Normals", "grey")
        elif mode == "supernormals" and features_viz and 's1' in features_viz:
            _add_vector_traces(fig, points_viz, features_viz['s1'], vector_length, "Supernormals", "red")
    
    # Update layout
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.5, y=1.5, z=1.5),
    )
    if ortho_view:
        camera["projection"] = dict(type="orthographic")
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=camera
        ),
        width=width,
        height=height,
        showlegend=True,
        hovermode=False
    )
    
    return fig


def _add_point_trace(fig, points, color, name, size, symbol="circle", 
                    colorscale=None, colorbar_title=None):
    """Helper function to add point traces to figure."""
    trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(
            size=size,
            color=color,
            opacity=0.8,
            symbol=symbol,
            colorscale=colorscale,
            colorbar=dict(title=colorbar_title) if colorbar_title else None,
        ),
        name=name,
        hoverinfo="none"
    )
    fig.add_trace(trace)


def _add_vector_traces(fig, points, vectors, length, name, color):
    """Helper function to add vector traces to figure."""
    vector_ends = points + vectors * length
    
    for i in range(len(points)):
        if i == 0:  # Only show legend for first vector
            show_legend = True
            legend_name = name
        else:
            show_legend = False
            legend_name = None
            
        trace = go.Scatter3d(
            x=[points[i, 0], vector_ends[i, 0]],
            y=[points[i, 1], vector_ends[i, 1]],
            z=[points[i, 2], vector_ends[i, 2]],
            mode="lines",
            line=dict(color=color, width=1.5),
            showlegend=show_legend,
            name=legend_name,
            hoverinfo="none"
        )
        fig.add_trace(trace)


def save_html(fig: go.Figure, path: Union[str, Path]) -> None:
    """Save Plotly figure as standalone HTML file."""
    fig.write_html(path)


def _generate_colors(n: int) -> list:
    """Generate distinct colors for visualization."""
    import matplotlib.pyplot as plt
    
    tab10 = plt.cm.get_cmap('tab10')
    colors = []
    for i in range(n):
        color_idx = i % 10
        rgb = tab10(color_idx)[:3]
        colors.append(f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})')
    
    return colors 