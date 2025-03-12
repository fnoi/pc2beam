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
    max_points: int = 10000,
) -> go.Figure:
    """
    Create an interactive 3D visualization of point cloud data.
    
    Args:
        points: Point coordinates of shape (N, 3)
        normals: Optional normal vectors of shape (N, 3)
        instances: Optional instance labels of shape (N,)
        show_normals: Whether to show normal vectors
        color_by: How to color points ('instance' or 'uniform')
        normal_length: Length of normal vector arrows
        title: Plot title
        width: Figure width in pixels
        height: Figure height in pixels
        max_points: Maximum number of points to display
        
    Returns:
        Plotly figure object that can be displayed in notebook or saved to HTML
    """
    # Get total number of points
    total_points = len(points)
    
    # Sample points if there are too many
    if total_points > max_points:
        # Use fixed seed for reproducibility
        np.random.seed(42)
        sample_idx = np.random.choice(total_points, max_points, replace=False)
        points_viz = points[sample_idx]
        normals_viz = normals[sample_idx] if normals is not None else None
        instances_viz = instances[sample_idx] if instances is not None else None
        
        # Enhanced title with sampling info
        enhanced_title = f"{title} | Points: {max_points} (of {total_points})"
    else:
        points_viz = points
        normals_viz = normals
        instances_viz = instances
        sample_idx = np.arange(total_points)
        
        # Enhanced title with point count
        enhanced_title = f"{title} | Points: {total_points}"
    
    # Create figure with subplots for legend
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "scene"}]],
        subplot_titles=[enhanced_title]
    )
    
    # Prepare colors
    if color_by == "instance" and instances_viz is not None:
        # Generate color map for instances
        unique_instances = np.unique(instances_viz)
        colors = _generate_colors(len(unique_instances))
        color_map = dict(zip(unique_instances, colors))
        
        # Add points by instance for better legend
        for instance_id in unique_instances:
            mask = instances_viz == instance_id
            fig.add_trace(
                go.Scatter3d(
                    x=points_viz[mask, 0],
                    y=points_viz[mask, 1],
                    z=points_viz[mask, 2],
                    mode="markers",
                    marker=dict(
                        size=2,
                        color=color_map[instance_id],
                        opacity=0.8
                    ),
                    name=f"Instance {instance_id}"
                )
            )
    else:
        # Uniform coloring
        fig.add_trace(
            go.Scatter3d(
                x=points_viz[:, 0],
                y=points_viz[:, 1],
                z=points_viz[:, 2],
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
    if show_normals and normals_viz is not None:
        # Sample points for normal visualization (avoid cluttering)
        n_points = len(points_viz)
                
        # Create arrows for normals
        normal_end = points_viz + normals_viz * normal_length
        
        # Add normal vectors as lines
        for i in range(n_points):
            fig.add_trace(
                go.Scatter3d(
                    x=[points_viz[i, 0], normal_end[i, 0]],
                    y=[points_viz[i, 1], normal_end[i, 1]],
                    z=[points_viz[i, 2], normal_end[i, 2]],
                    mode="lines",
                    line=dict(color="grey", width=1.5),
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
    """
    Generate distinct colors for visualization using matplotlib's tab10 colormap.
    
    Args:
        n: Number of colors to generate
        
    Returns:
        List of RGB color strings
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    # Get tab10 colormap (10 distinct colors)
    tab10 = plt.cm.get_cmap('tab10')
    
    colors = []
    for i in range(n):
        # Cycle through tab10 colors if n > 10
        color_idx = i % 10
        rgb = tab10(color_idx)[:3]  # Extract RGB (ignore alpha)
        colors.append(f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})')
    
    return colors 