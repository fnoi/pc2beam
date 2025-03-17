"""
Visualization utilities for point cloud data.
"""

from pathlib import Path
from typing import Union, Optional, Dict

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
                    name=f"Instance {instance_id}",
                    hoverinfo="none"
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
                name="Points",
                hoverinfo="none"
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
                    name="Normals" if i == 0 else None,
                    hoverinfo="none"
                )
            )
        
        # Update title with normal count
        fig.layout.annotations[0].text = f"{enhanced_title} | Normals: {n_points}"
    
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
        showlegend=True,
        hovermode=False
    )
    
    return fig


def plot_point_cloud_with_supernormals(
    points: np.ndarray,
    features: Dict[str, np.ndarray],
    instances: Optional[np.ndarray] = None,
    show_supernormals: bool = True,
    color_by: str = "instance",
    normal_length: float = 0.1,
    title: str = "Point Cloud with Supernormals",
    width: int = 1000,
    height: int = 800,
    max_points: int = 10000,
    color_by_s1: bool = False,
) -> go.Figure:
    """
    Create an interactive 3D visualization of point cloud data with supernormal vectors.
    
    Args:
        points: Point coordinates of shape (N, 3)
        features: Dictionary containing s1 feature values including 's1', 'sigma1', etc.
        instances: Optional instance labels of shape (N,)
        show_supernormals: Whether to show supernormal vectors
        color_by: How to color points ('instance', 'uniform', or 's1' if color_by_s1 is True)
        normal_length: Length of supernormal vector arrows
        title: Plot title
        width: Figure width in pixels
        height: Figure height in pixels
        max_points: Maximum number of points to display
        color_by_s1: Whether to color points by s1 feature value
        
    Returns:
        Plotly figure object that can be displayed in notebook or saved to HTML
    """
    # Get total number of points
    total_points = len(points)
    
    # Check if s1 feature is available
    if show_supernormals and ('s1' not in features or features['s1'] is None):
        raise ValueError("S1 feature is required for visualization with supernormals")
    
    # Get s1 vectors from features
    s1_vectors = features['s1']
    s1_magnitudes = np.linalg.norm(s1_vectors, axis=1)
    
    # Sample points if there are too many
    if total_points > max_points:
        # Use fixed seed for reproducibility
        np.random.seed(42)
        sample_idx = np.random.choice(total_points, max_points, replace=False)
        points_viz = points[sample_idx]
        s1_vectors_viz = s1_vectors[sample_idx]
        s1_magnitudes_viz = s1_magnitudes[sample_idx]
        instances_viz = instances[sample_idx] if instances is not None else None
        
        # Enhanced title with sampling info
        enhanced_title = f"{title} | Points: {max_points} (of {total_points})"
    else:
        points_viz = points
        s1_vectors_viz = s1_vectors
        s1_magnitudes_viz = s1_magnitudes
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
    if color_by_s1:
        # Create a colormap from s1 magnitudes
        colorscale = 'Viridis'
        fig.add_trace(
            go.Scatter3d(
                x=points_viz[:, 0],
                y=points_viz[:, 1],
                z=points_viz[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=s1_magnitudes_viz,
                    colorscale=colorscale,
                    opacity=0.8,
                    colorbar=dict(title="S1 Magnitude"),
                ),
                name="Points (S1 Value)",
                hoverinfo="none"
            )
        )
    elif color_by == "instance" and instances_viz is not None:
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
                    name=f"Instance {instance_id}",
                    hoverinfo="none"
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
                name="Points",
                hoverinfo="none"
            )
        )
    
    # Add supernormal vectors if requested
    if show_supernormals:
        # Sample points for supernormal visualization (avoid cluttering)
        n_points = len(points_viz)
        n_samples = min(1000, n_points)
        
        # Use fixed seed for reproducibility
        np.random.seed(43)
        s1_idx = np.random.choice(n_points, n_samples, replace=False)
        
        # Create arrows for supernormals
        s1_ends = points_viz[s1_idx] + s1_vectors_viz[s1_idx] * normal_length
        
        # Add supernormal vectors as lines
        for i in range(n_samples):
            fig.add_trace(
                go.Scatter3d(
                    x=[points_viz[s1_idx[i], 0], s1_ends[i, 0]],
                    y=[points_viz[s1_idx[i], 1], s1_ends[i, 1]],
                    z=[points_viz[s1_idx[i], 2], s1_ends[i, 2]],
                    mode="lines",
                    line=dict(
                        color="red", 
                        width=1.5,
                    ),
                    showlegend=True if i == 0 else False,
                    name="Supernormals" if i == 0 else None,
                    hoverinfo="none"
                )
            )
        
        # Update title with supernormal count
        fig.layout.annotations[0].text = f"{enhanced_title} | Supernormals: {n_samples}"
    
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
        showlegend=True,
        hovermode=False
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