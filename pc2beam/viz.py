"""
Visualization utilities for point cloud data.
"""

from pathlib import Path
from typing import Union, Optional, Dict, Literal, List, Tuple

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
    show_normals: Optional[bool] = None,
    normal_length: Optional[float] = None,
) -> go.Figure:
    """
    Unified function for point cloud visualization with various modes.
    
    Args:
        points: Point coordinates of shape (N, 3)
        normals: Optional normal vectors of shape (N, 3)
        instances: Optional instance labels of shape (N,)
        features: Optional dict; ``'s1'`` may be per-point scalars ``(N,)`` (e.g. intensity)
            or supernormal vectors ``(N, 3)``. ``mode='supernormals'`` requires ``(N, 3)``.

        mode: Visualization mode ('points', 'supernormals')
        color_by: How to color points ('instance', 'uniform', 's1'); ``'s1'`` uses ``features['s1']``
            as scalars ``(N,)`` or vector magnitudes from ``(N, 3)``.
        show_vectors: Whether to show normal/supernormal vectors
        vector_length: Length of vector arrows
        title: Plot title
        width: Figure width in pixels
        height: Figure height in pixels
        max_points: Maximum number of points to display
        ortho_view: If True, use orthographic projection
        point_size: Size of points in the visualization
        show_normals: Backward-compatible alias for show_vectors
        normal_length: Backward-compatible alias for vector_length
        
    Returns:
        Plotly figure object
    """
    # Backward-compatible argument aliases used by notebooks/older code.
    if show_normals is not None:
        show_vectors = show_normals
    if normal_length is not None:
        vector_length = normal_length

    # Validate inputs based on mode
    if mode == "supernormals" and (features is None or 's1' not in features):
        raise ValueError("S1 feature required for supernormals mode")
    if mode == "supernormals":
        s1_chk = np.asarray(features["s1"])
        if s1_chk.ndim != 2 or s1_chk.shape[1] != 3 or s1_chk.shape[0] != len(points):
            raise ValueError(
                "supernormals mode requires features['s1'] of shape (N, 3) with N equal to len(points)."
            )

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
    elif color_by == "s1" and features_viz and "s1" in features_viz:
        s1 = np.asarray(features_viz["s1"])
        n = len(points_viz)
        if s1.ndim == 1:
            if s1.shape[0] != n:
                raise ValueError(
                    f"features['s1'] length {s1.shape[0]} does not match number of points {n}"
                )
            s1_colors = s1
            trace_name = "Points (intensity)"
            cbar_title = "Intensity"
        elif s1.ndim == 2 and s1.shape[1] == 3:
            if s1.shape[0] != n:
                raise ValueError(
                    f"features['s1'] shape {s1.shape} incompatible with {n} points"
                )
            s1_colors = np.linalg.norm(s1, axis=1)
            trace_name = "Points (S1 Value)"
            cbar_title = "S1 Magnitude"
        else:
            raise ValueError(
                "features['s1'] must have shape (N,) for scalars or (N, 3) for vectors"
            )
        _add_point_trace(
            fig,
            points_viz,
            s1_colors,
            trace_name,
            point_size,
            colorscale="Viridis",
            colorbar_title=cbar_title,
        )

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


def plot_skeleton_with_points(points: np.ndarray, instances: np.ndarray, skeleton: "Skeleton", 
                            title: str = "Skeleton with Point Cloud", 
                            width: int = 1000, height: int = 800):
    """Plot skeleton lines together with point cloud instances."""
    fig = go.Figure()
    
    # Add point cloud traces by instance
    unique_instances = np.unique(instances)
    colors = _generate_colors(len(unique_instances))
    color_map = dict(zip(unique_instances, colors))
    
    for instance_id in unique_instances:
        mask = instances == instance_id
        instance_points = points[mask]
        
        fig.add_trace(go.Scatter3d(
            x=instance_points[:, 0],
            y=instance_points[:, 1], 
            z=instance_points[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=color_map[instance_id],
                opacity=0.7
            ),
            name=f"Instance {instance_id}",
            hoverinfo="none"
        ))
    
    # Add skeleton lines
    for line in skeleton.lines:
        for line_id, (start, end) in line.items():
            fig.add_trace(go.Scatter3d(
                x=[start[0], end[0]], 
                y=[start[1], end[1]], 
                z=[start[2], end[2]], 
                mode='lines',
                line=dict(color='red', width=4),
                name=f"Skeleton Line {line_id}",
                showlegend=True
            ))
    
    # Calculate extents for equal scaling
    all_x = points[:, 0].tolist()
    all_y = points[:, 1].tolist()
    all_z = points[:, 2].tolist()
    
    # Add skeleton line endpoints to extents calculation
    for line in skeleton.lines:
        for line_id, (start, end) in line.items():
            all_x.extend([start[0], end[0]])
            all_y.extend([start[1], end[1]])
            all_z.extend([start[2], end[2]])
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    z_min, z_max = min(all_z), max(all_z)
    
    # Calculate the maximum range to ensure equal scaling
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    
    # Center the ranges
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    
    # Set equal ranges for all axes
    x_range = [x_center - max_range/2, x_center + max_range/2]
    y_range = [y_center - max_range/2, y_center + max_range/2]
    z_range = [z_center - max_range/2, z_center + max_range/2]
    
    # Set layout with equal aspect ratio
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=x_range, visible=False),
            yaxis=dict(range=y_range, visible=False),
            zaxis=dict(range=z_range, visible=False),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        width=width,
        height=height,
        showlegend=True
    )
    
    return fig


def plot_skeleton(skeleton: "Skeleton"):
    """Plot skeleton."""
    fig = go.Figure()
    for line in skeleton.lines:
        for line_id, (start, end) in line.items():
            fig.add_trace(go.Scatter3d(
                x=[start[0], end[0]], 
                y=[start[1], end[1]], 
                z=[start[2], end[2]], 
                mode='lines',
                name=f"Line {line_id}"
            ))
    
    # Set equal aspect ratio for all axes
    fig.update_layout(
        scene=dict(
            aspectmode='cube',  # This ensures equal scale in all axes
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        width=800,
        height=600
    )
    
    return fig


def scanner_forward_vector(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    Local +X axis after intrinsic yaw (Z), pitch (Y), roll (X) in degrees.
    Used to visualize nominal scanner boresight together with HELIOS++ tripod legs.
    """
    yr, pr, rr = np.radians([yaw_deg, pitch_deg, roll_deg])
    cx, sx = np.cos(rr), np.sin(rr)
    cy, sy = np.cos(pr), np.sin(pr)
    cz, sz = np.cos(yr), np.sin(yr)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    r = rz @ ry @ rx
    v = r @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def plot_triangle_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    title: str = "Triangle mesh",
    max_triangles: int = 25_000,
    color: str = "lightgray",
    opacity: float = 1.0,
    width: int = 1000,
    height: int = 800,
    ortho_view: bool = True,
) -> go.Figure:
    """Plotly Mesh3d for OBJ-style vertices and triangular faces (N,3) int indices."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) > max_triangles:
        rng = np.random.default_rng(42)
        pick = rng.choice(len(f), size=max_triangles, replace=False)
        f = f[pick]
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                i=f[:, 0],
                j=f[:, 1],
                k=f[:, 2],
                color=color,
                opacity=opacity,
                flatshading=True,
            )
        ]
    )
    cam = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.5, y=1.5, z=1.2),
        projection=dict(type="orthographic" if ortho_view else "perspective"),
    )
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        scene=dict(aspectmode="data", camera=cam),
    )
    return fig


def plot_scanners_and_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    scanner_positions: List[Tuple[float, float, float]],
    scanner_orientations_deg: Optional[List[Tuple[float, float, float]]] = None,
    scanner_labels: Optional[List[str]] = None,
    vector_length: float = 3.0,
    mesh_title: str = "IFC beams + scanners",
    max_triangles: int = 25_000,
) -> go.Figure:
    """Overlay tessellated beams (mesh) with scanner standpoints and forward vectors."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) > max_triangles:
        rng = np.random.default_rng(42)
        pick = rng.choice(len(f), size=max_triangles, replace=False)
        f = f[pick]

    data = [
        go.Mesh3d(
            x=v[:, 0],
            y=v[:, 1],
            z=v[:, 2],
            i=f[:, 0],
            j=f[:, 1],
            k=f[:, 2],
            color="lightgray",
            opacity=0.85,
            name="beams",
            flatshading=True,
        )
    ]

    ori = scanner_orientations_deg or [(0.0, 0.0, 0.0)] * len(scanner_positions)
    labels = scanner_labels or [f"S{i}" for i in range(len(scanner_positions))]

    sx, sy, sz = [], [], []
    for (px, py, pz) in scanner_positions:
        sx.append(px)
        sy.append(py)
        sz.append(pz)
    data.append(
        go.Scatter3d(
            x=sx,
            y=sy,
            z=sz,
            mode="markers+text",
            marker=dict(size=8, color="red"),
            text=labels,
            textposition="top center",
            name="scanners",
        )
    )

    for i, ((px, py, pz), (yw, pt, rl), lab) in enumerate(
        zip(scanner_positions, ori, labels)
    ):
        fwd = scanner_forward_vector(yw, pt, rl) * vector_length
        data.append(
            go.Scatter3d(
                x=[px, px + fwd[0]],
                y=[py, py + fwd[1]],
                z=[pz, pz + fwd[2]],
                mode="lines",
                line=dict(color="crimson", width=6),
                name=f"boresight {lab}",
                showlegend=(i == 0),
            )
        )

    fig = go.Figure(data=data)
    cam = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.5, y=1.5, z=1.2),
        projection=dict(type="orthographic"),
    )
    fig.update_layout(
        title=mesh_title,
        width=1100,
        height=750,
        scene=dict(aspectmode="data", camera=cam),
    )
    return fig
