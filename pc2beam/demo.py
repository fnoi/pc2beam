"""
PC2Beam Demonstrator Module

This module contains all the functionality needed for the Google Colab demonstrator,
including dummy data generation, processing functions, and visualizations.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


def generate_dummy_data():
    """Generate dummy point cloud datasets for demonstration"""
    datasets = {}
    np.random.seed(42)
    
    # Dataset 1: Simple beam-like structure
    n_points = 1000
    x = np.random.uniform(0, 10, n_points)
    y = np.random.normal(0, 0.5, n_points)
    z = np.random.normal(0, 0.3, n_points)
    points = np.column_stack([x, y, z])
    datasets['simple_beam'] = {'points': points}
    
    # Dataset 2: Complex structure with multiple beams
    n_points = 2000
    x1 = np.random.uniform(0, 8, n_points // 2)
    y1 = np.random.normal(0, 0.4, n_points // 2)
    z1 = np.random.normal(0, 0.2, n_points // 2)
    x2 = np.random.uniform(2, 6, n_points // 2)
    y2 = np.random.normal(2, 0.3, n_points // 2)
    z2 = np.random.normal(0, 0.2, n_points // 2)
    points = np.vstack([
        np.column_stack([x1, y1, z1]),
        np.column_stack([x2, y2, z2])
    ])
    datasets['complex_structure'] = {'points': points}
    
    # Dataset 3: Dense point cloud
    n_points = 5000
    x = np.random.uniform(0, 12, n_points)
    y = np.random.normal(0, 0.6, n_points)
    z = np.random.normal(0, 0.4, n_points)
    noise = np.random.normal(0, 0.1, n_points)
    y += noise * np.sin(x * 0.5)
    z += noise * np.cos(x * 0.3)
    points = np.column_stack([x, y, z])
    datasets['dense_cloud'] = {'points': points}
    
    return datasets


def process_s1_features(points, voxel_size=0.1, min_points=10):
    """Extract S1 features from point cloud"""
    features = []
    
    # Voxelize points
    voxel_coords = np.floor(points / voxel_size).astype(int)
    unique_voxels, counts = np.unique(voxel_coords, axis=0, return_counts=True)
    
    # Filter voxels with enough points
    valid_voxels = unique_voxels[counts >= min_points]
    
    for voxel in valid_voxels:
        # Find points in this voxel
        mask = np.all(voxel_coords == voxel, axis=1)
        voxel_points = points[mask]
        
        if len(voxel_points) >= min_points:
            # Calculate basic features
            centroid = np.mean(voxel_points, axis=0)
            std_dev = np.std(voxel_points, axis=0)
            density = len(voxel_points) / (voxel_size ** 3)
            
            # PCA for orientation
            if len(voxel_points) > 3:
                pca = PCA(n_components=3)
                pca.fit(voxel_points)
                eigenvalues = pca.explained_variance_
            else:
                eigenvalues = np.zeros(3)
            
            feature = {
                'centroid_x': centroid[0],
                'centroid_y': centroid[1],
                'centroid_z': centroid[2],
                'std_x': std_dev[0],
                'std_y': std_dev[1],
                'std_z': std_dev[2],
                'density': density,
                'eigenvalue_1': eigenvalues[0],
                'eigenvalue_2': eigenvalues[1],
                'eigenvalue_3': eigenvalues[2],
                'point_count': len(voxel_points)
            }
            features.append(feature)
    
    return features


def process_s2_features(points):
    """Extract S2 features from point cloud"""
    # Global statistics
    centroid = np.mean(points, axis=0)
    std_dev = np.std(points, axis=0)
    
    # Distance-based features
    distances = np.linalg.norm(points - centroid, axis=1)
    
    # Clustering
    if len(points) > 10:
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(points)
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    else:
        n_clusters = 1
    
    # PCA for global orientation
    pca = PCA(n_components=3)
    pca.fit(points)
    eigenvalues = pca.explained_variance_
    
    feature = {
        'global_centroid_x': centroid[0],
        'global_centroid_y': centroid[1],
        'global_centroid_z': centroid[2],
        'global_std_x': std_dev[0],
        'global_std_y': std_dev[1],
        'global_std_z': std_dev[2],
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'max_distance': np.max(distances),
        'n_clusters': n_clusters,
        'eigenvalue_1': eigenvalues[0],
        'eigenvalue_2': eigenvalues[1],
        'eigenvalue_3': eigenvalues[2],
        'total_points': len(points)
    }
    
    return [feature]


def project_to_beam(points):
    """Project 3D points to 2D beam coordinates"""
    # Find the principal direction (assumed to be the beam direction)
    pca = PCA(n_components=3)
    pca.fit(points)
    
    # Project points onto the principal components
    projected = pca.transform(points)
    
    # Create centerline (simplified)
    x_range = np.linspace(projected[:, 0].min(), projected[:, 0].max(), 50)
    centerline = np.column_stack([x_range, np.zeros_like(x_range), np.zeros_like(x_range)])
    
    # Transform back to original space
    centerline_original = pca.inverse_transform(centerline)
    
    return {
        'projected_points': points,  # Keep original points for visualization
        'centerline': centerline_original,
        'principal_components': pca.components_,
        'explained_variance': pca.explained_variance_
    }


def create_3d_beam_projection(beam_projection):
    """Create 3D visualization of beam projection"""
    points = np.array(beam_projection['projected_points'])
    centerline = np.array(beam_projection['centerline'])
    
    fig = go.Figure()
    
    # Add projected points
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=points[:, 2],  # Color by z-coordinate
            colorscale='Viridis',
            opacity=0.7
        ),
        name='Projected Points'
    ))
    
    # Add centerline
    fig.add_trace(go.Scatter3d(
        x=centerline[:, 0],
        y=centerline[:, 1],
        z=centerline[:, 2],
        mode='lines',
        line=dict(
            color='red',
            width=5
        ),
        name='Beam Centerline'
    ))
    
    fig.update_layout(
        title='3D Beam Projection',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=800,
        height=600
    )
    
    return fig


def create_xy_scatter_plot(beam_projection):
    """Create XY scatter plot of projected points"""
    points = np.array(beam_projection['projected_points'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=points[:, 0],
        y=points[:, 1],
        mode='markers',
        marker=dict(
            size=3,
            color=points[:, 2],
            colorscale='Viridis',
            opacity=0.7,
            colorbar=dict(title='Z-coordinate')
        ),
        name='Projected Points'
    ))
    
    fig.update_layout(
        title='XY Projection of Beam Points',
        xaxis_title='X',
        yaxis_title='Y',
        width=800,
        height=600
    )
    
    return fig


def process_pipeline(points, voxel_size=0.1, min_points=10):
    """Run the complete PC2Beam processing pipeline"""
    results = {}
    
    # S1 features
    results['s1_features'] = process_s1_features(points, voxel_size, min_points)
    
    # S2 features
    results['s2_features'] = process_s2_features(points)
    
    # Beam projection
    results['beam_projection'] = project_to_beam(points)
    
    return results 