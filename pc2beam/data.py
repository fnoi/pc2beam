"""
Point cloud data structures and processing utilities.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import open3d as o3d
import plotly.graph_objects as go

from . import viz
from . import processing


class PointCloud:
    """
    Point cloud data structure with support for coordinates, normals, and instance labels.
    
    Attributes:
        points (np.ndarray): Point coordinates of shape (N, 3)
        normals (Optional[np.ndarray]): Normal vectors of shape (N, 3)
        instances (np.ndarray): Instance labels of shape (N,)
        metadata (dict): Additional information about the point cloud
        features (dict): Point-wise features including s1
    """
    
    def __init__(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None,
        instances: Optional[np.ndarray] = None,
    ):
        """
        Initialize point cloud data structure.
        
        Args:
            points: Point coordinates of shape (N, 3)
            normals: Optional normal vectors of shape (N, 3)
            instances: Optional instance labels of shape (N,)
        """
        self.points = self._validate_points(points)
        self.normals = self._validate_normals(normals) if normals is not None else None
        self.instances = self._validate_instances(instances) if instances is not None else None
        self.metadata = {}
        self.features = {}
        
    def _validate_points(self, points: np.ndarray) -> np.ndarray:
        """Validate point coordinates."""
        if not isinstance(points, np.ndarray):
            raise TypeError("Points must be a numpy array")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must have shape (N, 3)")
        return points.astype(np.float32)
    
    def _validate_normals(self, normals: np.ndarray) -> np.ndarray:
        """Validate normal vectors."""
        if not isinstance(normals, np.ndarray):
            raise TypeError("Normals must be a numpy array")
        if normals.shape != self.points.shape:
            raise ValueError("Normals must have same shape as points")
        # Ensure unit length
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        return (normals / norms).astype(np.float32)
    
    def _validate_instances(self, instances: np.ndarray) -> np.ndarray:
        """Validate instance labels."""
        if not isinstance(instances, np.ndarray):
            raise TypeError("Instance labels must be a numpy array")
        if instances.shape != (self.points.shape[0],):
            raise ValueError("Instance labels must have shape (N,)")
        return instances.astype(np.int32)
    
    @property
    def has_normals(self) -> bool:
        """Check if point cloud has normal vectors."""
        return self.normals is not None
    
    @property
    def has_instances(self) -> bool:
        """Check if point cloud has instance labels."""
        return self.instances is not None
    
    @property
    def size(self) -> int:
        """Get number of points."""
        return len(self.points)
    
    def compute_normals(
        self,
        radius: Optional[float] = None,
        k: int = 30,
        orientation_reference: Optional[np.ndarray] = None
    ) -> None:
        """
        Compute normal vectors if not present.
        
        Args:
            radius: Search radius for normal estimation. If None, use k-nearest neighbors
            k: Number of nearest neighbors for normal estimation
            orientation_reference: Optional reference point for normal orientation
        """
        # Convert to Open3D format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        
        # Estimate normals
        if radius is not None:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius)
            )
        else:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
            )
            
        # Orient normals if reference point provided
        if orientation_reference is not None:
            pcd.orient_normals_towards_points(
                orientation_reference.reshape(-1, 3)
            )
            
        self.normals = np.asarray(pcd.normals).astype(np.float32)
    
    @classmethod
    def from_txt(cls, path: Union[str, Path]) -> "PointCloud":
        """
        Load point cloud from text file.
        
        Expected format: X Y Z [Nx Ny Nz] instance_label
        """
        data = np.loadtxt(path)
        
        if data.shape[1] == 7:  # With normals
            points = data[:, :3]
            normals = data[:, 3:6]
            instances = data[:, 6]
            return cls(points, normals, instances)
        elif data.shape[1] == 4:  # Without normals
            points = data[:, :3]
            instances = data[:, 3]
            return cls(points, instances=instances)
        else:
            raise ValueError("Invalid file format")
    
    def visualize(self, **kwargs) -> go.Figure:
        """
        Create an interactive 3D visualization using Plotly.
        
        Args:
            **kwargs: Additional arguments passed to viz.plot_point_cloud()
            
        Returns:
            Plotly figure object that can be displayed in notebook or saved to HTML
        """
        return viz.plot_point_cloud(
            points=self.points,
            normals=self.normals,
            instances=self.instances,
            **kwargs
        )
    
    def visualize_with_supernormals(self, **kwargs) -> go.Figure:
        """
        Create an interactive 3D visualization using Plotly, showing supernormals instead of normal vectors.
        
        Args:
            **kwargs: Additional arguments passed to viz.plot_point_cloud_with_supernormals()
            
        Returns:
            Plotly figure object that can be displayed in notebook or saved to HTML
        """
        if not self.has_s1_feature:
            raise ValueError("S1 feature must be calculated first. Call calculate_s1() before visualization.")
            
        return viz.plot_point_cloud_with_supernormals(
            points=self.points,
            features=self.features,
            instances=self.instances,
            **kwargs
        )
    
    def save_html(self, path: Union[str, Path], **kwargs) -> None:
        """
        Save visualization as standalone HTML file.
        
        Args:
            path: Output path for HTML file
            **kwargs: Additional arguments passed to visualize()
        """
        fig = self.visualize(**kwargs)
        viz.save_html(fig, path) 

    def calculate_s1(self, radius: float = 0.1, k: int = 30, use_radius: bool = True) -> None:
        """
        Calculate and store local orientation supernormal feature s1 for each point.
        Requires normals to be present.
        
        Args:
            radius: Radius for spherical neighborhood search
            k: Number of nearest neighbors if not using radius-based search
            use_radius: Whether to use radius-based search (True) or k-nearest neighbors (False)
        """
        if not self.has_normals:
            raise ValueError("Normals are required to calculate s1 feature")
            
        features = processing.calculate_s1(
            self.points, 
            self.normals, 
            radius=radius,
            k=k,
            use_radius=use_radius
        )
        
        # Store all features
        self.features.update(features)

    @property
    def has_s1_feature(self) -> bool:
        """Check if point cloud has s1 feature computed."""
        return "s1" in self.features

    def calculate_s2(
        self, 
        distance_threshold: float = 0.01,
        ransac_n: int = 3,
        num_iterations: int = 1000
    ) -> None:
        """
        Calculate segment orientation feature s2 for each cluster and store for each point.
        Requires instance labels to be present.
        
        Args:
            distance_threshold: Maximum distance a point can be from the plane model
            ransac_n: Number of points to randomly sample for each RANSAC iteration
            num_iterations: Number of RANSAC iterations
        """
        if not self.has_instances:
            raise ValueError("Instance labels are required to calculate s2 feature")
        
        print('pre-processing')

        features = processing.calculate_s2(
            self.points, 
            self.instances,
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations            
        )
        
        # Store all features
        self.features.update(features)

    @property
    def has_s2_feature(self) -> bool:
        """Check if point cloud has s2 feature computed."""
        return "s2" in self.features