"""
Point cloud data structure and processing utilities.
"""

import numpy as np
import open3d as o3d
from typing import Optional, Union
from pathlib import Path
from . import processing


class PointCloud:
    """Point cloud data structure."""

    def __init__(self, points: np.ndarray, normals: Optional[np.ndarray] = None,
                 instances: Optional[np.ndarray] = None):
        self.points = self._validate(points, (None, 3), np.float32)
        self.normals = (
            self._validate(normals, self.points.shape, np.float32, True)
            if normals is not None else None
        )
        self.instances = (
            self._validate(instances, (self.points.shape[0],), np.int32, True)
            if instances is not None else None
        )
        self.metadata = {}
        self.features = {}

    def _validate(self, arr, shape, dtype, flatten=False):
        """Validate array shape and type."""
        if arr is None:
            return None
        arr = np.asarray(arr)
        if shape[0] is not None and arr.shape[0] != shape[0]:
            raise ValueError("Shape mismatch")
        if len(shape) > 1 and arr.shape[1:] != shape[1:]:
            raise ValueError("Shape mismatch")
        arr = arr.astype(dtype)
        if flatten:
            arr = arr.reshape(shape)
        return arr

    @property
    def has_normals(self):
        return self.normals is not None
    
    @property
    def has_instances(self):
        return self.instances is not None
    
    @property
    def size(self):
        return len(self.points)
    
    @classmethod
    def from_txt(cls, path: Union[str, Path]) -> "PointCloud":
        data = np.loadtxt(path)
        
        # Check if we have more than 3 columns
        if data.shape[1] > 3:
            # First 3 columns are points
            points = data[:, :3]
            # Next 3 columns are normals (if available)
            normals = data[:, 3:6] if data.shape[1] >= 6 else None
            # Last column is instance (if available)
            instances = data[:, 6] if data.shape[1] >= 7 else None
        else:
            # Only points available
            points = data
            normals = None
            instances = None
        
        return PointCloud(points, normals, instances)
    
    def compute_normals(self, radius=None, k=30, orientation_reference=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        if radius:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=k))
        else:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=k))
        self.normals = np.asarray(pcd.normals)
        return self
    
    def compute_s1(self, radius=None, k=None, use_radius=True):
        s1 = processing.calculate_s1(self.points, self.normals, radius, k, use_radius)
        self.features["s1"] = s1
        return self
    
    def compute_s2(self, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
        s2 = processing.calculate_s2(self.points, self.instances, distance_threshold, ransac_n, num_iterations)
        self.features["s2"] = s2
        return self 
    
    def visualize(self, mode="points", **kwargs):
        """Visualize point cloud with various modes."""
        if mode == "supernormals" and "s1" in self.features:
            return viz.plot_point_cloud(
                self.points, 
                features=self.features,
                instances=self.instances,
                mode=mode,
                **kwargs
            )
        else:
            return viz.plot_point_cloud(
                self.points, 
                self.normals, 
                self.instances,
                mode=mode,
                **kwargs
            )
        
    def to_skeleton(self):
        skeleton = Skeleton()
        for instance in np.unique(self.instances):
            projected_points = processing.project_points_to_line(
                self.points[self.instances == instance],
                self.features["s2"][instance]["s2_vector"],
                self.features["s2"][instance]["s2_point"]
                )
            skeleton.add_line(instance, projected_points[0], projected_points[1])
        return skeleton

    
class Skeleton:
    """Skeleton data structure."""

    def __init__(self):
        self.lines = []

    # initiate lines with id and start and end points
    def add_line(self, id: int, start: np.ndarray, end: np.ndarray):
        self.lines.append({id: [start, end]})
        return self
    