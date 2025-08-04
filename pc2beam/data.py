"""
Point cloud data structure and processing utilities.
"""

import numpy as np
import open3d as o3d
from typing import Optional
from . import processing


class PointCloud:
    """Point cloud data structure."""

    def __init__(self, points: np.ndarray, normals: Optional[np.ndarray] = None,
                 instances: Optional[np.ndarray] = None):
        self.points = points
        self.normals = normals
        self.instances = instances

    @property
    def has_normals(self):
        return self.normals is not None
    
    @property
    def has_instances(self):
        return self.instances is not None
    
    @property
    def size(self):
        return len(self.points)
    
    def from_txt(self, path: str):
        points = np.loadtxt(path)
        return PointCloud(points)
    
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
        pcd = processing.calculate_s1(self.points, self.normals, radius, k, use_radius)
        return pcd
    
    def compute_s2(self, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
        pcd = processing.calculate_s2(self.points, self.instances, distance_threshold, ransac_n, num_iterations)
        return pcd
    