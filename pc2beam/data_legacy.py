"""
Point cloud data structures and processing utilities.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import open3d as o3d
import plotly.graph_objects as go

from . import viz


class S2Features:
    """S2 segment orientation features."""
    
    def __init__(self):
        self.s2_vectors = {}
        self.line_points = {}
        self.instance_ids = set()
    
    def add_instance_features(self, instance_id: int, s2: np.ndarray,
                             line_point: np.ndarray):
        self.s2_vectors[instance_id] = s2
        self.line_points[instance_id] = line_point
        self.instance_ids.add(instance_id)
    
    def get_s2_vector(self, instance_id: int) -> np.ndarray:
        return self.s2_vectors[instance_id]
    
    def get_line_point(self, instance_id: int) -> np.ndarray:
        return self.line_points[instance_id]
    
    def has_instance(self, instance_id: int) -> bool:
        return instance_id in self.s2_vectors
    
    def get_all_instances(self) -> list:
        return list(self.instance_ids)
    
    def to_dict(self) -> Dict:
        return {
            i: {"s2": self.s2_vectors[i], "line_point": self.line_points[i]}
            for i in self.instance_ids
            }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "S2Features":
        s2f = cls()
        for i, f in data.items():
            s2f.add_instance_features(i, f["s2"], f["line_point"])
        return s2f


class PointCloud:
    """Point cloud with coordinates, normals, instances, and features."""
    
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
    
    def compute_normals(self, radius=None, k=30, orientation_reference=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        if radius:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius)
            )
        else:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
            )
        if orientation_reference is not None:
            pcd.orient_normals_towards_points(orientation_reference.reshape(-1, 3))
        self.normals = np.asarray(pcd.normals).astype(np.float32)
    
    @classmethod
    def from_txt(cls, path: Union[str, Path]) -> "PointCloud":
        data = np.loadtxt(path)
        if data.shape[1] == 7:
            return cls(data[:, :3], data[:, 3:6], data[:, 6])
        elif data.shape[1] == 4:
            raise ValueError("Invalid file format")
    
    def visualize(self, **kwargs) -> go.Figure:
        return viz.plot_point_cloud(
            points=self.points,
            normals=self.normals,
            instances=self.instances,
            **kwargs
        )
    
    def visualize_with_supernormals(self, **kwargs) -> go.Figure:
        if "s1" not in self.features:
            raise ValueError("S1 feature missing")
        return viz.plot_point_cloud_with_supernormals(
            points=self.points,
            features=self.features,
            instances=self.instances,
            **kwargs
        )
    
    def save_html(self, path: Union[str, Path], **kwargs):
        viz.save_html(self.visualize(**kwargs), path)

    def calculate_s1(self, radius=0.1, k=30, use_radius=True):
        if not self.has_normals:
            raise ValueError("Normals required")
        from . import processing
        self.features.update(
            processing.calculate_s1(
                self.points, self.normals, radius=radius, k=k, use_radius=use_radius
        )
        )

    @property
    def has_s1_feature(self):
        return "s1" in self.features

    def calculate_s2(self, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
        if not self.has_instances:
            raise ValueError("Instances required")
        from . import processing
        # implement s2 estimation here
        

    @property
    def has_s2_feature(self):
        return "s2_features" in self.features

    def project_to_beam(self, min_points_per_instance=10):
        if not self.has_instances or not self.has_s2_feature:
            raise ValueError("Instances and S2 required")
        from . import processing
        s2f = self.features["s2_features"]
        unique = np.unique(self.instances)
        s2_vectors = np.zeros((len(self.points), 3))
        line_points = np.zeros((len(self.points), 3))
        for i in unique:
            if s2f.has_instance(i):
                mask = self.instances == i
                s2_vectors[mask] = s2f.get_s2_vector(i)
                line_points[mask] = s2f.get_line_point(i)
        self.features.update(
            processing.project_to_line(
            self.points,
            self.instances,
            s2_vectors,
            line_points,
            min_points_per_instance=min_points_per_instance
        )
        )
        
    @property
    def has_beam_projection(self):
        return "projected_points" in self.features and "instance_info" in self.features

    def visualize_beam_projection(self, **kwargs) -> go.Figure:
        if not self.has_beam_projection:
            raise ValueError("Beam projection missing")
        return viz.plot_beam_projection(
            points=self.points,
            projected_points=self.features["projected_points"],
            instances=self.instances,
            instance_info=self.features["instance_info"],
            **kwargs
        )

    def project_to_centerline(self):
        if not self.has_instances or not self.has_s2_feature:
            raise ValueError("Instances and S2 required")
        from . import processing
        s2f = self.features["s2_features"]
        self.features.update(
            processing.project_to_centerline(
            self.points,
            self.instances,
                s2f.to_dict()
            )
        )
        
    @property
    def has_centerline_projection(self):
        return "distances" in self.features and "centerlines" in self.features

    def visualize_centerlines(self, **kwargs) -> go.Figure:
        if not self.has_centerline_projection:
            raise ValueError("Centerline projection missing")
        return viz.plot_centerlines(
            points=self.points,
            distances=self.features["distances"],
            instances=self.instances,
            centerlines=self.features["centerlines"],
            **kwargs
        )

    def visualize_projected_points_xy(self, **kwargs) -> go.Figure:
        if not self.has_beam_projection:
            raise ValueError("Beam projection missing")
        return viz.plot_projected_points_xy(
            projected_points=self.features["projected_points"],
            instances=self.instances,
            **kwargs
        )