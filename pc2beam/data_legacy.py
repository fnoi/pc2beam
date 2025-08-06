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
