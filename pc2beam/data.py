"""
Point cloud data structure and processing utilities.
"""

import numpy as np
import open3d as o3d
from typing import Optional, Union
from pathlib import Path
from . import processing, viz
from .ifc_io import load_ishape_catalogue
from .catalog_fit import solve_w_nsga_style, FitConfig

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
    
    def compute_s2(
        self,
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=1000,
        **kwargs,
    ):
        s2 = processing.calculate_s2(
            self.points,
            self.instances,
            distance_threshold,
            ransac_n,
            num_iterations,
            **kwargs,
        )
        self.features["s2"] = s2
        return self 

    def compute_legacy_projection(
        self,
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=1000,
        min_plane_inliers=10,
    ):
        if self.instances is None:
            raise ValueError("Instances are required for legacy projection flow.")
        if self.normals is None:
            raise ValueError("Normals are required for legacy projection flow.")

        projection = {}
        for instance_id in np.unique(self.instances):
            mask = self.instances == instance_id
            result = processing.project_points_by_plane_alignment(
                self.points[mask],
                self.normals[mask],
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations,
                min_plane_inliers=min_plane_inliers,
            )
            projection[int(instance_id)] = result
        self.features["legacy_projection"] = projection
        return self

    def fit_cross_sections_from_catalogue(
        self,
        ifc_catalogue_path: Union[str, Path],
        n_pop: int = 100,
        n_gen: int = 20,
        random_seed: int = 42,
    ):
        if "legacy_projection" not in self.features:
            raise ValueError("Run compute_legacy_projection before catalogue fitting.")
        _, catalogue_df = load_ishape_catalogue(ifc_catalogue_path)
        if catalogue_df.empty:
            raise ValueError("IfcIShapeProfileDef catalogue is empty.")

        fit_results = {}
        fit_cfg = FitConfig(n_pop=n_pop, n_gen=n_gen, random_seed=random_seed)
        for instance_id, projection in self.features["legacy_projection"].items():
            if not projection.get("ok"):
                fit_results[int(instance_id)] = {
                    "ok": False,
                    "status": f"projection_failed:{projection.get('status', 'unknown')}",
                }
                continue
            points_2d = np.asarray(projection["points_2d"], dtype=np.float64)
            normals_2d = np.asarray(projection["normals_2d"], dtype=np.float64)
            if len(points_2d) < 8:
                fit_results[int(instance_id)] = {"ok": False, "status": "insufficient_points"}
                continue
            fit = solve_w_nsga_style(points_2d, normals_2d, catalogue_df, fit_cfg=fit_cfg)
            fit_results[int(instance_id)] = {"ok": True, **fit}
        self.features["catalogue_fit"] = fit_results
        return self

    def visualize_legacy_projection(self, instance_id: int):
        if "legacy_projection" not in self.features:
            raise ValueError("Run compute_legacy_projection before visualization.")
        result = self.features["legacy_projection"].get(int(instance_id))
        if result is None or not result.get("ok"):
            raise ValueError(f"No successful projection for instance {instance_id}.")
        projection_plane = processing.normal_and_point_to_plane(result["vector_3d"], result["left_3d"])
        fig3d = viz.plot_segment_planes_3d(
            points=self.points[self.instances == int(instance_id)],
            planes=result["planes"],
            projection_plane=projection_plane,
            title=f"Instance {instance_id} points with fitted planes",
        )
        fig2d = viz.plot_projection_2d(
            result["points_2d"],
            title=f"Instance {instance_id} aligned 2D projection",
        )
        return {"fig3d": fig3d, "fig2d": fig2d}
    
    def visualize(
        self,
        mode="points",
        downsample_enabled: bool = True,
        downsample_max_points: int = 20000,
        downsample_seed: int = 42,
        **kwargs,
    ):
        """Visualize point cloud with various modes."""
        if mode == "supernormals" and "s1" in self.features:
            return viz.plot_point_cloud(
                self.points, 
                features=self.features,
                instances=self.instances,
                mode=mode,
                downsample_enabled=downsample_enabled,
                downsample_max_points=downsample_max_points,
                downsample_seed=downsample_seed,
                **kwargs
            )
        else:
            return viz.plot_point_cloud(
                self.points, 
                self.normals, 
                self.instances,
                mode=mode,
                downsample_enabled=downsample_enabled,
                downsample_max_points=downsample_max_points,
                downsample_seed=downsample_seed,
                **kwargs
            )
        
    def to_skeleton(self):
        skeleton = Skeleton()
        for instance in np.unique(self.instances):
            feature = self.features["s2"].get(instance)
            if feature is None:
                continue
            if feature.get("s2_vector") is None or feature.get("s2_point") is None:
                continue
            projected_points = processing.project_points_to_line(
                self.points[self.instances == instance],
                feature["s2_vector"],
                feature["s2_point"]
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
    
    def visualize(self):
        from . import viz
        fig = viz.plot_skeleton(self)
        fig.show()
    
    def visualize_with_points(self, point_cloud):
        """Visualize skeleton together with point cloud instances."""
        from . import viz
        fig = viz.plot_skeleton_with_points(point_cloud.points, point_cloud.instances, self)
        fig.show()
    