"""
Point cloud data structure and processing utilities.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import open3d as o3d
from typing import Optional, Union
from pathlib import Path
from tqdm import tqdm
from . import processing, viz
from .ifc_io import load_ishape_catalogue, resolve_catalogue_csv_path
from .catalog_fit import solve_w_nsga_style, FitConfig
from .cs_geometry import kmeans_points_normals_2D


def _fit_catalogue_instance_job(
    instance_id: int,
    projection: dict,
    catalogue_df,
    fit_cfg: FitConfig,
    n_downsample: int,
    km_seed: int,
    show_generation_progress: bool,
    progress_min_interval: float,
):
    points_2d = np.asarray(projection["points_2d"], dtype=np.float64)
    normals_2d = np.asarray(projection["normals_2d"], dtype=np.float64)
    n_in = len(points_2d)
    subsampling_meta = {
        "subsampling_method": "none",
        "n_input_points": int(n_in),
        "n_fitting_points": int(n_in),
    }
    pts_fit = points_2d
    nrm_fit = normals_2d
    cluster_weights = None
    if int(n_downsample) > 0 and int(n_downsample) < n_in:
        reps, rep_normals, cw, _labels = kmeans_points_normals_2D(
            points_2d,
            normals_2d,
            int(n_downsample),
            random_state=km_seed,
        )
        pts_fit = reps
        nrm_fit = rep_normals
        cluster_weights = cw
        subsampling_meta = {
            "subsampling_method": "kmeans",
            "n_input_points": int(n_in),
            "n_fitting_points": int(len(reps)),
        }

    fit = solve_w_nsga_style(
        pts_fit,
        nrm_fit,
        catalogue_df,
        fit_cfg=fit_cfg,
        cluster_weights=cluster_weights,
        show_progress=show_generation_progress,
        progress_label=f"instance {int(instance_id)}",
        progress_min_interval=progress_min_interval,
    )
    return int(instance_id), {**subsampling_meta, **fit}


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
            points=self.points,
            instances=self.instances,
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
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
        ransac_max_points: Optional[int] = None,
    ):
        if self.instances is None:
            raise ValueError("Instances are required for legacy projection flow.")
        if self.normals is None:
            raise ValueError("Normals are required for legacy projection flow.")

        projection = {}
        for instance_id in np.unique(self.instances):
            mask = self.instances == instance_id
            result = processing.project_instance_to_section_2d(
                self.points[mask],
                self.normals[mask],
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations,
                min_plane_inliers=min_plane_inliers,
                ransac_max_points=ransac_max_points,
            )
            projection[int(instance_id)] = result
        self.features["legacy_projection"] = projection
        return self

    def fit_cross_sections_from_catalogue(
        self,
        catalogue_csv_path: Optional[Union[str, Path]] = None,
        ifc_catalogue_path: Optional[Union[str, Path]] = None,
        catalogue_region: str = "eur",
        n_pop: int = 100,
        n_gen: int = 20,
        random_seed: int = 42,
        show_progress: bool = True,
        show_generation_progress: bool = False,
        progress_min_interval: float = 0.25,
        n_downsample: int = 0,
        polygon_subdivision: bool = False,
        edge_subdivision_lmax: float = 0.01,
        use_cluster_weights: bool = True,
        kmeans_random_seed: Optional[int] = None,
        n_jobs: int = -1,
    ):
        if "legacy_projection" not in self.features:
            raise ValueError("Run compute_legacy_projection before catalogue fitting.")
        catalogue_path = resolve_catalogue_csv_path(
            catalogue_region=catalogue_region,
            override_path=(
                catalogue_csv_path
                if catalogue_csv_path is not None
                else ifc_catalogue_path
            ),
        )
        _, catalogue_df = load_ishape_catalogue(
            catalogue_path,
            catalogue_region=catalogue_region,
        )
        if catalogue_df.empty:
            raise ValueError("Catalogue is empty after CSV normalization.")

        fit_results = {}
        km_seed = int(random_seed) if kmeans_random_seed is None else int(kmeans_random_seed)
        fit_cfg = FitConfig(
            n_pop=n_pop,
            n_gen=n_gen,
            random_seed=random_seed,
            polygon_subdivision=polygon_subdivision,
            edge_subdivision_lmax=edge_subdivision_lmax,
            use_cluster_weights=use_cluster_weights,
        )
        stats = {"processed": 0, "ok": 0, "failed": 0, "skipped": 0}
        jobs = int(n_jobs)
        if jobs == 0:
            jobs = 1
        if jobs < 0:
            jobs = max(1, os.cpu_count() or 1)

        pending = []
        iterator = self.features["legacy_projection"].items()
        for instance_id, projection in iterator:
            stats["processed"] += 1
            if not projection.get("ok"):
                fit_results[int(instance_id)] = {
                    "ok": False,
                    "status": f"projection_failed:{projection.get('status', 'unknown')}",
                }
                stats["failed"] += 1
                continue
            points_2d = np.asarray(projection["points_2d"], dtype=np.float64)
            if len(points_2d) < 8:
                fit_results[int(instance_id)] = {"ok": False, "status": "insufficient_points"}
                stats["skipped"] += 1
                continue
            pending.append((int(instance_id), projection))

        if show_progress:
            tqdm.write(
                f"[catalogue-fit] scheduling {len(pending)} fits with n_jobs={jobs}"
            )
        progress = None
        if show_progress:
            progress = tqdm(
                total=len(pending),
                desc="catalogue-fit instances",
                unit="inst",
                mininterval=max(0.0, float(progress_min_interval)),
            )
            progress.set_postfix(stats, refresh=False)

        if jobs == 1:
            for instance_id, projection in pending:
                iid, fit = _fit_catalogue_instance_job(
                    instance_id,
                    projection,
                    catalogue_df,
                    fit_cfg,
                    n_downsample,
                    km_seed,
                    show_generation_progress,
                    progress_min_interval,
                )
                fit_results[int(iid)] = {"ok": True, **fit}
                stats["ok"] += 1
                if progress is not None:
                    score = fit.get("fitness", {}).get("score")
                    if score is not None:
                        tqdm.write(
                            f"[catalogue-fit] instance={int(iid)} done score={float(score):.6f}"
                        )
                    progress.update(1)
                    progress.set_postfix(stats, refresh=False)
        else:
            # Keep stdout readable: generation-level tqdm from many processes gets noisy.
            if show_generation_progress and show_progress:
                tqdm.write("[catalogue-fit] disabling per-generation bars in parallel mode")
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                futures = [
                    ex.submit(
                        _fit_catalogue_instance_job,
                        instance_id,
                        projection,
                        catalogue_df,
                        fit_cfg,
                        n_downsample,
                        km_seed,
                        False,
                        progress_min_interval,
                    )
                    for instance_id, projection in pending
                ]
                for fut in as_completed(futures):
                    iid, fit = fut.result()
                    fit_results[int(iid)] = {"ok": True, **fit}
                    stats["ok"] += 1
                    if progress is not None:
                        score = fit.get("fitness", {}).get("score")
                        if score is not None:
                            tqdm.write(
                                f"[catalogue-fit] instance={int(iid)} done score={float(score):.6f}"
                            )
                        progress.update(1)
                        progress.set_postfix(stats, refresh=False)

        if progress is not None:
            progress.close()
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
    