"""
Main entry point for PC2Beam (function-based, no CLI parsing).
"""

import numpy as np
from pathlib import Path
from pc2beam import config_io
from pc2beam.data import PointCloud, Skeleton
from pc2beam.evaluation import (
    evaluate_s2_against_ground_truth,
    summarize_s2_metrics,
    summarize_catalogue_fit,
)


def run_pc2beam(
    input_file,
    config_path="config/default.yaml",
    entry="instance",
    ground_truth_yaml=None,
    evaluate=False,
    metrics_output_csv=None,
    run_legacy_projection=False,
    run_catalogue_fit=False,
    catalogue_csv_path=None,
    ifc_catalogue_path=None,
    catalogue_fit_overrides=None,
    visualize=True,
    run_s1=True,
):
    # Load configuration
    if not Path(config_path).exists():
        print(f"Warning: Config file {config_path} not found. Using default values.")
        config = {}
    else:
        config = config_io.load_config(config_path)

    if entry != "instance":
        raise ValueError("Only 'instance' entry is supported.")

    # Load point cloud data
    print(f"Loading point cloud from {input_file}")
    point_cloud = PointCloud.from_txt(input_file)

    # Display point cloud info
    print(f"Loaded point cloud with {point_cloud.size} points")
    if point_cloud.has_normals:
        print("Point cloud has normal vectors")
    if point_cloud.has_instances:
        print(f"Point cloud has {len(np.unique(point_cloud.instances))} instances")

    if run_s1:
        # Run s1 processing with configuration parameters
        if 's1' in config:
            radius = config['s1']['radius']
            k = config['s1']['k']
            use_radius = config['s1']['use_radius']
            print(f"Computing s1 features with radius={radius}, k={k}, use_radius={use_radius}")
        else:
            radius = 0.1
            k = 30
            use_radius = True
            print(f"Using default s1 parameters: radius={radius}, k={k}, use_radius={use_radius}")

        point_cloud.compute_s1(radius=radius, k=k, use_radius=use_radius)
        print("s1 features computed successfully")
    else:
        print("Skipping s1 feature computation (run_s1=False)")

    # Run s2 processing with configuration parameters
    if 's2' in config:
        distance_threshold = config['s2']['distance_threshold']
        distance_threshold_schedule = config['s2'].get('distance_threshold_schedule')
        ransac_n = config['s2']['ransac_n']
        num_iterations = config['s2']['num_iterations']
        quality_cfg = dict(config['s2'].get('quality', {}) or {})
        print(f"Computing s2 features with distance_threshold={distance_threshold}, ransac_n={ransac_n}, num_iterations={num_iterations}")
    else:
        distance_threshold = 0.01
        distance_threshold_schedule = None
        ransac_n = 3
        num_iterations = 1000
        quality_cfg = {}
        print(f"Using default s2 parameters: distance_threshold={distance_threshold}, ransac_n={ransac_n}, num_iterations={num_iterations}")

    point_cloud.compute_s2(
        distance_threshold=distance_threshold,
        distance_threshold_schedule=distance_threshold_schedule,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
        quality=quality_cfg,
    )
    print("s2 features computed successfully")

    if not run_legacy_projection:
        run_legacy_projection = bool(config.get("legacy_projection", {}).get("enabled", False))
    if not run_catalogue_fit:
        run_catalogue_fit = bool(config.get("catalogue_fit", {}).get("enabled", False))
    if catalogue_csv_path is None:
        fit_cfg = config.get("catalogue_fit", {})
        catalogue_csv_path = fit_cfg.get("catalogue_csv_path")
        if catalogue_csv_path is None:
            # Backward compatibility for older configs.
            catalogue_csv_path = fit_cfg.get("ifc_catalogue_path")

    if run_legacy_projection:
        print("Computing legacy-style projection/alignment per instance")
        leg = config.get("legacy_projection", {}) or {}
        rmp = leg.get("ransac_max_points")
        ransac_max_pts = int(rmp) if rmp is not None else None
        point_cloud.compute_legacy_projection(
            distance_threshold=float(leg.get("distance_threshold", distance_threshold)),
            ransac_n=int(leg.get("ransac_n", ransac_n)),
            num_iterations=int(leg.get("num_iterations", num_iterations)),
            min_plane_inliers=int(leg.get("min_plane_inliers", 10)),
            ransac_max_points=ransac_max_pts,
        )
        print("Legacy projection completed")

    catalogue_fit_summary = None
    if run_catalogue_fit:
        if "legacy_projection" not in point_cloud.features:
            leg = config.get("legacy_projection", {}) or {}
            rmp = leg.get("ransac_max_points")
            ransac_max_pts = int(rmp) if rmp is not None else None
            point_cloud.compute_legacy_projection(
                distance_threshold=float(leg.get("distance_threshold", distance_threshold)),
                ransac_n=int(leg.get("ransac_n", ransac_n)),
                num_iterations=int(leg.get("num_iterations", num_iterations)),
                min_plane_inliers=int(leg.get("min_plane_inliers", 10)),
                ransac_max_points=ransac_max_pts,
            )
        print("Running catalogue-based cross-section fitting")
        fit_cfg = dict(config.get("catalogue_fit", {}) or {})
        if catalogue_fit_overrides:
            fit_cfg.update(catalogue_fit_overrides)
        km_rs = fit_cfg.get("kmeans_random_seed")
        point_cloud.fit_cross_sections_from_catalogue(
            catalogue_csv_path=catalogue_csv_path,
            ifc_catalogue_path=ifc_catalogue_path,
            catalogue_region=str(fit_cfg.get("catalogue_region", "eur")),
            n_pop=int(fit_cfg.get("n_pop", 100)),
            n_gen=int(fit_cfg.get("n_gen", 20)),
            show_progress=bool(fit_cfg.get("show_progress", True)),
            show_generation_progress=bool(fit_cfg.get("show_generation_progress", False)),
            progress_min_interval=float(fit_cfg.get("progress_min_interval", 0.25)),
            n_downsample=int(fit_cfg.get("n_downsample", 0)),
            polygon_subdivision=bool(fit_cfg.get("polygon_subdivision", False)),
            edge_subdivision_lmax=float(fit_cfg.get("edge_subdivision_lmax", 0.01)),
            use_cluster_weights=bool(fit_cfg.get("use_cluster_weights", True)),
            kmeans_random_seed=int(km_rs) if km_rs is not None else None,
            n_jobs=int(fit_cfg.get("n_jobs", -1)),
        )
        catalogue_fit_summary = summarize_catalogue_fit(point_cloud.features["catalogue_fit"])
        print("Catalogue fitting summary:")
        for key, value in catalogue_fit_summary.items():
            print(f"  - {key}: {value}")

    skeleton = point_cloud.to_skeleton()
    print("skeleton initiated")

    metrics_df = None
    metrics_summary = None
    if evaluate:
        if ground_truth_yaml is None:
            raise ValueError("ground_truth_yaml is required when evaluate=True.")
        metrics_df = evaluate_s2_against_ground_truth(
            points=point_cloud.points,
            instances=point_cloud.instances,
            s2_features=point_cloud.features["s2"],
            ground_truth_yaml=ground_truth_yaml,
        )
        metrics_summary = summarize_s2_metrics(metrics_df)
        print("S2 evaluation summary:")
        for key, value in metrics_summary.items():
            print(f"  - {key}: {value}")
        if metrics_output_csv:
            metrics_df.to_csv(metrics_output_csv, index=False)
            print(f"Wrote per-instance metrics to {metrics_output_csv}")

    if visualize:
        skeleton.visualize()
        skeleton.visualize_with_points(point_cloud)

    # roadmap:
    # 2. extend to merge
    # 3. project points to plane
    # 4. polygon setup and fitting
    # 5. model reconstruction
    # 6. evaluation

    return {
        "point_cloud": point_cloud,
        "skeleton": skeleton,
        "metrics_df": metrics_df,
        "metrics_summary": metrics_summary,
        "catalogue_fit_summary": catalogue_fit_summary,
    }


if __name__ == "__main__":
    run_pc2beam(
        input_file="data/test_points.txt",
        config_path="config/default.yaml",
        entry="instance"
    )
