"""
Main entry point for PC2Beam (function-based, no CLI parsing).
"""

import numpy as np
from pathlib import Path
from pc2beam import config_io
from pc2beam.data import PointCloud, Skeleton


def run_pc2beam(
    input_file,
    config_path="config/default.yaml",
    entry="instance"
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

    # Run s2 processing with configuration parameters
    if 's2' in config:
        distance_threshold = config['s2']['distance_threshold']
        ransac_n = config['s2']['ransac_n']
        num_iterations = config['s2']['num_iterations']
        print(f"Computing s2 features with distance_threshold={distance_threshold}, ransac_n={ransac_n}, num_iterations={num_iterations}")
    else:
        distance_threshold = 0.01
        ransac_n = 3
        num_iterations = 1000
        print(f"Using default s2 parameters: distance_threshold={distance_threshold}, ransac_n={ransac_n}, num_iterations={num_iterations}")

    point_cloud.compute_s2(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    print("s2 features computed successfully")

    skeleton = point_cloud.to_skeleton()
    print("skeleton initiated")

    skeleton.visualize()
    skeleton.visualize_with_points(point_cloud)

    # roadmap:
    # 2. extend to merge
    # 3. project points to plane
    # 4. polygon setup and fitting
    # 5. model reconstruction
    # 6. evaluation


if __name__ == "__main__":
    run_pc2beam(
        input_file="data/test_points.txt",
        config_path="config/default.yaml",
        entry="instance"
    )
