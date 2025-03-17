"""
Main entry point for PC2Beam.
"""

import argparse
import numpy as np
from pathlib import Path
from pc2beam import config_io
from pc2beam.data import PointCloud

def main():
    parser = argparse.ArgumentParser(description="PC2Beam - Point Cloud to Beam Model Converter")
    parser.add_argument("input_file", help="Path to input point cloud file")
    parser.add_argument("--config", help="Path to configuration file", default="config/default.yaml")
    parser.add_argument("--entry", help="entry point to run")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Load configuration
    config_path = args.config
    if not Path(config_path).exists():
        print(f"Warning: Config file {config_path} not found. Using default values.")
        config = {}
    else:
        config = config_io.load_config(config_path)
    
    # check if entry is valid
    if args.entry not in ["instance"]: # raw, semantic not supported yet
        parser.error("Invalid entry point")

    if args.entry == "raw": # input data is point cloud
        pass # semantic segmentation not supported

    if args.entry == "semantic": # input data is semantic segmentation result
        pass # instance segmentation not supported

    if args.entry == "instance": # input data is instance segmentation result
        # Load point cloud data
        print(f"Loading point cloud from {args.input_file}")
        point_cloud = PointCloud.from_txt(args.input_file)
        
        # Display point cloud info
        print(f"Loaded point cloud with {point_cloud.size} points")
        if point_cloud.has_normals:
            print("Point cloud has normal vectors")
        if point_cloud.has_instances:
            print(f"Point cloud has {len(np.unique(point_cloud.instances))} instances")
        
        # Run s1 processing with configuration parameters
        if 's1' in config:
            radius = config.s1.radius
            k = config.s1.k
            use_radius = config.s1.use_radius
            print(f"Computing S1 features with radius={radius}, k={k}, use_radius={use_radius}")
        else:
            radius = 0.1
            k = 30
            use_radius = True
            print(f"Using default S1 parameters: radius={radius}, k={k}, use_radius={use_radius}")
            
        point_cloud.calculate_s1(radius=radius, k=k, use_radius=use_radius)
        print("S1 features computed successfully")
        
        # Additional processing steps would go here
        # ...


if __name__ == "__main__":
    main()
