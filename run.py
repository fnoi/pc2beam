"""
Main entry point for PC2Beam.
"""

import argparse
from pc2beam import config_io

def main():
    parser = argparse.ArgumentParser(description="PC2Beam - Point Cloud to Beam Model Converter")
    parser.add_argument("input_file", help="Path to input point cloud file")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--entry", help="entry point to run")
    parser.add_argument("--debug", help="debug mode")
    args = parser.parse_args()

    config = config_io.load_config(args.config)
    
    # check if entry is valid
    if args.entry not in ["instance"]: # raw, semantic not supported yet
        parser.error("Invalid entry point")

    if args.entry == "raw": # input data is point cloud
        pass # semantic segmentation not supported

    if args.entry == "semantic": # input data is semantic segmentation result
        pass # instance segmentation not supported

    if args.entry == "instance": # input data is instance segmentation result
        # run s1
        a = 0
        # run s2
        # point projection and fit
        # model reconstruction




if __name__ == "__main__":
    main()
