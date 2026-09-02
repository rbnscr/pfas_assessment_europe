import argparse
import os
import sys
from pathlib import Path

import geopandas as gpd

import logging

from pfas_assessment_europe.logging_config import configure_logging

log_path = configure_logging(naming = "preprocess")
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Template for a command-line tool using argparse."
    )

    # Mandatory path argument
    parser.add_argument(
        "-I",
        "--input-path",
        type=str,
        required=True,
        help="Path to the input file.",
    )

    # Optional path argument
    parser.add_argument(
        "-O",
        "--output-path",
        type=str,
        required=True,
        help="Path to the output directory.",
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.input_path):
        print(f"Error: The input path '{args.input_path}' does not exist.")
        sys.exit(1)

    if not os.path.isdir(os.path.dirname(args.output_path)):
        print(
            f"Error: The directory for the output path '{args.output_path}' does not exist."
        )
        sys.exit(1)

    return args


def preprocess(input_path: str, output_path: str) -> None:
    input = Path(input_path)
    output = Path(output_path)

    logger.info("Load full GeoPackage")
    gdf = gpd.read_file(input)

    logger.info("Crop GeoPackage to time >2018")
    gdf_timeframe = gdf[(gdf.year > 2018)]

    logger.info("Saving new GeoPackage to Ouput Path")
    gdf_timeframe.to_file(output, driver="GPKG")

    

def main() -> None:
    args = parse_arguments()

    logger.info("Starting PREPROCESSING")
    logger.info(f"Log file: {log_path}")

    logger.info(f"Input Path: {args.input_path}")
    logger.info(f"Output Path: {args.output_path}")

    preprocess(input_path = args.input_path, output_path = args.output_path)
    logger.info("Finished PREPROCESSING")

if __name__ == "__main__":
    main()
