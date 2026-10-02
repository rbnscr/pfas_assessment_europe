"""Run the PFAS assessment workflow from the command line.

The workflow loads PFAS observations from a Parquet file and HydroBASINS
level-04 and level-12 geometries from shapefiles. It derives a day-of-year
field, filters observations to years after 2018, and creates a subset that
excludes observations marked as below the limit of detection.

The processed data are used for ecological risk evaluation, factor analysis,
statistical summaries, concentration-ratio analysis, map generation, and
supplementary-table generation. Input and output paths are provided as
command-line arguments.

Logging is configured when this module is imported, and progress messages are
written throughout the workflow.
"""

import argparse
import logging
import os
import sys
from pathlib import (
    Path,
)

import geopandas as gpd
import pandas as pd

from pfas_assessment_europe.concentration_ratios import (
    conc_ratio,
)
from pfas_assessment_europe.ecological_risk_evaluation import (
    eco_risk_eval,
)
from pfas_assessment_europe.factor_analysis import (
    factor_analysis,
)
from pfas_assessment_europe.general_statistics import (
    general_info,
)
from pfas_assessment_europe.logging_config import (
    configure_logging,
)
from pfas_assessment_europe.maps import (
    eu_map,
    hotspot_maps,
)
from pfas_assessment_europe.supplement_table import (
    create_supplementary_table,
)

log_path = configure_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse and validate the command-line paths.

    Requires ``--input-path`` and ``--output-path`` arguments. The input path must
    exist, and the parent directory of the output path must exist.

    Returns:
        argparse.Namespace: Parsed arguments with ``input_path`` and
        ``output_path`` attributes.

    Raises:
        SystemExit: Exits with status code ``1`` if the input path does not exist
            or the output path's parent directory does not exist.

    Notes:
        This function does not create directories or verify that the output path
        itself is a directory.
    """
    parser = argparse.ArgumentParser(
        description="Template for a command-line tool using argparse."
    )

    # Mandatory path argument
    parser.add_argument(
        "-I",
        "--input-path",
        type=str,
        required=True,
        help="Path to the input directory.",
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


def load_files(input_path: Path) -> dict:
    """Load the PFAS observations and HydroBASINS geometries.

    Reads ``pfas_data.parquet`` and constructs a point geometry from its ``lon``
    and ``lat`` columns using the ``EPSG:4326`` coordinate reference system. It
    also reads the level-04 and level-12 HydroBASINS shapefiles.

    Args:
        input_path: Directory containing the input files.

    Returns:
        dict: Loaded data under these keys:

            - ``"gdf"``: PFAS observations as a GeoDataFrame.
            - ``"basins_lev04"``: Level-04 HydroBASINS polygons.
            - ``"basins_lev12"``: Level-12 HydroBASINS polygons.

    Raises:
        Exceptions from pandas or GeoPandas if an input file cannot be read, or
        if the required columns or geospatial data are invalid.

    """
    logger.info("--- Starting Loading Files ---")
    logger.info("Loading PFAS concentration GeoPackage")
    df = pd.read_parquet(input_path.joinpath("pfas_data.parquet"))
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")

    logger.info("Loading HydroBasins")
    basins_lev04 = gpd.read_file(input_path.joinpath("hybas_eu_lev04_v1c.shp"))
    basins_lev12 = gpd.read_file(input_path.joinpath("hybas_eu_lev12_v1c.shp"))
    input_data = {
        "gdf": gdf,
        "basins_lev04": basins_lev04,
        "basins_lev12": basins_lev12,
    }
    logger.info(input_data.keys())
    logger.info("--- Finished Loading Files ---")
    return input_data


def preprocess(loaded_data: dict) -> dict:
    """Prepare PFAS observations and basin geometries for analysis.

    Adds a ``dayofyear`` column derived from the observations' ``date`` column,
    selects observations whose ``year`` is greater than 2018, and creates a
    second subset excluding rows where ``less_than`` is true. The level-12
    HydroBASINS GeoDataFrame is reprojected to the CRS of the filtered
    observations.

    Args:
        loaded_data: Dictionary returned by :func:`load_files`. It must contain
            ``"gdf"``, ``"basins_lev04"``, and ``"basins_lev12"``.

    Returns:
        dict: Preprocessed data under these keys:

            - ``"gdf_timeframe"``: Observations from years after 2018.
            - ``"gdf_timeframe_dropped"``: Those observations excluding rows
            marked as below the limit of detection.
            - ``"basins_lev04"``: Level-04 basin geometries reprojected to the
            CRS of ``gdf_timeframe``.
            - ``"basins_lev12"``: Level-12 basin geometries reprojected to the
            CRS of ``gdf_timeframe``.
    """
    logger.info("--- Starting Preprocessing ---")

    gdf: gpd.GeoDataFrame = loaded_data["gdf"]
    gdf["dayofyear"] = gdf["date"].dt.dayofyear
    gdf_timeframe = gdf[(gdf.year > 2018)].copy()
    logger.info("Unifying Basins")
    basins_lev04: gpd.GeoDataFrame = loaded_data["basins_lev04"]
    basins_lev12: gpd.GeoDataFrame = loaded_data["basins_lev12"]
    basins_lev04 = basins_lev04.to_crs(gdf_timeframe.crs)
    basins_lev12 = basins_lev12.to_crs(gdf_timeframe.crs)

    gdf_timeframe_dropped = gdf_timeframe[~gdf_timeframe["less_than"]].copy()

    output: dict = {
        "gdf_timeframe": gdf_timeframe,
        "gdf_timeframe_dropped": gdf_timeframe_dropped,
        "basins_lev04": basins_lev04,
        "basins_lev12": basins_lev12,
    }

    logger.info("--- Finished Preprocessing ---")
    return output


def main() -> None:
    """Run the complete PFAS assessment workflow.

    The function configures the input and output paths from command-line
    arguments, loads the required datasets, preprocesses them, and executes
    all analysis and visualisation functions.

    The workflow runs the following analyses:

    - :func:`general_info`
    - :func:`eco_risk_eval`
    - :func:`factor_analysis`
    - :func:`conc_ratio`
    - :func:`eu_map`
    - :func:`hotspot_maps`
    - :func:`create_supplementary_table`

    Results are written to the output path supplied through
    ``--output-path``.

    Returns:
        None.

    Notes:
        Each workflow receives a copy of the preprocessed
        GeoDataFrame. The individual analysis functions may create files,
        modify their local copies, and write progress information to the
        configured log.

        If argument validation fails, :func:`parse_arguments` terminates the
        process with status code ``1``. Errors raised while loading data,
        preprocessing datasets, or running an analysis are not caught by this
        function and propagate to the caller.
    """
    logger.info(f"Log file: {log_path}")
    logger.info("STARTING ANALYSIS")
    logger.info("Parsing arguments")
    args = parse_arguments()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    logger.info(f"Input Path: {input_path}")
    logger.info(f"Output Path: {output_path}")

    loaded_data = load_files(input_path)
    preprocessed_data = preprocess(loaded_data)
    general_info(
        preprocessed_data["gdf_timeframe"].copy(),
        output_path,
    )
    eco_risk_eval(
        preprocessed_data["gdf_timeframe"].copy(),
        preprocessed_data["basins_lev12"].copy(),
        output_path,
    )
    factor_analysis_report = factor_analysis(
        preprocessed_data["gdf_timeframe_dropped"].copy(),
        output_path
    )
    
    conc_ratio(
        preprocessed_data["gdf_timeframe"].copy(),
        output_path
    )
    eu_map(
        preprocessed_data["gdf_timeframe"].copy(),
        preprocessed_data["basins_lev04"].copy(),
        output_path
    )
    hotspot_maps(
        preprocessed_data["gdf_timeframe"].copy(),
        preprocessed_data["basins_lev04"].copy(),
        output_path
    )
    additional_information = {
        "Factor Analysis" : factor_analysis_report
    }
    # additional_information = {} # Exit the 'create_supplementary_table' function early. This is just for trying out.
    create_supplementary_table(
        preprocessed_data["gdf_timeframe"].copy(),
        # preprocessed_data["basins_lev04"].copy(),
        additional_information,
        output_path,
    )
    logger.info("FINISHED ANALYSIS")


if __name__ == "__main__":
    main()
