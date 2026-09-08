"""Run the complete PFAS assessment workflow from the command line.

This module loads PFAS concentration data, HydroBASINS geometries, and country
boundaries; preprocesses the datasets; and runs the available analysis and
visualisation workflows.

The workflow includes:

- Ecological risk evaluation.
- Factor analysis.
- General statistical summaries.
- Concentration-ratio analysis.
- European monitoring-coverage maps.
- PFAS hotspot maps.
- Supplementary Excel-table generation.

Input and output directories are supplied through command-line arguments.
Logging is configured when the module is imported, and progress messages are
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
    """Parse and validate command-line arguments.

    The function expects an existing input directory or path supplied through
    ``--input-path`` and an output path supplied through ``--output-path``.
    The parent directory of the output path must already exist.

    Returns:
        argparse.Namespace: Parsed command-line arguments with the
        ``input_path`` and ``output_path`` attributes.

    Notes:
        If the input path does not exist, an error message is printed and the
        process exits with status code ``1``. If the parent directory of the
        output path does not exist, an error message is printed and the process
        also exits with status code ``1``.

        The function does not create the output directory or otherwise verify
        that the output path itself is a directory.
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
    """Load PFAS, basin, and country boundary datasets.

    The function reads the PFAS observations, HydroBASINS level-04 and
    level-12 geometries, and country boundaries from ``input_path``.

    The following filenames are expected:

    - ``pfas_data.gpkg``
    - ``hybas_eu_lev04_v1c.shp``
    - ``hybas_eu_lev12_v1c.shp``
    - ``ne_10m_admin_0_countries.shp``

    Args:
        input_path: Directory containing the input datasets.

    Returns:
        dict: Dictionary containing the loaded GeoDataFrames under the
        following keys:

        - ``"gdf"``: PFAS concentration observations.
        - ``"basins_lev04"``: Level-04 HydroBASINS polygons.
        - ``"basins_lev12"``: Level-12 HydroBASINS polygons.
        - ``"countries"``: Country boundary polygons.

    Notes:
        The input data are loaded but not reprojected or otherwise modified by
        this function.

        Missing files, unsupported file formats, invalid geospatial data, or
        unreadable paths may result in exceptions raised by GeoPandas or its
        underlying file-reading libraries.
    """
    logger.info("--- Starting Loading Files ---")
    logger.info("Loading PFAS concentration GeoPackage")
    # gdf = gpd.read_file(input_path.joinpath("test_cutout.gpkg"))
    gdf = gpd.read_file(input_path.joinpath("pfas_data.gpkg"))
    logger.info("Loading HydroBasins")
    basins_lev04 = gpd.read_file(input_path.joinpath("hybas_eu_lev04_v1c.shp"))
    basins_lev12 = gpd.read_file(input_path.joinpath("hybas_eu_lev12_v1c.shp"))
    logger.info("Loading country shapes")
    countries = gpd.read_file(input_path.joinpath("ne_10m_admin_0_countries.shp"))
    input_data = {
        "gdf": gdf,
        "basins_lev04": basins_lev04,
        "basins_lev12": basins_lev12,
        "countries": countries,
    }
    logger.info(input_data.keys())
    logger.info("--- Finished Loading Files ---")
    return input_data


def preprocess(loaded_data: dict) -> dict:
    """Prepare the loaded datasets for the assessment workflows.

    The function derives a ``dayofyear`` column from the PFAS observation
    dates, selects observations collected after 2018, and creates a second
    subset containing only observations that are not marked as below the limit
    of detection.

    The level-04 and level-12 HydroBASINS GeoDataFrames, together with the
    country GeoDataFrame, are reprojected to the coordinate reference system
    of the cropped PFAS dataset.

    Args:
        loaded_data: Dictionary containing the loaded datasets. It must
            contain the following keys:

            - ``"gdf"``: Complete PFAS observations.
            - ``"basins_lev04"``: Level-04 HydroBASINS geometries.
            - ``"basins_lev12"``: Level-12 HydroBASINS geometries.
            - ``"countries"``: Country boundary geometries.

    Returns:
        dict: Dictionary containing the preprocessed datasets:

            - ``"gdf"``: Complete PFAS dataset with a derived
              ``dayofyear`` column.
            - ``"gdf_timeframe"``: Observations from years after 2018.
            - ``"gdf_timeframe_dropped"``: Observations from years after 2018
              that are not below the limit of detection.
            - ``"basins_lev04"``: Reprojected level-04 basin geometries.
            - ``"basins_lev12"``: Reprojected level-12 basin geometries.
            - ``"countries"``: Reprojected country geometries.
    """
    logger.info("--- Starting Preprocessing ---")

    logger.info("Crop GeoPackage to time > 2018")
    gdf: gpd.GeoDataFrame = loaded_data["gdf"]
    # logger.info(f"Length of dataset: {len(gdf)}")
    # gdf["month"] = pd.to_datetime(gdf["date"], format="ISO8601").dt.month
    gdf["dayofyear"] = pd.to_datetime(gdf["date"], format="ISO8601").dt.dayofyear
    logger.info(f"Length of dataset: {len(gdf)}")

    gdf_timeframe = gdf[(gdf.year > 2018)].copy()
    logger.info(f"Length of cropped dataset: {len(gdf_timeframe)}")

    logger.info("Unifying Basins")
    basins_lev04: gpd.GeoDataFrame = loaded_data["basins_lev04"]
    basins_lev12: gpd.GeoDataFrame = loaded_data["basins_lev12"]
    basins_lev04 = basins_lev04.to_crs(gdf_timeframe.crs)
    basins_lev12 = basins_lev12.to_crs(gdf_timeframe.crs)

    gdf_timeframe_dropped = gdf_timeframe[~gdf_timeframe["less_than"]].copy()
    countries: gpd.GeoDataFrame = loaded_data["countries"]
    countries = countries.to_crs(gdf_timeframe.crs)
    output: dict = {
        "gdf": gdf,
        "gdf_timeframe": gdf_timeframe,
        "gdf_timeframe_dropped": gdf_timeframe_dropped,
        "basins_lev04": basins_lev04,
        "basins_lev12": basins_lev12,
        "countries": countries,
    }

    logger.info("--- Finished Preprocessing ---")
    return output


def main() -> None:
    """Run the complete PFAS assessment workflow.

    The function configures the input and output paths from command-line
    arguments, loads the required datasets, preprocesses them, and executes
    all analysis and visualisation functions.

    The workflow runs the following analyses:

    - :func:`eco_risk_eval`
    - :func:`factor_analysis`
    - :func:`general_info`
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

    eco_risk_eval(
        preprocessed_data["gdf_timeframe"].copy(),
        preprocessed_data["basins_lev12"].copy(),
        output_path,
    )
    factor_analysis(
        preprocessed_data["gdf_timeframe_dropped"].copy(),
        preprocessed_data["basins_lev04"].copy(),
        output_path,
    )
    general_info(
        preprocessed_data["gdf"].copy(),
        preprocessed_data["gdf_timeframe"].copy(),
        output_path,
    )
    conc_ratio(
        preprocessed_data["gdf_timeframe"].copy(),
        preprocessed_data["basins_lev04"].copy(),
        output_path,
    )
    eu_map(
        preprocessed_data["gdf_timeframe"].copy(),
        preprocessed_data["basins_lev04"].copy(),
        output_path,
    )
    hotspot_maps(
        preprocessed_data["gdf_timeframe"].copy(),
        preprocessed_data["basins_lev04"].copy(),
        output_path,
    )
    create_supplementary_table(
        preprocessed_data["gdf_timeframe"].copy(),
        preprocessed_data["basins_lev04"].copy(),
        preprocessed_data["countries"].copy(),
        output_path,
    )
    logger.info("FINISHED ANALYSIS")


if __name__ == "__main__":
    main()
