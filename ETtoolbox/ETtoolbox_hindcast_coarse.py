"""
ETtoolbox_hindcast_coarse.py

This module provides functionality to generate ET Toolbox hindcast and forecast products
for a specified tile and date range using various remote sensing and meteorological datasets.
It includes a main function for command-line execution and a core function for processing
a single tile.

Functions:
    ET_toolbox_hindcast_coarse_tile: Generates ET Toolbox products for a given tile and date range.
    main: Entry point for command-line execution.

Usage:
    python ETtoolbox_hindcast_coarse.py <tile> [--working <dir>] [--static <dir>] [--SRTM <dir>]
                                         [--VIIRS <dir>] [--GEOS5FP <dir>] [--output <dir>]
                                         [--spacetrack <file>] [--ERS <file>]
"""

import logging
import sys
from datetime import datetime, timedelta, date
from os.path import join, exists, expanduser
from typing import List, Callable, Union

import colored_logging
from gedi_canopy_height import GEDICanopyHeight
from GEOS5FP import GEOS5FP
from MODISCI import MODISCI
from PTJPL import PTJPL
from NASADEM import NASADEMConnection
from soil_capacity_wilting import SoilGrids
from rasters import Raster, RasterGrid
from sentinel_tiles import sentinel_tiles

from .constants import *
from .VIIRS_GEOS5FP_NRT import VIIRS_GEOS5FP_NRT, GEOS5FPNotAvailableError

logger = logging.getLogger(__name__)

def ET_toolbox_hindcast_coarse_tile(
        tile: str,
        geometry: RasterGrid = None,
        present_date: Union[date, str] = None,
        water: Raster = None,
        model: PTJPL = None,
        model_name: str = "PTJPL",
        working_directory: str = None,
        static_directory: str = None,
        VIIRS_download_directory: str = None,
        output_directory: str = None,
        output_bucket_name: str = None,
        SRTM_connection: NASADEMConnection = None,
        SRTM_download: str = None,
        GEOS5FP_connection: GEOS5FP = None,
        GEOS5FP_download: str = None,
        GEOS5FP_products: str = None,
        GEDI_connection: GEDICanopyHeight = None,
        GEDI_download: str = None,
        ORNL_connection: MODISCI = None,
        CI_directory: str = None,
        soil_grids_connection: SoilGrids = None,
        soil_grids_download: str = None,
        intermediate_directory: str = None,
        ANN_model: Callable = None,
        ANN_model_filename: str = None,
        spacetrack_credentials_filename: str = None,
        ERS_credentials_filename: str = None,
        resampling: str = RESAMPLING,
        meso_cell_size: float = MESO_CELL_SIZE,
        coarse_cell_size: float = COARSE_CELL_SIZE,
        downscale_air: bool = DOWNSCALE_AIR,
        downscale_humidity: bool = DOWNSCALE_HUMIDITY,
        downscale_moisture: bool = DOWNSCALE_MOISTURE,
        save_intermediate: bool = False,
        show_distribution: bool = True,
        load_previous: bool = True,
        target_variables: List[str] = None):
    """
    Generate ET Toolbox hindcast and forecast products for a given tile.

    Args:
        tile (str): The tile identifier.
        geometry (RasterGrid, optional): The spatial grid for processing.
        present_date (date or str, optional): The central date for processing.
        water (Raster, optional): Water mask raster.
        model (PTJPL, optional): Model instance to use.
        model_name (str, optional): Name of the model.
        working_directory (str, optional): Working directory for intermediate files.
        static_directory (str, optional): Directory for static data.
        VIIRS_download_directory (str, optional): Directory for VIIRS downloads.
        output_directory (str, optional): Directory for output products.
        output_bucket_name (str, optional): Cloud bucket for output.
        SRTM_connection (NASADEMConnection, optional): SRTM data connection.
        SRTM_download (str, optional): Directory for SRTM downloads.
        GEOS5FP_connection (GEOS5FP, optional): GEOS5FP data connection.
        GEOS5FP_download (str, optional): Directory for GEOS5FP downloads.
        GEOS5FP_products (str, optional): GEOS5FP product types.
        GEDI_connection (GEDICanopyHeight, optional): GEDI data connection.
        GEDI_download (str, optional): Directory for GEDI downloads.
        ORNL_connection (MODISCI, optional): MODISCI data connection.
        CI_directory (str, optional): Directory for climate indices.
        soil_grids_connection (SoilGrids, optional): Soil grids connection.
        soil_grids_download (str, optional): Directory for soil grids downloads.
        intermediate_directory (str, optional): Directory for intermediate files.
        ANN_model (Callable, optional): Artificial Neural Network model.
        ANN_model_filename (str, optional): Filename for ANN model.
        spacetrack_credentials_filename (str, optional): SpaceTrack credentials file.
        ERS_credentials_filename (str, optional): ERS credentials file.
        resampling (str, optional): Resampling method.
        meso_cell_size (float, optional): Meso cell size.
        coarse_cell_size (float, optional): Coarse cell size.
        downscale_air (bool, optional): Whether to downscale air temperature.
        downscale_humidity (bool, optional): Whether to downscale humidity.
        downscale_moisture (bool, optional): Whether to downscale soil moisture.
        save_intermediate (bool, optional): Whether to save intermediate files.
        show_distribution (bool, optional): Whether to show distribution plots.
        load_previous (bool, optional): Whether to load previous results.
        target_variables (List[str], optional): List of target variables.

    Returns:
        None
    """
    if present_date is None:
        present_date = datetime.utcnow().date()

    logger.info(
        f"generating ET Toolbox hindcast and forecast at tile {colored_logging.place(tile)} centered on present date: {colored_logging.time(present_date)}")

    # If geometry is not provided, generate it from the sentinel tile grid
    if geometry is None:
        geometry = sentinel_tiles.grid(tile, cell_size=meso_cell_size)

    # Use default target variables if not provided
    if target_variables is None:
        target_variables = TARGET_VARIABLES

    # Create GEOS5FP connection if not provided
    if GEOS5FP_connection is None:
        GEOS5FP_connection = GEOS5FP(
            working_directory=working_directory,
            download_directory=GEOS5FP_download
        )

    # Set default SpaceTrack credentials if not provided
    default_spacetrack_credentials_filename = join(expanduser("~"), ".spacetrack_credentials")
    if spacetrack_credentials_filename is None and exists(default_spacetrack_credentials_filename):
        spacetrack_credentials_filename = default_spacetrack_credentials_filename

    # Set default ERS credentials if not provided
    default_ERS_credentials_filename = join(expanduser("~"), ".ERS_credentials")
    if ERS_credentials_filename is None and exists(default_ERS_credentials_filename):
        ERS_credentials_filename = default_ERS_credentials_filename

    # Create SRTM connection if not provided
    if SRTM_connection is None:
        SRTM_connection = NASADEMConnection(
            working_directory=working_directory,
            download_directory=SRTM_download,
            offline_ok=True
        )

    missing_dates = []

    # Loop over the past 7 days up to present date
    for relative_days in range(-7, 1):
        target_date = present_date + timedelta(days=relative_days)
        logger.info(f"VIIRS GEOS-5 FP target date: {colored_logging.time(target_date)} ({colored_logging.time(relative_days)} days)")

        # Set target solar time (approximate overpass time)
        time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
        logger.info(f"VIIRS target time solar: {colored_logging.time(time_solar)}")

        try:
            # Run the VIIRS GEOS5FP NRT processing for this date
            VIIRS_GEOS5FP_NRT(
                target_date=target_date,
                geometry=geometry,
                target=tile,
                working_directory=working_directory,
                static_directory=static_directory,
                SRTM_connection=SRTM_connection,
                SRTM_download=SRTM_download,
                GEOS5FP_connection=GEOS5FP_connection,
                GEOS5FP_download=GEOS5FP_download,
                GEOS5FP_products=GEOS5FP_products,
                GEDI_connection=GEDI_connection,
                GEDI_download=GEDI_download,
                ORNL_connection=ORNL_connection,
                CI_directory=CI_directory,
                soil_grids_connection=soil_grids_connection,
                soil_grids_download=soil_grids_download,
                VIIRS_download_directory=VIIRS_download_directory,
                VIIRS_output_directory=output_directory,
                output_bucket_name=output_bucket_name,
                intermediate_directory=intermediate_directory,
                ANN_model=ANN_model,
                ANN_model_filename=ANN_model_filename,
                model=model,
                model_name=model_name,
                water=water,
                coarse_cell_size=coarse_cell_size,
                target_variables=target_variables,
                downscale_air=downscale_air,
                downscale_humidity=downscale_humidity,
                downscale_moisture=downscale_moisture,
                resampling=resampling,
                show_distribution=show_distribution,
                load_previous=load_previous,
                save_intermediate=save_intermediate,
                spacetrack_credentials_filename=spacetrack_credentials_filename
            )
        except GEOS5FPNotAvailableError as e:
            logger.warning(e)
            logger.warning(f"VIIRS GEOS-5 FP cannot be processed for date: {target_date}")
            missing_dates.append(target_date)
            continue
        except Exception as e:
            logger.exception(e)
            logger.warning(f"VIIRS GEOS-5 FP cannot be processed for date: {target_date}")
            missing_dates.append(target_date)
            continue

    logger.info("missing VIIRS GEOS-5 FP dates: " + ", ".join(colored_logging.time(d) for d in missing_dates))


def main(argv=sys.argv):
    """
    Command-line entry point for ET Toolbox hindcast coarse tile processing.

    Args:
        argv (list): List of command-line arguments.

    Returns:
        None
    """
    tile = argv[1]

    # Parse command-line arguments for directories and credentials
    if "--working" in argv:
        working_directory = argv[argv.index("--working") + 1]
    else:
        working_directory = "."

    if "--static" in argv:
        static_directory = argv[argv.index("--static") + 1]
    else:
        static_directory = join(working_directory, "PTJPL_static")

    if "--SRTM" in argv:
        SRTM_download = argv[argv.index("--SRTM") + 1]
    else:
        SRTM_download = join(working_directory, "SRTM_download_directory")

    if "--VIIRS" in argv:
        VIIRS_download_directory = argv[argv.index("--VIIRS") + 1]
    else:
        VIIRS_download_directory = join(working_directory, "VIIRS_download_directory")

    if "--GEOS5FP" in argv:
        GEOS5FP_download = argv[argv.index("--GEOS5FP") + 1]
    else:
        GEOS5FP_download = join(working_directory, "GEOS5FP_download_directory")

    if "--output" in argv:
        output_directory = argv[argv.index("--output") + 1]
    else:
        output_directory = join(working_directory, "output_directory")

    if "--spacetrack" in argv:
        spacetrack_credentials_filename = argv[argv.index("--spacetrack") + 1]
    else:
        spacetrack_credentials_filename = None

    if "--ERS" in argv:
        ERS_credentials_filename = argv[argv.index("--ERS") + 1]
    else:
        ERS_credentials_filename = None

    # Call the main processing function
    ET_toolbox_hindcast_coarse_tile(
        tile=tile,
        working_directory=working_directory,
        static_directory=static_directory,
        output_directory=output_directory,
        SRTM_download=SRTM_download,
        VIIRS_download_directory=VIIRS_download_directory,
        GEOS5FP_download=GEOS5FP_download,
        spacetrack_credentials_filename=spacetrack_credentials_filename,
        ERS_credentials_filename=ERS_credentials_filename
    )