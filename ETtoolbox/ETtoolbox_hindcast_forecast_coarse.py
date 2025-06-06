"""
ETtoolbox_hindcast_forecast_coarse.py

This module provides functions to generate hindcast and forecast evapotranspiration (ET) products
for a given tile using various remote sensing and meteorological data sources. It includes
functions to process both GEOS-5 FP and GFS forecast data, handling missing data and logging
progress and issues. The main entry point allows command-line execution with configurable
directories for working, static, SRTM, VIIRS, GEOS5FP, and output data.

Functions:
    ET_toolbox_hindcast_forecast_tile: Generates hindcast and forecast ET products for a tile.
    main: Command-line interface for running the ET toolbox hindcast/forecast process.

Usage:
    python ETtoolbox_hindcast_forecast_coarse.py <tile> [--working <dir>] [--static <dir>] [--SRTM <dir>]
        [--VIIRS <dir>] [--GEOS5FP <dir>] [--output <dir>]
"""

import sys
import logging
from datetime import datetime, timedelta, date
from typing import List, Callable, Union
from os.path import join

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
from .VIIRS_GFS_forecast import VIIRS_GFS_forecast
from .ET_toolbox_hindcast_tile import ET_toolbox_hindcast_tile

logger = logging.getLogger(__name__)

def ET_toolbox_hindcast_forecast_tile(
        tile: str,
        geometry: RasterGrid = None,
        present_date: Union[date, str] = None,
        water: Raster = None,
        model: PTJPL = None,
        model_name: str = "PTJPL",
        working_directory: str = None,
        static_directory: str = None,
        VIIRS_download: str = None,
        VIIRS_output_directory: str = None,
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
        preview_quality: int = PREVIEW_QUALITY,
        ANN_model: Callable = None,
        ANN_model_filename: str = None,
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
    Generate hindcast and forecast ET products for a given tile.

    Args:
        tile (str): Tile identifier.
        geometry (RasterGrid, optional): Geometry grid for the tile.
        present_date (date or str, optional): Date to center the forecast/hindcast on.
        water (Raster, optional): Water mask raster.
        model (PTJPL, optional): ET model instance.
        model_name (str, optional): Name of the ET model.
        working_directory (str, optional): Working directory for intermediate files.
        static_directory (str, optional): Directory for static data.
        VIIRS_download (str, optional): Directory for VIIRS downloads.
        VIIRS_output_directory (str, optional): Directory for VIIRS outputs.
        SRTM_connection (NASADEMConnection, optional): SRTM data connection.
        SRTM_download (str, optional): Directory for SRTM downloads.
        GEOS5FP_connection (GEOS5FP, optional): GEOS5FP data connection.
        GEOS5FP_download (str, optional): Directory for GEOS5FP downloads.
        GEOS5FP_products (str, optional): Directory for GEOS5FP products.
        GEDI_connection (GEDICanopyHeight, optional): GEDI data connection.
        GEDI_download (str, optional): Directory for GEDI downloads.
        ORNL_connection (MODISCI, optional): MODISCI data connection.
        CI_directory (str, optional): Directory for climate indices.
        soil_grids_connection (SoilGrids, optional): Soil grids data connection.
        soil_grids_download (str, optional): Directory for soil grids downloads.
        intermediate_directory (str, optional): Directory for intermediate files.
        preview_quality (int, optional): Preview quality setting.
        ANN_model (Callable, optional): Artificial Neural Network model.
        ANN_model_filename (str, optional): Filename for ANN model.
        resampling (str, optional): Resampling method.
        meso_cell_size (float, optional): Meso cell size.
        coarse_cell_size (float, optional): Coarse cell size.
        downscale_air (bool, optional): Whether to downscale air temperature.
        downscale_humidity (bool, optional): Whether to downscale humidity.
        downscale_moisture (bool, optional): Whether to downscale moisture.
        save_intermediate (bool, optional): Whether to save intermediate results.
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

    if geometry is None:
        geometry = sentinel_tiles.grid(tile, cell_size=meso_cell_size)

    if target_variables is None:
        target_variables = TARGET_VARIABLES

    if GEOS5FP_connection is None:
        GEOS5FP_connection = GEOS5FP(
            working_directory=working_directory,
            download_directory=GEOS5FP_download,
            products_directory=GEOS5FP_products
        )

    if SRTM_connection is None:
        # FIXME: fix handling of credentials here
        SRTM_connection = NASADEMConnection(
            working_directory=working_directory,
            download_directory=SRTM_download,
            offline_ok=True
        )

    missing_dates = []

    # Process hindcast for the previous 7 days (GEOS-5 FP)
    for relative_days in range(-7, 1):
        target_date = present_date + timedelta(days=relative_days)
        logger.info(f"VIIRS GEOS-5 FP target date: {colored_logging.time(target_date)} ({colored_logging.time(relative_days)} days)")

        time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
        logger.info(f"VIIRS target time solar: {colored_logging.time(time_solar)}")

        try:
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
                VIIRS_download_directory=VIIRS_download,
                VIIRS_output_directory=VIIRS_output_directory,
                intermediate_directory=intermediate_directory,
                preview_quality=preview_quality,
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
                save_intermediate=save_intermediate
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

    # Prepare forecast dates: missing hindcast dates + next 8 days
    forecast_dates = missing_dates + [present_date + timedelta(days=d) for d in range(8)]

    # Process forecast for the next 8 days (GFS)
    for target_date in forecast_dates:
        relative_days = target_date - present_date
        logger.info(f"GFS VIIRS target date: {colored_logging.time(target_date)} ({colored_logging.time(relative_days)} days)")

        try:
            VIIRS_GFS_forecast(
                target_date=target_date,
                geometry=geometry,
                target=tile,
                working_directory=working_directory,
                static_directory=static_directory,
                SRTM_connection=SRTM_connection,
                SRTM_download=SRTM_download,
                GEDI_connection=GEDI_connection,
                GEDI_download=GEDI_download,
                ORNL_connection=ORNL_connection,
                CI_directory=CI_directory,
                soil_grids_connection=soil_grids_connection,
                soil_grids_download=soil_grids_download,
                VIIRS_download_directory=VIIRS_download,
                intermediate_directory=intermediate_directory,
                preview_quality=preview_quality,
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
                save_intermediate=save_intermediate
            )
        except Exception as e:
            logger.exception(e)
            logger.warning(f"VIIRS GFS cannot be processed for date: {target_date}")
            continue

def main(argv=sys.argv):
    """
    Command-line interface for running the ET toolbox hindcast/forecast process.

    Args:
        argv (list): List of command-line arguments.

    Returns:
        None
    """
    tile = argv[1]

    # Parse command-line arguments for directories, with defaults
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

    # Call the hindcast/forecast tile function (calls ET_toolbox_hindcast_tile, not the function above)
    ET_toolbox_hindcast_tile(
        tile=tile,
        working_directory=working_directory,
        static_directory=static_directory,
        output_directory=output_directory,
        SRTM_download=SRTM_download,
        VIIRS_download=VIIRS_download_directory,
        GEOS5FP_download=GEOS5FP_download,
    )