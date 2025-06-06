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

logger = logging.getLogger(__name__)


def ET_toolbox_hindcast_tile(
        tile: str,
        geometry: RasterGrid = None,
        present_date: Union[date, str] = None,
        water: Raster = None,
        model: PTJPL = None,
        model_name: str = "PTJPL",
        working_directory: str = None,
        static_directory: str = None,
        VIIRS_download: str = None,
        output_directory: str = None,
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
    Generate ET Toolbox hindcast and forecast products for a given Sentinel-2 tile.

    This function processes a time series of days (default: 7 days before present to present)
    for a specified Sentinel-2 tile, generating ET and related products using VIIRS and GEOS-5 FP data.
    It handles data connections, downloads, and invokes the main processing routine for each day.
    Missing or unavailable dates are logged.

    Args:
        tile (str): Sentinel-2 tile identifier.
        geometry (RasterGrid, optional): Spatial grid geometry. If None, uses default for tile.
        present_date (date or str, optional): The central date for processing. Defaults to today (UTC).
        water (Raster, optional): Water mask raster.
        model (PTJPL, optional): Model instance to use for ET calculation.
        model_name (str, optional): Name of the ET model. Defaults to "PTJPL".
        working_directory (str, optional): Directory for working files.
        static_directory (str, optional): Directory for static files.
        VIIRS_download (str, optional): Directory for VIIRS downloads.
        output_directory (str, optional): Directory for output products.
        SRTM_connection (NASADEMConnection, optional): Connection for SRTM/NASA DEM data.
        SRTM_download (str, optional): Directory for SRTM downloads.
        GEOS5FP_connection (GEOS5FP, optional): Connection for GEOS-5 FP data.
        GEOS5FP_download (str, optional): Directory for GEOS-5 FP downloads.
        GEOS5FP_products (str, optional): Directory for GEOS-5 FP products.
        GEDI_connection (GEDICanopyHeight, optional): Connection for GEDI data.
        GEDI_download (str, optional): Directory for GEDI downloads.
        ORNL_connection (MODISCI, optional): Connection for MODIS CI data.
        CI_directory (str, optional): Directory for CI data.
        soil_grids_connection (SoilGrids, optional): Connection for soil grids data.
        soil_grids_download (str, optional): Directory for soil grids downloads.
        intermediate_directory (str, optional): Directory for intermediate files.
        preview_quality (int, optional): Quality setting for previews.
        ANN_model (Callable, optional): Artificial Neural Network model for ET estimation.
        ANN_model_filename (str, optional): Filename for ANN model.
        resampling (str, optional): Resampling method.
        meso_cell_size (float, optional): Cell size for meso grid.
        coarse_cell_size (float, optional): Cell size for coarse grid.
        downscale_air (bool, optional): Whether to downscale air temperature.
        downscale_humidity (bool, optional): Whether to downscale humidity.
        downscale_moisture (bool, optional): Whether to downscale soil moisture.
        save_intermediate (bool, optional): Whether to save intermediate files.
        show_distribution (bool, optional): Whether to show distribution plots.
        load_previous (bool, optional): Whether to load previous results.
        target_variables (List[str], optional): List of target variables to process.

    Returns:
        None
    """
    if present_date is None:
        present_date = datetime.utcnow().date()

    logger.info(
        f"generating ET Toolbox hindcast and forecast at tile {colored_logging.place(tile)} centered on present date: {colored_logging.time(present_date)}")

    # If no geometry is provided, use the default Sentinel-2 tile grid
    if geometry is None:
        geometry = sentinel_tiles.grid(tile, cell_size=meso_cell_size)

    # Use default target variables if not specified
    if target_variables is None:
        target_variables = TARGET_VARIABLES

    # Initialize GEOS5FP connection if not provided
    if GEOS5FP_connection is None:
        GEOS5FP_connection = GEOS5FP(
            working_directory=working_directory,
            download_directory=GEOS5FP_download,
            products_directory=GEOS5FP_products
        )

    # Initialize SRTM/NASA DEM connection if not provided
    if SRTM_connection is None:
        # FIXME: fix handling of credentials here
        SRTM_connection = NASADEMConnection(
            working_directory=working_directory,
            download_directory=SRTM_download,
            offline_ok=True
        )

    missing_dates = []

    # Loop over the past 7 days up to present (inclusive)
    for relative_days in range(-7, 1):
        target_date = present_date + timedelta(days=relative_days)
        logger.info(f"VIIRS GEOS-5 FP target date: {colored_logging.time(target_date)} ({colored_logging.time(relative_days)} days)")

        # Set target solar time to 13:30 local
        time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
        logger.info(f"VIIRS target time solar: {colored_logging.time(time_solar)}")

        try:
            # Main processing routine for this date
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
                VIIRS_output_directory=output_directory,
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

    # Log any dates for which processing failed
    logger.info("missing VIIRS GEOS-5 FP dates: " + ", ".join(colored_logging.time(d) for d in missing_dates))