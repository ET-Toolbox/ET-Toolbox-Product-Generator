from typing import Union

from datetime import date, timedelta
from dateutil import parser
import numpy as np
import logging
from rasters import Raster, RasterGrid

from LandsatL2C2 import LandsatL2C2

import colored_logging as cl

from .constants import *

logger = logging.getLogger(__name__)

def generate_landsat_ST_C_prior(
        date_UTC: Union[date, str],
        geometry: RasterGrid,
        target_name: str,
        landsat: LandsatL2C2 = None,
        working_directory: str = None,
        download_directory: str = None,
        landsat_initialization_days: int = LANDSAT_INITIALIZATION_DAYS) -> Raster:
    """
    Generate a prior composite of Landsat surface temperature (ST_C) images.

    This function creates a composite raster of surface temperature (ST_C) from Landsat imagery
    for a given date and geometry. It searches for available Landsat scenes within a specified
    initialization window before the target date, retrieves the ST_C product for each date,
    and computes the median composite.

    Args:
        date_UTC (Union[date, str]): The target date (as a date object or ISO string).
        geometry (RasterGrid): The spatial grid/geometry for the output raster.
        target_name (str): The name/identifier for the target area.
        landsat (LandsatL2C2, optional): An initialized LandsatL2C2 object. If None, a new one is created.
        working_directory (str, optional): Directory for temporary files.
        download_directory (str, optional): Directory for Landsat downloads.
        landsat_initialization_days (int, optional): Number of days before date_UTC to search for scenes.

    Returns:
        Raster: A Raster object containing the median composite of ST_C images.

    Raises:
        Exception: If no valid ST_C images are found in the initialization window.
    """
    # Parse date string if necessary
    if isinstance(date_UTC, str):
        date_UTC = parser.parse(date_UTC).date()

    # Initialize LandsatL2C2 object if not provided

    if landsat is None:
        landsat = LandsatL2C2(
            working_directory=working_directory,
            download_directory=download_directory
        )

    # Define the search window for Landsat scenes
    landsat_start = date_UTC - timedelta(days=landsat_initialization_days)
    landsat_end = date_UTC - timedelta(days=1)
    logger.info(f"generating Landsat temperature composite from {cl.time(landsat_start)} to {cl.time(landsat_end)}")

    # Search for available Landsat scenes in the window
    landsat_listing = landsat.scene_search(start=landsat_start, end=landsat_end, target_geometry=geometry)
    landsat_composite_dates = sorted(set(landsat_listing.date_UTC))
    logger.info(f"found Landsat granules on dates: {', '.join([cl.time(d) for d in landsat_composite_dates])}")

    ST_C_images = []

    # Retrieve the ST_C product for each available date
    for date_UTC in landsat_composite_dates:
        try:
            ST_C = landsat.product(acquisition_date=date_UTC, product="ST_C", geometry=geometry, target_name=target_name)
            ST_C_images.append(ST_C)
        except Exception as e:
            logger.warning(e)
            continue

    # Compute the median composite across all valid ST_C images
    composite = Raster(np.nanmedian(np.stack(ST_C_images), axis=0), geometry=geometry)

    return composite