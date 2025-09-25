from typing import Union
from glob import glob
from os.path import abspath, expanduser, join, splitext, basename
from dateutil import parser
from datetime import date
import logging

import rasters as rt

import colored_logging as cl

from .generate_Landsat_output_directory import generate_Landsat_output_directory

logger = logging.getLogger(__name__)

def load_Landsat(Landsat_output_directory: str, target_date: Union[date, str], target: str):
    """
    Loads Landsat raster products from a specified output directory for a given date and target.

    Args:
        Landsat_output_directory (str): Base directory where Landsat outputs are stored.
        target_date (Union[date, str]): Date of the Landsat data to load. Can be a date object or a string.
        target (str): Target identifier (e.g., site or region name).

    Returns:
        dict: A dictionary mapping product names (str) to raster image objects (rt.Raster).
    """
    dataset = {}    

    # Generate the directory path for the specified Landsat output
    directory = generate_Landsat_output_directory(
        Landsat_output_directory=Landsat_output_directory,
        target_date=target_date,
        target=target
    )

    # Find all GeoTIFF files in the directory
    filenames = glob(join(directory, "*.tif"))

    for filename in filenames:
        logger.info(f"loading Landsat GEOS-5 FP file: {cl.file(filename)}")
        # Extract the product name from the filename (assumes format: ..._<product>.tif)
        product = splitext(basename(filename))[0].split("_")[-1]
        # Open the raster file
        image = rt.Raster.open(filename)
        # Store the image in the dataset dictionary
        dataset[product] = image
    
    return dataset