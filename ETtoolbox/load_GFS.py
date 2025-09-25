from typing import Union, List
from os.path import abspath, expanduser, join, splitext, basename
from glob import glob
from dateutil import parser
from datetime import date, datetime
import logging

from rasters import Raster

import colored_logging as cl

from .generate_GFS_output_directory import generate_GFS_output_directory

logger = logging.getLogger(__name__)

def load_GFS(GFS_output_directory: str, target_date: Union[date, str], target: str, products: List[str] = None):
    """
    Loads GFS product raster files from a specified output directory for a given date and target.

    Args:
        GFS_output_directory (str): Base directory where GFS outputs are stored.
        target_date (Union[date, str]): The date for which to load products. Can be a date object or a string.
        target (str): The target identifier (e.g., region or site name).
        products (List[str], optional): List of product names to load. If None, loads all products.

    Returns:
        dict: A dictionary mapping product names to Raster objects.
    """
    dataset = {}

    # Generate the directory path for the specified date and target
    directory = generate_GFS_output_directory(
        GFS_output_directory=GFS_output_directory,
        target_date=target_date,
        target=target
    )

    # Search for all .tif files in the directory
    pattern = join(directory, "*.tif")
    logger.info(f"searching for GFS products: {cl.val(pattern)}")
    filenames = glob(pattern)

    for filename in filenames:
        logger.info(f"loading GFS VIIRS file: {cl.file(filename)}")
        # Extract the product name from the filename
        product = splitext(basename(filename))[0].split("_")[-1]

        # If a product filter is provided, skip files not in the list
        if products is not None and product not in products:
            continue

        # Open the raster file and add it to the dataset
        image = Raster.open(filename)
        dataset[product] = image

    return dataset