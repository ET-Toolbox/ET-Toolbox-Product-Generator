from typing import Union, List
from os.path import exists, splitext, basename, join
from glob import glob
from datetime import date
import logging

import rasters as rt

import colored_logging as cl

from .generate_VIIRS_output_directory import generate_VIIRS_output_directory

logger = logging.getLogger(__name__)


def load_VIIRS(VIIRS_output_directory: str, target_date: Union[date, str], target: str, products: List[str] = None):
    """
    Load VIIRS GEOS-5 FP products for a given target and date.

    Args:
        VIIRS_output_directory (str): Root directory where VIIRS output is stored.
        target_date (Union[date, str]): The date for which to load products.
        target (str): The target location or identifier.
        products (List[str], optional): List of product names to load. If None, load all products.

    Returns:
        dict: Dictionary mapping product names to loaded raster images.
    """
    logger.info(f"loading VIIRS GEOS-5 FP products for {cl.place(target)} on {cl.time(target_date)}")

    dataset = {}

    # Generate the directory path for the given date and target
    directory = generate_VIIRS_output_directory(
        VIIRS_output_directory=VIIRS_output_directory,
        target_date=target_date,
        target=target
    )

    # Search for all .tif files in the directory
    pattern = join(directory, "*.tif")
    logger.info(f"searching for VIIRS product: {cl.val(pattern)}")
    filenames = glob(pattern)
    logger.info(f"found {cl.val(len(filenames))} VIIRS files")

    for filename in filenames:
        # Extract product name from filename (assumes product is last underscore-separated part before extension)
        product = splitext(basename(filename))[0].split("_")[-1]

        # Skip file if product is not in the requested list
        if products is not None and product not in products:
            continue

        logger.info(f"loading VIIRS GEOS-5 FP file: {cl.file(filename)}")
        # Load the raster image using the rasters library
        image = rt.Raster.open(filename)
        dataset[product] = image

    return dataset