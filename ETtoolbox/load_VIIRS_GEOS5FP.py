from typing import Union, List
from os.path import exists, splitext, basename, join
from glob import glob
import logging

import rasters as rt

import colored_logging as cl

from .generate_VIIRS_GEOS5FP_output_directory import generate_VIIRS_GEOS5FP_output_directory

logger = logging.getLogger(__name__)

def load_VIIRS_GEOS5FP(VIIRS_GEOS5FP_output_directory: str, target_date: Union[date, str], target: str,
                       products: List[str] = None):
    """
    Load VIIRS GEOS-5 FP product rasters for a given target and date.

    Args:
        VIIRS_GEOS5FP_output_directory (str): Base directory where VIIRS GEOS-5 FP outputs are stored.
        target_date (Union[date, str]): Date for which to load products.
        target (str): Target location or identifier.
        products (List[str], optional): List of product names to load. If None, load all products found.

    Returns:
        dict: Dictionary mapping product names to loaded raster images.
    """
    logger.info(f"loading VIIRS GEOS-5 FP products for {cl.place(target)} on {cl.time(target_date)}")

    dataset = {}

    # Generate the directory path for the specified target and date
    directory = generate_VIIRS_GEOS5FP_output_directory(
        VIIRS_GEOS5FP_output_directory=VIIRS_GEOS5FP_output_directory,
        target_date=target_date,
        target=target
    )

    # Search for all .tif files in the directory
    pattern = join(directory, "*.tif")
    logger.info(f"searching for VIIRS GEOS-5 FP product: {cl.val(pattern)}")
    filenames = glob(pattern)
    logger.info(f"found {cl.val(len(filenames))} VIIRS GEOS-5 FP files")

    for filename in filenames:
        # Extract product name from filename (assumes product is last underscore-separated part before extension)
        product = splitext(basename(filename))[0].split("_")[-1]

        # If a product filter is provided, skip files not in the list
        if products is not None and product not in products:
            continue

        logger.info(f"loading VIIRS GEOS-5 FP file: {cl.file(filename)}")
        # Load the raster image using the rasters package
        image = rt.Raster.open(filename)
        dataset[product] = image

    return dataset