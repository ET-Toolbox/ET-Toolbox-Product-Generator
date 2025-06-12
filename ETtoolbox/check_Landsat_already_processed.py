from typing import Union, List
from os.path import exists
from datetime import date, datetime
import logging

import colored_logging as cl

from .generate_Landsat_output_filename import generate_Landsat_output_filename

logger = logging.getLogger(__name__)

def check_Landsat_already_processed(
        Landsat_output_directory: str, 
        target_date: Union[date, str], 
        time_UTC: Union[datetime, str], 
        target: str,
        products: List[str]) -> bool:
    """
    Check if all specified Landsat products have already been processed for a given date, time, and target.

    Args:
        Landsat_output_directory (str): Directory where Landsat output files are stored.
        target_date (Union[date, str]): The date for which to check processing (can be a date object or string).
        time_UTC (Union[datetime, str]): The UTC time for which to check processing (can be a datetime object or string).
        target (str): The target location or identifier.
        products (List[str]): List of product names to check.

    Returns:
        bool: True if all products have already been processed (files exist), False otherwise.
    """
    already_processed = True
    logger.info(
        f"Checking if Landsat GEOS-5 FP has previously been processed at {cl.place(target)} on {cl.time(target_date)}"
    )
    
    for product in products:
        # Generate the expected output filename for the product
        filename = generate_Landsat_output_filename(
            Landsat_output_directory=Landsat_output_directory,
            target_date=target_date,
            time_UTC=time_UTC,
            target=target,
            product=product
        )

        # Check if the file exists
        if exists(filename):
            logger.info(
                f"Found previous Landsat GEOS-5 FP {cl.name(product)} at {cl.place(target)} on {cl.time(target_date)}: {cl.file(filename)}"
            )
        else:
            logger.info(
                f"Did not find previous Landsat GEOS-5 FP {cl.name(product)} at {cl.place(target)} on {cl.time(target_date)}"
            )
            already_processed = False  # At least one product is missing
    
    return already_processed