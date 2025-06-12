from typing import Union, List
from os.path import abspath, expanduser, join, exists
from dateutil import parser
from datetime import date, datetime
import logging

import colored_logging as cl

logger = logging.getLogger(__name__)

from .generate_VIIRS_output_filename import generate_VIIRS_output_filename

def check_VIIRS_already_processed(
        VIIRS_output_directory: str,
        target_date: Union[date, str],
        time_UTC: Union[datetime, str],
        target: str,
        products: List[str]) -> bool:
    """
    Check if VIIRS GEOS-5 FP products have already been processed for a given target and date.

    Args:
        VIIRS_output_directory (str): Directory where VIIRS output files are stored.
        target_date (Union[date, str]): The date for which to check processed products.
        time_UTC (Union[datetime, str]): The UTC time associated with the products.
        target (str): The target location or identifier.
        products (List[str]): List of product names to check.

    Returns:
        bool: True if all products have already been processed (files exist), False otherwise.
    """
    already_processed = True
    logger.info(
        f"checking if VIIRS GEOS-5 FP has previously been processed at {cl.place(target)} on {cl.time(target_date)}")

    # Iterate through each product and check if its output file exists
    for product in products:
        filename = generate_VIIRS_output_filename(
            VIIRS_output_directory=VIIRS_output_directory,
            target_date=target_date,
            time_UTC=time_UTC,
            target=target,
            product=product
        )

        if exists(filename):
            logger.info(
                f"found previous VIIRS GEOS-5 FP {cl.name(product)} at {cl.place(target)} on {cl.time(target_date)}: {cl.file(filename)}")
        else:
            logger.info(
                f"did not find previous VIIRS GEOS-5 FP {cl.name(product)} at {cl.place(target)} on {cl.time(target_date)}")
            already_processed = False  # If any product is missing, set flag to False

    return already_processed