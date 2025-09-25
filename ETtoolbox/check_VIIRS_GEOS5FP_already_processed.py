from typing import Union, List
from os.path import abspath, expanduser, join, exists
from dateutil import parser
from datetime import date, datetime
import logging

import colored_logging as cl

from .generate_VIIRS_GEOS5FP_output_filename import generate_VIIRS_GEOS5FP_output_filename

logger = logging.getLogger(__name__)

def check_VIIRS_GEOS5FP_already_processed(
        VIIRS_GEOS5FP_output_directory: str,
        target_date: Union[date, str],
        time_UTC: Union[datetime, str],
        target: str,
        products: List[str]) -> bool:
    """
    Check if VIIRS GEOS-5 FP products have already been processed for a given date, time, and target.

    Args:
        VIIRS_GEOS5FP_output_directory (str): Directory where output files are stored.
        target_date (Union[date, str]): The date for which to check processing (can be a date object or string).
        time_UTC (Union[datetime, str]): The UTC time for which to check processing (can be a datetime object or string).
        target (str): The target location or identifier.
        products (List[str]): List of product names to check.

    Returns:
        bool: True if all products have already been processed (files exist), False otherwise.
    """
    already_processed = True
    logger.info(
        f"checking if VIIRS GEOS-5 FP has previously been processed at {cl.place(target)} on {cl.time(target_date)}")

    for product in products:
        # Generate the expected output filename for the product
        filename = generate_VIIRS_GEOS5FP_output_filename(
            VIIRS_GEOS5FP_output_directory=VIIRS_GEOS5FP_output_directory,
            target_date=target_date,
            time_UTC=time_UTC,
            target=target,
            product=product
        )

        # Check if the output file exists
        if exists(filename):
            logger.info(
                f"found previous VIIRS GEOS-5 FP {cl.name(product)} at {cl.place(target)} on {cl.time(target_date)}: {cl.file(filename)}")
        else:
            logger.info(
                f"did not find previous VIIRS GEOS-5 FP {cl.name(product)} at {cl.place(target)} on {cl.time(target_date)}")
            already_processed = False

    return already_processed