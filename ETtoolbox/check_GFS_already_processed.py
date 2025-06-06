from typing import Union, List
from os.path import exists
from datetime import date, datetime
import logging

import colored_logging as cl

from .generate_GFS_output_filename import generate_GFS_output_filename

logger = logging.getLogger(__name__)

def check_GFS_already_processed(
        GFS_output_directory: str,
        target_date: Union[date, str],
        time_UTC: Union[datetime, str],
        target: str,
        products: List[str]) -> bool:
    """
    Check if GFS VIIRS products have already been processed for a given target and date.

    Args:
        GFS_output_directory (str): Directory where GFS output files are stored.
        target_date (Union[date, str]): The date for which to check processing.
        time_UTC (Union[datetime, str]): The UTC time for which to check processing.
        target (str): The target location or identifier.
        products (List[str]): List of product names to check.

    Returns:
        bool: True if all products have already been processed (files exist), False otherwise.
    """
    already_processed = True
    logger.info(
        f"checking if GFS VIIRS has previously been processed at {cl.place(target)} on {cl.time(target_date)}"
    )

    # Iterate over each product and check if its output file exists
    for product in products:
        filename = generate_GFS_output_filename(
            GFS_output_directory=GFS_output_directory,
            target_date=target_date,
            time_UTC=time_UTC,
            target=target,
            product=product
        )

        if exists(filename):
            logger.info(
                f"found previous GFS VIIRS {cl.name(product)} at {cl.place(target)} on {cl.time(target_date)}: {cl.file(filename)}"
            )
        else:
            logger.info(
                f"did not find previous GFS VIIRS {cl.name(product)} at {cl.place(target)} on {cl.time(target_date)}"
            )
            already_processed = False  # Mark as not fully processed if any file is missing

    return already_processed