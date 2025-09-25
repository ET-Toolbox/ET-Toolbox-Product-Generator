from typing import Union
from os.path import abspath, expanduser, join
from dateutil import parser
from datetime import date
from datetime import datetime

from .generate_VIIRS_output_directory import generate_VIIRS_output_directory

def generate_VIIRS_output_filename(
        VIIRS_output_directory: str,
        target_date: Union[date, str],
        time_UTC: Union[datetime, str],
        target: str,
        product: str):
    """
    Generate the full output filename for a VIIRS product.

    Args:
        VIIRS_output_directory (str): Base directory for VIIRS outputs.
        target_date (Union[date, str]): The target date as a date object or string.
        time_UTC (Union[datetime, str]): The UTC time as a datetime object or string.
        target (str): The target location or identifier.
        product (str): The VIIRS product name.

    Returns:
        str: The absolute path to the output file, including the filename.
    """
    # Parse target_date if it's a string
    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()

    # Parse time_UTC if it's a string
    if isinstance(time_UTC, str):
        time_UTC = parser.parse(time_UTC)

    # Generate the output directory path
    directory = generate_VIIRS_output_directory(
        VIIRS_output_directory=VIIRS_output_directory,
        target_date=target_date,
        target=target
    )

    # Construct the output filename using the specified format
    filename = join(directory, f"VIIRS_{time_UTC:%Y.%m.%d.%H.%M.%S}_{target}_{product}.tif")

    return filename