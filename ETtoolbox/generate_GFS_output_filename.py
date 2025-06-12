from typing import Union
from os.path import abspath, expanduser, join
from dateutil import parser
from datetime import date, datetime

from .generate_GFS_output_directory import generate_GFS_output_directory

def generate_GFS_output_filename(
        GFS_output_directory: str,
        target_date: Union[date, str],
        time_UTC: Union[datetime, str],
        target: str,
        product: str) -> str:
    """
    Generate the full output filename for a GFS product.

    Args:
        GFS_output_directory (str): Base directory for GFS output.
        target_date (Union[date, str]): The target date as a date object or string.
        time_UTC (Union[datetime, str]): The UTC time as a datetime object or string.
        target (str): The target location or identifier.
        product (str): The product name or type.

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
    directory = generate_GFS_output_directory(
        GFS_output_directory=GFS_output_directory,
        target_date=target_date,
        target=target
    )

    # Construct the output filename with timestamp, target, and product
    filename = join(directory, f"GFS_{time_UTC:%Y.%m.%d.%H.%M.%S}_{target}_{product}.tif")

    return filename