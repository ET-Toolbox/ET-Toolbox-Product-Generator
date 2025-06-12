from typing import Union
from os.path import join
from datetime import date, datetime
from dateutil import parser

from .generate_Landsat_output_directory import generate_Landsat_output_directory

def generate_Landsat_output_filename(
        Landsat_output_directory: str, 
        target_date: Union[date, str], 
        time_UTC: Union[datetime, str], 
        target: str,
        product: str) -> str:
    """
    Generate a Landsat output filename based on the provided directory, date, time, target, and product.

    Args:
        Landsat_output_directory (str): Base directory for Landsat outputs.
        target_date (Union[date, str]): The target date as a date object or ISO string.
        time_UTC (Union[datetime, str]): The UTC time as a datetime object or ISO string.
        target (str): The target location or identifier.
        product (str): The product type or name.

    Returns:
        str: The full path to the generated Landsat output file.
    """
    # Parse target_date if it's a string
    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()
    
    # Parse time_UTC if it's a string
    if isinstance(time_UTC, str):
        time_UTC = parser.parse(time_UTC)

    # Generate the output directory path
    directory = generate_Landsat_output_directory(
        Landsat_output_directory=Landsat_output_directory,
        target_date=target_date,
        target=target
    )

    # Construct the filename using the specified format
    filename = join(
        directory, 
        f"Landsat_{time_UTC:%Y.%m.%d.%H.%M.%S}_{target}_{product}.tif"
    )

    return filename