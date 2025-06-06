from typing import Union
from os.path import abspath, expanduser, join
from dateutil import parser
from datetime import date

def generate_VIIRS_output_directory(
        VIIRS_output_directory: str,
        target_date: Union[date, str],
        target: str) -> str:
    """
    Generate the output directory path for VIIRS products.

    Args:
        VIIRS_output_directory (str): Base directory for VIIRS outputs. Can include '~' for home directory.
        target_date (Union[date, str]): The date for which the output is generated. Can be a date object or a string.
        target (str): The target identifier to include in the directory name.

    Returns:
        str: The absolute path to the VIIRS output directory for the given date and target.
    """
    # If target_date is a string, parse it into a date object
    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()

    # Construct the output directory path using the base directory, date, and target
    directory = join(
        abspath(expanduser(VIIRS_output_directory)),
        f"{target_date:%Y-%m-%d}",
        f"VIIRS_{target_date:%Y-%m-%d}_{target}",
    )

    return directory