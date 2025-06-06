from typing import Union
from os.path import abspath, expanduser, join
from dateutil import parser
from datetime import date

def generate_GFS_output_directory(
        GFS_output_directory: str,
        target_date: Union[date, str],
        target: str) -> str:
    """
    Generate the full output directory path for GFS products.

    Args:
        GFS_output_directory (str): Base directory for GFS output.
        target_date (Union[date, str]): Date for which to generate the directory.
            Can be a datetime.date object or a string in a parseable date format.
        target (str): Target identifier to include in the directory name.

    Returns:
        str: The absolute path to the GFS output directory for the given date and target.
    """
    # If target_date is a string, parse it into a date object
    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()

    # Construct the directory path using the base directory, date, and target
    directory = join(
        abspath(expanduser(GFS_output_directory)),
        f"{target_date:%Y-%m-%d}",
        f"GFS_{target_date:%Y-%m-%d}_{target}",
    )

    return directory