from typing import Union
from os.path import abspath, expanduser, join
from dateutil import parser
from datetime import date

def generate_VIIRS_GEOS5FP_output_directory(
        VIIRS_GEOS5FP_output_directory: str,
        target_date: Union[date, str],
        target: str) -> str:
    """
    Generate the output directory path for VIIRS GEOS-5 FP products.

    Args:
        VIIRS_GEOS5FP_output_directory (str): Base directory for VIIRS GEOS-5 FP output.
        target_date (Union[date, str]): Date for which the output is generated (date object or ISO string).
        target (str): Target identifier to include in the directory name.

    Returns:
        str: Full path to the output directory for the specified date and target.

    Raises:
        ValueError: If VIIRS_GEOS5FP_output_directory is None.
    """
    if VIIRS_GEOS5FP_output_directory is None:
        raise ValueError("no VIIRS GEOS-5 FP output directory given")

    # Parse target_date if it's a string
    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()

    # Construct the output directory path
    directory = join(
        abspath(expanduser(VIIRS_GEOS5FP_output_directory)),
        f"{target_date:%Y-%m-%d}",
        f"VIIRS-GEOS5FP_{target_date:%Y-%m-%d}_{target}",
    )

    return directory
