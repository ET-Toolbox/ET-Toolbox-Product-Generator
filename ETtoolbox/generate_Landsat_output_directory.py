from typing import Union
from os.path import join
from datetime import date, datetime

def generate_Landsat_output_directory(
        Landsat_output_directory: str, 
        target_date: Union[date, str], 
        target: str) -> str:
    """
    Generate the output directory path for Landsat products.

    Args:
        Landsat_output_directory (str): Base directory for Landsat outputs.
        target_date (Union[date, str]): The date for which the output is generated. 
            Can be a datetime.date object or a string in 'YYYY-MM-DD' format.
        target (str): The target identifier (e.g., scene or product name).

    Returns:
        str: The full output directory path for the specified date and target.
    """
    # If target_date is a string, parse it to a date object
    if isinstance(target_date, str):
        try:
            target_date_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError as e:
            raise ValueError(f"target_date string must be in 'YYYY-MM-DD' format: {e}")
    else:
        target_date_obj = target_date

    # Construct the output directory path
    return join(
        Landsat_output_directory, 
        f"{target_date_obj:%Y-%m-%d}", 
        f"Landsat_{target_date_obj:%Y-%m-%d}_{target}", 
    )