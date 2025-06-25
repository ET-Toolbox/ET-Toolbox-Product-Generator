import os, sys, shutil, datetime
from os.path import join
import pip_system_certs.wrapt_requests
from datetime import datetime, timezone, date, timedelta
from ETtoolbox import ET_toolbox_hindcast_forecast_tile


def _cleanup_folder(s_folder_path: str, s_delimiter: str, i_day_offset: int = 8):
    """
    This function is intended to clean-up input and output data that is beyond the needed time frame of the ET estimates. This minimizes the storage space required for the
    toolbox.

    Parameters
    ----------
    s_folder_path: str
        Path to the folder to be cleaned
    s_delimiter: str
        Delimiter to use to parse the folder dates.
    i_day_offset: int
        Number of days behind the current eday to keep

    Returns
    -------
    None. Folders are deleted from the disk

    """

    # Try/except is intended to handle first cast after the build. Subsequent runs triggering the block should be considered errors.
    try:
        # Get the files in the folder
        sl_files = os.listdir(s_folder_path)

        # Convert to date
        ol_dates = [datetime.strptime(x, '%Y' + s_delimiter + "%m" + s_delimiter + '%d').date() for x in sl_files]

        # Clean the folder
        for i_entry_folder in range(0, len(ol_dates), 1):
            # Check for days before a target offset
            if ol_dates[i_entry_folder] < date.today() - timedelta(days=i_day_offset):
                # Create the file path
                s_target_path = os.path.join(s_folder_path, sl_files[i_entry_folder])

                # Remove the folder
                shutil.rmtree(s_target_path)

    except:
        # Error cleaning up the file
        print('Error cleaning up ' + s_folder_path)


if __name__ == '__main__':

    ### Input information ###
    # Set the tile information for the domain
    RIO_GRANDE_TILES = ["13SBB", "13SCB", "13SDB", "13SCA", "13SDA",
                        "13SCV", "13SDV", "13SCU", "13SBT", "13SCT",
                        "13SBS", "13SCS", "13SCR"]

    # Set the directories
    s_working_directory = os.getcwd()
    s_static_directory = join(s_working_directory, "ptjpl_static")
    s_srtm_download = join(s_working_directory, "srtm_download_directory")
    s_lance_download_directory = join(s_working_directory, "lance_download_directory")
    s_geos5fp_download = join(s_working_directory, "geos5fp_download_directory")

    ### Cleanup previous runs ###
    # Cleanup historical values
    _cleanup_folder(s_geos5fp_download, '.')
    _cleanup_folder('HLS2_download', '.', i_day_offset=16)
    _cleanup_folder(os.path.join(s_lance_download_directory, 'VNP21_NRT'), '-')
    _cleanup_folder(os.path.join(s_lance_download_directory, 'VNP43IA4N'), '-')
    _cleanup_folder(os.path.join(s_lance_download_directory, 'VNP43MA4N'), '-')
    _cleanup_folder("LANCE_output", '-')
    _cleanup_folder("landsat_download", '-', i_day_offset=30)
    _cleanup_folder(s_geos5fp_download, '.')

    # Cleanup forecast values
    _cleanup_folder("GFS_download_directory", '-', i_day_offset=-10)
    _cleanup_folder("GFS_output", '-', i_day_offset=-10)

    ### Get the current datetime ###
    o_datetime_utc = datetime.now(timezone.utc)

    ### Process each tile ###
    for tile in RIO_GRANDE_TILES:
        ET_toolbox_hindcast_forecast_tile(
            tile=tile,
            s_working_directory=s_working_directory,
            s_static_directory=s_static_directory,
            s_srtm_download_directory=s_srtm_download,
            s_lance_download_directory=s_lance_download_directory,
            s_geos5fp_download_directory=s_geos5fp_download,
            o_present_date=o_datetime_utc
        )
