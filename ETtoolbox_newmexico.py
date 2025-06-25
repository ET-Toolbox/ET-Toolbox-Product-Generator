import logging
import sys, shutil, datetime
from os.path import join
from time import sleep
import os
import pip_system_certs.wrapt_requests
from ETtoolbox import ET_toolbox_hindcast_coarse_tile

logger = logging.getLogger(__name__)


def _cleanup_folder(s_folder_path: str, s_delimiter: str, i_day_offset: int =7):
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
        ol_dates = [datetime.datetime.strptime(x, '%Y' + s_delimiter + "%m" + s_delimiter + '%d').date() for x in sl_files]

        # Clean the folder
        for i_entry_folder in range(0, len(ol_dates), 1):
            # Check for days before a target offset
            if ol_dates[i_entry_folder] < datetime.date.today() - datetime.timedelta(days=i_day_offset):
                # Create the file path
                s_target_path = os.path.join(s_folder_path, sl_files[i_entry_folder])

                # Remove the folder
                shutil.rmtree(s_target_path)

    except:
        # Error cleaning up the file
        print('Error cleaning up ' + s_folder_path)


if __name__=="__main__":
    # Set the directories
    s_working_directory = os.getcwd()
    s_static_directory = join(s_working_directory, "ptjpl_static")
    s_srtm_download = join(s_working_directory, "srtm_download_directory")
    s_lance_download_directory = join(s_working_directory, "lance_download_directory")
    s_geos5fp_download = join(s_working_directory, "geos5fp_download_directory")

    logger.info("starting New Mexico VIIRS data production")

    logger.info(f"working directory: {s_working_directory}")
    logger.info(f"static directory: {s_static_directory}")
    logger.info(f"LANCE directory: {s_lance_download_directory}")
    logger.info(f"SRTM directory: {s_srtm_download}")
    logger.info(f"GEOS-5 FP directory: {s_geos5fp_download}")


    ### Cleanup previous runs ###
    # Cleanup historical values
    _cleanup_folder(s_geos5fp_download, '.')
    _cleanup_folder('HLS2_download', '.', i_day_offset=16)
    _cleanup_folder(os.path.join(s_lance_download_directory, 'VNP21_NRT'), '-')
    _cleanup_folder(os.path.join(s_lance_download_directory, 'VNP43IA4N'), '-')
    _cleanup_folder(os.path.join(s_lance_download_directory, 'VNP43MA4N'), '-')
    _cleanup_folder("GFS_download_directory", '-')
    _cleanup_folder("LANCE_output", '-')
    _cleanup_folder("landsat_download", '-', i_day_offset=30)
    _cleanup_folder(s_geos5fp_download, '.')

    # Cleanup forecast values
    _cleanup_folder("GFS_output", '-', i_day_offset=0)

    ### Process the tiles ###
    # Define the New Mexico tiles
    tiles = ['12RXV', '12RYV', '12SXA', '12SXB', '12SXC', '12SXD', '12SXE', '12SXF', '12SXG', '12SYA',
             '12SYB', '12SYC', '12SYD', '12SYE', '12SYF', '12SYG', '13SBA', '13SBB', '13SBR', '13SBS',
             '13SBT', '13SBU', '13SBV', '13SCA', '13SCB', '13SCR', '13SCS', '13SCT', '13SCU', '13SCV',
             '13SDA', '13SDB', '13SDR', '13SDS', '13SDT', '13SDU', '13SDV', '13SEA', '13SEB', '13SER',
             '13SES', '13SET', '13SEU', '13SEV', '13SFA', '13SFB', '13SFR', '13SFS', '13SFT', '13SFU',
             '13SFV']

    # Process the tiles
    for tile in tiles:
        ET_toolbox_hindcast_coarse_tile(
            tile=tile,
            working_directory=s_working_directory,
            output_directory=s_working_directory,
            static_directory=s_static_directory,
            SRTM_download=s_srtm_download,
            LANCE_download=s_lance_download_directory,
            GEOS5FP_download=s_geos5fp_download,
        )




