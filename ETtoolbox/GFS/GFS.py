import shutil, time, os, posixpath, pygrib
from typing import List
from matplotlib.colors import LinearSegmentedColormap
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta, date, timezone
from dateutil import parser
import rasters as rt
import numpy as np
import logging
import colored_logging
import boto3
from botocore import UNSIGNED
from botocore.client import Config

logger = logging.getLogger(__name__)

GFS_SM_MESSAGE = 565
GFS_TA_MESSAGE = 581
GFS_RH_MESSAGE = 584
GFS_U_WIND_MESSAGE = 588
GFS_V_WIND_MESSAGE = 589
GFS_SWIN_MESSAGE = 653



SM_CMAP = LinearSegmentedColormap.from_list("SM", ["#f6e8c3", "#d8b365", "#99894a", "#2d6779", "#6bdfd2", "#1839c5"])


def create_gfs_urls(o_target_datetime) -> tuple[List, datetime]:
    """
    Finds the most recent GFS forecast within a UTC date and constructs the filenames for it. If forecasts aren't available, then it returns a None as the list values.

    Parameters
    ----------
    o_target_datetime: datetime
        Target datetime for the GFS forecast

    Returns
    -------
    sl_urls: list
        Contains the paths to the GFS forecast objects
    o_datetime_utc: datetime
        Contains the time in UTC of the forecast issuance being utilized

    """


    # Define the AWS URL
    s_url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"

    # Get the current datetime in utc format to match the issuance
    o_datetime_utc = o_target_datetime
    o_date_utc_previous = o_datetime_utc - timedelta(days=1)

    # Set a variable to control the loop
    b_continue = True

    # Default to having no filenames
    sl_urls = None

    # Attempt to request the filenames
    while b_continue and o_datetime_utc > o_date_utc_previous:

        # Calculate the hour
        i_request_hour = int(np.floor((o_datetime_utc.hour / 6)) * 6)

        # Convert the request hour to a string to create the URL
        s_request_hour = str(i_request_hour)
        if len(s_request_hour) < 2:
            s_request_hour = '0' + s_request_hour

        # Construct the path to the folder within the bucket
        s_prefix = 'gfs.' + o_datetime_utc.strftime("%Y%m%d") + "/" + s_request_hour + "/atmos/gfs.t" + s_request_hour + "z.pgrb2.0p25"

        # Create the client and query the AWS bucket to determine if the date/issuance is valid
        o_s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED), region_name='us-east-1')
        o_response = o_s3_client.list_objects(Bucket='noaa-gfs-bdp-pds', Prefix=s_prefix)

        # Process the request results based on the contents of the response
        if "Contents" in o_response:
            # The response contains contents, indicating that it's valid. Parse the filename from the contents to construct the URLS
            # Find the correct files in the bucket
            sl_files = [x['Key'] for x in o_response['Contents'] if '.idx' not in x['Key'] and '.anl' not in x['Key']]

            # Construct the urls
            sl_urls = [s_url + x for x in sl_files]

            # Break from the look
            b_continue = False

            # Return the correct forecast datetime
            o_forecast_datetime = datetime(year=o_datetime_utc.year, month=o_datetime_utc.month, day=o_datetime_utc.day, hour=i_request_hour)

        else:
            # Forecast is not available for this issuance yet. Decrement by 6 hours and try again
            o_datetime_utc = o_datetime_utc - timedelta(hours=6)
            o_forecast_datetime = None

    # Return to the calling function
    return sl_urls, o_forecast_datetime


def get_gfs_listing(o_target_datetime) -> pd.DataFrame:
    """
    Requests the GFS forecast and formats the information for subsequent use, sorting the forecast files by date

    Parameters
    ----------
    o_target_datetime: datetime
        Target datetime for the GFS forecast

    Returns
    -------
    df_address: pd.DataFrame
        Contains the URLs and the dates to the GFS forecast files

    """

    ### Request the most recentURLS and the forecast dates ###
    sl_urls, o_source_datetime = create_gfs_urls(o_target_datetime)

    ### Process the files ###
    # Set the URLs into the dataframe
    df_address = pd.DataFrame({"address": sl_urls})

    # Ensure that data is available to process
    if sl_urls is not None:
        # A valid forecast is available for the current day. Process it.

        # Extract out just the filename portion
        df_address["basename"] = df_address.address.apply(lambda address: posixpath.basename(address))

        # Set the initial date information
        df_address["source_date_UTC"] = o_source_datetime.date()
        df_address["source_hour"] = o_source_datetime.hour
        df_address["source_datetime_UTC"] = o_source_datetime.replace(tzinfo=None)

        # Set the forecast times into the dataframe
        df_address["forecast_hours"] = df_address.basename.apply(lambda basename: int(basename.split(".")[4][1:]))
        df_address["forecast_time_UTC"] = df_address.apply(lambda row: o_source_datetime.replace(tzinfo=None) + timedelta(hours=row.forecast_hours), axis=1)

        # Sort the values by time
        df_address.sort_values(by=["forecast_time_UTC", "source_hour"], inplace=True)

        # Remove any duplicates
        df_address.drop_duplicates(subset="forecast_time_UTC", keep="last", inplace=True)

        # Keep just the urls and the forecast time in UTC
        df_address = df_address[["forecast_time_UTC", "address"]]

    else:
        # A valid forecast is not available for the current day. Set the forecast dates as None into the dataframe as a flag.
        df_address['forecast_time_UTC'] = None

    ### Return to the calling function ###
    return df_address


def gfs_download(s_url: str, o_datetime: datetime , s_filename: str = None, s_directory: str = None) -> str:
    """
    Downloads a GEFS forecast file given by the input URL

    Parameters
    ----------
    s_url: str
        URL to the target download file
    o_datetime: str
        Forecast issuance date of the file being downloaded
    s_filename: str
        Local filename for the file
    s_directory: str
        Local directory into which the file should be saved

    Returns
    -------
    s_filename: str
        Local path to the downloaded file

    """

    ### Set the default directory ###
    if s_directory is None:
        s_directory = "."

    ### Parse the date from the target URL ###
    o_date_utc = o_datetime.date()

    ### Create the download directory ###
    # Get the full path from the input path
    s_directory = os.path.expanduser(s_directory)

    # Construct the path by concatenating with the date
    s_target_directory = os.path.join(s_directory, o_date_utc.strftime("%Y-%m-%d"))

    # Make the nested directory
    os.makedirs(s_target_directory, exist_ok=True)

    ### Construct the filename ###
    # Construct the filename
    if s_filename is None:
        s_filename = os.path.join(s_target_directory, posixpath.basename(s_url))

    # Handle an existing file
    if os.path.exists(s_filename) and os.path.getsize(s_filename) > 0:
        # File exists and has nonzero size. It is likely good, so keep it assuming the download is complete
        logger.info(f"file already downloaded: {colored_logging.file(s_filename)} Assuming valid and skipping redownload.")
        return s_filename

    elif os.path.exists(s_filename) and os.path.getsize(s_filename) == 0:
        # File exists and is of zero size. Assume it's bad and redownload
        # Remove the file
        logger.info(f"file already downloaded but likely corrupted: {colored_logging.file(s_filename)} Likely not valid and redownloading.")
        os.remove(s_filename)

    # Log the filename
    logger.info(f"downloading URL: {colored_logging.URL(s_url)}")

    ### Attempt the download ###
    # Create a temporary filename
    s_partial_filename = s_filename + ".download"

    # Create the control variables for the download loop
    i_attempts = 3
    i_attempt_counter = 0

    # Loop on the download
    while i_attempt_counter < i_attempts:
        # Attempt the download. If the download fails, pause in case it's a server issue which can be resolved by waiting a bit
        try:
            # Construct the download command
            command = "curl -o " + s_partial_filename + " " + s_url

            # Log the command
            logger.info(command)

            # Issue the command to the system for download
            os.system(command)

            # Download successful. Break from the loop as further calls are not required.
            break

        except:
            # Log the error
            logger.info(f"error downloading file: {colored_logging.file(s_filename)} Attempting reentry.")

            # Increment the counter
            i_attempt_counter += 1

            # Pause a bit to let the server clear. Hopefully the next time works...
            time.sleep(10)

            # If a partial download occurred, remove it to allow for another download attempt
            if os.path.exists(s_partial_filename):
                os.remove(s_partial_filename)

    # Handle download results
    if os.path.exists(s_partial_filename):
        # File exists continue
        # Move the file from the temporary filename to the final filename
        shutil.move(s_partial_filename, s_filename)

        # Log the file status
        logger.info(f"downloaded file: {colored_logging.file(s_filename)}")

    else:
        # Raise an issue if the download failed
        logger.error(f"unable to download URL: {s_url}")

        # Set the filename to None to indicate failure
        s_filename = None

    ### Return to the calling function ###
    return s_filename


def read_gfs(filename: str, message: int, geometry: rt.RasterGeometry = None, resampling ="cubic") -> rt.Raster:
    with pygrib.open(filename) as file:
        data = file.message(message).values

    rows, cols = data.shape
    data = np.roll(data, int(cols / 2), axis=1)
    grid = rt.RasterGrid.from_bbox(rt.BBox(-180, -90, 180, 90), data.shape)
    image = rt.Raster(data, geometry=grid)

    if geometry is not None:
        image = image.to_geometry(geometry, resampling=resampling)
    
    return image

def gfs_interpolate(i_message: int, o_datetime_utc: datetime, o_geometry: rt.RasterGeometry = None, s_resampling: str = "cubic", s_directory: str = None,
                    df_forecast: pd.DataFrame = None) -> rt.Raster:
    """
    Determines which files are between the target time, downloads the associated files, and interpolates between the timesteps

    Parameters
    ----------
    i_message: int
        Variable/band to pull from the GFS file
    o_datetime_utc: datetime
        Time as which to download and parse the data
    o_geometry: rt.RasterGeometry
        Domain extents
    s_resampling: str
        Resampling approach
    s_directory: str
        Working directory into which to download the files
    df_forecast: pd.DateFrame
        List of forecast files with associated times from forecast issuance

    Returns
    -------
    o_interpolated_image: rt.Raster
        Raster product interpolated between the bounding timestamps

    """

    ### Find and download the forecast file before the current time ###
    # Find the most recent timestamp before the current time
    i_before_index = np.max(np.argwhere(df_forecast['forecast_time_UTC'] <= o_datetime_utc).flatten())

    # Log what file will be used
    logger.info(f"before URL: {colored_logging.URL(df_forecast['address'].iloc[i_before_index])}")
    s_before_time = parser.parse(str(df_forecast['forecast_time_UTC'].iloc[i_before_index]))

    # Attempt to download the file. Error handling is done within the download function
    s_before_filename = gfs_download(s_url=df_forecast['address'].iloc[i_before_index], o_datetime=df_forecast['forecast_time_UTC'].iloc[0], s_directory=s_directory)
    
    # Read in the file
    if s_before_filename is not None:
        # Download is successful. Read the file.
        try:
            # Attempt to read the file from disk.
            o_before_image = read_gfs(filename=s_before_filename, message=i_message, geometry=o_geometry, resampling=s_resampling)

        except:
            # Read has failed. Flag the failure
            o_before_image = None

    else:
        # File has not downloaded correctly and read will not be successful. Flag the failure
        o_before_image = None

    ### Find and download the forecast file after the current time ###
    # Find the first timestamp after the current time
    i_after_index = np.min(np.argwhere(df_forecast['forecast_time_UTC'] > o_datetime_utc).flatten())

    # Log what file will be used
    logger.info(f"after URL: {colored_logging.URL(df_forecast['address'].iloc[i_after_index])}")
    s_after_time = parser.parse(str(df_forecast['forecast_time_UTC'].iloc[i_after_index]))

    # Attempt to download the file. Error handling is done within the download function
    s_after_filename = gfs_download(s_url=df_forecast['address'].iloc[i_after_index], o_datetime=df_forecast['forecast_time_UTC'].iloc[0], s_directory=s_directory)

    # Read in the file
    if s_before_filename is not None:
        # Download is successful. Read the file.
        try:
            # Attempt to read the file from disk.
            o_after_image = read_gfs(filename=s_after_filename, message=i_message, geometry=o_geometry, resampling=s_resampling)

        except:
            # Read has failed. Flag the failure
            o_after_image = None

    else:
        # File has not downloaded correctly and read will not be successful. Flag the failure
        o_after_image = None

    ### Do the math on the forecasts ###
    if o_before_image is not None and o_after_image is not None:
        # Data is available and valid.
        # Difference the files based on teh timestamp
        o_source_diff = o_after_image - o_before_image

        # Calculate the fractional time of the current time between the files
        s_time_fraction = (parser.parse(str(o_datetime_utc)) - parser.parse(str(s_before_time))) / (parser.parse(str(s_after_time)) - parser.parse(str(s_before_time)))

        # Linearly interpolate between the timestamps using the current time
        o_interpolated_image = o_before_image + o_source_diff * s_time_fraction

    else:
        # Data is not available. Set none into the image to flag an issue.
        o_interpolated_image = None

    ### Return to the calling function ###
    return o_interpolated_image


def forecast_Ta_K(time_UTC: datetime, geometry: rt.RasterGeometry = None, resampling: str = "cubic", directory: str = None, listing: pd.DataFrame = None) -> rt.Raster:
    return gfs_interpolate(i_message=GFS_TA_MESSAGE, o_datetime_utc=time_UTC, o_geometry=geometry, s_resampling=resampling, s_directory=directory, df_forecast=listing)


def forecast_Ta_C(time_UTC: datetime, geometry: rt.RasterGeometry = None, resampling: str = "cubic", directory: str = None, listing: pd.DataFrame = None) -> rt.Raster:
    return forecast_Ta_K(time_UTC=time_UTC, geometry=geometry, resampling=resampling, directory=directory, listing=listing) - 273.15


def forecast_RH(time_UTC: datetime, geometry: rt.RasterGeometry = None, resampling: str = "cubic", directory: str = None, listing: pd.DataFrame = None) -> rt.Raster:
    return rt.clip(gfs_interpolate(i_message=GFS_RH_MESSAGE, o_datetime_utc=time_UTC, o_geometry=geometry, s_resampling=resampling, s_directory=directory, df_forecast=listing) / 100, 0, 1)


def forecast_SM(time_UTC: datetime, geometry: rt.RasterGeometry = None, resampling: str = "cubic", directory: str = None, listing: pd.DataFrame = None) -> rt.Raster:
    SM = rt.clip(gfs_interpolate(i_message=GFS_SM_MESSAGE, o_datetime_utc=time_UTC, o_geometry=geometry, s_resampling=resampling, s_directory=directory, df_forecast=listing) / 10000, 0, 1)
    SM.cmap = SM_CMAP

    return SM

def forecast_SWin(time_UTC: datetime, geometry: rt.RasterGeometry = None, resampling: str = "cubic", directory: str = None, listing: pd.DataFrame = None) -> rt.Raster:
    return rt.clip(gfs_interpolate(i_message=GFS_SWIN_MESSAGE, o_datetime_utc=time_UTC, o_geometry=geometry, s_resampling=resampling, s_directory=directory, df_forecast=listing), 0, None)

def forecast_wind(time_UTC: datetime, geometry: rt.RasterGeometry = None, resampling: str = "cubic", directory: str = None, listing: pd.DataFrame = None) -> rt.Raster:
    U = gfs_interpolate(i_message=GFS_U_WIND_MESSAGE, o_datetime_utc=time_UTC, o_geometry=geometry, s_resampling=resampling, s_directory=directory, df_forecast=listing)
    V = gfs_interpolate(i_message=GFS_V_WIND_MESSAGE, o_datetime_utc=time_UTC, o_geometry=geometry, s_resampling=resampling, s_directory=directory, df_forecast=listing)
    wind_speed = rt.clip(np.sqrt(U ** 2.0 + V ** 2.0), 0.0, None)

    return wind_speed
