import os, time
import posixpath
import warnings
from os import makedirs, remove
from os.path import join, getsize, basename
from shutil import move
from typing import List
from datetime import timezone
import xarray as xr

import h5py
import pandas as pd
import requests
from bs4 import BeautifulSoup
from matplotlib.colors import LinearSegmentedColormap
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3 import Retry

import colored_logging
import rasters as rt
from modland import find_modland_tiles, generate_modland_grid, parsehv
from solar_apparent_time import UTC_to_solar
from ETtoolbox.VIIRS_orbit import *
from ..credentials import get_earthdata_credentials

DEFAULT_REMOTE = "https://nrt4.modaps.eosdis.nasa.gov/api/v2/content/archives/allData"
DEFAULT_READ_TIMEOUT = 60
DEFAULT_RETRIES = 3
ARCHIVE = "5200"

NDVI_CMAP = LinearSegmentedColormap.from_list(
    name="NDVI",
    colors=[(0, "#0000ff"), (0.4, "#000000"), (0.5, "#745d1a"), (0.6, "#e1dea2"), (0.8, "#45ff01"), (1, "#325e32")])

logger = logging.getLogger(__name__)


class LANCENotAvailableError(Exception):
    pass


def HTTP_listing(URL: str,timeout: float = None, retries: int = None, username: str = None, password: str = None, **kwargs):
    """
    Get the directory listing from an FTP-like HTTP data dissemination system.
    There is no standard for listing directories over HTTP, and this was designed
    for use with the USGS data dissemination system.
    HTTP connections are typically made for brief, single-use periods of time.
    :param url: URL of URL HTTP directory
    :param timeout:
    :param retries:
    :param username: username string (optional)
    :param password: password string (optional)
    :param kwargs:
    :return:
    """
    if timeout is None:
        timeout = DEFAULT_READ_TIMEOUT

    if retries is None:
        retries = DEFAULT_RETRIES

    retries = Retry(total=retries, backoff_factor=3, status_forcelist=[500, 502, 503, 504])

    if not username is None and not password is None:
        auth = HTTPBasicAuth(username, password)
    else:
        auth = None
    with warnings.catch_warnings(), requests.Session() as s:
        warnings.filterwarnings("ignore")
        # too many retries in too short a time may cause the server to refuse connections
        s.mount('http://', HTTPAdapter(max_retries=retries))
        response = s.get(URL, auth=auth, timeout=timeout)

    if response.status_code != 200:
        raise LANCENotAvailableError(f"LANCE server not available with status {response.status_code} at URL: {URL}")

    # there was a conflict between Unicode markup and from_encoding
    soup = BeautifulSoup(response.text, 'html.parser')
    links = list(soup.find_all('a', href=True))

    # get directory names from links on http site
    directories = [link['href'] for link in links]

    if len(directories) == 0:
        logger.error(response.text)
        raise LANCENotAvailableError(f"no links found at LANCE URL: {URL}")

    return directories


def available_LANCE_dates(product: str, archive: str, remote=DEFAULT_REMOTE) -> List[date]:
    year = datetime.utcnow().year
    URL = posixpath.join(remote, archive, product, f"{year:04d}")
    listing = HTTP_listing(URL)
    listing = [x.split('/') for x in listing]
    dates = [x[-3] + x[-2] for x in listing]
    dates = sorted([datetime.strptime(x, "%Y%j").date() for x in dates])

    # dates = sorted([datetime.strptime(f"{year:04d}{posixpath.basename(item)}", "%Y%j").date() for item in listing])

    if len(dates) == 0:
        raise LANCENotAvailableError(f"no dates for LANCE products found at URL: {URL}")

    return dates


def get_LANCE_download_directory(directory: str, product: str, date_UTC: Union[date, str]) -> str:
    return join(expanduser(directory), product, f"{date_UTC:%Y-%m-%d}")


def download_lance_viirs(s_url: str, s_directory: str, i_number_of_retries: int = 3, i_wait_seconds: int = 30) -> Union[str, None]:
    """
    Download the LANCE data from the server to disk

    Parameters
    ----------
    s_url: str
        URL to download from the server
    s_directory: str
        Local folder into which to download the files into
    i_number_of_retries: int
        Number of times to attempt redownload
    i_wait_seconds: int
        Number of seconds to wait between attempts

    Returns
    -------
    s_filename: str or None
        Path to the successfully downloaded file, otherwise None if download is not successful

    """

    ### Setup the local disk ###
    # Construct the base URL from the provided
    s_product = posixpath.basename(s_url).split(".")[0]

    # Get the date in UTC format
    o_date_utc = datetime.strptime(posixpath.basename(s_url).split(".")[1][1:], "%Y%j").date()

    # Construct the full download path within the target local folder
    s_destination_directory = get_LANCE_download_directory(directory=s_directory, product=s_product, date_UTC=o_date_utc)

    # Construct the directory structure
    makedirs(s_destination_directory, exist_ok=True)

    # Construct the filename
    s_filename = join(s_destination_directory, posixpath.basename(s_url))

    ### Attempt the download ###
    # Continue to to download while there are reentries available
    while i_number_of_retries > 0:
        # Decrement the reentry counter
        i_number_of_retries -= 1

        # Attempt the download
        try:
            # Check if a previous download attempt exists
            if exists(s_filename) and getsize(s_filename) == 0:
                # A previous download attempt exists. Cleanup up after the failed attempt
                logger.warning(f"removing zero-size file: {s_filename}")
                remove(s_filename)

            # Check if the file exists. If it does and it passed the nonzero check above, then assume it's a successful download
            if exists(s_filename):
                # File exists.
                logger.info(f"file already downloaded: {colored_logging.file(s_filename)}")

                # Check that the file can be opened
                try:
                    # Attempt to open the file
                    with xr.open_dataset(s_filename) as o_file:
                        if not np.all(np.isnan(o_file)):
                            # Passed the opening check. Return the filename.
                            return s_filename

                        else:
                            # Raise an error to delete the file
                            raise LANCENotAvailableError('File contains invalid contents')

                except Exception as e:
                    # Opening the file failed. Remove it and attempt to redownload
                    logger.warning(f"removing corrupted LANCE file: {s_filename}")
                    os.remove(s_filename)

            # Log the download attempt
            logger.info(f"downloading URL: {colored_logging.URL(s_url)}")

            # Construct the temporary local filename to use
            s_partial_filename = s_filename + ".download"

            # Call for credentials to ensure the enviromental variables are set
            c_credentials = get_earthdata_credentials()

            # Construct the command and call it as a system subprocess
            s_command = 'wget -e robots=off -c -nc -np -nH --no-directories ' + s_url + ' --header "Authorization: Bearer ' + os.environ['EARTHDATA_TOKEN'] + '" -O ' \
                        + s_partial_filename
            logger.info(s_command)
            os.system(s_command)

            # Take action based on the outcome of the system command
            if not exists(s_partial_filename):
                # File does not exist on the disk. Log that there was an error with teh download
                raise ConnectionError(f"unable to download URL: {s_url}")

            elif exists(s_partial_filename) and getsize(s_partial_filename) == 0:
                # Download exists but is of zero size. Assume the file is bad.
                # Log the error
                logger.warning(f"removing zero-size corrupted LANCE file: {s_partial_filename}")

                # Remove the bad file
                os.remove(s_partial_filename)

                # Raise a connection error
                raise ConnectionError(f"unable to download URL: {s_url}")

            # Checks successful. Accept the file by moving it from the temporary filename to the final filename
            move(s_partial_filename, s_filename)

            # Check that the move was successful. Otherwise error out
            if not exists(s_filename):
                raise ConnectionError(f"unable to download URL: {s_url}")

            # Log the successful download
            logger.info(f"successfully downloaded file: {colored_logging.file(s_filename)} ({getsize(s_filename)})")

            # Return the path to the file to signal a clean download
            return s_filename

        except Exception as e:
            # An exception was thrown. Determine next steps by the number of reentries available.
            if i_number_of_retries == 0:
                # Number of reentries has been exceeded. Return a None type to flag an improper download
                return None

            else:
                # Rentries still exist. Log warning and attempt to reentry
                logger.warning(e)
                logger.warning(f"waiting {i_wait_seconds} for M2M retry")
                time.sleep(i_wait_seconds)
                continue


def generate_VNP21NRT_URL(datetime_UTC: datetime, remote: str = DEFAULT_REMOTE, archive: str = "5200", collection: str = "002"):
    year = datetime_UTC.year
    doy = datetime_UTC.timetuple().tm_yday
    granule = f"{datetime_UTC:%H%M}"
    URL = posixpath.join(remote, archive, "VNP21_NRT", f"{year:04d}", f"{doy:03d}", f"VNP21_NRT.A{year:04d}{doy:03d}.{granule}.{collection}.nc")

    return URL


def read_VNP21NRT_latitude(filename: str) -> np.ndarray:
    dataset_name = "VIIRS_Swath_LSTE/Geolocation Fields/latitude"

    try:
        with h5py.File(filename, "r") as file:
            return np.array(file[dataset_name])
    except Exception as e:
        logger.exception(e)
        raise IOError(f"unable to load dataset {dataset_name} from file {filename}")


def read_VNP21NRT_longitude(filename: str) -> np.ndarray:
    dataset_name = "VIIRS_Swath_LSTE/Geolocation Fields/longitude"

    try:
        with h5py.File(filename, "r") as file:
            return np.array(file[dataset_name])
    except Exception as e:
        logger.exception(e)
        raise IOError(f"unable to load dataset {dataset_name} from file {filename}")


def read_VNP21NRT_geometry(filename: str) -> rt.RasterGeometry:
    return rt.RasterGeolocation(x=read_VNP21NRT_longitude(filename=filename), y=read_VNP21NRT_latitude(filename=filename))


def read_VNP21NRT_DN(filename: str, variable: str) -> rt.Raster:
    dataset_name = f"VIIRS_Swath_LSTE/Data Fields/{variable}"

    # FIXME remove corrupted files here

    try:
        with h5py.File(filename, "r") as file:
            data = np.array(file[dataset_name])
    except Exception as e:
        logger.exception(e)
        raise IOError(f"unable to load dataset {dataset_name} from file {filename}")

    geometry = read_VNP21NRT_geometry(filename=filename)
    image = rt.Raster(data, geometry=geometry)

    return image


def read_VNP21NRT_attribute(filename: str, variable: str, attribute: str) -> Union[str, float, int]:
    dataset_name = f"VIIRS_Swath_LSTE/Data Fields/{variable}"

    try:
        with h5py.File(filename, "r") as file:
            dataset = file[dataset_name]

            if attribute not in dataset.attrs.keys():
                raise ValueError(f"attribute {attribute} not found in layer {dataset} in file {filename}")

            attribute = dataset.attrs[attribute]

            if len(attribute) == 1:
                attribute = attribute[0]

            return attribute
    except Exception as e:
        logger.exception(e)
        raise IOError(f"unable to load dataset {dataset_name} from file {filename}")


def read_VNP21NRT_QC(filename: str) -> rt.Raster:
    return read_VNP21NRT_DN(filename, "QC")


def read_VNP21NRT_cloud(filename: str) -> rt.Raster:
    return read_VNP21NRT_QC(filename) >> 1 & 1 == 1


def read_VNP21NRT_layer(filename: str, variable: str, geometry: rt.RasterGeometry = None, resampling: str = None) -> rt.Raster:

    data = read_VNP21NRT_DN(filename, variable)
    fill_value = read_VNP21NRT_attribute(filename, variable, "_FillValue")
    scale_factor = read_VNP21NRT_attribute(filename, variable, "scale_factor")
    offset = read_VNP21NRT_attribute(filename, variable, "add_offset")
    data = rt.where(data == fill_value, np.nan, data * scale_factor + offset)
    cloud = read_VNP21NRT_cloud(filename)
    data = rt.where(cloud, np.nan, data)

    if geometry is not None:
        data = data.to_geometry(geometry.grid.rescale(1000), resampling="average").to_geometry(geometry, resampling=resampling)

    return data


def retrieve_vnp21nrt(o_geometry: rt.RasterGeometry, o_date_solar: date = None, s_variable: str = None, s_resampling: str = None, s_directory: str = None,
                      s_spacetrack_credentials_filename: str = None) -> Union[rt.Raster, None]:
    """
    Retrieves the vnp21nrt tiles values from EOSDIS

    Parameters
    ----------
    o_geometry: rt.RasterGeometry
        Gives the spatial extents of the requested data
    o_date_solar: date
        Solar date of the requested tiles
    s_variable: str
        Variable name to extract from the tiles
    s_resampling: str
        Resampling scheme, if applicable
    s_directory: str
        Local directory to save the data into
    s_spacetrack_credentials_filename: str
        Credentials for spacetrack authentication

    Returns
    -------
    o_composite_image: Union[rt.RasterGeometry, None]
        Returns raster geometry if the download was either fully or partially successful. Otherwise returns None to indicate a failed download
    o_date_solar: datetime
        Datetime associated with the data request

    """

    # Check that a variable is provided. If not, assume LST
    if s_variable is None:
        s_variable = "LST"

    # Convert a string date to a datetime object
    if isinstance(o_date_solar, str):
        o_date_solar = parser.parse(o_date_solar).date()

    # If no time is provided, assume the time is now.
    if o_date_solar is None:
        o_date_solar = UTC_to_solar(datetime.now(timezone.utc), rt.wrap_geometry(o_geometry).centroid_latlon).date()

    # Set the starting date of the request
    o_datetime_solar = datetime(o_date_solar.year, o_date_solar.month, o_date_solar.day, 13, 30)

    # Step backward, attempting to request the days
    i_number_of_reentries = 3
    while i_number_of_reentries > 0:
        try:
            # Convert the datetime solar to UTC
            o_datetime_UTC = solar_to_UTC(o_datetime_solar, o_geometry.centroid.latlon.x)

            # Determine what swaths are available
            o_swaths = find_VIIRS_swaths(o_date_solar, o_geometry.corner_polygon_latlon.geometry, filter_geometry=True,
                                         spacetrack_credentials_filename=s_spacetrack_credentials_filename)

            # Define an empty raster to fill with requested values
            o_composite_image = rt.Raster(np.full(o_geometry.shape, np.nan), geometry=o_geometry)

            # Loop and request data for each swath
            for i_entry, (o_swath_datetime_utc, o_swath_datetime_solar, s_swath_name, s_swath_geometry) in o_swaths.iterrows():
                # Create the URL
                s_url = generate_VNP21NRT_URL(o_swath_datetime_utc)

                # Download the file and receive the local filename
                s_filename = download_lance_viirs(s_url=s_url, s_directory=s_directory)

                # Check that there was content in the package.
                if s_filename is not None:
                    # Content exists. Add it into the raster
                    o_image = read_VNP21NRT_layer(filename=s_filename, variable=s_variable, geometry=o_geometry)
                    o_composite_image = o_composite_image.fill(o_image)

            # Check for the composite image to have content
            if np.all(np.isnan(o_composite_image)):
                # No data exists. Return a None type to flag an improper parse
                o_composite_image = None

            # Break from the loop
            break

        except:
            # There as been an unknown issue. Decrement the counter and try again.
            i_number_of_reentries -= 1

    # Handle missing data
    if np.all(np.isnan(o_composite_image)):
        o_composite_image = None

    # Return to the calling function
    return o_composite_image, o_date_solar


def retrieve_vnp21nrt_st(o_geometry: rt.RasterGeometry, o_date_solar: date = None, s_resampling: str = None, s_directory: str = None,
                         s_spacetrack_credentials_filename: str = None) -> Union[rt.Raster, None]:
    """
    Attempts to retrieve the LST attribute from the LANCE data

    Parameters
    ----------
    o_geometry: rt.RasterGeometry
        Gives the spatial extents of the requested data
    o_date_solar: date
        Solar date of the requested tiles
    s_resampling: str
        Resampling scheme, if applicable
    s_directory: str
        Local directory to save the data into
    s_spacetrack_credentials_filename: str
        Credentials for spacetrack authentication

    Returns
    -------
    o_composite_image: Union[rt.RasterGeometry, None]
        Returns raster geometry if the download was either fully or partially successful. Otherwise returns None to indicate a failed download
    o_date_solar: datetime
        Datetime associated with the data request

    """

    # Call the download function
    o_composite_image, o_date_solar = retrieve_vnp21nrt(o_geometry=o_geometry, o_date_solar=o_date_solar, s_variable="LST", s_directory=s_directory,
                                          s_spacetrack_credentials_filename=s_spacetrack_credentials_filename)

    # Returns the composite image to the calling function
    return o_composite_image, o_date_solar


def retrieve_vnp21nrt_emissivity(o_geometry: rt.RasterGeometry, o_date_solar: date = None, s_resampling: str = None, s_directory: str = None,
                                 s_spacetrack_credentials_filename: str = None) -> Union[rt.Raster, None]:

    """
    Attempts to retrieve the LST attribute from the LANCE data

    Parameters
    ----------
    o_geometry: rt.RasterGeometry
        Gives the spatial extents of the requested data
    o_date_solar: date
        Solar date of the requested tiles
    s_resampling: str
        Resampling scheme, if applicable
    s_directory: str
        Local directory to save the data into
    s_spacetrack_credentials_filename: str
        Credentials for spacetrack authentication

    Returns
    -------
    o_composite_image: Union[rt.RasterGeometry, None]
        Returns raster geometry if the download was either fully or partially successful. Otherwise returns None to indicate a failed download
    o_date_solar: datetime
        Datetime associated with the data request

    """

    # Call the download function
    o_composite_image, o_valid_date = retrieve_vnp21nrt(o_geometry=o_geometry, o_date_solar=o_date_solar, s_variable="Emis_ASTER", s_directory=s_directory,
                                                        s_spacetrack_credentials_filename=s_spacetrack_credentials_filename)

    # Return to the calling function
    return o_composite_image, o_valid_date


def generate_VNP43IA4N_date_URL(date_UTC: Union[date, str], remote: str = DEFAULT_REMOTE, archive: str = ARCHIVE) -> str:
    if isinstance(date_UTC, str):
        date_UTC = parser.parse(date_UTC).date()

    year = date_UTC.year
    doy = date_UTC.timetuple().tm_yday
    URL = posixpath.join(remote, archive, "VNP43IA4N", f"{year:04d}", f"{doy:03d}")

    return URL


def list_VNP43IA4N_URLs( date_UTC: Union[date, str], tiles: List[str] = None, remote: str = DEFAULT_REMOTE) -> pd.DataFrame:
    date_URL = generate_VNP43IA4N_date_URL(date_UTC=date_UTC, remote=remote)
    URLs = HTTP_listing(date_URL)
    df = pd.DataFrame({"URL": URLs})
    df.insert(0, "tile", df["URL"].apply(lambda URL: posixpath.basename(URL).split(".")[2]))

    if tiles is not None:
        df = df[df["tile"].apply(lambda tile: tile in tiles)]

    return df


def read_VNP43IA4N_DN(filename: str, variable: str) -> rt.Raster:
    dataset_name = f"HDFEOS/GRIDS/VIIRS_Grid_BRDF/Data Fields/{variable}"

    tile = basename(filename).split(".")[2]

    try:
        with h5py.File(filename, "r") as file:
            data = np.array(file[dataset_name])
    except Exception as e:
        logger.exception(e)
        raise IOError(f"unable to load dataset {dataset_name} from file {filename}")

    geometry = generate_modland_grid(*parsehv(tile), data.shape[0])
    image = rt.Raster(data, geometry=geometry)

    return image


def read_VNP43IA4N_attribute(filename: str, variable: str, attribute: str) -> Union[str, float, int]:
    dataset_name = f"HDFEOS/GRIDS/VIIRS_Grid_BRDF/Data Fields/{variable}"

    try:
        with h5py.File(filename, "r") as file:
            dataset = file[dataset_name]

            if attribute not in dataset.attrs.keys():
                raise ValueError(f"attribute {attribute} not found in layer {dataset} in file {filename}")

            attribute = dataset.attrs[attribute]

            if len(attribute) == 1:
                attribute = attribute[0]

            return attribute
    except Exception as e:
        logger.exception(e)
        raise IOError(f"unable to load dataset {dataset_name} from file {filename}")


def read_VNP43IA4N_QC(filename: str, band: int) -> rt.Raster:
    return read_VNP43IA4N_DN(filename, f"BRDF_Albedo_Band_Mandatory_Quality_I{band}")


def read_VNP43IA4N_reflectance(filename: str, band: int, geometry: rt.RasterGeometry = None, resampling: str = None) -> rt.Raster:
    variable = f"Nadir_Reflectance_I{band}"
    data = read_VNP43IA4N_DN(filename, variable)
    fill_value = read_VNP43IA4N_attribute(filename, variable, "_FillValue")
    scale_factor = read_VNP43IA4N_attribute(filename, variable, "scale_factor")
    offset = read_VNP43IA4N_attribute(filename, variable, "add_offset")
    data = rt.where(data == fill_value, np.nan, data * scale_factor + offset)
    QC = read_VNP43IA4N_QC(filename, band)
    data = rt.where(QC >> 1 == 0, data, np.nan)

    if geometry is not None:
        data = data.to_geometry(geometry, resampling=resampling)

    return data


def read_VNP43IA4N_NDVI(filename: str) -> rt.Raster:
    red = read_VNP43IA4N_reflectance(filename, 1)
    NIR = read_VNP43IA4N_reflectance(filename, 2)
    NDVI = (NIR - red) / (NIR + red)

    return NDVI


def read_VNP43IA4N_variable(filename: str, variable: str, geometry: rt.RasterGeometry = None, resampling: str = None) -> rt.Raster:
    if variable == "NDVI":
        image = read_VNP43IA4N_NDVI(filename)
    else:
        raise ValueError(f"unrecognized VNP43IA4N variable: {variable}")

    if geometry is not None:
        image = image.to_geometry(geometry, resampling=resampling)

    image.cmap = NDVI_CMAP

    return image


def retrieve_vnp43ia4n(o_geometry: rt.RasterGeometry, o_date_utc: date = None, s_variable: str = None, s_resampling: str = "cubic",
                       s_directory: str = None) -> Union[rt.Raster, None]:
    """
    Retrieves the vnp43ia4n tiles values from EOSDIS

    Parameters
    ----------
    o_geometry: rt.RasterGeometry
        Gives the spatial extents of the requested data
    o_date_utc: date
        Solar date of the requested tiles
    s_variable: str
        Variable name to extract from the tiles
    s_resampling: str
        Resampling scheme, if applicable
    s_directory: str
        Local directory to save the data into

    Returns
    -------
    o_composite_image: Union[rt.RasterGeometry, None]
        Returns raster geometry if the download was either fully or partially successful. Otherwise returns None to indicate a failed download

    """

    # Check that a variable is provided. If not, assume NDVI
    if s_variable is None:
        s_variable = "NDVI"

    # Provide a working directory
    if s_directory is None:
        s_directory = "."

    # Convert the datettime into UTC format
    if isinstance(o_date_utc, str):
        o_date_utc = parser.parse(o_date_utc).date()

    # Request the list of tiles
    o_tiles = find_modland_tiles(o_geometry.corner_polygon_latlon.geometry)

    # Construct an empty raster to receive the tile information
    o_composite_image = rt.Raster(np.full(o_geometry.shape, np.nan), geometry=o_geometry)

    # Get the urls from the tiles
    df_listing = list_VNP43IA4N_URLs(date_UTC=o_date_utc, tiles=o_tiles)

    # Loop and request each of the tiles
    for i_entry, (s_tile, s_url) in df_listing.iterrows():
        logger.info(f"processing VNP43IA4 tile: {s_tile} URL: {s_url}")

        # Download the file and receive the local filename
        s_filename = download_lance_viirs(s_url=s_url, s_directory=s_directory)

        # Check that there was content in the package.
        if s_filename is not None:
            o_image = read_VNP43IA4N_variable(filename=s_filename, variable=s_variable, geometry=o_geometry, resampling=s_resampling)
            o_composite_image = o_composite_image.fill(o_image)

    # Check for the composite image to have content
    if np.all(np.isnan(o_composite_image)):
        # No data exists. Return a None type to flag an improper parse
        o_composite_image = None

    # Return to the calling function
    return o_composite_image


def generate_VNP43MA4N_date_URL(date_UTC: Union[date, str], remote: str = DEFAULT_REMOTE, archive: str = ARCHIVE) -> str:
    if isinstance(date_UTC, str):
        date_UTC = parser.parse(date_UTC).date()

    year = date_UTC.year
    doy = date_UTC.timetuple().tm_yday
    URL = posixpath.join(remote, archive, "VNP43MA4N", f"{year:04d}", f"{doy:03d}")

    return URL


def list_VNP43MA4N_URLs(date_UTC: Union[date, str], tiles: List[str] = None, remote: str = DEFAULT_REMOTE, archive: str = ARCHIVE) -> pd.DataFrame:
    date_URL = generate_VNP43MA4N_date_URL(date_UTC=date_UTC, remote=remote, archive=archive)
    URLs = HTTP_listing(date_URL)
    df = pd.DataFrame({"URL": URLs})
    df.insert(0, "tile", df["URL"].apply(lambda URL: posixpath.basename(URL).split(".")[2]))

    if tiles is not None:
        df = df[df["tile"].apply(lambda tile: tile in tiles)]

    return df


def read_VNP43MA4N_DN(filename: str, variable: str) -> rt.Raster:
    dataset_name = f"HDFEOS/GRIDS/VIIRS_Grid_BRDF/Data Fields/{variable}"

    tile = basename(filename).split(".")[2]

    try:
        with h5py.File(filename, "r") as file:
            data = np.array(file[dataset_name])
    except Exception as e:
        logger.exception(e)
        raise IOError(f"unable to load dataset {dataset_name} from file {filename}")

    geometry = generate_modland_grid(*parsehv(tile), data.shape[0])
    image = rt.Raster(data, geometry=geometry)

    return image


def read_VNP43MA4N_attribute(filename: str, variable: str, attribute: str) -> Union[str, float, int]:
    dataset_name = f"HDFEOS/GRIDS/VIIRS_Grid_BRDF/Data Fields/{variable}"

    try:
        with h5py.File(filename, "r") as file:
            dataset = file[dataset_name]

            if attribute not in dataset.attrs.keys():
                raise ValueError(f"attribute {attribute} not found in layer {dataset} in file {filename}")

            attribute = dataset.attrs[attribute]

            if len(attribute) == 1:
                attribute = attribute[0]

            return attribute
    except Exception as e:
        logger.exception(e)
        raise IOError(f"unable to load dataset {dataset_name} from file {filename}")


def read_VNP43MA4N_QC(filename: str, band: int) -> rt.Raster:
    return read_VNP43MA4N_DN(filename, f"BRDF_Albedo_Band_Mandatory_Quality_M{band}")


def read_VNP43MA4N_reflectance(filename: str, band: int, geometry: rt.RasterGeometry = None, resampling: str = None) -> rt.Raster:
    variable = f"Nadir_Reflectance_M{band}"
    data = read_VNP43MA4N_DN(filename, variable)
    fill_value = read_VNP43MA4N_attribute(filename, variable, "_FillValue")
    scale_factor = read_VNP43MA4N_attribute(filename, variable, "scale_factor")
    offset = read_VNP43MA4N_attribute(filename, variable, "add_offset")
    data = rt.where(data == fill_value, np.nan, data * scale_factor + offset)
    QC = read_VNP43MA4N_QC(filename, band)
    data = rt.where(QC >> 1 == 0, data, np.nan)

    if geometry is not None:
        data = data.to_geometry(geometry, resampling=resampling)

    return data


def read_vnp43ma4n_ndvi(s_filename: str) -> rt.Raster:
    """
    Reads the NDVI variable from a vnp43ma4n file

    Parameters
    ----------
    s_filename: str
        Path to the file

    Returns
    -------
    dm_ndvi: np.ndarray
        Estimate calculated from the files on disk

    """

    # Read in the bands
    o_red = read_VNP43MA4N_reflectance(s_filename, 5)
    o_nir = read_VNP43MA4N_reflectance(s_filename, 7)

    # Calculate the estimate
    o_ndvi = (o_nir - o_red) / (o_nir + o_red)

    # Return to the calling function
    return o_ndvi


def read_vnp43ma4n_albedo(s_filename: str) -> rt.Raster:
    """
    Estimates albedo from the vnp43ma4n files on disk. https://lpdaac.usgs.gov/documents/193/VNP43_User_Guide_V1.pdf

    Parameters
    ----------
    s_filename: str
        Path to the file on disk

    Returns
    -------
    o_albedo: rt.Raster
        Albedo estimate calculated from the files on disk

    """

    # Read in the bands
    o_m1 = read_VNP43MA4N_reflectance(s_filename, 1)
    o_m2 = read_VNP43MA4N_reflectance(s_filename, 2)
    o_m3 = read_VNP43MA4N_reflectance(s_filename, 3)
    o_m4 = read_VNP43MA4N_reflectance(s_filename, 4)
    o_m5 = read_VNP43MA4N_reflectance(s_filename, 5)
    o_m7 = read_VNP43MA4N_reflectance(s_filename, 7)
    o_m8 = read_VNP43MA4N_reflectance(s_filename, 8)
    o_m10 = read_VNP43MA4N_reflectance(s_filename, 10)
    o_m11 = read_VNP43MA4N_reflectance(s_filename, 11)

    # Calculate the albedo from the bands
    o_albedo = -0.0131 + (o_m1 * 0.2418) + (o_m2 * -0.201) + (o_m3 * 0.2093) + (o_m4 * 0.1146) + (o_m5 * 0.1348) + (o_m7 * 0.2251) + (o_m8 * 0.1123) + (o_m10 * 0.086) + \
              (o_m11 * 0.0803)

    # Return to the calling function
    return o_albedo


def read_vnp43ma4n_variable(s_filename: str, s_variable: str, o_geometry: rt.RasterGeometry = None, s_resampling: str = None) -> rt.Raster:
    """
    Reads the vnp43ma4n from disk

    Parameters
    ----------
    s_filename: str
        Local path to the file
    s_variable: str
        Varible to be extracted
    o_geometry: rt.RasterGeometry
        Raster geometry bounds
    s_resampling: str
        Resampling method, if any

    Returns
    -------
    o_image: rt.Raster or None
        Object containing the data

    """

    # Define a placeholder for the dataset
    o_image = None

    # Attempt to read data from an existing file
    try:
        if s_variable == "NDVI":
            # Read the NDVI variable
            o_image = read_vnp43ma4n_ndvi(s_filename)

        elif s_variable == "albedo":
            # Read the albedo variable
            o_image = read_vnp43ma4n_albedo(s_filename)

        else:
            # Request is not understood. Raise an error
            raise ValueError(f"unrecognized VNP43MA4N variable: {s_variable}")

        # Clip the raster to the geometry, if provided
        if o_geometry is not None:
            o_image = o_image.to_geometry(o_geometry, resampling=s_resampling)

    except:
        # File read was not successful. Return None to indicate failure
        o_image = None

    # Return to the calling function
    return o_image


def retrieve_vnp43ma4n(o_geometry: rt.RasterGeometry, o_date_utc: date = None, s_variable: str = None, s_resampling: str = "cubic",
                       s_directory: str = None) -> Union[rt.Raster, None]:
    """
    Retrieves the vnp43ma4n tiles values from EOSDIS

    Parameters
    ----------
    o_geometry: rt.RasterGeometry
        Gives the spatial extents of the requested data
    o_date_utc: date
        Solar date of the requested tiles
    s_variable: str
        Variable name to extract from the tiles
    s_resampling: str
        Resampling scheme, if applicable
    s_directory: str
        Local directory to save the data into

    Returns
    -------
    o_composite_image: Union[rt.RasterGeometry, None]
        Returns raster geometry if the download was either fully or partially successful. Otherwise returns None to indicate a failed download

    """

    # Assign a default variable, if not provided
    if s_variable is None:
        s_variable = "NDVI"

    # Assign a working directory, if not provided
    if s_directory is None:
        s_directory = "."

    # Convert a potential string date to a date object
    if isinstance(o_date_utc, str):
        o_date_utc = parser.parse(o_date_utc).date()

    # Request the tiles
    o_tiles = find_modland_tiles(o_geometry.corner_polygon_latlon.geometry)

    # Create an empty raster to receive the data
    o_composite_image = rt.Raster(np.full(o_geometry.shape, np.nan), geometry=o_geometry)

    # Request the URLs from the tiles
    listing = list_VNP43MA4N_URLs(date_UTC=o_date_utc, tiles=o_tiles)

    for i_entry, (s_tile, s_url) in listing.iterrows():
        logger.info(f"processing VNP43MA4 tile: {s_tile} URL: {s_url}")
        s_filename = download_lance_viirs(s_url=s_url, s_directory=s_directory)

        # Check that there was content in the package.
        if s_filename is not None:
            o_image = read_vnp43ma4n_variable(s_filename=s_filename, s_variable=s_variable, o_geometry=o_geometry, s_resampling=s_resampling)
            o_composite_image = o_composite_image.fill(o_image)

    # Check for the composite image to have content
    if np.all(np.isnan(o_composite_image)):
        # No data exists. Return a None type to flag an improper parse
        o_composite_image = None

    # Return to the calling function
    return o_composite_image
