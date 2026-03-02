import posixpath
import warnings
from os import makedirs, remove
from os.path import join, getsize, basename
from shutil import move
from time import sleep
from typing import List

import h5py
import pandas as pd
import requests
from bs4 import BeautifulSoup
from matplotlib.colors import LinearSegmentedColormap
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3 import Retry

from ETtoolbox.ERS_credentials.ERS_credentials import get_ERS_credentials
import colored_logging
import rasters as rt
from modland import find_modland_tiles, generate_modland_grid, parsehv
from solar_apparent_time import UTC_to_solar
from ETtoolbox.VIIRS_orbit import *

DEFAULT_REMOTE = "https://nrt4.modaps.eosdis.nasa.gov/api/v2/content/archives/allData"
DEFAULT_READ_TIMEOUT = 60
DEFAULT_RETRIES = 3
ARCHIVE = "5200"

NDVI_CMAP = LinearSegmentedColormap.from_list(
    name="NDVI",
    colors=[
        (0, "#0000ff"),
        (0.4, "#000000"),
        (0.5, "#745d1a"),
        (0.6, "#e1dea2"),
        (0.8, "#45ff01"),
        (1, "#325e32")
    ]
)

logger = logging.getLogger(__name__)


class LANCENotAvailableError(Exception):
    pass


def HTTP_listing(
        URL: str,
        timeout: float = None,
        retries: int = None,
        username: str = None,
        password: str = None,
        **kwargs):
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

    retries = Retry(
        total=retries,
        backoff_factor=3,
        status_forcelist=[500, 502, 503, 504]
    )

    if not username is None and not password is None:
        auth = HTTPBasicAuth(username, password)
    else:
        auth = None
    with warnings.catch_warnings(), requests.Session() as s:
        warnings.filterwarnings("ignore")
        # too many retries in too short a time may cause the server to refuse connections
        s.mount('http://', HTTPAdapter(max_retries=retries))
        response = s.get(
            URL,
            auth=auth,
            timeout=timeout
        )

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

    dates = sorted([datetime.strptime(f"{year:04d}{posixpath.basename(posixpath.dirname(item))}", "%Y%j").date() for item in listing])

    if len(dates) == 0:
        raise LANCENotAvailableError(f"no dates for LANCE products found at URL: {URL}")

    return dates


def get_LANCE_download_directory(directory: str, product: str, date_UTC: Union[date, str]) -> str:
    return join(expanduser(directory), product, f"{date_UTC:%Y-%m-%d}")


def download_LANCE_VIIRS(
        URL: str,
        directory: str,
        ERS_credentials_filename: str,
        retries: int = 3,
        wait_seconds: int = 30) -> str:
    credentials = get_ERS_credentials(filename=ERS_credentials_filename)
    ERS_token = credentials["token"]

    header = f"Authorization: Bearer {ERS_token}"
    product = posixpath.basename(URL).split(".")[0]
    date_UTC = datetime.strptime(posixpath.basename(URL).split(".")[1][1:], "%Y%j").date()

    destination_directory = get_LANCE_download_directory(
        directory=directory,
        product=product,
        date_UTC=date_UTC
    )

    makedirs(destination_directory, exist_ok=True)
    filename = join(destination_directory, posixpath.basename(URL))

    while retries > 0:
        retries -= 1

        try:
            if exists(filename) and getsize(filename) == 0:
                logger.warning(f"removing zero-size file: {filename}")
                remove(filename)

            if exists(filename):
                logger.info(f"file already downloaded: {colored_logging.file(filename)}")

                try:
                    with h5py.File(filename) as file:
                        pass

                    return filename
                except Exception as e:
                    logger.warning(f"removing corrupted LANCE file: {filename}")
                    os.remove(filename)

            logger.info(f"downloading URL: {colored_logging.URL(URL)}")
            partial_filename = filename + ".download"
            command = f'wget -e robots=off -c -nc -np -nH --no-directories --header "{header}" -O "{partial_filename}" "{URL}"'
            logger.info(command)
            os.system(command)

            if not exists(partial_filename):
                raise ConnectionError(f"unable to download URL: {URL}")
            elif exists(partial_filename) and getsize(partial_filename) == 0:
                logger.warning(f"removing zero-size corrupted LANCE file: {partial_filename}")
                os.remove(partial_filename)
                raise ConnectionError(f"unable to download URL: {URL}")

            move(partial_filename, filename)

            if not exists(filename):
                raise ConnectionError(f"unable to download URL: {URL}")

            logger.info(f"successfully downloaded file: {colored_logging.file(filename)} ({getsize(filename)})")

            return filename
        except Exception as e:
            if retries == 0:
                raise e

            logger.warning(e)
            logger.warning(f"waiting {wait_seconds} for M2M retry")
            sleep(wait_seconds)
            continue


def generate_VNP21NRT_URL(
        datetime_UTC: datetime,
        remote: str = DEFAULT_REMOTE,
        archive: str = "5200",
        collection: str = "002"):
    year = datetime_UTC.year
    doy = datetime_UTC.timetuple().tm_yday
    granule = f"{datetime_UTC:%H%M}"
    URL = posixpath.join(remote, archive, "VNP21_NRT", f"{year:04d}", f"{doy:03d}",
                         f"VNP21_NRT.A{year:04d}{doy:03d}.{granule}.{collection}.nc")

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
    return rt.RasterGeolocation(
        x=read_VNP21NRT_longitude(filename=filename),
        y=read_VNP21NRT_latitude(filename=filename)
    )


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


def read_VNP21NRT_layer(filename: str, variable: str, geometry: rt.RasterGeometry = None,
                        resampling: str = None) -> rt.Raster:
    data = read_VNP21NRT_DN(filename, variable)
    fill_value = read_VNP21NRT_attribute(filename, variable, "_FillValue")
    scale_factor = read_VNP21NRT_attribute(filename, variable, "scale_factor")
    offset = read_VNP21NRT_attribute(filename, variable, "add_offset")
    data = rt.where(data == fill_value, np.nan, data * scale_factor + offset)
    cloud = read_VNP21NRT_cloud(filename)
    data = rt.where(cloud, np.nan, data)

    if geometry is not None:
        data = data.to_geometry(geometry.grid.rescale(1000), resampling="average").to_geometry(geometry,
                                                                                               resampling=resampling)

    return data


def retrieve_VNP21NRT(
        geometry: rt.RasterGeometry,
        date_solar: date = None,
        variable: str = None,
        resampling: str = None,
        directory: str = None,
        spacetrack_credentials_filename: str = None,
        ERS_credentials_filename: str = None) -> rt.Raster:
    if variable is None:
        variable = "LST"

    if isinstance(date_solar, str):
        date_solar = parser.parse(date_solar).date()

    if date_solar is None:
        date_solar = UTC_to_solar(datetime.utcnow(), rt.wrap_geometry(geometry).centroid_latlon).date()

    datetime_solar = datetime(date_solar.year, date_solar.month, date_solar.day, 13, 30)
    datetime_UTC = solar_to_UTC(datetime_solar, geometry.centroid.latlon.x)
    swaths = find_VIIRS_swaths(date_solar, geometry.corner_polygon_latlon.geometry, filter_geometry=True, spacetrack_credentials_filename=spacetrack_credentials_filename)
    composite_image = rt.Raster(np.full(geometry.shape, np.nan), geometry=geometry)

    for i, (swath_datetime_UTC, swath_datetime_solar, swath_name, swath_geometry) in swaths.iterrows():
        URL = generate_VNP21NRT_URL(swath_datetime_UTC)
        filename = download_LANCE_VIIRS(URL=URL, directory=directory, ERS_credentials_filename=ERS_credentials_filename)
        image = read_VNP21NRT_layer(filename=filename, variable=variable, geometry=geometry)
        composite_image = composite_image.fill(image)

    return composite_image


def retrieve_VNP21NRT_ST(
        geometry: rt.RasterGeometry,
        date_solar: date = None,
        resampling: str = None,
        directory: str = None,
        spacetrack_credentials_filename: str = None,
        ERS_credentials_filename: str = None) -> rt.Raster:
    return retrieve_VNP21NRT(
        geometry=geometry,
        date_solar=date_solar,
        variable="LST",
        directory=directory,
        spacetrack_credentials_filename=spacetrack_credentials_filename,
        ERS_credentials_filename=ERS_credentials_filename
    )

def retrieve_VNP21NRT_emissivity(
        geometry: rt.RasterGeometry,
        date_solar: date = None,
        resampling: str = None,
        directory: str = None,
        spacetrack_credentials_filename: str = None) -> rt.Raster:
    return retrieve_VNP21NRT(
        geometry=geometry,
        date_solar=date_solar,
        variable="Emis_ASTER",
        directory=directory,
        spacetrack_credentials_filename=spacetrack_credentials_filename
    )


def generate_VNP43IA4N_date_URL(
        date_UTC: Union[date, str],
        remote: str = DEFAULT_REMOTE,
        archive: str = ARCHIVE) -> str:
    if isinstance(date_UTC, str):
        date_UTC = parser.parse(date_UTC).date()

    year = date_UTC.year
    doy = date_UTC.timetuple().tm_yday
    URL = posixpath.join(remote, archive, "VNP43IA4N", f"{year:04d}", f"{doy:03d}")

    return URL


def list_VNP43IA4N_URLs(
        date_UTC: Union[date, str],
        tiles: List[str] = None,
        remote: str = DEFAULT_REMOTE) -> pd.DataFrame:
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


def read_VNP43IA4N_reflectance(filename: str, band: int, geometry: rt.RasterGeometry = None,
                               resampling: str = None) -> rt.Raster:
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


def read_VNP43IA4N_variable(
        filename: str,
        variable: str,
        geometry: rt.RasterGeometry = None,
        resampling: str = None) -> rt.Raster:
    if variable == "NDVI":
        image = read_VNP43IA4N_NDVI(filename)
    else:
        raise ValueError(f"unrecognized VNP43IA4N variable: {variable}")

    if geometry is not None:
        image = image.to_geometry(geometry, resampling=resampling)

    image.cmap = NDVI_CMAP

    return image


def retrieve_VNP43IA4N(
        geometry: rt.RasterGeometry,
        date_UTC: date = None,
        variable: str = None,
        resampling: str = "cubic",
        directory: str = None,
        ERS_credentials_filename: str = None) -> rt.Raster:
    if variable is None:
        variable = "NDVI"

    if directory is None:
        directory = "."

    if isinstance(date_UTC, str):
        date_UTC = parser.parse(date_UTC).date()

    tiles = find_modland_tiles(geometry.corner_polygon_latlon.geometry)
    composite_image = rt.Raster(np.full(geometry.shape, np.nan), geometry=geometry)

    listing = list_VNP43IA4N_URLs(date_UTC=date_UTC, tiles=tiles)

    for i, (tile, URL) in listing.iterrows():
        logger.info(f"processing VNP43IA4 tile: {tile} URL: {URL}")
        filename = download_LANCE_VIIRS(URL=URL, directory=directory, ERS_credentials_filename=ERS_credentials_filename)
        image = read_VNP43IA4N_variable(filename=filename, variable=variable, geometry=geometry, resampling=resampling)
        composite_image = composite_image.fill(image)

    return composite_image


def generate_VNP43MA4N_date_URL(
        date_UTC: Union[date, str],
        remote: str = DEFAULT_REMOTE,
        archive: str = ARCHIVE) -> str:
    if isinstance(date_UTC, str):
        date_UTC = parser.parse(date_UTC).date()

    year = date_UTC.year
    doy = date_UTC.timetuple().tm_yday
    URL = posixpath.join(remote, archive, "VNP43MA4N", f"{year:04d}", f"{doy:03d}")

    return URL


def list_VNP43MA4N_URLs(
        date_UTC: Union[date, str],
        tiles: List[str] = None,
        remote: str = DEFAULT_REMOTE,
        archive: str = ARCHIVE) -> pd.DataFrame:
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


def read_VNP43MA4N_reflectance(filename: str, band: int, geometry: rt.RasterGeometry = None,
                               resampling: str = None) -> rt.Raster:
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


def read_VNP43MA4N_NDVI(filename: str) -> rt.Raster:
    red = read_VNP43MA4N_reflectance(filename, 5)
    NIR = read_VNP43MA4N_reflectance(filename, 7)
    NDVI = (NIR - red) / (NIR + red)

    return NDVI


def read_VNP43MA4N_albedo(filename: str) -> rt.Raster:
    # https://lpdaac.usgs.gov/documents/193/VNP43_User_Guide_V1.pdf
    M1 = read_VNP43MA4N_reflectance(filename, 1)
    M2 = read_VNP43MA4N_reflectance(filename, 2)
    M3 = read_VNP43MA4N_reflectance(filename, 3)
    M4 = read_VNP43MA4N_reflectance(filename, 4)
    M5 = read_VNP43MA4N_reflectance(filename, 5)
    M7 = read_VNP43MA4N_reflectance(filename, 7)
    M8 = read_VNP43MA4N_reflectance(filename, 8)
    M10 = read_VNP43MA4N_reflectance(filename, 10)
    M11 = read_VNP43MA4N_reflectance(filename, 11)

    albedo = -0.0131 + \
             (M1 * 0.2418) + \
             (M2 * -0.201) + \
             (M3 * 0.2093) + \
             (M4 * 0.1146) + \
             (M5 * 0.1348) + \
             (M7 * 0.2251) + \
             (M8 * 0.1123) + \
             (M10 * 0.086) + \
             (M11 * 0.0803)

    return albedo


def read_VNP43MA4N_variable(
        filename: str,
        variable: str,
        geometry: rt.RasterGeometry = None,
        resampling: str = None) -> rt.Raster:
    if variable == "NDVI":
        image = read_VNP43MA4N_NDVI(filename)
    elif variable == "albedo":
        image = read_VNP43MA4N_albedo(filename)
    else:
        raise ValueError(f"unrecognized VNP43MA4N variable: {variable}")

    if geometry is not None:
        image = image.to_geometry(geometry, resampling=resampling)

    return image


def retrieve_VNP43MA4N(
        geometry: rt.RasterGeometry,
        date_UTC: date = None,
        variable: str = None,
        resampling: str = "cubic",
        directory: str = None) -> rt.Raster:
    if variable is None:
        variable = "NDVI"

    if directory is None:
        directory = "."

    if isinstance(date_UTC, str):
        date_UTC = parser.parse(date_UTC).date()

    tiles = find_modland_tiles(geometry.corner_polygon_latlon.geometry)
    composite_image = rt.Raster(np.full(geometry.shape, np.nan), geometry=geometry)

    listing = list_VNP43MA4N_URLs(date_UTC=date_UTC, tiles=tiles)

    for i, (tile, URL) in listing.iterrows():
        logger.info(f"processing VNP43MA4 tile: {tile} URL: {URL}")
        filename = download_LANCE_VIIRS(URL=URL, directory=directory, ERS_token=API_key)
        image = read_VNP43MA4N_variable(filename=filename, variable=variable, geometry=geometry, resampling=resampling)
        composite_image = composite_image.fill(image)

    return composite_image
