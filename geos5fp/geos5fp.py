import json
import logging
import os
import posixpath
import warnings
from datetime import datetime, time, timedelta, date
from time import sleep
from os import makedirs
from os.path import dirname, basename, splitext, expanduser, getsize
from os.path import exists
from os.path import join
from shutil import move
from typing import List, Any, Union

import numpy as np
import pandas as pd
import rasterio
import requests
from bs4 import BeautifulSoup
from dateutil import parser
from matplotlib.colors import LinearSegmentedColormap
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3 import Retry

import cl
import rasters as rt
from downscaling import linear_downscale, DEFAULT_UPSAMPLING, DEFAULT_DOWNSAMPLING, bias_correct
from rasters import Raster, RasterGeometry
from timer import Timer

__author__ = 'Gregory Halverson'

logger = logging.getLogger(__name__)

# requests.adapters.DEFAULT_RETRIES = 5

DEFAULT_READ_TIMEOUT = 60
DEFAULT_RETRIES = 3

DEFAULT_WORKING_DIRECTORY = "."
DEFAULT_DOWNLOAD_DIRECTORY = "GEOS5FP_download_directory"
DEFAULT_PRODUCTS_DIRECTORY = "GEOS5FP_products"
DEFAULT_USE_HTTP_LISTING = False
DEFAULT_COARSE_CELL_SIZE_METERS = 27440

SM_CMAP = LinearSegmentedColormap.from_list("SM", [
    "#f6e8c3",
    "#d8b365",
    "#99894a",
    "#2d6779",
    "#6bdfd2",
    "#1839c5"
])

NDVI_CMAP = LinearSegmentedColormap.from_list(
    name="LAI",
    colors=[
        "#000000",
        "#745d1a",
        "#e1dea2",
        "#45ff01",
        "#325e32"
    ]
)


class GEOS5FPGranuleNotAvailable(Exception):
    pass


class GEOS5FPDayNotAvailable(Exception):
    pass


class GEOS5FPMonthNotAvailable(Exception):
    pass


class GEOS5FPYearNotAvailable(Exception):
    pass


def HTTP_listing(
        url: str,
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

    with requests.Session() as s:
        # too many retries in too short a time may cause the server to refuse connections
        s.mount('http://', HTTPAdapter(max_retries=retries))
        response = s.get(
            url,
            auth=auth,
            timeout=timeout
        )

    # there was a conflict between Unicode markup and from_encoding
    soup = BeautifulSoup(response.text, 'html.parser')
    links = list(soup.find_all('a', href=True))

    # get directory names from links on http site
    directories = [link['href'].replace('/', '') for link in links]

    return directories


class GEOS5FPGranule:
    logger = logging.getLogger(__name__)

    DEFAULT_RESAMPLING_METHOD = "cubic"

    def __init__(
            self,
            filename: str,
            working_directory: str = None,
            products_directory: str = None,
            save_products: bool = False):
        if not exists(filename):
            raise IOError(f"GEOS-5 FP file does not exist: {filename}")

        if working_directory is None:
            working_directory = DEFAULT_WORKING_DIRECTORY

        if working_directory.startswith("~"):
            working_directory = expanduser(working_directory)

        logger.info(f"GEOS-5 FP working directory: {cl.dir(working_directory)}")

        if products_directory is None:
            products_directory = join(working_directory, DEFAULT_PRODUCTS_DIRECTORY)

        if products_directory.startswith("~"):
            products_directory = expanduser(products_directory)

        logger.info(f"GEOS-5 FP products directory: {cl.dir(products_directory)}")

        self.working_directory = working_directory
        self.products_directory = products_directory
        self.filename = filename
        self.save_products = save_products

    @property
    def product(self) -> str:
        return str(splitext(basename(self.filename))[0].split(".")[-3])

    @property
    def time_UTC(self) -> datetime:
        return datetime.strptime(splitext(basename(self.filename))[0].split(".")[-2], "%Y%m%d_%H%M")

    @property
    def product_directory(self):
        if self.products_directory is None:
            return None
        else:
            return join(self.products_directory, self.product)

    def variable_directory(self, variable):
        if self.product_directory is None:
            return None
        else:
            return join(self.product_directory, variable)

    @property
    def filename_stem(self):
        return splitext(basename(self.filename))[0]

    def variable_filename(self, variable):
        variable_directory = self.variable_directory(variable)

        if variable_directory is None:
            return None
        else:
            return join(variable_directory, f"{self.filename_stem}_{variable}.tif")

    def read(
            self,
            variable: str,
            geometry: RasterGeometry = None,
            resampling: str = None,
            nodata: Any = None,
            min_value: Any = None,
            max_value: Any = None,
            exclude_values=None) -> Raster:
        if resampling is None:
            resampling = self.DEFAULT_RESAMPLING_METHOD

        if nodata is None:
            nodata = np.nan

        variable_filename = self.variable_filename(variable)

        if variable_filename is not None and exists(variable_filename):
            data = Raster.open(variable_filename, nodata=nodata)
        else:
            try:
                data = Raster.open(f'netcdf:"{self.filename}":{variable}', nodata=nodata)
            except Exception as e:
                logger.error(e)
                os.remove(self.filename)

                raise GEOS5FPGranuleNotAvailable(f"removed corrupted GEOS-5 FP file: {self.filename}")

        if exclude_values is not None:
            for exclusion_value in exclude_values:
                data = rt.where(data == exclusion_value, np.nan, data)

        data = rt.clip(data, min_value, max_value)

        if self.save_products and variable_filename is not None and not exists(variable_filename):
            data.to_geotiff(variable_filename)

        if geometry is not None:
            data = data.to_geometry(geometry, resampling=resampling)

        return data


class FailedGEOS5FPDownload(ConnectionError):
    pass


class GEOS5FP:
    logger = logging.getLogger(__name__)

    DEFAULT_URL_BASE = "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das"
    DEFAULT_TIMEOUT_SECONDS = 500
    DEFAULT_RETRIES = 3

    def __init__(
            self,
            working_directory: str = None,
            download_directory: str = None,
            products_directory: str = None,
            remote: str = None,
            save_products: bool = False):
        if working_directory is None:
            working_directory = DEFAULT_WORKING_DIRECTORY

        if working_directory.startswith("~"):
            working_directory = expanduser(working_directory)

        logger.info(f"GEOS-5 FP working directory: {cl.dir(working_directory)}")

        if download_directory is None:
            download_directory = join(working_directory, DEFAULT_DOWNLOAD_DIRECTORY)

        if download_directory.startswith("~"):
            download_directory = expanduser(download_directory)

        logger.info(f"GEOS-5 FP download directory: {cl.dir(download_directory)}")

        if products_directory is None:
            products_directory = join(working_directory, DEFAULT_PRODUCTS_DIRECTORY)

        if products_directory.startswith("~"):
            products_directory = expanduser(products_directory)

        logger.info(f"GEOS-5 FP products directory: {cl.dir(products_directory)}")

        if remote is None:
            remote = self.DEFAULT_URL_BASE

        self.working_directory = working_directory
        self.download_directory = download_directory
        self.products_directory = products_directory
        self.remote = remote
        self._listings = {}
        self.filenames = set([])
        self.save_products = save_products

    def __repr__(self):
        display_dict = {
            "URL": self.remote,
            "download_directory": self.download_directory,
            "products_directory": self.products_directory
        }

        display_string = json.dumps(display_dict, indent=2)

        return display_string

    def _check_remote(self):
        logger.info(f"checking URL: {cl.URL(self.remote)}")
        response = requests.head(self.remote)
        status = response.status_code
        duration = response.elapsed.total_seconds()

        if status == 200:
            logger.info(f"remote verified with status {cl.val(200)} in " + cl.time(
                f"{duration:0.2f}") + " seconds: {cl.URL(self.remote)}")
        else:
            raise IOError(f"status: {status} URL: {self.remote}")

    @property
    def years_available(self) -> List[date]:
        listing = self.list_URL(self.remote)
        dates = sorted([date(int(item[1:]), 1, 1) for item in listing if item.startswith("Y")])

        return dates

    def year_URL(self, year: int) -> str:
        return posixpath.join(self.remote, f"Y{year:04d}") + "/"

    def is_year_available(self, year: int) -> bool:
        return requests.head(self.year_URL(year)).status_code != 404

    def months_available_in_year(self, year) -> List[date]:
        year_URL = self.year_URL(year)

        if requests.head(year_URL).status_code == 404:
            raise GEOS5FPYearNotAvailable(f"GEOS-5 FP year not available: {year_URL}")

        listing = self.list_URL(year_URL)
        dates = sorted([date(year, int(item[1:]), 1) for item in listing if item.startswith("M")])

        return dates

    def month_URL(self, year: int, month: int) -> str:
        return posixpath.join(self.remote, f"Y{year:04d}", f"M{month:02d}") + "/"

    def is_month_available(self, year: int, month: int) -> bool:
        return requests.head(self.month_URL(year, month)).status_code != 404

    def dates_available_in_month(self, year, month) -> List[date]:
        month_URL = self.month_URL(year, month)

        if requests.head(month_URL).status_code == 404:
            raise GEOS5FPMonthNotAvailable(f"GEOS-5 FP month not available: {month_URL}")

        listing = self.list_URL(month_URL)
        dates = sorted([date(year, month, int(item[1:])) for item in listing if item.startswith("D")])

        return dates

    def day_URL(self, date_UTC: Union[date, str]) -> str:
        if isinstance(date_UTC, str):
            date_UTC = parser.parse(date_UTC).date()

        year = date_UTC.year
        month = date_UTC.month
        day = date_UTC.day
        URL = posixpath.join(self.remote, f"Y{year:04d}", f"M{month:02d}", f"D{day:02d}") + "/"

        return URL

    def is_day_available(self, date_UTC: Union[date, str]) -> bool:
        return requests.head(self.day_URL(date_UTC)).status_code != 404

    @property
    def latest_date_available(self) -> date:
        date_UTC = datetime.utcnow().date()
        year = date_UTC.year
        month = date_UTC.month

        if self.is_day_available(date_UTC):
            return date_UTC

        if self.is_month_available(year, month):
            return self.dates_available_in_month(year, month)[-1]

        if self.is_year_available(year):
            return self.dates_available_in_month(year, self.months_available_in_year(year)[-1].month)[-1]

        available_year = self.years_available[-1].year
        available_month = self.months_available_in_year(available_year)[-1].month
        available_date = self.dates_available_in_month(available_year, available_month)[-1]

        return available_date

    @property
    def latest_time_available(self) -> datetime:
        retries = 3
        wait_seconds = 30

        while retries > 0:
            retries -= 1

            try:
                return self.http_listing(self.latest_date_available).sort_values(by="time_UTC").iloc[-1].time_UTC.to_pydatetime()
            except Exception as e:
                logger.warning(e)
                sleep(wait_seconds)
                continue


    def time_from_URL(self, URL: str) -> datetime:
        return datetime.strptime(URL.split(".")[-3], "%Y%m%d_%H%M")

    def list_URL(self, URL: str, timeout: float = None, retries: int = None) -> List[str]:
        if URL in self._listings:
            return self._listings[URL]
        else:
            listing = HTTP_listing(URL, timeout=timeout, retries=retries)
            self._listings[URL] = listing

            return listing

    def http_listing(
            self,
            date_UTC: datetime or str,
            product_name: str = None,
            timeout: float = None,
            retries: int = None) -> pd.DataFrame:
        if timeout is None:
            timeout = self.DEFAULT_TIMEOUT_SECONDS

        if retries is None:
            retries = self.DEFAULT_RETRIES

        day_URL = self.day_URL(date_UTC)

        if requests.head(day_URL).status_code == 404:
            raise GEOS5FPDayNotAvailable(f"GEOS-5 FP day not available: {day_URL}")

        logger.info(f"listing URL: {cl.URL(day_URL)}")
        # listing = HTTP_listing(day_URL, timeout=timeout, retries=retries)
        listing = self.list_URL(day_URL, timeout=timeout, retries=retries)

        if product_name is None:
            URLs = sorted([
                posixpath.join(day_URL, filename)
                for filename
                in listing
                if filename.endswith(".nc4")
            ])
        else:
            URLs = sorted([
                posixpath.join(day_URL, filename)
                for filename
                in listing
                if product_name in filename and filename.endswith(".nc4")
            ])

        df = pd.DataFrame({"URL": URLs})
        df["time_UTC"] = df["URL"].apply(
            lambda URL: datetime.strptime(posixpath.basename(URL).split(".")[4], "%Y%m%d_%H%M"))
        df["product"] = df["URL"].apply(lambda URL: posixpath.basename(URL).split(".")[3])
        df = df[["time_UTC", "product", "URL"]]

        return df

    def generate_filenames(
            self,
            date_UTC: datetime or str,
            product_name: str,
            interval: int,
            expected_hours: List[float] = None) -> pd.DataFrame:
        if isinstance(date_UTC, str):
            date_UTC = parser.parse(date_UTC).date()

        # day_URL = self.day_URL(date_UTC)
        # logger.info(f"generating URLs under: {cl.URL(day_URL)}")

        if expected_hours is None:
            if interval == 1:
                expected_hours = np.arange(0.5, 24.5, 1)
            elif interval == 3:
                expected_hours = np.arange(0.0, 24.0, 3)
            else:
                raise ValueError(f"unrecognized GEOS-5 FP interval: {interval}")

        rows = []

        expected_times = [datetime.combine(date_UTC - timedelta(days=1), time(0)) + timedelta(
            hours=float(expected_hours[-1]))] + [
                             datetime.combine(date_UTC, time(0)) + timedelta(hours=float(hour))
                             for hour
                             in expected_hours
                         ] + [datetime.combine(date_UTC + timedelta(days=1), time(0)) + timedelta(
            hours=float(expected_hours[0]))]

        for time_UTC in expected_times:
            # time_UTC = datetime.combine(date_UTC, time(0)) + timedelta(hours=float(hour))
            filename = f"GEOS.fp.asm.{product_name}.{time_UTC:%Y%m%d_%H%M}.V01.nc4"
            day_URL = self.day_URL(time_UTC.date())
            URL = posixpath.join(day_URL, filename)
            rows.append([time_UTC, URL])

        df = pd.DataFrame(rows, columns=["time_UTC", "URL"])

        return df

    def product_listing(
            self,
            date_UTC: datetime or str,
            product_name: str,
            interval: int,
            expected_hours: List[float] = None,
            timeout: float = None,
            retries: int = None,
            use_http_listing: bool = False) -> pd.DataFrame:
        if use_http_listing:
            return self.http_listing(
                date_UTC=date_UTC,
                product_name=product_name,
                timeout=timeout,
                retries=retries
            )
        elif expected_hours is not None or interval is not None:
            return self.generate_filenames(
                date_UTC=date_UTC,
                product_name=product_name,
                interval=interval,
                expected_hours=expected_hours
            )
        else:
            raise ValueError("must use HTTP listing if not supplying expected hours")

    def date_download_directory(self, time_UTC: datetime) -> str:
        return join(self.download_directory, f"{time_UTC:%Y.%m.%d}")

    def download_filename(self, URL: str) -> str:
        time_UTC = self.time_from_URL(URL)
        download_directory = self.date_download_directory(time_UTC)
        filename = join(download_directory, posixpath.basename(URL))

        return filename

    def download_file(self, URL: str, filename: str = None, retries: int = 3, wait_seconds: int = 30) -> GEOS5FPGranule:
        if filename is None:
            filename = self.download_filename(URL)

        if exists(filename) and getsize(filename) == 0:
            logger.warning(f"removing previously created zero-size corrupted GEOS-5 FP file: {filename}")
            os.remove(filename)

        while retries > 0:
            retries -= 1

            try:
                if requests.head(URL).status_code == 404:
                    directory_URL = posixpath.dirname(URL)

                    if requests.head(directory_URL).status_code == 404:
                        raise GEOS5FPDayNotAvailable(f"GEOS-5 FP day not available: {directory_URL}")
                    else:
                        raise GEOS5FPGranuleNotAvailable(f"GEOS-5 FP granule not available: {URL}")

                if exists(filename):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")

                            with rasterio.open(filename, "r") as file:
                                pass
                    except Exception as e:
                        logger.exception(f"unable to open GEOS-5 FP file: {filename}")
                        logger.warning(f"removing corrupted GEOS-5 FP file: {filename}")
                        os.remove(filename)

                if exists(filename):
                    logger.info(f"GEOS-5 FP file found: {cl.file(filename)}")
                else:
                    logger.info(f"downloading GEOS-5 FP: {cl.URL(URL)} -> {cl.file(filename)}")
                    makedirs(dirname(filename), exist_ok=True)
                    partial_filename = f"{filename}.{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.download"

                    if exists(partial_filename) and getsize(partial_filename) == 0:
                        logger.warning(f"removing zero-size corrupted GEOS-5 FP file: {partial_filename}")
                        os.remove(partial_filename)

                    command = f'wget -c -O "{partial_filename}" "{URL}"'
                    # logger.info(command)
                    timer = Timer()
                    os.system(command)

                    if not exists(partial_filename):
                        raise IOError(f"unable to download URL: {URL}")

                    if not exists(partial_filename):
                        raise FailedGEOS5FPDownload(f"GEOS-5 FP partial download file not found: {URL} -> {partial_filename}")
                    elif exists(partial_filename) and getsize(partial_filename) == 0:
                        logger.warning(f"removing zero-size corrupted GEOS-5 FP file: {partial_filename}")
                        os.remove(partial_filename)
                        raise FailedGEOS5FPDownload(f"zero-size file from GEOS-5 FP download: {URL} -> {partial_filename}")

                    move(partial_filename, filename)

                    if not exists(filename):
                        raise FailedGEOS5FPDownload(f"GEOS-5 FP final download file not found: {URL} -> {filename}")

                    try:
                        with rasterio.open(filename, "r") as file:
                            pass
                    except Exception as e:
                        logger.exception(f"unable to open GEOS-5 FP file: {filename}")
                        logger.warning(f"removing corrupted GEOS-5 FP file: {filename}")
                        os.remove(filename)
                        raise FailedGEOS5FPDownload(f"GEOS-5 FP corrupted download: {URL} -> {filename}")

                    logger.info("GEOS-5 FP download completed: " + cl.file(filename) + " (" + cl.val(
                        f"{(getsize(filename) / 1000000):0.2f}") + " mb) (" + cl.time(timer.duration) + " seconds)")

                granule = GEOS5FPGranule(
                    filename=filename,
                    working_directory=self.working_directory,
                    products_directory=self.products_directory,
                    save_products=self.save_products
                )

                return granule

            except Exception as e:
                if retries == 0:
                    raise e

                self.logger.warning(e)
                self.logger.warning(f"waiting {wait_seconds} for M2M retry")
                sleep(wait_seconds)
                continue

    def before_and_after(
            self,
            time_UTC: datetime or str,
            product: str,
            interval: int = None,
            expected_hours: List[float] = None,
            timeout: float = None,
            retries: int = None,
            use_http_listing: bool = DEFAULT_USE_HTTP_LISTING) -> (datetime, Raster, datetime, Raster):
        if isinstance(time_UTC, str):
            time_UTC = parser.parse(time_UTC)

        search_date = time_UTC.date()
        logger.info(f"searching GEOS-5 FP {cl.name(product)} at " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M:%S} UTC"))

        product_listing = self.product_listing(
            search_date,
            product,
            interval=interval,
            expected_hours=expected_hours,
            timeout=timeout,
            retries=retries,
            use_http_listing=use_http_listing
        )

        if len(product_listing) == 0:
            raise IOError(f"no {product} files found for {time_UTC}")

        before_listing = product_listing[product_listing.time_UTC < time_UTC]

        if len(before_listing) == 0:
            raise IOError(f"no {product} files found preceeding {time_UTC}")

        before_time_UTC, before_URL = before_listing.iloc[-1][["time_UTC", "URL"]]
        after_listing = product_listing[product_listing.time_UTC > time_UTC]

        if len(after_listing) == 0:
            after_listing = self.product_listing(
                search_date + timedelta(days=1),
                product,
                interval=interval,
                expected_hours=expected_hours,
                timeout=timeout,
                retries=retries,
                use_http_listing=use_http_listing
            )
            # raise IOError(f"no {product} files found after {time_UTC}")

        after_time_UTC, after_URL = after_listing.iloc[0][["time_UTC", "URL"]]
        before_granule = self.download_file(before_URL)
        after_granule = self.download_file(after_URL)

        return before_granule, after_granule

    def interpolate(
            self,
            time_UTC: datetime or str,
            product: str,
            variable: str,
            geometry: RasterGeometry = None,
            resampling: str = None,
            cmap=None,
            min_value: Any = None,
            max_value: Any = None,
            exclude_values=None,
            interval: int = None,
            expected_hours: List[float] = None,
            timeout: float = None,
            retries: int = None,
            use_http_listing: bool = DEFAULT_USE_HTTP_LISTING) -> Raster:
        if interval is None:
            if product == "tavg1_2d_rad_Nx":
                interval = 1
            elif product == "tavg1_2d_slv_Nx":
                interval = 1
            elif product == "inst3_2d_asm_Nx":
                interval = 3

        if interval is None and expected_hours is None:
            raise ValueError(f"interval or expected hours not given for {product}")

        before_granule, after_granule = self.before_and_after(
            time_UTC,
            product,
            interval=interval,
            expected_hours=expected_hours,
            timeout=timeout,
            retries=retries,
            use_http_listing=use_http_listing
        )

        logger.info(
            f"interpolating GEOS-5 FP {cl.name(product)} {cl.name(variable)} " +
            f"from " + cl.time(f"{before_granule.time_UTC:%Y-%m-%d %H:%M} UTC ") +
            f"and " + cl.time(f"{after_granule.time_UTC:%Y-%m-%d %H:%M} UTC") + " to " + cl.time(
                f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        with Timer() as timer:
            before = before_granule.read(
                variable,
                geometry=geometry,
                resampling=resampling,
                min_value=min_value,
                max_value=max_value,
                exclude_values=exclude_values
            )

            after = after_granule.read(
                variable,
                geometry=geometry,
                resampling=resampling,
                min_value=min_value,
                max_value=max_value,
                exclude_values=exclude_values
            )

            time_fraction = (time_UTC - before_granule.time_UTC) / (after_granule.time_UTC - before_granule.time_UTC)
            source_diff = after - before
            interpolated_data = before + source_diff * time_fraction
            logger.info(f"GEOS-5 FP interpolation complete ({timer:0.2f} seconds)")

        before_filename = before_granule.filename
        after_filename = after_granule.filename
        filenames = [before_filename, after_filename]
        self.filenames = set(self.filenames) | set(filenames)
        interpolated_data["filenames"] = filenames

        if cmap is not None:
            interpolated_data.cmap = cmap

        return interpolated_data

    def SFMC(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        top soil layer moisture content cubic meters per cubic meters
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of soil moisture
        """
        NAME = "top layer soil moisture"
        PRODUCT = "tavg1_2d_lnd_Nx"
        VARIABLE = "SFMC"
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        return self.interpolate(
            time_UTC=time_UTC,
            product=PRODUCT,
            variable=VARIABLE,
            geometry=geometry,
            resampling=resampling,
            interval=1,
            min_value=0,
            max_value=1,
            exclude_values=[1],
            cmap=SM_CMAP
        )

    SM = SFMC

    def LAI(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        leaf area index
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of LAI
        """
        NAME = "leaf area index"
        PRODUCT = "tavg1_2d_lnd_Nx"
        VARIABLE = "LAI"
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        return self.interpolate(
            time_UTC=time_UTC,
            product=PRODUCT,
            variable=VARIABLE,
            geometry=geometry,
            resampling=resampling,
            interval=1,
            min_value=0,
            max_value=10,
            cmap=NDVI_CMAP
        )

    def NDVI(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        normalized difference vegetation index
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of NDVI
        """
        LAI = self.LAI(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        NDVI = rt.clip(1.05 - np.exp(-0.5 * LAI), 0, 1)

        return NDVI

    def LHLAND(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        latent heat flux in watts per square meter
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of soil moisture
        """
        NAME = "latent heat flux land"
        PRODUCT = "tavg1_2d_lnd_Nx"
        VARIABLE = "LHLAND"
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        return self.interpolate(
            time_UTC=time_UTC,
            product=PRODUCT,
            variable=VARIABLE,
            geometry=geometry,
            resampling=resampling,
            interval=1,
            exclude_values=[1.e+15]
        )

    def EFLUX(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        latent heat flux in watts per square meter
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of soil moisture
        """
        NAME = "total latent energy flux"
        PRODUCT = "tavg1_2d_flx_Nx"
        VARIABLE = "EFLUX"
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        return self.interpolate(
            time_UTC=time_UTC,
            product=PRODUCT,
            variable=VARIABLE,
            geometry=geometry,
            resampling=resampling,
            interval=1
        )

    def PARDR(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        Surface downward PAR beam flux in watts per square meter
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of soil moisture
        """
        NAME = "PARDR"
        PRODUCT = "tavg1_2d_lnd_Nx"
        VARIABLE = "PARDR"
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        image = self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=geometry, resampling=resampling)
        image = rt.clip(image, 0, None)

        return image

    def PARDF(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        Surface downward PAR diffuse flux in watts per square meter
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of soil moisture
        """
        NAME = "PARDF"
        PRODUCT = "tavg1_2d_lnd_Nx"
        VARIABLE = "PARDF"
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        image = self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=geometry, resampling=resampling)
        image = rt.clip(image, 0, None)

        return image

    def AOT(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        aerosol optical thickness (AOT) extinction coefficient
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of AOT
        """
        NAME = "AOT"
        PRODUCT = "tavg3_2d_aer_Nx"
        VARIABLE = "TOTEXTTAU"
        # 1:30, 4:30, 7:30, 10:30, 13:30, 16:30, 19:30, 22:30 UTC
        EXPECTED_HOURS = [1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5]

        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        return self.interpolate(
            time_UTC=time_UTC,
            product=PRODUCT,
            variable=VARIABLE,
            geometry=geometry,
            resampling=resampling,
            expected_hours=EXPECTED_HOURS
        )

    def COT(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        cloud optical thickness (COT) extinction coefficient
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of COT
        """
        NAME = "COT"
        PRODUCT = "tavg1_2d_rad_Nx"
        VARIABLE = "TAUTOT"
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        return self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=geometry, resampling=resampling)

    def Ts_K(
            self,
            time_UTC: datetime,
            geometry: RasterGeometry = None,
            resampling: str = None) -> Raster:
        """
        surface temperature (Ts) in Kelvin
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of Ta
        """
        NAME = "Ts"
        PRODUCT = "tavg1_2d_slv_Nx"
        VARIABLE = "TS"
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        return self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=geometry, resampling=resampling)

    def Ta_K(
            self,
            time_UTC: datetime,
            geometry: RasterGeometry = None,
            ST_K: Raster = None,
            water: Raster = None,
            coarse_geometry: RasterGeometry = None,
            coarse_cell_size_meters: int = DEFAULT_COARSE_CELL_SIZE_METERS,
            resampling: str = None,
            upsampling: str = None,
            downsampling: str = None,
            apply_scale: bool = True,
            apply_bias: bool = True,
            return_scale_and_bias: bool = False) -> Raster:
        """
        near-surface air temperature (Ta) in Kelvin
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of Ta
        """
        NAME = "Ta"
        PRODUCT = "tavg1_2d_slv_Nx"
        VARIABLE = "T2M"

        if isinstance(time_UTC, str):
            time_UTC = parser.parse(time_UTC)

        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        if coarse_cell_size_meters is None:
            coarse_cell_size_meters = DEFAULT_COARSE_CELL_SIZE_METERS

        if ST_K is None:
            return self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=geometry, resampling=resampling)
        else:
            if geometry is None:
                geometry = ST_K.geometry

            if coarse_geometry is None:
                coarse_geometry = geometry.rescale(coarse_cell_size_meters)

            Ta_K_coarse = self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=coarse_geometry, resampling=resampling)
            filenames = Ta_K_coarse["filenames"]

            ST_K_water = None

            if water is not None:
                ST_K_water = rt.where(water, ST_K, np.nan)
                ST_K = rt.where(water, np.nan, ST_K)

            scale = None
            bias = None

            Ta_K = linear_downscale(
                coarse_image=Ta_K_coarse,
                fine_image=ST_K,
                upsampling=upsampling,
                downsampling=downsampling,
                apply_scale=apply_scale,
                apply_bias=apply_bias,
                return_scale_and_bias=return_scale_and_bias
            )

            if water is not None:
                # Ta_K_smooth = self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=geometry, resampling="linear")
                Ta_K_water = linear_downscale(
                    coarse_image=Ta_K_coarse,
                    fine_image=ST_K_water,
                    upsampling=upsampling,
                    downsampling=downsampling,
                    apply_scale=apply_scale,
                    apply_bias=apply_bias,
                    return_scale_and_bias=False
                )

                Ta_K = rt.where(water, Ta_K_water, Ta_K)

            Ta_K.filenames = filenames

            return Ta_K

    def Tmin_K(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        minimum near-surface air temperature (Ta) in Kelvin
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of Ta
        """
        NAME = "Tmin"
        PRODUCT = "inst3_2d_asm_Nx"
        VARIABLE = "T2MMIN"
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        return self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=geometry, resampling=resampling)

    # def SVP_mb(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
    #     # https://cran.r-project.org/web/packages/humidity/vignettes/humidity-measures.html
    #     Rw = 461.52  # J/(kgK)
    #     T0 = 273.15  # K
    #     L = 2.5 * 10 ** 6  # J/kg
    #     Ta_K = self.Ta_K(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
    #     SVP_mb = 6.11 * np.exp((L / Rw) * ((1 / T0) - (1 / Ta_K)))
    #
    #     return SVP_mb

    def SVP_Pa(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        Ta_C = self.Ta_C(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        SVP_Pa = 0.6108 * np.exp((17.27 * Ta_C) / (Ta_C + 237.3)) * 1000  # [Pa]

        return SVP_Pa

    def SVP_kPa(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        return self.SVP_Pa(time_UTC=time_UTC, geometry=geometry, resampling=resampling) / 1000

    def Ta_C(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        return self.Ta_K(time_UTC=time_UTC, geometry=geometry, resampling=resampling) - 273.15

    def PS(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        surface pressure in Pascal
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of surface pressure
        """
        NAME = "surface pressure"
        PRODUCT = "tavg1_2d_slv_Nx"
        VARIABLE = "PS"
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        return self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=geometry, resampling=resampling)

    def Q(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        near-surface specific humidity (Q) in kilograms per kilogram
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of Q
        """
        NAME = "Q"
        PRODUCT = "tavg1_2d_slv_Nx"
        VARIABLE = "QV2M"
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        return self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=geometry, resampling=resampling)

    # def Ea_mb(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
    #     # https://cran.r-project.org/web/packages/humidity/vignettes/humidity-measures.html
    #     Q = self.Q(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
    #     PS = self.PS(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
    #     Ea_mb = (Q * PS) / 0.622 + 0.378 * Q
    #
    #     return Ea_mb

    def Ea_Pa(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        RH = self.RH(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        SVP_Pa = self.SVP_Pa(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        Ea_Pa = RH * SVP_Pa

        return Ea_Pa

    def Td_K(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        Ta_K = self.Ta_K(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        RH = self.RH(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        Td_K = Ta_K - (100 - (RH * 100)) / 5

        return Td_K

    def Td_C(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        return self.Td_K(time_UTC=time_UTC, geometry=geometry, resampling=resampling) - 273.15

    def Cp(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        Ps_Pa = self.PS(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        Ea_Pa = self.Ea_Pa(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        Cp = 0.24 * 4185.5 * (1.0 + 0.8 * (0.622 * Ea_Pa / (Ps_Pa - Ea_Pa)))  # [J kg-1 K-1]

        return Cp

    def VPD_Pa(
            self,
            time_UTC: datetime,
            ST_K: Raster = None,
            geometry: RasterGeometry = None,
            coarse_geometry: RasterGeometry = None,
            coarse_cell_size_meters: int = DEFAULT_COARSE_CELL_SIZE_METERS,
            resampling: str = None,
            upsampling: str = None,
            downsampling: str = None,
            return_scale_and_bias: bool = False) -> Raster:
        if ST_K is None:
            Ea_Pa = self.Ea_Pa(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
            SVP_Pa = self.SVP_Pa(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
            VPD_Pa = rt.clip(SVP_Pa - Ea_Pa, 0, None)

            return VPD_Pa
        else:
            if geometry is None:
                geometry = ST_K.geometry

            if coarse_geometry is None:
                coarse_geometry = geometry.rescale(coarse_cell_size_meters)

            Ea_Pa = self.Ea_Pa(time_UTC=time_UTC, geometry=coarse_geometry, resampling=resampling)
            SVP_Pa = self.SVP_Pa(time_UTC=time_UTC, geometry=coarse_geometry, resampling=resampling)
            VPD_Pa = rt.clip(SVP_Pa - Ea_Pa, 0, None)

            return linear_downscale(
                coarse_image=VPD_Pa,
                fine_image=ST_K,
                upsampling=upsampling,
                downsampling=downsampling,
                return_scale_and_bias=return_scale_and_bias
            )

    def VPD_kPa(
            self,
            time_UTC: datetime,
            ST_K: Raster = None,
            geometry: RasterGeometry = None,
            coarse_geometry: RasterGeometry = None,
            coarse_cell_size_meters: int = DEFAULT_COARSE_CELL_SIZE_METERS,
            resampling: str = None,
            upsampling: str = None,
            downsampling: str = None) -> Raster:
        VPD_Pa = self.VPD_Pa(
            time_UTC=time_UTC,
            ST_K=ST_K,
            geometry=geometry,
            coarse_geometry=coarse_geometry,
            coarse_cell_size_meters=coarse_cell_size_meters,
            resampling=resampling,
            upsampling=upsampling,
            downsampling=downsampling
        )

        VPD_kPa = VPD_Pa / 1000

        return VPD_kPa

    # def RH(
    #         self,
    #         time_UTC: datetime,
    #         geometry: RasterGeometry = None,
    #         SM: Raster = None,
    #         ST_K: Raster = None,
    #         water: Raster = None,
    #         coarse_geometry: RasterGeometry = None,
    #         coarse_cell_size_meters: int = DEFAULT_COARSE_CELL_SIZE_METERS,
    #         resampling: str = None,
    #         upsampling: str = None,
    #         downsampling: str = None,
    #         apply_bias: bool = True,
    #         return_scale_and_bias: bool = False) -> Raster:
    #
    #     if SM is None:
    #         Q = self.Q(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
    #         Ps_Pa = self.PS(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
    #         SVP_Pa = self.SVP_Pa(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
    #         Mw = 18.015268  # g / mol
    #         Md = 28.96546e-3  # kg / mol
    #         epsilon = Mw / (Md * 1000)
    #         w = Q / (1 - Q)
    #         ws = epsilon * SVP_Pa / (Ps_Pa - SVP_Pa)
    #         RH = rt.clip(w / ws, 0, 1)
    #     else:
    #         if geometry is None:
    #             geometry = SM.geometry
    #
    #         if coarse_geometry is None:
    #             coarse_geometry = geometry.rescale(coarse_cell_size_meters)
    #
    #         RH_coarse = self.RH(time_UTC=time_UTC, geometry=coarse_geometry, resampling=resampling)
    #
    #         # if water is not None:
    #         #     SM = rt.where(water, 1, SM)
    #         scale = None
    #         bias = None
    #
    #         if return_scale_and_bias:
    #             RH, scale, bias = linear_downscale(
    #                 coarse_image=RH_coarse,
    #                 fine_image=SM,
    #                 upsampling=upsampling,
    #                 downsampling=downsampling,
    #                 apply_bias=apply_bias,
    #                 return_scale_and_bias=True
    #             )
    #         else:
    #             RH = linear_downscale(
    #                 coarse_image=RH_coarse,
    #                 fine_image=SM,
    #                 upsampling=upsampling,
    #                 downsampling=downsampling,
    #                 apply_bias=apply_bias,
    #                 return_scale_and_bias=False
    #             )
    #
    #         if water is not None:
    #             if ST_K is not None:
    #                 ST_K_water = rt.where(water, ST_K, np.nan)
    #                 RH_coarse_complement = 1 - RH_coarse
    #                 RH_complement_water = linear_downscale(
    #                     coarse_image=RH_coarse_complement,
    #                     fine_image=ST_K_water,
    #                     upsampling=upsampling,
    #                     downsampling=downsampling,
    #                     apply_bias=apply_bias,
    #                     return_scale_and_bias=False
    #                 )
    #
    #                 RH_water = 1 - RH_complement_water
    #                 RH = rt.where(water, RH_water, RH)
    #             else:
    #                 RH_smooth = self.RH(time_UTC=time_UTC, geometry=geometry, resampling="linear")
    #                 RH = rt.where(water, RH_smooth, RH)
    #
    #     RH = rt.clip(RH, 0, 1)
    #
    #     if return_scale_and_bias:
    #         return RH, scale, bias
    #     else:
    #         return RH

    def RH(
            self,
            time_UTC: datetime,
            geometry: RasterGeometry = None,
            SM: Raster = None,
            ST_K: Raster = None,
            VPD_kPa: Raster = None,
            water: Raster = None,
            coarse_geometry: RasterGeometry = None,
            coarse_cell_size_meters: int = DEFAULT_COARSE_CELL_SIZE_METERS,
            resampling: str = None,
            upsampling: str = None,
            downsampling: str = None,
            sharpen_VPD: bool = True,
            return_bias: bool = False) -> Raster:
        if upsampling is None:
            upsampling = DEFAULT_UPSAMPLING

        if downsampling is None:
            downsampling = DEFAULT_DOWNSAMPLING

        bias_fine = None

        if SM is None:
            Q = self.Q(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
            Ps_Pa = self.PS(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
            SVP_Pa = self.SVP_Pa(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
            Mw = 18.015268  # g / mol
            Md = 28.96546e-3  # kg / mol
            epsilon = Mw / (Md * 1000)
            w = Q / (1 - Q)
            ws = epsilon * SVP_Pa / (Ps_Pa - SVP_Pa)
            RH = rt.clip(w / ws, 0, 1)
        else:
            if geometry is None:
                geometry = SM.geometry

            if coarse_geometry is None:
                coarse_geometry = geometry.rescale(coarse_cell_size_meters)

            RH_coarse = self.RH(time_UTC=time_UTC, geometry=coarse_geometry, resampling=resampling)

            if VPD_kPa is None:
                if sharpen_VPD:
                    VPD_fine_distribution = ST_K
                else:
                    VPD_fine_distribution = None

                VPD_kPa = self.VPD_kPa(
                    time_UTC=time_UTC,
                    ST_K=VPD_fine_distribution,
                    geometry=geometry,
                    coarse_geometry=coarse_geometry,
                    coarse_cell_size_meters=coarse_cell_size_meters,
                    resampling=resampling,
                    upsampling=upsampling,
                    downsampling=downsampling
                )

            RH_estimate_fine = SM ** (1 / VPD_kPa)

            if return_bias:
                RH, bias_fine = bias_correct(
                    coarse_image=RH_coarse,
                    fine_image=RH_estimate_fine,
                    upsampling=upsampling,
                    downsampling=downsampling,
                    return_bias=True
                )
            else:
                RH = bias_correct(
                    coarse_image=RH_coarse,
                    fine_image=RH_estimate_fine,
                    upsampling=upsampling,
                    downsampling=downsampling,
                    return_bias=False
                )

            if water is not None:
                if ST_K is not None:
                    ST_K_water = rt.where(water, ST_K, np.nan)
                    RH_coarse_complement = 1 - RH_coarse
                    RH_complement_water = linear_downscale(
                        coarse_image=RH_coarse_complement,
                        fine_image=ST_K_water,
                        upsampling=upsampling,
                        downsampling=downsampling,
                        apply_bias=True,
                        return_scale_and_bias=False
                    )

                    RH_water = 1 - RH_complement_water
                    RH = rt.where(water, RH_water, RH)
                else:
                    RH_smooth = self.RH(time_UTC=time_UTC, geometry=geometry, resampling="linear")
                    RH = rt.where(water, RH_smooth, RH)

        RH = rt.clip(RH, 0, 1)

        if return_bias:
            return RH, bias_fine
        else:
            return RH

    def Ea_kPa(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        return self.Ea_Pa(time_UTC=time_UTC, geometry=geometry, resampling=resampling) / 1000

    def vapor_kgsqm(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        total column water vapor (vapor_gccm) in kilograms per square meter
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of vapor_gccm
        """
        NAME = "vapor_gccm"
        PRODUCT = "inst3_2d_asm_Nx"
        VARIABLE = "TQV"
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        vapor = self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=geometry, resampling=resampling)
        vapor = np.clip(vapor, 0, None)

        return vapor

    def vapor_gccm(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        total column water vapor (vapor_gccm) in grams per square centimeter
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of vapor_gccm
        """
        return self.vapor_kgsqm(time_UTC=time_UTC, geometry=geometry, resampling=resampling) / 10

    def ozone_dobson(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        total column ozone (ozone_cm) in Dobson units
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of vapor_gccm
        """
        NAME = "ozone_cm"
        PRODUCT = "inst3_2d_asm_Nx"
        VARIABLE = "TO3"
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        ozone_dobson = self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=geometry, resampling=resampling)
        ozone_dobson = np.clip(ozone_dobson, 0, None)

        return ozone_dobson

    def ozone_cm(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        total column ozone (ozone_cm) in centimeters
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of vapor_gccm
        """
        return self.ozone_dobson(time_UTC=time_UTC, geometry=geometry, resampling=resampling) / 1000

    def U2M(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        eastward wind at 2 meters in meters per second
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of vapor_gccm
        """
        NAME = "U2M"
        PRODUCT = "inst3_2d_asm_Nx"
        VARIABLE = "U2M"
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        U2M = self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=geometry, resampling=resampling)

        return U2M

    def V2M(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        northward wind at 2 meters in meters per second
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of vapor_gccm
        """
        NAME = "V2M"
        PRODUCT = "inst3_2d_asm_Nx"
        VARIABLE = "V2M"
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        V2M = self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=geometry, resampling=resampling)

        return V2M

    def CO2SC(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        carbon dioxide suface concentration in ppm or micromol per mol
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of vapor_gccm
        """
        NAME = "CO2SC"
        PRODUCT = "tavg3_2d_chm_Nx"
        VARIABLE = "CO2SC"
        # 1:30, 4:30, 7:30, 10:30, 13:30, 16:30, 19:30, 22:30 UTC
        EXPECTED_HOURS = [1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5]

        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        CO2SC = self.interpolate(
            time_UTC=time_UTC,
            product=PRODUCT,
            variable=VARIABLE,
            geometry=geometry,
            resampling=resampling,
            expected_hours=EXPECTED_HOURS
        )

        return CO2SC

    def wind_speed(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        wind speed in meters per second
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of vapor_gccm
        """
        U = self.U2M(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        V = self.V2M(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        wind_speed = rt.clip(np.sqrt(U ** 2.0 + V ** 2.0), 0.0, None)

        return wind_speed

    def SWin(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        incoming shortwave radiation (SWin) in watts per square meter
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of SWin
        """
        NAME = "SWin"
        PRODUCT = "tavg1_2d_rad_Nx"
        VARIABLE = "SWGNT"
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        SWin = self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=geometry, resampling=resampling)
        SWin = np.clip(SWin, 0, None)

        return SWin

    def SWTDN(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        top of atmosphere incoming shortwave radiation (SWin) in watts per square meter
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of SWin
        """
        NAME = "SWTDN"
        PRODUCT = "tavg1_2d_rad_Nx"
        VARIABLE = "SWTDN"
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        SWin = self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=geometry, resampling=resampling)
        SWin = np.clip(SWin, 0, None)

        return SWin

    def ALBVISDR(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        Direct beam VIS-UV surface albedo
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of direct visible albedo
        """
        NAME = "ALBVISDR"
        PRODUCT = "tavg1_2d_rad_Nx"
        VARIABLE = "ALBVISDR"
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        image = self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=geometry, resampling=resampling)
        image = np.clip(image, 0, 1)

        return image

    def ALBVISDF(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        Diffuse beam VIS-UV surface albedo
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of direct visible albedo
        """
        NAME = "ALBVISDF"
        PRODUCT = "tavg1_2d_rad_Nx"
        VARIABLE = "ALBVISDF"
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        image = self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=geometry, resampling=resampling)
        image = np.clip(image, 0, 1)

        return image

    def ALBNIRDF(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        Diffuse beam NIR surface albedo
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of direct visible albedo
        """
        NAME = "ALBNIRDF"
        PRODUCT = "tavg1_2d_rad_Nx"
        VARIABLE = "ALBNIRDF"
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        image = self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=geometry, resampling=resampling)
        image = np.clip(image, 0, 1)

        return image

    def ALBNIRDR(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        Direct beam NIR surface albedo
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of direct visible albedo
        """
        NAME = "ALBNIRDR"
        PRODUCT = "tavg1_2d_rad_Nx"
        VARIABLE = "ALBNIRDR"
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        image = self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=geometry, resampling=resampling)
        image = np.clip(image, 0, 1)

        return image

    def ALBEDO(self, time_UTC: datetime, geometry: RasterGeometry = None, resampling: str = None) -> Raster:
        """
        Surface albedo
        :param time_UTC: date/time in UTC
        :param geometry: optional target geometry
        :param resampling: optional sampling method for resampling to target geometry
        :return: raster of direct visible albedo
        """
        NAME = "ALBEDO"
        PRODUCT = "tavg1_2d_rad_Nx"
        VARIABLE = "ALBEDO"
        logger.info(
            f"retrieving {cl.name(NAME)} "
            f"from GEOS-5 FP {cl.name(PRODUCT)} {cl.name(VARIABLE)} " +
            "for " + cl.time(f"{time_UTC:%Y-%m-%d %H:%M} UTC")
        )

        image = self.interpolate(time_UTC, PRODUCT, VARIABLE, geometry=geometry, resampling=resampling)
        image = np.clip(image, 0, 1)

        return image
