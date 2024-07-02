from fnmatch import fnmatch
import os
from os import makedirs, system
from typing import List
from matplotlib.colors import LinearSegmentedColormap
import requests
from bs4 import BeautifulSoup
import posixpath
import re
import pandas as pd
from os.path import exists
from os.path import splitext
from os.path import join
from os.path import expanduser
import pygrib
from shutil import move
from datetime import datetime
from datetime import timedelta
from datetime import date
from dateutil import parser
import rasters as rt
import numpy as np
import logging
import colored_logging

logger = logging.getLogger(__name__)

GFS_SM_MESSAGE = 565
GFS_TA_MESSAGE = 581
GFS_RH_MESSAGE = 584
GFS_U_WIND_MESSAGE = 588
GFS_V_WIND_MESSAGE = 589
GFS_SWIN_MESSAGE = 653

REMOTE = "https://www.ncei.noaa.gov/data/global-forecast-system/access/grid-004-0.5-degree/forecast"

SM_CMAP = LinearSegmentedColormap.from_list("SM", [
    "#f6e8c3",
    "#d8b365",
    "#99894a",
    "#2d6779",
    "#6bdfd2",
    "#1839c5"
])

def generate_GFS_date_URL(date_UTC: date) -> str:
    if isinstance(date_UTC, str):
        date_UTC = parser.parse(date_UTC).date()

    URL = f"https://www.ncei.noaa.gov/data/global-forecast-system/access/grid-004-0.5-degree/forecast/202109/"

    return URL

def GFS_month_addresses(URL: str = None) -> List[str]:
    if URL is None:
        URL = REMOTE    

    response = requests.get(URL)
    soup = BeautifulSoup(response.text, "html.parser")
    links = [link.get("href") for link in soup.find_all("a")]
    directories = [link for link in links if re.compile("[0-9]{6}").match(link)]
    addresses = sorted([posixpath.join(URL, directory) for directory in directories])
    
    return addresses

def GFS_most_recent_month_address(year: int = None, month: int = None) -> str:
    addresses = GFS_month_addresses()

    if year is not None and month is not None:
        addresses = [address for address in addresses if datetime.strptime(posixpath.basename(address.strip("/")), "%Y%m") <= datetime(year, month, 1)]

    most_recent_address = addresses[-1]

    return most_recent_address

def GFS_date_addresses_from_URL(month_URL: str = None) -> List[str]:
    if month_URL is None:
        month_URL = GFS_most_recent_month_address()
        
    response = requests.get(month_URL)
    soup = BeautifulSoup(response.text, "html.parser")
    links = [link.get("href") for link in soup.find_all("a")]
    directories = [link for link in links if re.compile("[0-9]{8}").match(link)]
    addresses = sorted([posixpath.join(month_URL, directory) for directory in directories])
    
    return addresses

def GFS_date_addresses(year: int = None, month: int = None) -> List[str]:
    month_URL = GFS_most_recent_month_address(year=year, month=month)
    addresses = GFS_date_addresses_from_URL(month_URL)

    return addresses

def GFS_most_recent_date_address(date_UTC: date = None) -> str:
    if date_UTC is None:
        date_UTC = datetime.utcnow()
    
    if isinstance(date_UTC, str):
        date_UTC = parser.parse(date_UTC).date()
    
    if isinstance(date_UTC, datetime):
        date_UTC = date_UTC.date()
    
    addresses = GFS_date_addresses(year=date_UTC.year, month=date_UTC.month)
    addresses = [address for address in addresses if datetime.strptime(posixpath.basename(address.strip("/")), "%Y%m%d").date() <= date_UTC]
    address = addresses[-1]

    return address

def GFS_file_addresses(date_URL: str = None, pattern: str = None) -> List[str]:
    if date_URL is None:
        date_URL = GFS_most_recent_date_address()
    
    if pattern is None:
        pattern = "*.grb2"
    
    response = requests.get(date_URL)
    soup = BeautifulSoup(response.text, "html.parser")
    links = [link.get("href") for link in soup.find_all("a")]
    filenames = [link for link in links if fnmatch(link, pattern)]
    addresses = [posixpath.join(date_URL, filename) for filename in filenames]
    
    return addresses

def get_GFS_listing(date_URL: str = None) -> pd.DataFrame:
    if date_URL is None:
        date_URL = GFS_most_recent_date_address()
    
    addresses = GFS_file_addresses(date_URL)
    address_df = pd.DataFrame({"address": addresses})
    address_df["basename"] = address_df.address.apply(lambda address: posixpath.basename(address))
    address_df["source_date_UTC"] = address_df.basename.apply(lambda basename: datetime.strptime(basename.split("_")[2], "%Y%m%d"))
    address_df["source_hour"] = address_df.basename.apply(lambda basename: int(basename.split("_")[3][:2]))
    address_df["source_datetime_UTC"] = address_df.apply(lambda row: row.source_date_UTC + timedelta(hours=row.source_hour), axis=1)
    address_df["forecast_hours"] = address_df.basename.apply(lambda basename: int(splitext(basename)[0].split("_")[-1]))
    address_df["forecast_time_UTC"] = address_df.apply(lambda row: row.source_datetime_UTC + timedelta(hours=row.forecast_hours), axis=1)
    address_df.sort_values(by=["forecast_time_UTC", "source_hour"], inplace=True)
    address_df.drop_duplicates(subset="forecast_time_UTC", keep="last", inplace=True)
    address_df = address_df[["forecast_time_UTC", "address"]]
    
    return address_df

def GFS_before_after_addresses(time_UTC: datetime, listing: pd.DataFrame = None) -> pd.DataFrame:
    if not isinstance(time_UTC, datetime):
        time_UTC = parser.parse(time_UTC)

    if listing is None:
        listing = get_GFS_listing()

    # if len(listing) == 0:
    #     raise ValueError(f"zero-length GFS listing at time {time_UTC} UTC")
    #
    # min_time = min(listing.forecast_time_UTC)
    # max_time = max(listing.forecast_time_UTC)

    # logger.info(f"selecting GFS files for time {time_UTC} UTC between {min_time} UTC and {max_time} UTC")

    # before = listing[listing.forecast_time_UTC <= time_UTC].iloc[[-1]]

    before = listing[listing.forecast_time_UTC.apply(lambda forecast_time_UTC: str(forecast_time_UTC) <= str(time_UTC))].iloc[[-1]]
    after = listing[listing.forecast_time_UTC.apply(lambda forecast_time_UTC: str(forecast_time_UTC) > str(time_UTC))].iloc[[0]]
    before_after = pd.concat([before, after])

    return before_after

def GFS_download(URL: str, filename: str = None, directory: str = None) -> str:
    if directory is None:
        directory = "."

    date_UTC = datetime.strptime(posixpath.basename(URL).split("_")[2], "%Y%m%d").date()
    directory = expanduser(directory)

    target_directory = join(directory, date_UTC.strftime("%Y-%m-%d"))
    makedirs(target_directory, exist_ok=True)

    if filename is None:
        filename = join(target_directory, posixpath.basename(URL))
    
    if exists(filename):
        logger.info(f"file already downloaded: {colored_logging.file(filename)}")
        return filename
    
    logger.info(f"downloading URL: {colored_logging.URL(URL)}")

    partial_filename = filename + ".download"

    command = f'wget -c -O "{partial_filename}" "{URL}"'
    logger.info(command)
    system(command)

    move(partial_filename, filename)

    if not exists(filename):
        raise ConnectionError(f"unable to download URL: {URL}")

    logger.info(f"downloaded file: {colored_logging.file(filename)}")
    
    return filename

def read_GFS(filename: str, message: int, geometry: rt.RasterGeometry = None, resampling = "cubic") -> rt.Raster:
    with pygrib.open(filename) as file:
        data = file.message(message).values

    rows, cols = data.shape
    data = np.roll(data, int(cols / 2), axis=1)
    grid = rt.RasterGrid.from_bbox(rt.BBox(-180, -90, 180, 90), data.shape)
    image = rt.Raster(data, geometry=grid)

    if geometry is not None:
        image = image.to_geometry(geometry, resampling=resampling)
    
    return image

def GFS_interpolate(
        message: int,
        time_UTC: datetime,
        geometry: rt.RasterGeometry = None,
        resampling: str = "cubic",
        directory: str = None,
        listing: pd.DataFrame = None) -> rt.Raster:
    before_after = GFS_before_after_addresses(time_UTC, listing=listing)
    
    before_address = before_after.iloc[0].address
    logger.info(f"before URL: {colored_logging.URL(before_address)}")
    before_time = parser.parse(str(before_after.iloc[0].forecast_time_UTC))
    before_filename = GFS_download(URL=before_address, directory=directory)
    
    try:
        before_image = read_GFS(filename=before_filename, message=message, geometry=geometry, resampling=resampling)
    except Exception as e:
        logger.warning(e)
        os.remove(before_filename)

    before_filename = GFS_download(URL=before_address, directory=directory)
    before_image = read_GFS(filename=before_filename, message=message, geometry=geometry, resampling=resampling)

    after_address = before_after.iloc[-1].address
    logger.info(f"after URL: {colored_logging.URL(after_address)}")
    after_time = parser.parse(str(before_after.iloc[-1].forecast_time_UTC))
    after_filename = GFS_download(URL=after_address, directory=directory)
    
    try:
        after_image = read_GFS(filename=after_filename, message=message, geometry=geometry, resampling=resampling)
    except Exception as e:
        logger.warning(e)
        os.remove(after_filename)

    after_filename = GFS_download(URL=after_address, directory=directory)
    after_image = read_GFS(filename=after_filename, message=message, geometry=geometry, resampling=resampling)
    
    source_diff = after_image - before_image
    time_fraction = (parser.parse(str(time_UTC)) - parser.parse(str(before_time))) / (parser.parse(str(after_time)) - parser.parse(str(before_time)))
    interpolated_image = before_image + source_diff * time_fraction

    return interpolated_image

def forecast_Ta_K(
        time_UTC: datetime,
        geometry: rt.RasterGeometry = None,
        resampling: str = "cubic",
        directory: str = None,
        listing: pd.DataFrame = None) -> rt.Raster:
    return GFS_interpolate(message=GFS_TA_MESSAGE, time_UTC=time_UTC, geometry=geometry, resampling=resampling, directory=directory, listing=listing)

def forecast_Ta_C(
        time_UTC: datetime,
        geometry: rt.RasterGeometry = None,
        resampling: str = "cubic",
        directory: str = None,
        listing: pd.DataFrame = None) -> rt.Raster:
    return forecast_Ta_K(time_UTC=time_UTC, geometry=geometry, resampling=resampling, directory=directory, listing=listing) - 273.15

def forecast_RH(
        time_UTC: datetime,
        geometry: rt.RasterGeometry = None,
        resampling: str = "cubic",
        directory: str = None,
        listing: pd.DataFrame = None) -> rt.Raster:
    return rt.clip(GFS_interpolate(message=GFS_RH_MESSAGE, time_UTC=time_UTC, geometry=geometry, resampling=resampling, directory=directory, listing=listing) / 100, 0, 1)

def forecast_SM(
        time_UTC: datetime,
        geometry: rt.RasterGeometry = None,
        resampling: str = "cubic",
        directory: str = None,
        listing: pd.DataFrame = None) -> rt.Raster:
    SM = rt.clip(GFS_interpolate(message=GFS_SM_MESSAGE, time_UTC=time_UTC, geometry=geometry, resampling=resampling, directory=directory, listing=listing) / 10000, 0, 1)
    SM.cmap = SM_CMAP

    return SM

def forecast_SWin(
        time_UTC: datetime,
        geometry: rt.RasterGeometry = None,
        resampling: str = "cubic",
        directory: str = None,
        listing: pd.DataFrame = None) -> rt.Raster:
    return rt.clip(GFS_interpolate(message=GFS_SWIN_MESSAGE, time_UTC=time_UTC, geometry=geometry, resampling=resampling, directory=directory, listing=listing), 0, None)

def forecast_wind(
        time_UTC: datetime,
        geometry: rt.RasterGeometry = None,
        resampling: str = "cubic",
        directory: str = None,
        listing: pd.DataFrame = None) -> rt.Raster:
    U = GFS_interpolate(message=GFS_U_WIND_MESSAGE, time_UTC=time_UTC, geometry=geometry, resampling=resampling, directory=directory, listing=listing)
    V = GFS_interpolate(message=GFS_V_WIND_MESSAGE, time_UTC=time_UTC, geometry=geometry, resampling=resampling, directory=directory, listing=listing)
    wind_speed = rt.clip(np.sqrt(U ** 2.0 + V ** 2.0), 0.0, None)

    return wind_speed
