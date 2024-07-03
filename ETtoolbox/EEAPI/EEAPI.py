"""
This module handles searching URLs in the Earth Explorer API.

Developed by Gregory Halverson at the Jet Propulsion Laboratory.
"""
import json
import logging
import os
import tarfile
from datetime import datetime, date
from glob import glob
from os import makedirs, system
from os.path import join, dirname, expanduser, splitext, exists, abspath
from shutil import move
from time import perf_counter
from typing import List
from urllib.parse import urljoin

import geopandas as gpd
import pandas as pd
import requests
import shapely.wkt
from dateutil import parser
from shapely.geometry import Point, Polygon, shape

import colored_logging
from ETtoolbox.ERS_credentials import get_ERS_credentials
from rasters import RasterGrid
from time import sleep

from ETtoolbox.M2M_credentials.M2M_credentials import get_M2M_credentials

class M2MAPIUnavailableError(Exception):
    pass


class EEAPI:
    logger = logging.getLogger(__name__)

    _M2MHOST = "https://m2m.cr.usgs.gov/api/api/json/stable/"
    _DEFAULT_MAX_RESULTS = 1000
    _LOGIN_ENDPOINT = "login"
    _LOGOUT_ENDPOINT = "logout"
    _SCENE_SEARCH_ENDPOINT = "scene-search"
    _DOWNLOAD_OPTIONS_ENDPOINT = "download-options"
    _DOWNLOAD_REQUEST_ENDPOINT = "download-request"
    _DOWNLOAD_RETRIEVE_ENDPOINT = "download-retrieve"
    _DEFAULT_DOWNLOAD_DIRECTORY = "earth_explorer_download"

    def __init__(
            self,
            username: str = None,
            password: str = None,
            API_key: str = None,
            host_URL: str = None,
            download_directory: str = None):
        if host_URL is None:
            host_URL = self._M2MHOST

        if username is None or password is None:
            credentials = get_M2M_credentials()
            username = credentials["username"]
            password = credentials["password"]

        self._username = username
        self._password = password
        self._API_key = API_key

        self.host_URL = host_URL

        if download_directory is None:
            download_directory = self._DEFAULT_DOWNLOAD_DIRECTORY

        self.download_directory = expanduser(download_directory)

    def __enter__(self):
        self.login()
        return self

    def __exit__(self, type, value, tb):
        self.logout()

    def __repr__(self):
        return json.dumps(
            {
                "host": self.host_URL,
                # "key": self.API_key,
                "download_directory": self.download_directory
            },
            indent=2
        )

    def request(self, URL: str, request_dict: dict, retries: int = 3, wait_seconds: int = 30) -> dict:
        while retries > 0:
            retries -= 1

            try:
                request_JSON = json.dumps(request_dict)

                if self._API_key is None:
                    response = requests.post(URL, request_JSON)
                else:
                    headers = {'X-Auth-Token': self.API_key}
                    response = requests.post(URL, request_JSON, headers=headers)

                if response is None:
                    raise IOError(f"no response from URL: {URL}")

                HTTP_status_code = response.status_code

                if not HTTP_status_code == 200:
                    self.logger.warning(f"response code {HTTP_status_code} from URL: {URL}")

                if HTTP_status_code == 503:
                    raise M2MAPIUnavailableError(f"M2M API is unavailable ({HTTP_status_code}) at URL: {URL}")

                try:
                    response_dict = json.loads(response.text)
                except Exception as e:
                    self.logger.exception(e)
                    raise M2MAPIUnavailableError(f"unable to parse response: {response.text}")

                if "errorCode" in response_dict:
                    error_code = response_dict["errorCode"]
                else:
                    error_code = None

                if "errorMessage" in response_dict:
                    error_message = response_dict["errorMessage"]
                else:
                    error_message = None

                if error_code is not None and error_message is not None:
                    error_code = response_dict['errorCode']
                    error_message = response_dict['errorMessage']

                    raise IOError(f"{error_code}: {error_message}")

                if HTTP_status_code == 404:
                    raise M2MAPIUnavailableError(f"HTTP 404 not found: {URL}")

                elif HTTP_status_code == 403:
                    raise M2MAPIUnavailableError(f"HTTP 403 forbidden: {URL}")

                elif HTTP_status_code == 401:
                    raise M2MAPIUnavailableError(f"HTTP 401 unauthorized: {URL}")

                elif HTTP_status_code == 400:
                    raise M2MAPIUnavailableError(f"HTTP 400: {URL}")

                response.close()
                result = response_dict['data']

                return result
            except Exception as e:
                if retries == 0:
                    raise e

                self.logger.warning(e)
                self.logger.warning(f"waiting {wait_seconds} for M2M retry")
                sleep(wait_seconds)
                continue

    @property
    def login_URL(self):
        return urljoin(self.host_URL, self._LOGIN_ENDPOINT)

    def login(self):
        request_dict = {
            "username": self._username,
            "password": self._password
        }

        URL = self.login_URL
        self._API_key = self.request(URL, request_dict)

    @property
    def API_key(self):
        if self._API_key is None:
            self.login()

        return self._API_key

    @property
    def logout_URL(self):
        return urljoin(self.host_URL, self._LOGOUT_ENDPOINT)

    def logout(self):
        if self._API_key is None:
            return
        else:
            self.request(self.logout_URL, {"apiKey": self.API_key})

    @property
    def scene_search_URL(self):
        return urljoin(self.host_URL, self._SCENE_SEARCH_ENDPOINT)

    def scene_search_API(self, request_dict: dict) -> dict:
        return self.request(self.scene_search_URL, request_dict)

    def scene_search(
            self,
            start_date: date or datetime or str,
            target_geometry: Point or Polygon or RasterGrid,
            datasets: str or list,
            end_date: date or datetime or str = None,
            max_results: int = None,
            cloud_percent_min: float = 0,
            cloud_percent_max: float = 100,
            ascending: bool = True):
        if isinstance(start_date, str):
            start_date = parser.parse(start_date).date()

        if end_date is None:
            end_date = start_date

        if isinstance(end_date, str):
            end_date = parser.parse(end_date).date()

        if isinstance(datasets, str):
            datasets = [datasets]

        if max_results is None:
            max_results = self._DEFAULT_MAX_RESULTS

        if isinstance(target_geometry, str):
            target_geometry = shapely.wkt.loads(target_geometry)

        if isinstance(target_geometry, str):
            target_geometry = shapely.wkt.loads(target_geometry)

        if isinstance(target_geometry, Point):
            lon = target_geometry.x
            lat = target_geometry.y

            lower_left = {
                "latitude": lat,
                "longitude": lon
            }

            upper_right = {
                "latitude": lat,
                "longitude": lon
            }

        elif isinstance(target_geometry, Polygon):
            x_min, y_min, x_max, y_max = target_geometry.bounds

            lower_left = {
                "latitude": y_min,
                "longitude": x_min
            }

            upper_right = {
                "latitude": y_max,
                "longitude": x_max
            }
        elif isinstance(target_geometry, RasterGrid):
            x_min, y_min, x_max, y_max = target_geometry.bbox_latlon

            lower_left = {
                "latitude": y_min,
                "longitude": x_min
            }

            upper_right = {
                "latitude": y_max,
                "longitude": x_max
            }
        else:
            raise ValueError("invalid target geometry for EE search")

        results_dict_list = []

        spatial_filter = {
            "filterType": "mbr",
            "lowerLeft": lower_left,
            "upperRight": upper_right
        }

        cloud_cover_filter = {
            "max": cloud_percent_max,
            "min": cloud_percent_min,
            "includeUnknown": True
        }

        acquisition_filter = {
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d")
        }

        for dataset in datasets:
            request_dict = {
                "maxResults": max_results,
                "datasetName": dataset,
                "node": "EE",
                "apiKey": self.API_key,
                "sceneFilter": {
                    "spatialFilter": spatial_filter,
                    "metadataFilter": None,
                    "cloudCoverFilter": cloud_cover_filter,
                    "acquisitionFilter": acquisition_filter
                },
                "metadataType": "summary",
                "sortDirection": "ASC" if ascending else "DESC",
                "sortField": "displayId",
                "startingNumber": 1
            }

            response_dict = self.scene_search_API(request_dict)

            for results_dict in response_dict["results"]:
                if len(results_dict.keys()) > 0:
                    results_dict_list.append(results_dict)

        listing = pd.DataFrame(results_dict_list)

        if len(listing) == 0:
            return listing

        listing["date_UTC"] = listing["temporalCoverage"].apply(
            lambda temporal_coverage: parser.parse(temporal_coverage["startDate"]).date())
        listing["display_ID"] = listing["displayId"]
        listing["sensor"] = listing["display_ID"].apply(lambda display_ID: display_ID.split("_")[0])
        listing["entity_ID"] = listing["entityId"]
        listing["cloud"] = listing["cloudCover"]

        geometry = listing["spatialCoverage"].apply(lambda spatial_coverage: shape(spatial_coverage))

        listing = listing[[
            "date_UTC",
            "display_ID",
            "entity_ID",
            "cloud"
        ]]

        listing = gpd.GeoDataFrame(listing, geometry=geometry, crs="EPSG:4326")
        listing = listing.sort_values(by=["date_UTC", "display_ID"], ascending=ascending)

        return listing

    @property
    def download_options_URL(self):
        return urljoin(self.host_URL, self._DOWNLOAD_OPTIONS_ENDPOINT)

    def download_options_API(self, request_dict: dict) -> dict:
        return self.request(self.download_options_URL, request_dict)

    def download_options(
            self,
            dataset: str,
            entity_IDs: List[str] or str,
            granule_systems: str = None,
            band_systems: str = None):
        if isinstance(entity_IDs, str):
            entity_IDs = [entity_IDs]

        if isinstance(granule_systems, str):
            granule_systems = [granule_systems]

        if isinstance(band_systems, str):
            band_systems = [band_systems]

        request_dict = {
            "datasetName": dataset,
            "entityIds": entity_IDs
        }

        # print("request_dict")
        # print(request_dict)

        response_dict = self.download_options_API(request_dict)

        # print("response_dict")
        # print(response_dict)

        granule_item_dict_list = []
        band_item_dict_list = []

        for granule_item_dict in response_dict:
            # print("granule_item_dict")
            # print(granule_item_dict)

            if not granule_item_dict["available"]:
                # print("unavailable")
                continue

            if granule_systems is not None and granule_item_dict["downloadSystem"] not in granule_systems:
                # print(f'granule system {granule_item_dict["downloadSystem"]} not in {", ".join(granule_systems)}')
                continue

            band_items = granule_item_dict.pop("secondaryDownloads")
            granule_item_dict_list.append(granule_item_dict)
            granule_ID = granule_item_dict["displayId"]

            for band_item_dict in band_items:
                if band_systems is not None and band_item_dict["downloadSystem"] not in band_systems:
                    continue

                del (band_item_dict["secondaryDownloads"])
                band_item_dict["granule_ID"] = granule_ID
                band_item_dict_list.append(band_item_dict)

        # print(granule_item_dict_list)

        granule_items = pd.DataFrame(granule_item_dict_list)

        # print(granule_items)

        granule_items = granule_items.rename(columns={
            "id": "product_ID",
            "displayId": "display_ID",
            "entityId": "entity_ID",
            "datasetId": "dataset_ID"
        })

        # print(granule_items)

        granule_items["granule_ID"] = granule_items["display_ID"]
        band_items = pd.DataFrame(band_item_dict_list)

        band_items = band_items.rename(columns={
            "id": "product_ID",
            "displayId": "display_ID",
            "entityId": "entity_ID",
            "datasetId": "dataset_ID"
        })

        return granule_items, band_items

    @property
    def download_request_URL(self):
        return urljoin(self.host_URL, self._DOWNLOAD_REQUEST_ENDPOINT)

    def download_request_API(self, request_dict: dict) -> dict:
        return self.request(self.download_request_URL, request_dict)

    def download_request(self, downloads: pd.DataFrame):
        return self.download_request_API({
            "downloads": [
                {
                    "label": f"{item.entity_ID}-{item.product_ID}",
                    "productId": item.product_ID,
                    "entityId": item.entity_ID
                }
                for i, item
                in downloads.iterrows()
            ],
            "downloadApplication": "EE"
        })

    @property
    def download_retrieve_URL(self):
        return urljoin(self.host_URL, self._DOWNLOAD_RETRIEVE_ENDPOINT)

    def download_retrieve_API(self, request_dict: dict) -> dict:
        return self.request(self.download_request_URL, request_dict)

    def download_URL(self, product_ID: str, entity_ID: str) -> str or None:
        request_dict = {
            "downloads": [
                {
                    "productId": product_ID,
                    "entityId": entity_ID
                }
            ]
        }

        response_dict = self.download_request_API(request_dict)

        if "availableDownloads" in response_dict and len(response_dict["availableDownloads"]) > 0:
            downloads = response_dict["availableDownloads"]
        elif "preparingDownloads" in response_dict and len(response_dict["preparingDownloads"]) > 0:
            downloads = response_dict["preparingDownloads"]
        else:
            return None

        if len(downloads) == 0:
            return None

        if "url" not in downloads[0]:
            return None

        URL = downloads[0]["url"]

        return URL

    def granule_URLs(
            self,
            dataset: str,
            entity_IDs: List[str],
            granule_systems: str = None,
            band_systems: str = None):

        granules, bands = self.download_options(
            dataset=dataset,
            entity_IDs=entity_IDs,
            granule_systems=granule_systems,
            band_systems=band_systems
        )

        granules["URL"] = granules.apply(
            lambda item: self.download_URL(product_ID=item.product_ID, entity_ID=item.entity_ID), axis=1)

        return granules

    def band_URLs(
            self,
            dataset: str,
            entity_IDs: List[str] or str,
            band_names: List[str] = None,
            granule_system: str = None,
            band_system: str = None):
        if isinstance(entity_IDs, str):
            entity_IDs = [entity_IDs]

        granules, bands = self.download_options(
            dataset=dataset,
            entity_IDs=entity_IDs,
            granule_systems=granule_system,
            band_systems=band_system
        )

        if len(bands) == 0:
            return None

        if band_names is not None:
            def identify_band(entity_ID: str, band_names: List[str]) -> str or None:
                for band_name in band_names:
                    if band_name in entity_ID:
                        return band_name

                return None

            bands["band"] = bands.entity_ID.apply(lambda entity_ID: identify_band(entity_ID, band_names))
            bands = bands[bands.band.apply(lambda band: band is not None)]

        bands["URL"] = bands.apply(
            lambda item: self.download_URL(product_ID=item.product_ID, entity_ID=item.entity_ID),
            axis=1
        )

        bands = bands[bands.URL.apply(lambda URL: URL is not None)]
        bands["filename"] = bands.display_ID.apply(
            lambda display_ID: f"{splitext(display_ID)[0]}{splitext(display_ID)[1].lower()}")

        return bands

    def download_file(self, URL: str, filename: str):
        if exists(filename):
            self.logger.info(f"file already downloaded: {colored_logging.file(filename)}")
            return filename

        self.logger.info(f"downloading: {colored_logging.URL(URL)} -> {colored_logging.file(filename)}")
        directory = dirname(filename)
        makedirs(directory, exist_ok=True)
        partial_filename = f"{filename}.download"
        command = f'wget -c -O "{partial_filename}" "{URL}"'
        download_start = perf_counter()
        system(command)
        download_end = perf_counter()
        download_duration = download_end - download_start
        self.logger.info(
            "completed download in " + colored_logging.val(f"{download_duration:0.2f}") + " seconds: " + colored_logging.file(filename))

        if not exists(partial_filename):
            raise IOError(f"unable to download URL: {URL}")

        move(partial_filename, filename)

        if not exists(filename):
            raise IOError(f"failed to download file: {filename}")

        return filename

    def date_directory(self, dataset: str, date_UTC: date) -> str:
        return join(self.download_directory, dataset, f"{date_UTC:%Y-%m-%d}")

    def granule_directory(self, dataset: str, date_UTC: date, granule_ID: str) -> str:
        return join(self.date_directory(dataset, date_UTC), granule_ID)

    def download_bands(
            self,
            dataset: str,
            date_UTC: date or str,
            entity_ID: str,
            bands: List[str] or str = None) -> pd.DataFrame or None:
        if isinstance(date_UTC, str):
            date_UTC = parser.parse(date_UTC).date()

        if isinstance(bands, str):
            bands = [bands]

        band_listing = self.band_URLs(
            dataset=dataset,
            entity_IDs=entity_ID,
            band_names=bands
        )

        if band_listing is None:
            return None

        results = []

        include_band_names = "band" in band_listing.columns

        for i, band_item in band_listing.iterrows():
            if include_band_names:
                band_name = band_item.band
            else:
                band_name = None

            granule_ID = band_item.granule_ID
            URL = band_item.URL
            # filename = band_item.directory
            filename = band_item.filename
            filename_extension = splitext(filename)[1]
            filename = filename.replace(filename_extension, filename_extension.lower())
            directory = self.granule_directory(dataset=dataset, date_UTC=date_UTC, granule_ID=granule_ID)
            download_filename = join(directory, filename)

            try:
                self.download_file(URL, download_filename)
            except Exception as e:
                self.logger.exception(e)
                self.logger.warning(f"failed to download file: {download_filename}")
                continue

            if not exists(download_filename):
                self.logger.warning(f"failed to download file: {download_filename}")

            if include_band_names:
                results.append([dataset, date_UTC, entity_ID, band_name, download_filename])
            else:
                results.append([dataset, date_UTC, entity_ID, download_filename])

        if include_band_names:
            columns = ["dataset", "date_UTC", "entity_ID", "band", "filename"]
        else:
            columns = ["dataset", "date_UTC", "entity_ID", "filename"]

        results = pd.DataFrame(results, columns=columns)

        return results

    def retrieve_granule(
            self,
            dataset: str,
            date_UTC: date or str,
            granule_ID: str,
            entity_ID: str,
            bands: List[str] = None) -> str or None:
        if isinstance(date_UTC, str):
            date_UTC = parser.parse(date_UTC).date()

        granule_directory = self.granule_directory(dataset=dataset, date_UTC=date_UTC, granule_ID=granule_ID)

        if self.validate_granule_retrieval(
                dataset=dataset,
                date_UTC=date_UTC,
                granule_ID=granule_ID,
                band_names=bands):
            self.logger.info(f"granule {colored_logging.val(granule_ID)} already retrieved: {colored_logging.dir(granule_directory)}")

            return granule_directory

        band_listing = None

        if bands is not None:
            self.logger.info(
                "attempting to download" +
                " bands: " + colored_logging.val(', '.join(bands)) +
                " entity ID: " + colored_logging.val(entity_ID)
            )

            band_listing = self.download_bands(
                dataset=dataset,
                date_UTC=date_UTC,
                entity_ID=entity_ID,
                bands=bands
            )

        if bands is not None and band_listing is None:
            self.logger.info(
                "unable to directly download" +
                " bands: " + colored_logging.val(', '.join(bands)) +
                " entity ID: " + colored_logging.val(entity_ID)
            )

        if bands is None or band_listing is None:
            granule_listing = self.granule_URLs(dataset=dataset, entity_IDs=[entity_ID])

            if len(granule_listing) == 0:
                return

            granule = granule_listing.iloc[0]

            if granule.granule_ID != granule_ID:
                raise ValueError(f"entity ID: {entity_ID} does not match granule ID: {granule_ID}")

            URL = granule.URL
            tarfile_directory = self.date_directory(dataset=dataset, date_UTC=date_UTC)
            filename = f"{granule_ID}.tar"
            tarfile_filename = join(tarfile_directory, filename)

            if self.validate_granule_retrieval(
                    dataset=dataset,
                    date_UTC=date_UTC,
                    granule_ID=granule_ID,
                    band_names=bands):
                self.logger.info(f"granule {colored_logging.val(granule_ID)} already retrieved: {colored_logging.dir(granule_directory)}")

                if exists(tarfile_filename):
                    self.logger.info("removing archive: " + colored_logging.file(tarfile_filename))

                return granule_directory

            try:
                self.logger.info(f"attempting download of entire granule: {colored_logging.val(entity_ID)} URL: {colored_logging.URL(URL)}")
                self.download_file(URL, tarfile_filename)

                with tarfile.open(tarfile_filename) as file:
                    pass

            except Exception as e:
                self.logger.exception(e)
                os.remove(tarfile_filename)
                raise IOError(f"failed to download file: {tarfile_filename}")

            try:
                self.logger.info(f"extracting: {colored_logging.file(tarfile_filename)} -> {colored_logging.dir(granule_directory)}")

                with tarfile.open(tarfile_filename) as file:
                    file.extractall(granule_directory)
                    filenames = sorted(glob(join(granule_directory, "*")))

                    self.logger.info(f"extraced {colored_logging.val(len(filenames))} files")

                    for filename in filenames:
                        filename_base, extension = splitext(filename)
                        extension_lower = extension.lower()
                        filename_lower = filename_base + extension_lower

                        if filename != filename_lower:
                            # self.logger.info("fixing filename extension: " + colored_logging.file(filename_lower))
                            os.rename(filename, filename_lower)

                        self.logger.info(f"* {colored_logging.file(filename_lower)}")

                self.logger.info("removing archive: " + colored_logging.file(tarfile_filename))
                os.remove(tarfile_filename)
            except Exception as e:
                self.logger.exception(e)
                raise IOError(f"unable to extract {tarfile_filename} to {granule_directory}")

        if not self.validate_granule_retrieval(dataset=dataset, date_UTC=date_UTC, granule_ID=granule_ID,
                                               band_names=bands):
            raise IOError("failed to retrieve granule: " + colored_logging.val(granule_ID))

        return granule_directory

    def dates_available(
            self,
            start_date: date or datetime or str,
            end_date: date or datetime or str,
            target_geometry: Point or Polygon or RasterGrid,
            datasets: str or list = None,
            sensor_names: List[str] or str = None,
            max_results: int = None,
            cloud_percent_min: float = 0,
            cloud_percent_max: float = 100):
        scenes = self.scene_search(
            start_date=start_date,
            end_date=end_date,
            target_geometry=target_geometry,
            datasets=datasets,
            sensor_names=sensor_names,
            max_results=max_results,
            cloud_percent_min=cloud_percent_min,
            cloud_percent_max=cloud_percent_max
        )

        dates = sorted(set(scenes.date_UTC))

        return dates

    def download(
            self,
            start: date or datetime or str,
            end: date or datetime or str,
            geometry: Point or Polygon or RasterGrid,
            datasets: str or list = None,
            sensors: List[str] or str = None,
            bands: List[str] or str = None,
            max_results: int = None,
            cloud_percent_min: float = 0,
            cloud_percent_max: float = 100) -> pd.DataFrame:

        self.logger.info(
            "searching scenes" +
            " from " + colored_logging.time(f"{start:%Y-%m-%d}") +
            " to " + colored_logging.time(f"{end:%Y-%m-%d}")
        )

        # if datasets is not None:
        #     self.logger.info(f"datasets: {', '.join(datasets)}")
        #
        # if sensors is not None:
        #     self.logger.info(f"sensors: {', '.join(sensors)}")
        #
        # if bands is not None:
        #     self.logger.info(f"bands: {', '.join(bands)}")

        scenes = self.scene_search(
            start_date=start,
            end_date=end,
            target_geometry=geometry,
            datasets=datasets,
            sensor_names=sensors,
            max_results=max_results,
            cloud_percent_min=cloud_percent_min,
            cloud_percent_max=cloud_percent_max
        )

        self.logger.info(f"found {colored_logging.val(len(scenes))} scenes")

        downloads = []

        for i, scene in scenes.iterrows():
            dataset = scene.dataset
            date_UTC = scene.date_UTC
            granule_ID = scene.granule_ID
            entity_ID = scene.entity_ID

            try:
                download = self.retrieve_granule(
                    dataset=dataset,
                    date_UTC=date_UTC,
                    granule_ID=granule_ID,
                    entity_ID=entity_ID,
                    bands=bands
                )
            except Exception as e:
                download = None
                self.logger.exception(e)
                self.logger.warning(f"failed to download granule: {granule_ID}")
                continue

            downloads.append(download)

        scenes["download"] = downloads

        return scenes

    def validate_granule_retrieval(self, dataset: str, date_UTC: date, granule_ID: str,
                                   band_names: List[str] = None) -> bool:
        directory = self.granule_directory(dataset=dataset, date_UTC=date_UTC, granule_ID=granule_ID)
        filenames = glob(join(directory, "*"))

        if band_names is None:
            return len(filenames) > 0
        else:
            for band in band_names:
                if len(glob(join(directory, f"*{band}*"))) == 0:
                    self.logger.warning(f"band {band} file not found for granule: {granule_ID}")
                    return False

            return True
