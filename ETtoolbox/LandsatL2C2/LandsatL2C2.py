import json
import logging
import sys
import tarfile
import warnings
from datetime import date, datetime
from glob import glob
from os import makedirs, remove, listdir
from os.path import splitext, join, abspath, dirname, basename, expanduser, isdir, exists
from pathlib import Path
from shutil import rmtree
from typing import List, Union
import shapely
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from dateutil import parser
from matplotlib.colors import LinearSegmentedColormap
from pyproj import Proj, transform
from shapely.geometry import Polygon, Point

import colored_logging
import rasters
import rasters as rt
from ETtoolbox.EEAPI import EEAPI
from ETtoolbox.WRS2 import WRS2Descending
from rasters import RasterGrid, Raster, RasterGeometry

WRS2_FILENAME = join(abspath(dirname(__file__)), "WRS2_descending_centroids.geojson")

NDVI_COLORMAP = LinearSegmentedColormap.from_list(name="NDVI",
                                                  colors=[(0, "#0000ff"), (0.4, "#000000"), (0.5, "#745d1a"), (0.6, "#e1dea2"), (0.8, "#45ff01"), (1, "#325e32")])

ALBEDO_COLORMAP = LinearSegmentedColormap.from_list(name="albedo", colors=["black", "white"])

DEFAULT_COLORMAP = "jet"

WGS84 = Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')

logger = logging.getLogger(__name__)


class UnavailableError(Exception):
    pass


def parse_path_row(pathrow):
    if isinstance(pathrow, str) and len(pathrow) == 6:
        path = int(pathrow[:3])
        row = int(pathrow[3:])
    elif isinstance(pathrow, str) and len(pathrow) == 7 and (pathrow[3] == "." or pathrow[3] == "/"):
        path = int(pathrow[:3])
        row = int(pathrow[4:])
    else:
        raise ValueError(f"invalid Landsat path/row identifier: {pathrow}")

    return path, row


class BlankLandsatDNs(ValueError):
    pass


def parse_landsat_ID(landsat_id):
    """
    This function parses a Landsat ARD ID into its components
    sensor
        C: OLI/TIRS combined (Landsat 8)
        O: OLI-only (Landsat 8 shortwave)
        T: TIRS-only (Landsat 8 longwave)
        E: Enhanced Thematic Mapper (Landsat 7)
        T: Thematic Mapper (Landsat 4 or 5)
    satellite:
        4, 5, 7, 8
    grid:
        CU: Continental United States
        AK: Alaska
        HI: Hawaii
    h:
        horizontal tile index
    v:
        vertical tile index
    acquisition:
        date of satellite image acquisition
        YYYY-MM-DD
    production:
        date of product generation
        YYYY-MM-DD
    collection:
        collection number 1 or 2
    version:
        ARD version 1 or 2
    product_name:
        product name
    """
    sections = landsat_id.split("_")
    sensor = sections[0]
    satellite = int(sensor[2:4])
    level = sections[1]
    pathrow = sections[2]
    path = int(pathrow[:3])
    row = int(pathrow[3:])
    acquisition_date = parser.parse(sections[3]).strftime("%Y-%m-%d")
    production_date = parser.parse(sections[4]).strftime("%Y-%m-%d")
    collection = int(sections[5][1:3])
    version = int(sections[6][1:3])

    return {"sensor": sensor, "satellite": satellite, "tile": pathrow, "path": path, "row": row, "date_UTC": acquisition_date, "production_date": production_date,
            "collection": collection, "version": version, "level": level}


def load_coords(filename):
    """
    This function generates a latitude and longitude coordinate field from the affine transform in a Landsat 8 GeoTIFF.
    :param filename: GeoTIFF filename
    :return: two-tuple of two-dimensional arrays of longitude and latitude
    """
    with rasterio.open(filename) as f:
        affine = f.transform
        rows, cols = f.shape
        x, y = np.meshgrid(np.arange(cols), np.arange(rows)) * affine
        lon, lat = transform(Proj(f.crs), WGS84, x, y)

    return lon, lat


def evaluate_value(value):
    """
    This function evaluates a value read from a Landsat 8 MTL file.
    :param value:
    :return:
    """
    try:
        return int(value)
    except:
        pass

    try:
        return float(value)
    except:
        pass

    try:
        return value.strip().strip('"')
    except:
        pass

    return value


class MTL(object):
    """
    This class encapsulates a Landsat 8 metadata MTL text file.
    """

    def __init__(self, filename):
        """
        :param filename: filename of Landsat 8 metadata MTL text file
        """
        self.metadata = {}

        extension = splitext(basename(filename))[-1].lower()

        if extension == ".json":
            with open(filename, "r") as file:
                self.metadata = json.loads(file.read())

        elif extension == ".txt":
            parent = [None]
            group = self.metadata

            with open(filename) as f:
                for line in f.readlines():
                    if line.strip().startswith('GROUP'):
                        group_name = line.split('=')[-1].strip()
                        parent.append(group)
                        group[group_name] = {}
                        group = group[group_name]
                    elif line.strip().startswith('END_GROUP'):
                        group = parent.pop()
                    else:
                        split = line.split('=')
                        group[split[0].strip()] = evaluate_value(split[-1].strip())
        else:
            raise ValueError(f"unrecognized MTL file extension: {extension}")

    @property
    def date_acquired(self):
        return self.metadata["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"]["DATE_ACQUIRED"]

    @property
    def scene_center_time(self):
        return self.metadata["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"]["SCENE_CENTER_TIME"]

    def band_filename(self, band):
        return self.metadata['LANDSAT_METADATA_FILE']['PRODUCT_METADATA']['FILE_NAME_BAND_{}'.format(band)]

    @property
    def SR_parameters(self):
        return self.metadata['LANDSAT_METADATA_FILE']['LEVEL2_SURFACE_REFLECTANCE_PARAMETERS']

    def SR_mult(self, band):
        """
        This method loads the multiplicative factor to process TOA reflectance for a given band.
        :param band: band number
        :return: multiplicative factor
        """
        return self.SR_parameters['REFLECTANCE_MULT_BAND_{}'.format(band)]

    def SR_add(self, band):
        """
        This method loads the additive term to process TOA reflectance for a given band.
        :param band: band number
        :return: additive term
        """
        return self.SR_parameters['REFLECTANCE_ADD_BAND_{}'.format(band)]

    def SR_min(self, band):
        """
        This method loads the minimum reflectance value for a given band.
        :param band: band number
        :return: minimum reflectance value
        """
        return self.SR_parameters['REFLECTANCE_MINIMUM_BAND_{}'.format(band)]

    def SR_max(self, band):
        """
        This method loads the maximum reflectance value for a given band.
        :param band: band number
        :return: maximum reflectance value
        """
        return self.SR_parameters['REFLECTANCE_MAXIMUM_BAND_{}'.format(band)]

    def SR_DN_min(self, band):
        """
        This method loads the minimum reflectance DN value for a given band.
        :param band: band number
        :return: minimum reflectance value
        """
        return self.SR_parameters['QUANTIZE_CAL_MIN_BAND_{}'.format(band)]

    def SR_DN_max(self, band):
        """
        This method loads the maximum reflectance DN value for a given band.
        :param band: band number
        :return: maximum reflectance value
        """
        return self.SR_parameters['QUANTIZE_CAL_MAX_BAND_{}'.format(band)]

    @property
    def ST_parameters(self):
        return self.metadata['LANDSAT_METADATA_FILE']['LEVEL2_SURFACE_TEMPERATURE_PARAMETERS']

    def ST_max(self, band: int) -> float:
        return float(self.ST_parameters[f"TEMPERATURE_MAXIMUM_BAND_ST_B{band}"])

    def ST_min(self, band: int) -> float:
        return float(self.ST_parameters[f"TEMPERATURE_MINIMUM_BAND_ST_B{band}"])

    def ST_DN_max(self, band: int) -> float:
        return float(self.ST_parameters[f"QUANTIZE_CAL_MAXIMUM_BAND_ST_B{band}"])

    def ST_DN_min(self, band: int) -> float:
        return float(self.ST_parameters[f"QUANTIZE_CAL_MINIMUM_BAND_ST_B{band}"])

    def ST_mult(self, band: int) -> float:
        return float(self.ST_parameters[f"TEMPERATURE_MULT_BAND_ST_B{band}"])

    def ST_add(self, band: int) -> float:
        return float(self.ST_parameters[f"TEMPERATURE_ADD_BAND_ST_B{band}"])


class LandsatL2C2Granule(object):
    """
    This class encapsulates a Landsat ARD granule
    """

    _QA_BAND_NAME = "QA_PIXEL"
    _DEFAULT_PREVIEW_QUALITY = 20
    _DEFAULT_PRODUCTS_DIRECTORY = "LandsatL2C2_products"

    logger = logging.getLogger(__name__)

    def __init__(self, filename: str, products_directory: str = None, preview_quality: int = None):
        self._filename = filename

        if self.istar:
            with tarfile.TarFile(self.path) as tar:
                self._metadata = MTL(tar.extractfile(tar.getmember(self.metadata_filename)).read())
        elif self.isdir:
            self._metadata = MTL(self.metadata_filepath)
        else:
            raise ValueError("unrecognized granule filename '{}'".format(filename))

        self.tags = parse_landsat_ID(self.ID)
        self._rasterio_profile = None
        self._cloud = None
        self._water = None

        if products_directory is None:
            products_directory = self._DEFAULT_PRODUCTS_DIRECTORY

        products_directory = expanduser(products_directory)

        self.products_directory = products_directory

        if preview_quality is None:
            preview_quality = self._DEFAULT_PREVIEW_QUALITY

        self.preview_quality = preview_quality

    def __repr__(self):
        display_dict = {"filename": self.path, "products_directory": self.products_directory}

        display_string = json.dumps(display_dict, indent=2)

        return display_string

    def remove(self):
        self.logger.info("removing granule source: " + colored_logging.file(self.path))

        path = Path(self.path)

        if path.is_dir():
            rmtree(path)
        else:
            remove(path)

        parent = path.parent

        if len([item for item in listdir(parent) if not item.startswith(".")]) == 0:
            self.logger.info("removing empty directory: " + colored_logging.dir(parent))
            rmtree(parent)

    @property
    def time_UTC(self):
        return parser.parse(f"{self.MTL.date_acquired} {self.MTL.scene_center_time}")

    @property
    def date_UTC(self):
        return self.time_UTC.date()

    @property
    def path(self):
        """
        Location of ARD product as a directory or tar-file
        """
        return self._filename

    @property
    def isdir(self):
        """
        Is the product contained in a directory
        """
        return isdir(self.path)

    @property
    def istar(self):
        """
        Is the product contained in a tar-file
        """
        return self.extension == ".tar"

    @property
    def ID(self):
        """
        Landsat ARD granule ID
        """
        return splitext(basename(self.path))[0]

    @property
    def ID_base(self):
        """
        Landsat ARD granule ID without product name
        """
        return "_".join(self.ID.split("_")[:7])

    @property
    def metadata_filename(self):
        """
        Filename of XML metadata file
        """
        return "{}_MTL.txt".format(self.ID_base)

    @property
    def metadata_filepath(self):
        """
        Full path to XML metadata file, treating archive as directory
        """
        return join(self.path, self.metadata_filename)

    @property
    def MTL(self):
        """
        Metadata read from ARD metadata XML file, encapsulated in ARDMetadata class
        """
        return self._metadata

    @property
    def filenames(self):
        """
        List of filenames contained within ARD product
        """
        if self.isdir:
            return sorted(glob(join(self.path, "*")))
        elif self.istar:
            with tarfile.TarFile(self.path) as tar:
                return sorted([member.name for member in tar.getmembers()])

    @property
    def sensor(self):
        """
        Landsat sensor type:
            C: OLI/TIRS combined (Landsat 8)
            O: OLI-only (Landsat 8 shortwave)
            T: TIRS-only (Landsat 8 longwave)
            E: Enhanced Thematic Mapper (Landsat 7)
            T: Thematic Mapper (Landsat 4 or 5)
        """
        return self.tags["sensor"]

    @property
    def level(self):
        return self.tags["level"]

    @property
    def satellite(self):
        """
        Landsat satellite number:
            4, 5, 7, 8
        """
        return self.tags["satellite"]

    @property
    def grid(self):
        """
        Regional grid:
            CU: Continental United States
            AK: Alaska
            HI: Hawaii
        """
        return self.tags["grid"]

    @property
    def h(self):
        """
        Horizontal index of ARD tile
        """
        return self.tags["h"]

    @property
    def v(self):
        """
        Vertical index of ARD tile
        """
        return self.tags["v"]

    @property
    def region(self):
        return "{:03d}{:03d}".format(self.h, self.v)

    @property
    def acquisition_date(self):
        """
        Date of image acquisition on satellite
        """
        return parser.parse(self.tags["date_UTC"]).date()

    @property
    def acquisition_datetime(self):
        return datetime.combine(self.acquisition_date, self.scene_center_time)

    @property
    def production_date(self):
        """
        Date of ARD product generation at USGS
        """
        return self.tags["production_date"]

    @property
    def collection(self):
        """
        Landsat collection number
        """
        return self.tags["collection"]

    @property
    def version(self):
        """
        ARD version number
        """
        return self.tags["version"]

    @property
    def product_name(self):
        """
        ARD product name:
            ST: surface temperature
            SR: surface reflectance
        """
        return self.tags["product_name"]

    @property
    def extension(self):
        """
        File extension of product filename
        """
        return splitext(self.path)[1]

    def band_filename(self, band_name: str) -> str:
        return f"{self.ID_base}_{band_name}.tif"

    def band_filepath(self, band_name: str) -> str:
        return join(self.path, self.band_filename(band_name))

    def product_directory(self, product: str) -> str or None:
        if self.products_directory is None:
            return None
        else:
            return join(self.products_directory, product, f"{self.date_UTC:%Y.%m.%d}")

    def product_filename(self, product: str) -> str or None:
        if self.product_directory(product) is None:
            return None
        else:
            return join(self.product_directory(product),f"{self.ID_base}_{product}.tif")

    def save_product(self, image: Raster, product_name: str, save_preview: bool = True, product_filename: str = None, preview_filename: str = None,
                     preview_quality: int = None) -> str:

        if product_filename is None:
            product_filename = self.product_filename(product_name)

        if preview_filename is None:
            preview_filename = product_filename.replace(".tif", ".jpeg")

        if preview_quality is None:
            preview_quality = self.preview_quality

        self.logger.info(f"saving Landsat {colored_logging.val(product_name)}: {colored_logging.file(product_filename)}")
        image.to_geotiff(product_filename)

        if save_preview:
            self.logger.info(f"saving Landsat {colored_logging.val(product_name)} preview: {colored_logging.file(preview_filename)}")
            image.percentilecut.to_geojpeg(preview_filename, quality=preview_quality, remove_XML=True)

        return product_filename

    def SR_band(self, band: int) -> str:
        return f"SR_B{band}"

    def SR_filename(self, band: int) -> str:
        """
        Surface reflectance GeoTIFF filename
        :param band: band number
        :return: filename
        """
        return f"{self.ID_base}_{self.SR_band(band)}.TIF"

    def SR_filepath(self, band: int) -> str:
        """
        Surface reflectance GeoTIFF full file-path
        :param band: band number
        :return: filename
        """
        return join(self.path, self.SR_filename(band))

    def SR(self, band: int, apply_cloud: bool = True, apply_water: bool = False, apply_land: bool = False) -> Raster:
        band_name = self.SR_band(band=band)
        image = self.DN(band=band_name)

        if apply_cloud:
            image = rasters.where(self.cloud, np.nan, image)

        if apply_water:
            image = rasters.where(self.water, np.nan, image)
        elif apply_land:
            image = rasters.where(self.water, image, np.nan)

        image = rasters.where(image < self.MTL.SR_DN_min(band), np.nan, image)
        image = rasters.where(image > self.MTL.SR_DN_max(band), np.nan, image)
        image = image * self.MTL.SR_mult(band) + self.MTL.SR_add(band)
        image = rasters.where(image < self.MTL.SR_min(band), np.nan, image)
        image = rasters.where(image > self.MTL.SR_max(band), np.nan, image)

        return image

    @property
    def ST_band_number(self) -> int:
        if self.sensor in ("LC08", "LC09"):
            return 10
        else:
            return 6

    @property
    def ST_band_name(self) -> str:
        return f"ST_B{self.ST_band_number}"

    @property
    def ST_filename(self) -> str:
        """
        Surface temperature GeoTIFF filename
        :param band: band number
        :return: filename
        """
        return f"{self.ID_base}_{self.ST_band_name}.TIF"

    @property
    def ST_filepath(self) -> str:
        """
        Surface temperature GeoTIFF full file-path
        :param band: band number
        :return: filename
        """
        return join(self.path, self.ST_filename)

    def DN(self, band: int or str) -> Raster:
        """
        Product request_dict array
        """
        if isinstance(band, int):
            band = f"SR_B{band}"

        if self.isdir:
            filename = self.band_filepath(band)
            image = Raster.open(filename)

        elif self.istar:
            with tarfile.TarFile(self.path) as tar:
                member = tar.getmember(self.band_filename(band))
                extracted = str(tar.extractfile(member))
                image = Raster.open(extracted)

        return image

    @property
    def emissivity(self):
        DN = self.DN("ST_EMIS")
        fill_value = -9999
        scale_factor = 0.0001
        emissivity = rt.where(DN == fill_value, np.nan, DN * scale_factor)

        return emissivity

    @property
    def ST_DN(self):
        return self.DN(self.ST_band_name)

    @property
    def ST_max(self) -> float:
        return self.MTL.ST_max(self.ST_band_number)

    @property
    def ST_min(self) -> float:
        return self.MTL.ST_min(self.ST_band_number)

    @property
    def ST_DN_max(self) -> float:
        return self.MTL.ST_DN_max(self.ST_band_number)

    @property
    def ST_DN_min(self) -> float:
        return self.MTL.ST_DN_min(self.ST_band_number)

    @property
    def ST_mult(self) -> float:
        return self.MTL.ST_mult(self.ST_band_number)

    @property
    def ST_add(self) -> float:
        return self.MTL.ST_add(self.ST_band_number)

    def get_ST(self, apply_cloud: bool = True, apply_water: bool = False, apply_land: bool = False) -> Raster:
        image = self.ST_DN

        if apply_cloud:
            image = rasters.where(self.cloud, np.nan, image)

        if apply_water:
            image = rasters.where(self.water, np.nan, image)
        elif apply_land:
            image = rasters.where(self.water, image, np.nan)

        image = rasters.where(image < self.ST_DN_min, np.nan, image)
        image = rasters.where(image > self.ST_DN_max, np.nan, image)
        image = image * self.ST_mult + self.ST_add
        image = rasters.where(image < self.ST_min, np.nan, image)
        image = rasters.where(image > self.ST_max, np.nan, image)

        return image

    ST = property(get_ST)

    ST_K = ST

    @property
    def ST_C(self):
        return self.ST_K - 273.15

    @property
    def LST(self):
        return self.get_ST(apply_cloud=True, apply_water=True, apply_land=False)

    LST_K = LST

    @property
    def LST_C(self):
        return self.LST_K - 273.15

    @property
    def WST(self):
        return self.get_ST(apply_cloud=True, apply_water=False, apply_land=True)

    WST_K = WST

    @property
    def WST_C(self):
        return self.WST_K - 273.15

    @classmethod
    def product_cmap(cls, product_name: str):
        return LandsatL2C2.product_cmap(product_name)

    def product(self, product: str, save_data: bool = False, save_preview: bool = True, product_filename: str = None, preview_filename: str = None,
                preview_quality: int = None, geometry: RasterGeometry = None, return_filename: bool = False, return_raster: bool = True) -> Raster or (Raster, str):
        if product_filename is None:
            product_filename = self.product_filename(product)

        if product == "ST":
            product = "ST_K"
        elif product == "LST":
            product = "LST_K"
        elif product == "WST":
            product = "WST_K"

        if exists(product_filename):
            self.logger.info(f"loading Landsat {colored_logging.val(product)}: {colored_logging.file(product_filename)}")

            if return_raster:
                image = Raster.open(product_filename)
            else:
                image = None
        else:
            if product == "ST_K":
                image = self.ST_K
            elif product == "ST_C":
                image = self.ST_C
            elif product == "LST_K":
                image = self.LST_K
            elif product == "LST_C":
                image = self.LST_C
            elif product == "WST_K":
                image = self.WST_K
            elif product == "WST_C":
                image = self.WST_C
            elif product == "emissivity":
                image = self.emissivity
            elif product == "NDVI":
                image = self.NDVI
            elif product == "albedo":
                image = self.albedo
            elif product == "water":
                image = self.water
            else:
                raise ValueError(f"unrecognized product: {product}")

            if save_data:
                self.save_product(image=image, product_name=product, save_preview=save_preview, product_filename=product_filename, preview_filename=preview_filename,
                                  preview_quality=preview_quality)

        if geometry is not None:
            image = image.to_geometry(geometry)

        if image is not None:
            image.cmap = self.product_cmap(product)

        if return_filename:
            return image, product_filename
        else:
            return image

    @property
    def QA_filename(self) -> str:
        return self.band_filename(self._QA_BAND_NAME)

    @property
    def QA_filepath(self) -> str:
        return self.band_filepath(self._QA_BAND_NAME)

    @property
    def QA(self) -> Raster:
        # https://www.usgs.gov/core-science-systems/nli/landsat/landsat-collection-2-quality-assessment-bands
        return self.DN(self._QA_BAND_NAME)

    @property
    def cloud(self):
        if self._cloud is not None:
            return self._cloud

        # checking if dilated cloud, cirrus, cloud, or cloud shadow bits are set
        # https://www.usgs.gov/core-science-systems/nli/landsat/landsat-collection-2-quality-assessment-bands
        self._cloud = (self.QA >> 1) & 15 > 0

        return self._cloud

    @property
    def water(self):
        if self._water is not None:
            return self._water

        self._water = rasters.where((self.QA >> 7) & 1, True, False)

        return self._water

    @property
    def rasterio_profile(self):
        """
        rasterio profile dictionary loaded from BQA raster
        """
        if self._rasterio_profile is not None:
            return self._rasterio_profile

        if self.isdir:
            filename = self.band_filepath("QA")

            with rasterio.open(filename, "r") as f:
                self._rasterio_profile = f.profile

        elif self.istar:
            with tarfile.TarFile(self.path) as tar:
                member = tar.getmember(self.band_filename("QA"))
                extracted = tar.extractfile(member)

                with rasterio.open(extracted, "r") as f:
                    self._rasterio_profile = f.profile

        return self._rasterio_profile

    @property
    def cols(self):
        """
        Number of columns in raster
        """
        return self.rasterio_profile["width"]

    @property
    def rows(self):
        """
        Number of rows in raster
        """
        return self.rasterio_profile["height"]

    @property
    def crs(self):
        """
        Coordinate reference system
        """
        return self.rasterio_profile["crs"]

    @property
    def affine(self):
        """
        Affine transform of raster
        """
        return self.rasterio_profile["transform"]

    def to_geotiff(self, location, product_name, product_data=None, raw=False):
        if product_data is None:
            product_data = self.band(product_name, raw=raw)

        if splitext(location)[1] == ".tif":
            filename = location
        else:
            makedirs(location, exist_ok=True)
            filename = join(location, "{}_{}.tif".format(self.ID_base, self.product_name))

        profile = self.rasterio_profile

        if str(product_data.dtype) == "bool":
            product_data = product_data.astype("int16")

        profile["dtype"] = product_data.dtype

        with rasterio.open(filename, "w", **profile) as f:
            f.write(product_data, 1)

    @property
    def red(self):
        if self.satellite in (8, 9):
            return self.SR(4)
        elif self.satellite in (4, 5, 7):
            return self.SR(3)
        else:
            raise NotImplementedError("Landsat {} red not supported".format(self.satellite))

    @property
    def green(self):
        if self.satellite in (8, 9):
            return self.SR(3)
        elif self.satellite in (4, 5, 7):
            return self.SR(2)
        else:
            raise NotImplementedError("Landsat {} green not supported".format(self.satellite))

    @property
    def blue(self):
        if self.satellite in (8, 9):
            return self.SR(2)
        elif self.satellite in (4, 5, 7):
            return self.SR(1)
        else:
            raise NotImplementedError("Landsat {} blue not supported".format(self.satellite))

    @property
    def NIR(self):
        if self.satellite in (8, 9):
            return self.SR(5)
        elif self.satellite in (4, 5, 7):
            return self.SR(4)
        else:
            raise NotImplementedError("Landsat {} NIR not supported".format(self.satellite))

    @property
    def SWIR(self):
        if self.satellite in (8, 9):
            return self.SR(6)
        elif self.satellite in (4, 5, 7):
            return self.SR(5)
        else:
            raise NotImplementedError("Landsat {} SWIR not supported".format(self.satellite))

    @property
    def NDSI(self):
        warnings.filterwarnings("ignore")
        NDSI = (self.green - self.SWIR) / (self.green + self.SWIR)
        NDSI = np.clip(NDSI, -1, 1)
        return NDSI.astype(np.float32)

    @property
    def MNDWI(self):
        warnings.filterwarnings("ignore")
        MNDWI = (self.green - self.SWIR) / (self.green + self.SWIR)
        MNDWI = np.clip(MNDWI, -1, 1)
        return MNDWI.astype(np.float32)

    @property
    def NDWI(self):
        warnings.filterwarnings("ignore")
        NDWI = (self.green - self.NIR) / (self.green + self.NIR)
        NDWI = np.clip(NDWI, -1, 1)
        return NDWI.astype(np.float32)

    @property
    def WRI(self):
        warnings.filterwarnings("ignore")
        WRI = (self.green + self.red) / (self.NIR + self.SWIR)
        return WRI.astype(np.float32)

    @property
    def NDVI(self):
        warnings.filterwarnings("ignore")
        NDVI = (self.NIR - self.red) / (self.NIR + self.red)
        NDVI = np.clip(NDVI, -1, 1)
        NDVI = NDVI.astype(np.float32)
        NDVI.cmap = NDVI_COLORMAP

        return NDVI

    # @property
    # def water(self):
    #     return np.where(
    #         np.logical_and.reduce([
    #             self.NDWI > 0.0,
    #             self.MNDWI > 0.0,
    #             self.WRI > 1.0,
    #             self.NDVI < 0.0
    #         ]), 1, 0).astype("int16")

    @property
    def EVI(self):
        warnings.filterwarnings("ignore")
        evi = 2.5 * ((self.NIR - self.red) / (self.NIR + 6.0 * self.red - 7.5 * self.blue + 1.0))
        # evi = np.where(evi < 0.0, np.nan, evi)
        evi = np.clip(evi, 0, None)
        return evi.astype(np.float32)

    @property
    def albedo(self):
        warnings.filterwarnings("ignore")
        b = self.blue
        g = self.green
        r = self.red
        n = self.NIR
        s = self.SWIR
        albedo = ((0.356 * b) + (0.130 * g) + (0.373 * r) + (0.085 * n) + (0.072 * s) - 0.018) / 1.016
        albedo.nodata = np.nan
        albedo = rt.clip(albedo, 0, 1)
        albedo = albedo.astype(np.float32)
        albedo.cmap = "jet"

        return albedo


class LandsatL2C2(EEAPI):
    _LANDSAT_COLLECTION_2_DATASETS = ("landsat_tm_c2_l2", "landsat_etm_c2_l2", "landsat_ot_c2_l2")

    _GRANULE_DOWNLOAD_SYSTEM = ["dds_zip", "dds", "ls_zip"]
    _BAND_DOWNLOAD_SYSTEM = "dds"

    _DEFAULT_WORKING_DIRECTORY = "."
    _DEFAULT_DOWNLOAD_DIRECTORY = "landsat_download"
    _DEFAULT_PRODUCTS_DIRECTORY = "landsat_products"
    _DEFAULT_MOSAIC_DIRECTORY = "landsat_mosaic"

    def __init__(self, *args, working_directory: str = None, download_directory: str = None, products_directory: str = None, mosaic_directory: str = None,
                 preview_quality: int = None, remove_sources: bool = False, **kwargs):
        super(LandsatL2C2, self).__init__(*args, **kwargs)
        self.WRS2 = WRS2Descending()

        if working_directory is None:
            working_directory = self._DEFAULT_WORKING_DIRECTORY

        if download_directory is None:
            download_directory = join(working_directory, self._DEFAULT_DOWNLOAD_DIRECTORY)

        download_directory = expanduser(download_directory)

        if products_directory is None:
            products_directory = join(working_directory, self._DEFAULT_PRODUCTS_DIRECTORY)

        products_directory = expanduser(products_directory)

        if mosaic_directory is None:
            mosaic_directory = join(working_directory, self._DEFAULT_MOSAIC_DIRECTORY)

        mosaic_directory = expanduser(mosaic_directory)

        self.download_directory = download_directory
        self.products_directory = products_directory
        self.mosaic_directory = mosaic_directory
        self.preview_quality = preview_quality
        self.remove_sources = remove_sources

    def __repr__(self):
        return json.dumps({"host": self.host_URL, "key": self.API_key, "download_directory": self.download_directory, "products_directory": self.products_directory,
                                "mosaic_directory": self.mosaic_directory}, indent=2)

    def date_directory(self, dataset: str, date_UTC: date) -> str:
        return join(self.download_directory, f"{date_UTC:%Y-%m-%d}")

    def granule_directory(self, dataset: str, date_UTC: date, granule_ID: str) -> str:
        return join(self.date_directory(dataset, date_UTC), granule_ID)

    def mosaic_filename(self, product_name: str, time_UTC: Union[datetime, str], target_name: str, sensor: str):
        if isinstance(time_UTC, str):
            time_UTC = parser.parse(time_UTC)

        if not isinstance(time_UTC, date):
            raise ValueError(f"invalid date/time: {time_UTC}")

        if self.mosaic_directory is None:
            raise ValueError("no mosaic directory given")

        return join(self.mosaic_directory, target_name, product_name, f"{time_UTC:%Y.%m.%d}", f"{time_UTC:%Y.%m.%d.%H.%M.%S}_{sensor}_{target_name}_{product_name}.tif")

    def save_mosaic(self, image: Raster, product_name: str, date_UTC: date or str, target_name: str, sensor: str, save_preview: bool = True, product_filename: str = None,
                    preview_filename: str = None, preview_quality: int = None) -> str:

        if product_filename is None:
            product_filename = self.mosaic_filename(product_name=product_name, time_UTC=date_UTC, target_name=target_name, sensor=sensor)

        if preview_filename is None:
            preview_filename = product_filename.replace(".tif", ".jpeg")

        if preview_quality is None:
            preview_quality = self.preview_quality

        self.logger.info(f"saving Landsat {colored_logging.val(product_name)} mosaic: {colored_logging.file(product_filename)}")
        image.to_geotiff(product_filename)

        if save_preview:
            self.logger.info(f"saving Landsat {colored_logging.val(product_name)} mosaic preview: {colored_logging.file(preview_filename)}")
            image.percentilecut.to_geojpeg(preview_filename, quality=preview_quality, remove_XML=True)

        return product_filename

    def centroid(self, tile: str) -> Point:
        path, row = parse_path_row(tile)
        tile = f"{path:03d}.{row:03d}"
        centroid = self.WRS2.centroid(tile)

        return centroid

    def collection_2_datasets(self, start: date or str, end: date or str, sensors: List[str] = None) -> List[str]:
        LANDSAT_5 = "landsat_tm_c2_l2"
        LANDSAT_7 = "landsat_etm_c2_l2"
        LANDSAT_8 = "landsat_ot_c2_l2"

        if sensors is None:
            sensors = ["LT05", "LE07", "LC08"]

        datasets = []

        if isinstance(start, str):
            start = parser.parse(start).date()

        if isinstance(end, str):
            end = parser.parse(end).date()

        start_year = start.year
        end_year = end.year

        if "LT05" in sensors and end_year >= 1984 and start_year <= 2013:
            datasets.append(LANDSAT_5)

        if "LE07" in sensors and end_year >= 1999:
            datasets.append(LANDSAT_7)

        if "LC08" in sensors and end_year >= 2013:
            datasets.append(LANDSAT_8)

        return datasets

    def scene_search(self, start: date or datetime or str, end: date or datetime or str = None, tiles: List[str] = None,
                     target_geometry: Point or Polygon or RasterGrid = None, datasets: str or list = None, sensors: List[str] or str = None,
                     max_results: int = None, cloud_percent_min: float = 0, cloud_percent_max: float = 100, ascending: bool = True):
        if isinstance(start, str):
            start = parser.parse(start).date()

        if isinstance(end, str):
            end = parser.parse(end).date()

        if target_geometry is None and tiles is None:
            raise ValueError("no geometry or path/row given for scene search")

        # print(type(target_geometry))

        if isinstance(target_geometry, shapely.geometry.point.Point):
            target_geometry = rt.Point(target_geometry)

        if isinstance(target_geometry, shapely.geometry.polygon.Polygon):
            target_geometry = rt.Polygon(target_geometry)

        # print(type(target_geometry))
        # print(isinstance(target_geometry, rt.Polygon))

        if isinstance(target_geometry, rt.Point) or isinstance(target_geometry, rt.Polygon):
            target_vector = target_geometry
        elif isinstance(target_geometry, RasterGeometry):
            target_vector = target_geometry.boundary_latlon
        elif tiles is None:
            raise ValueError("no target vector")

        if tiles is None and target_geometry is not None:
            tiles = self.WRS2.tiles(target_vector, eliminate_redundancy=True).tile
            self.logger.info("Landsat path/row tiles: " + colored_logging.place(', '.join(tiles)))

        if isinstance(tiles, str):
            tiles = [tiles]

        if isinstance(sensors, str):
            sensors = [sensors]

        if datasets is None:
            datasets = self.collection_2_datasets(start=start, end=end, sensors=sensors)
        elif isinstance(datasets, str):
            datasets = [datasets]

        # scenes = None
        scenes = []

        for dataset in datasets:
            for tile in tiles:
                self.logger.info(f"searching dataset {colored_logging.val(dataset)} tile {colored_logging.place(tile)}" + " from " +
                                 colored_logging.time(f"{start:%Y-%m-%d}") + " to " + colored_logging.time(f"{end:%Y-%m-%d}"))

                search_results = super(LandsatL2C2, self).scene_search(start_date=start, end_date=end, target_geometry=self.WRS2.centroid(tile), datasets=dataset,
                                                                       max_results=max_results, cloud_percent_min=cloud_percent_min, cloud_percent_max=cloud_percent_max,
                                                                       ascending=ascending)

                self.logger.info(f"found {colored_logging.val(len(search_results))} scenes")

                search_results["dataset"] = dataset

                # if scenes is None:
                #     scenes = search_results
                # elif len(search_results) > 0:
                #     # FIXME replace deprecated append with concat
                #     scenes = scenes.append(search_results)

                scenes.append(search_results)

        scenes = pd.concat(scenes)

        if len(scenes) == 0:
            raise UnavailableError("no scenes found")

        geometry = scenes.pop("geometry")
        scenes["sensor"] = scenes.display_ID.apply(lambda display_ID: display_ID.split("_")[0])
        scenes["tile"] = scenes.display_ID.apply(lambda display_ID: display_ID.split("_")[2])
        scenes["date_UTC"] = scenes.display_ID.apply(lambda display_ID: parser.parse(display_ID.split("_")[3]).date())
        scenes["granule_ID"] = scenes["display_ID"]
        scenes = gpd.GeoDataFrame(scenes, geometry=geometry, crs="EPSG:4326")

        if sensors is not None:
            scenes = scenes[scenes.sensor.apply(lambda sensor: sensor in sensors)]

        scenes = scenes[scenes.date_UTC.apply(lambda date_UTC: date_UTC >= start or date_UTC <= end)]
        scenes = scenes.sort_values(by=["date_UTC", "display_ID"], ascending=ascending)

        if max_results is not None:
            scenes = scenes.iloc[:max_results]

        return scenes

    def granule_URLs(self, dataset: str, entity_IDs: List[str], sensor_names: List[str] or str = None, granule_systems: str = None, band_systems: str = None):
        if granule_systems is None:
            granule_systems = self._GRANULE_DOWNLOAD_SYSTEM

        if band_systems is None:
            band_systems = self._BAND_DOWNLOAD_SYSTEM

        if isinstance(sensor_names, str):
            sensor_names = [sensor_names]

        granules = super(LandsatL2C2, self).granule_URLs(dataset=dataset, entity_IDs=entity_IDs, granule_systems=granule_systems, band_systems=band_systems)

        date_UTC = granules.display_ID.apply(lambda display_ID: parser.parse(display_ID.split("_")[3]).date())

        if "date_UTC" not in granules.columns:
            granules.insert(0, "date_UTC", date_UTC)

        sensor = granules.display_ID.apply(lambda display_ID: display_ID.split("_")[0])
        granules.insert(1, "sensor", sensor)
        pathrow = granules.display_ID.apply(lambda display_ID: display_ID.split("_")[2])
        granules.insert(2, "tile", pathrow)
        granule_ID = granules.display_ID

        if "granule_ID" not in granules.columns:
            granules.insert(3, "granule_ID", granule_ID)

        if sensor_names is not None:
            granules = granules[granules.sensor.apply(lambda sensor: sensor in sensor_names)]

        return granules

    def translate_band_name(self, band_name: str, dataset: str) -> str:
        if dataset == "landsat_ot_c2_l2":
            if band_name == "blue":
                return "SR_B2"
            elif band_name == "green":
                return "SR_B3"
            elif band_name == "red":
                return "SR_B4"
            elif band_name == "NIR":
                return "SR_B5"
            elif band_name == "SWIR":
                return "SR_B6"
            elif band_name == "ST":
                return "ST_B10"
            elif band_name == "emissivity":
                return "ST_EMIS"
            else:
                return band_name
        elif dataset in ("landsat_tm_c2_l2", "landsat_etm_c2_l2"):
            if band_name == "blue":
                return "SR_B1"
            elif band_name == "green":
                return "SR_B2"
            elif band_name == "red":
                return "SR_B3"
            elif band_name == "NIR":
                return "SR_B4"
            elif band_name == "SWIR":
                return "SR_B5"
            elif band_name == "ST":
                return "ST_B6"
            else:
                return band_name
        else:
            return band_name

    def band_URLs(self, dataset: str, entity_IDs: List[str] or str, band_names: List[str] or str = None, granule_system: str = None, band_system: str = None):
        if granule_system is None:
            granule_system = self._GRANULE_DOWNLOAD_SYSTEM

        if band_system is None:
            band_system = self._BAND_DOWNLOAD_SYSTEM

        if band_names is not None:
            if "MTL" not in band_names:
                band_names.append("MTL")

            if "QA_PIXEL" not in band_names:
                band_names.append("QA_PIXEL")

        bands = super(LandsatL2C2, self).band_URLs(dataset=dataset, entity_IDs=entity_IDs, granule_system=granule_system, band_system=band_system)

        if bands is None or len(bands) == 0:
            return None

        date_UTC = bands.display_ID.apply(lambda display_ID: parser.parse(display_ID.split("_")[3]).date())
        bands.insert(0, "date_UTC", date_UTC)
        sensor = bands.display_ID.apply(lambda display_ID: display_ID.split("_")[0])
        bands.insert(1, "sensor", sensor)
        pathrow = bands.display_ID.apply(lambda display_ID: display_ID.split("_")[2])
        bands.insert(2, "tile", pathrow)
        band = bands.display_ID.apply(lambda display_ID: "_".join(splitext(display_ID)[0].split("_")[7:]))
        bands.insert(3, "band", band)
        granule_ID = bands.display_ID.apply(lambda display_ID: "_".join(splitext(display_ID)[0].split("_")[:7]))

        if "granule_ID" not in bands:
            bands.insert(4, "granule_ID", granule_ID)

        if band_names is not None:
            translated_band_names = [self.translate_band_name(band, dataset) for band in band_names]
            bands = bands[bands.band.apply(lambda band: band in translated_band_names)]

        return bands

    def retrieve_granule(self, dataset: str, date_UTC: date, granule_ID: str, entity_ID: str, bands: List[str] = None) -> LandsatL2C2Granule or None:

        if bands is None:
            self.logger.info(f"retrieving whole Landsat L2 C2 granule: {colored_logging.name(granule_ID)}")
        else:
            bands = [self.translate_band_name(band, dataset) for band in bands]
            self.logger.info(f"retrieving Landsat L2 C2 granule: {colored_logging.name(granule_ID)} bands: {', '.join(bands)}")

        directory = super(LandsatL2C2, self).retrieve_granule(dataset=dataset, date_UTC=date_UTC, granule_ID=granule_ID, entity_ID=entity_ID, bands=bands)

        if directory is None:
            return None

        granule = LandsatL2C2Granule(filename=directory, products_directory=self.products_directory, preview_quality=self.preview_quality)

        return granule

    def download(self, start_date: date or datetime or str, end_date: date or datetime or str = None, pathrow: str = None,
                 target_geometry: Point or Polygon or RasterGrid = None, datasets: str or list = None, band_names: List[str] or str = None,
                 sensor_names: List[str] or str = None, max_results: int = None, cloud_percent_min: float = 0, cloud_percent_max: float = 100) -> pd.DataFrame:

        if target_geometry is None and pathrow is None:
            raise ValueError("no target geometry or path/row given for scene search")
        elif target_geometry is None and isinstance(pathrow, str):
            target_geometry = self.centroid(pathrow)

        return super(LandsatL2C2, self).download(start=start_date, end=end_date, geometry=target_geometry, datasets=datasets, sensors=sensor_names, bands=band_names,
                                                 max_results=max_results, cloud_percent_min=cloud_percent_min, cloud_percent_max=cloud_percent_max)

    def required_bands(self, product_name: str) -> Union[List[str], None]:
        band_names = []

        if "ST" in product_name:
            band_names.append("ST")

        if "emissivity" in product_name:
            band_names.append("emissivity")

        if "NDVI" in product_name:
            band_names.append("red")
            band_names.append("NIR")

        if "albedo" in product_name:
            band_names.append("red")
            band_names.append("green")
            band_names.append("blue")
            band_names.append("NIR")
            band_names.append("SWIR")

        if len(band_names) == 0:
            return None
        else:
            return band_names

    def process_scene(self, product: str, dataset: str, date_UTC: date or str, granule_ID: str, entity_ID: str, geometry: RasterGeometry, band_names: List[str] = None,
                      return_raster: bool = True) -> rt.Raster:

        if band_names is None:
            band_names = self.required_bands(product)

        # self.logger.info(f"retrieving Landsat L2 C2 granule: {colored_logging.name(granule_ID)}")
        granule = self.retrieve_granule(dataset=dataset, date_UTC=date_UTC, granule_ID=granule_ID, entity_ID=entity_ID, bands=band_names)

        time_UTC = granule.time_UTC

        source_filename = granule.path
        self.logger.info(f"processing {colored_logging.val(product)} for granule: {colored_logging.val(granule_ID)}")

        image, product_filename = granule.product(product=product, geometry=geometry, return_filename=True, return_raster=return_raster)

        if self.remove_sources:
            granule.remove()

        pixel_count = np.count_nonzero(~np.isnan(image))
        self.logger.info(f"retrieved {colored_logging.val(product)} {colored_logging.val(pixel_count)} pixels from {colored_logging.val(granule_ID)}")

        return image, product_filename, source_filename, time_UTC

    def process(self, start: date or datetime or str, products: List[str], geometry: Point or Polygon or RasterGrid = None, target: str = None,
                tiles: List[str] = None, end: date or datetime or str = None, datasets: str or list = None, bands: List[str] or str = None, sensors: List[str] or str = None,
                max_results: int = None, cloud_percent_min: float = 0, cloud_percent_max: float = 100, resampling: str = "cubic"):

        generating_mosaic = False

        if geometry is not None and target is not None:
            self.logger.info(f"generating mosaic: {colored_logging.name(target)}")
            generating_mosaic = True

        scenes = self.scene_search(start=start, end=end, target_geometry=geometry, datasets=datasets, sensors=sensors, max_results=max_results,
                                   cloud_percent_min=cloud_percent_min, cloud_percent_max=cloud_percent_max, tiles=tiles)

        dates_available = sorted(set(scenes.date_UTC))
        results_rows = []
        self.logger.info(f"processing {colored_logging.val(len(scenes))} results")

        for i, date_UTC in enumerate(dates_available):
            product_filenames = {}
            mosaic_time_UTC = None
            self.logger.info(f"processing {colored_logging.val(len(products))} products: {colored_logging.val(', '.join(products))}")

            for product in products:
                self.logger.info(f"processing product: {colored_logging.name(product)}")
                day_scenes = scenes[scenes.date_UTC == date_UTC]
                sensors = sorted(np.unique(day_scenes.sensor))

                self.logger.info(f"processing {colored_logging.val(len(sensors))} sensors: {colored_logging.val(', '.join(sensors))}")

                for sensor in sensors:
                    mosaic_previously_generated = False
                    self.logger.info(f"processing sensor: {colored_logging.name(sensor)}")
                    image = None
                    scene_image = None
                    mosaic_filename = None

                    # if geometry is None and target is None:
                    if generating_mosaic:
                        mosaic_filename = None
                        self.logger.info(f"generating {colored_logging.val(sensor)} {colored_logging.val(product)} on " + colored_logging.time(f"{date_UTC:%Y-%m-%d} for "
                                         f"target {colored_logging.name(target)}"))
                    else:
                        self.logger.info(f"generating {colored_logging.val(sensor)} {colored_logging.val(product)} mosaic at " + colored_logging.place(target) +
                                         " on " + colored_logging.time(f"{date_UTC:%Y-%m-%d}"))

                    day_scene_count = len(day_scenes)
                    self.logger.info(f"processing {colored_logging.val(day_scene_count)} scenes for date {colored_logging.time(date_UTC)}")

                    for j, scene in day_scenes.iterrows():
                        granule_ID = scene.granule_ID
                        entity_ID = scene.entity_ID
                        date_UTC = scene.date_UTC
                        dataset = scene.dataset
                        self.logger.info(f"processing granule ({j + 1} / {day_scene_count}): {granule_ID}")

                        try:
                            self.logger.info(f"processing {colored_logging.val(product)} scene image for granule: {colored_logging.val(granule_ID)}")

                            scene_image, scene_product_filename, source_filename, time_UTC = \
                                self.process_scene(product=product, dataset=dataset, date_UTC=date_UTC, granule_ID=granule_ID, entity_ID=entity_ID, geometry=geometry,
                                                   band_names=bands)

                            if j == 0:
                                self.logger.info(f"date/time of first granule: {colored_logging.time(time_UTC)}")
                                mosaic_time_UTC = time_UTC
                                mosaic_filename = self.mosaic_filename(product, time_UTC, target, sensor)
                                product_filenames[product] = mosaic_filename

                                if exists(mosaic_filename):
                                    self.logger.info(f"Landsat mosaic already exists: {colored_logging.file(mosaic_filename)}")
                                    mosaic_previously_generated = True
                                    continue
                                else:
                                    self.logger.info(f"mosaic filename: {colored_logging.file(mosaic_filename)}")

                            if scene_image is None:
                                raise ValueError("failed to generate scene image")

                            if scene_image is not None and np.all(np.isnan(scene_image)):
                                self.logger.warning("no pixels retrieved over target geometry from granule: " + colored_logging.val(granule_ID))

                            if generating_mosaic:
                                if image is None:
                                    self.logger.info("initializing composite image with scene image")
                                    image = scene_image.to_geometry(geometry, resampling=resampling)
                                else:
                                    self.loger.info("filling composite image with scene image")
                                    image = image.fill(scene_image.to_geometry(geometry, resampling=resampling))

                        except Exception as e:
                            raise e

                    if generating_mosaic and day_scene_count > 0:
                        if image is None and not mosaic_previously_generated:
                            raise ValueError("failed to produce composite image")

                        if mosaic_filename is None:
                            raise ValueError("failed to generate mosaic filename")

                        if not exists(mosaic_filename):
                            image.cmap = self.product_cmap(product)
                            self.logger.info(f"writing Landsat {colored_logging.val(product)} mosaic: {colored_logging.file(mosaic_filename)}")
                            image.to_geotiff(mosaic_filename)

                        if mosaic_filename is not None and not exists(mosaic_filename):
                            raise IOError(f"failed too produce Landsat mosaic: {mosaic_filename}")

            results_row = [date_UTC, mosaic_time_UTC]

            for product in products:
                results_row.append(product_filenames[product])

            results_rows.append(results_row)

        columns = ["date_UTC", "time_UTC"] + products
        results = pd.DataFrame(results_rows, columns=columns)

        return results

    @classmethod
    def product_cmap(cls, product_name: str):
        if product_name == "NDVI":
            cmap = NDVI_COLORMAP
        elif product_name == "albedo":
            cmap = ALBEDO_COLORMAP
        else:
            cmap = DEFAULT_COLORMAP

        return cmap

    def product_bands(self, product_name: str) -> List[str] or None:
        if product_name == "NDVI":
            return ["NIR", "red"]
        elif product_name == "albedo":
            return ["blue", "green", "red", "NIR", "SWIR"]
        else:
            return None

    def product_directory(self, product: str, date_UTC: date or str) -> str or None:
        if isinstance(date_UTC, str):
            date_UTC = parser.parse(date_UTC).date()

        if self.products_directory is None:
            return None
        else:
            return join(self.products_directory, product, f"{date_UTC:%Y.%m.%d}")

    def product_filename(self, granule_ID: str, product: str, date_UTC: Union[date, str]) -> str or None:
        product_directory = self.product_directory(product=product, date_UTC=date_UTC)

        if product_directory is None:
            return None
        else:
            return join(product_directory, f"{granule_ID}_{product}.tif")

    def product(self, acquisition_date: date, product: str, geometry: RasterGeometry = None, target_name: str = None, bands: List[str] = None, save_mosaic: bool = False,
                return_raster: bool = True) -> Raster or None:

        if bands is None:
            bands = self.product_bands(product)

        scenes = self.scene_search(start=acquisition_date, end=acquisition_date, target_geometry=geometry)

        # if len(scenes) == 0:
        #     raise UnavailableError(f"Landsat is not available on {date_UTC} over target geometry")

        day_image = None

        sensors = np.unique(scenes.sensor)

        for sensor in sensors:
            mosaic_filename = self.mosaic_filename(product_name=product, time_UTC=acquisition_date, target_name=target_name, sensor=sensor)

            if exists(mosaic_filename):
                self.logger.info(f"loading Landsat {colored_logging.val(product)} mosaic: {colored_logging.file(mosaic_filename)}")

                if return_raster:
                    sensor_image = Raster.open(mosaic_filename)
                else:
                    sensor_image = None
            else:
                sensor_image = None

                for i, scene in scenes.iterrows():
                    dataset = scene.dataset
                    date_UTC = scene.date_UTC
                    granule_ID = scene.granule_ID
                    entity_ID = scene.entity_ID

                    try:
                        scene_image_filename = self.product_filename(granule_ID=granule_ID, product=product, date_UTC=date_UTC)

                        if exists(scene_image_filename):
                            scene_image = rt.Raster.open(scene_image_filename)
                        else:
                            granule = self.retrieve_granule(dataset=dataset, date_UTC=date_UTC, granule_ID=granule_ID, entity_ID=entity_ID, bands=bands)

                            self.logger.info(f"processing {colored_logging.val(product)} for granule: {colored_logging.val(granule_ID)}")
                            scene_image = granule.product(product=product, geometry=geometry)

                            pixel_count = np.count_nonzero(~np.isnan(scene_image))
                            self.logger.info(
                                f"retrieved {colored_logging.val(product)} {colored_logging.val(pixel_count)} pixels from {granule_ID}")

                            if self.remove_sources:
                                granule.remove()

                        if np.all(np.isnan(scene_image)):
                            self.logger.warning(
                                "no pixels retrieved over target geometry from granule: " + colored_logging.val(granule_ID))

                        if sensor_image is None:
                            sensor_image = scene_image
                        else:
                            sensor_image = sensor_image.fill(scene_image)

                    except Exception as e:
                        self.logger.exception(e)
                        self.logger.error(f"failed to download granule: {granule_ID}")
                        continue

                if np.all(np.isnan(sensor_image)):
                    self.logger.warning(f"no pixels retrieved over target geometry")

                if save_mosaic and target_name is not None:
                    self.save_mosaic(image=sensor_image, product_name=product, date_UTC=acquisition_date, target_name=target_name, sensor=sensor)

            sensor_image.cmap = self.product_cmap(product)

            if day_image is None:
                day_image = sensor_image
            else:
                day_image = day_image.fill(sensor_image)

        return day_image


def main(argv=sys.argv):
    start = parser.parse(argv[1]).date()
    end = parser.parse(argv[2]).date()
    tiles = argv[3].split(",")
    products = argv[4].split(",")

    if '--sensors' in argv:
        sensors = argv[argv.index('--sensors') + 1].split(",")
    else:
        sensors = None

    if '--directory' in argv:
        working_directory = argv[argv.index('--directory') + 1]
    else:
        working_directory = None

    if '--download' in argv:
        download_directory = argv[argv.index('--download') + 1]
    else:
        download_directory = None

    if '--products' in argv:
        products_directory = argv[argv.index('--products') + 1]
    else:
        products_directory = None

    if '--mosaic' in argv:
        mosaic_directory = argv[argv.index('--mosaic') + 1]
    else:
        mosaic_directory = None

    remove_sources = "--remove-sources" in argv

    with LandsatL2C2(
            working_directory=working_directory,
            download_directory=download_directory,
            products_directory=products_directory,
            mosaic_directory=mosaic_directory,
            remove_sources=remove_sources) as landsat:
        landsat.process(start=start, end=end, tiles=tiles, products=products, sensors=sensors)


if __name__ == "__main__":
    configure_logger()
    sys.exit(main(argv=sys.argv))
