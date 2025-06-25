import io
import json
import logging
import sys
import warnings
from datetime import datetime
from datetime import timedelta, date
from glob import glob
from math import floor
from os import makedirs
from os.path import abspath, dirname, expanduser
from os.path import basename
from os.path import join, exists
from os.path import splitext
from time import perf_counter
from typing import List, Set, Union, Tuple
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.crs
import shapely
import shapely.geometry
import shapely.wkt
import xmltodict
from affine import Affine
from dateutil import parser
import mgrs
from rasterio.features import rasterize
from rasterio.warp import Resampling
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt

from ETtoolbox.credentials import get_credentials
from rasters import Point, Polygon, BBox
from shapely.geometry.base import BaseGeometry
from six import string_types

import colored_logging
import rasters
from rasters import RasterGrid, Raster, RasterGeometry, CRS
from solar_apparent_time import UTC_to_solar

# from transform.UTM import UTM_proj4_from_latlon, UTM_proj4_from_zone

pd.options.mode.chained_assignment = None  # default='warn'

DEFAULT_ALBEDO_RESOLUTION = 10
DEFAULT_SEARCH_DAYS = 10
DEFAULT_CLOUD_MIN = 0
DEFAULT_CLOUD_MAX = 50
DEFAULT_ORDER_BY = "-beginposition"

SENTINEL_POLYGONS_FILENAME = join(abspath(dirname(__file__)), "sentinel2_tiles_world.geojson")

DEFAULT_WORKING_DIRECTORY = "."
DEFAULT_DOWNLOAD_DIRECTORY = "sentinel_download"
DEFAULT_PRODUCTS_DIRECTORY = "sentinel_products"

logger = logging.getLogger(__name__)

def UTM_proj4_from_latlon(lat: float, lon: float) -> str:
    UTM_zone = (floor((lon + 180) / 6) % 60) + 1
    UTM_proj4 = f"+proj=utm +zone={UTM_zone} {'+south ' if lat < 0 else ''}+ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    return UTM_proj4


def UTM_proj4_from_zone(zone: str):
    zone_number = int(zone[:-1])

    if zone[-1].upper() == "N":
        hemisphere = ""
    elif zone[-1].upper() == "S":
        hemisphere = "+south "
    else:
        raise ValueError(f"invalid hemisphere in zone: {zone}")

    UTM_proj4 = f"+proj=utm +zone={zone_number} {hemisphere}+datum=WGS84 +units=m +no_defs"

    return UTM_proj4


def load_geojson_as_wkt(geojson_filename: str) -> str:
    return geojson_to_wkt(read_geojson(geojson_filename))


def parse_sentinel_granule_id(granule_id: str) -> dict:
    # Compact Naming Convention
    #
    # The compact naming convention is arranged as follows:
    #
    # MMM_MSIXXX_YYYYMMDDHHMMSS_Nxxyy_ROOO_Txxxxx_<Product Discriminator>.SAFE
    #
    # The products contain two dates.
    #
    # The first date (YYYYMMDDHHMMSS) is the datatake sensing time.
    # The second date is the "<Product Discriminator>" field, which is 15 characters in length, and is used to distinguish between different end user products from the same datatake. Depending on the instance, the time in this field can be earlier or slightly later than the datatake sensing time.
    #
    # The other components of the filename are:
    #
    # MMM: is the mission ID(S2A/S2B)
    # MSIXXX: MSIL1C denotes the Level-1C product level/ MSIL2A denotes the Level-2A product level
    # YYYYMMDDHHMMSS: the datatake sensing start time
    # Nxxyy: the PDGS Processing Baseline number (e.g. N0204)
    # ROOO: Relative Orbit number (R001 - R143)
    # Txxxxx: Tile Number field
    # SAFE: Product Format (Standard Archive Format for Europe)
    #
    # Thus, the following filename
    #
    # S2A_MSIL1C_20170105T013442_N0204_R031_T53NMJ_20170105T013443.SAFE
    #
    # Identifies a Level-1C product acquired by Sentinel-2A on the 5th of January, 2017 at 1:34:42 AM. It was acquired over Tile 53NMJ(2) during Relative Orbit 031, and processed with PDGS Processing Baseline 02.04.
    #
    # In addition to the above changes, a a TCI (True Colour Image) in JPEG2000 format is included within the Tile folder of Level-1C products in this format. For more information on the TCI, see the Definitions page here.
    # https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/naming-convention
    parts = granule_id.split("_")

    return {"mission_id": parts[0], "product": parts[1], "date": parser.parse(parts[2]), "baseline": parts[3], "orbit": parts[4], "tile": parts[5][1:]}


def resize_affine(affine: Affine, cell_size: float) -> Affine:
    if not isinstance(affine, Affine):
        raise ValueError("invalid affine transform")

    new_affine = Affine(cell_size, affine.b, affine.c, affine.d, -cell_size, affine.f)

    return new_affine


class SentinelGranule:
    logger = logging.getLogger(__name__)

    def __init__(self, filename: str, working_directory: str = None, products_directory: str = None, target_resolution: float = None):

        if working_directory is None:
            working_directory = DEFAULT_WORKING_DIRECTORY

        if working_directory.startswith("~"):
            working_directory = expanduser(working_directory)

        if products_directory is None:
            products_directory = join(working_directory, DEFAULT_PRODUCTS_DIRECTORY)

        if products_directory.startswith("~"):
            products_directory = expanduser(products_directory)

        self.products_directory = products_directory
        self.target_resolution = target_resolution
        self._filename = abspath(filename)
        self._granule_id = None
        self._mission_id = None
        self._product = None
        self._time_UTC = None
        self._baseline = None
        self._orbit = None
        self._tile = None
        self._filenames = None
        self._image_filenames = None
        self._metadata = None
        self._dn_nodata = None
        self._dn_saturation = None
        self._dn_scale_factor = None
        self._cloud_df = None

        self._cloud_masks = {}
        self._resolution_dimensions = {}
        self._resolution_affine = {}
        self._resolution_grid = {}

    def __repr__(self):
        display_dict = {"filename": self.filename, "products_directory": self.products_directory}

        if self.target_resolution is not None:
            display_dict["target_resolution"] = self.target_resolution

        display_string = json.dumps(display_dict, indent=2)

        return display_string

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def filename_base(self) -> str:
        return basename(self.filename)

    @property
    def filename_stem(self) -> str:
        return splitext(self.filename_base)[0]

    @property
    def granule_id(self) -> str:
        if self._granule_id is None:
            granule_id = splitext(self.filename_base)[0].strip()

            if granule_id.endswith(".SAFE"):
                granule_id = granule_id[:-5]

            self._granule_id = granule_id

        return self._granule_id

    def _parse_granule_id(self):
        attributes = parse_sentinel_granule_id(self.granule_id)
        self._mission_id = attributes["mission_id"]
        self._product = attributes["product"]
        self._time_UTC = attributes["date"]
        self._baseline = attributes["baseline"]
        self._orbit = attributes["orbit"]
        self._tile = attributes["tile"]

    @property
    def mission_id(self) -> str:
        if self._mission_id is None:
            self._parse_granule_id()

        return self._mission_id

    @property
    def product(self) -> str:
        if self._product is None:
            self._parse_granule_id()

        return self._product

    @property
    def time_UTC(self) -> datetime:
        if self._time_UTC is None:
            self._parse_granule_id()

        return self._time_UTC

    @property
    def date_UTC(self):
        return self.time_UTC.date()

    @property
    def lon_center(self):
        return self.centroid_latlon.x

    @property
    def time_solar(self):
        return UTC_to_solar(self.time_UTC, self.lon_center)

    @property
    def date_solar(self):
        return self.time_solar.date()

    @property
    def baseline(self) -> str:
        if self._baseline is None:
            self._parse_granule_id()

        return self._baseline

    @property
    def orbit(self) -> int:
        if self._orbit is None:
            self._parse_granule_id()

        return self._orbit

    @property
    def tile(self) -> str:
        if self._tile is None:
            self._parse_granule_id()

        return self._tile

    @property
    def filenames(self) -> list:
        if self._filenames is None:
            with ZipFile(self.filename) as z:
                self._filenames = [item.filename for item in z.infolist()]

        return self._filenames

    @property
    def image_filenames(self) -> list:
        if self._image_filenames is None:
            self._image_filenames = [filename for filename in self.filenames if splitext(filename)[-1] == ".jp2"]

        return self._image_filenames

    # The list of band with their central wavelengths and resolutions are shown below:
    #
    # Sentinel-2 Bands	Central Wavelength (µm)	Resolution (m)
    # Band 1 - Coastal aerosol	0.443	60
    # Band 2 - Blue	0.490	10
    # Band 3 - Green	0.560	10
    # Band 4 - Red	0.665	10
    # Band 5 - Vegetation Red Edge	0.705	20
    # Band 6 - Vegetation Red Edge	0.740	20
    # Band 7 - Vegetation Red Edge	0.783	20
    # Band 8 - NIR	0.842	10
    # Band 8A - Vegetation Red Edge	0.865	20
    # Band 9 - Water vapour	0.945	60
    # Band 10 - SWIR - Cirrus	1.375	60
    # Band 11 - SWIR	1.610	20
    # Band 12 - SWIR	2.190	20
    # https://www.hatarilabs.com/ih-en/how-many-spectral-bands-have-the-sentinel-2-images

    def band_filename(self, band: int) -> str:
        if isinstance(band, int):
            band = f"B{band:02d}"

        filenames = sorted([filename for filename in self.image_filenames if band in filename])

        if len(filenames) == 0:
            return None

        return filenames[0]

    def band_URI(self, band: int) -> str:
        band_filename = self.band_filename(band)

        if band_filename is None:
            return None
        else:
            return f"zip://{self.filename}!/{band_filename}"

    def band_DN(self, band: int) -> Raster:
        URI = self.band_URI(band)

        try:
            with rasterio.Env(CPL_ZIP_ENCODING="UTF-8"):
                DN = Raster.open(URI)
        except Exception as e:
            self.logger.exception(e)
            raise IOError(f"unable to load sentinel band {band} from file: {URI}")

        return DN

    def band_profile(self, band: int) -> dict:
        with rasterio.open(self.band_URI(band), "r") as f:
            return f.profile

    @property
    def crs(self) -> rasterio.crs.CRS:
        with rasterio.open(self.band_URI(2), "r") as f:
            return f.crs

    def band_affine(self, band: int) -> Affine:
        if self.band_URI(band) is None:
            return None

        with rasterio.open(self.band_URI(band), "r") as f:
            return f.transform

    def band_grid(self, band: int) -> RasterGrid:
        if self.band_URI(band) is None:
            return None

        grid = RasterGrid.from_raster_file(self.band_URI(band))

        return grid

    @property
    def geometry(self):
        if self.target_resolution is not None:
            return self.resolution_grid(self.target_resolution)
        else:
            return self.band_grid(1)

    @property
    def centroid(self) -> Point:
        return self.geometry.centroid

    @property
    def centroid_latlon(self) -> Point:
        return self.centroid.latlon

    def band_dimensions(self, band: int) -> tuple:
        if self.band_URI(band) is None:
            return None

        with rasterio.open(self.band_URI(band), "r") as f:
            return f.height, f.width

    def resolution_dimensions(self, cell_size_meters: float) -> tuple:
        cell_size_meters = int(cell_size_meters)

        if cell_size_meters in self._resolution_dimensions:
            return self._resolution_dimensions[cell_size_meters]

        for band in range(1, 13):
            if self.band_cell_size(band) == cell_size_meters:
                dimensions = self.band_dimensions(band)
                self._resolution_dimensions[cell_size_meters] = dimensions

                return dimensions

        rows10, cols10 = self.resolution_dimensions(10)

        rows = int(rows10 * 10.0 / cell_size_meters)
        cols = int(cols10 * 10.0 / cell_size_meters)

        dimensions = (rows, cols)
        self._resolution_dimensions[cell_size_meters] = dimensions

        return dimensions

    def resolution_affine(self, cell_size_meters: float) -> Affine:
        cell_size_meters = int(cell_size_meters)

        if cell_size_meters in self._resolution_affine:
            return self._resolution_affine[cell_size_meters]

        for band in range(1, 13):
            if self.band_cell_size(band) == cell_size_meters:
                affine = self.band_affine(band)
                self._resolution_affine[cell_size_meters] = affine

                return affine

        affine10 = self.resolution_affine(10)
        affine = resize_affine(affine10, cell_size_meters)

        self._resolution_affine[cell_size_meters] = affine

        return affine

    def resolution_grid(self, cell_size_meters: float) -> RasterGrid:
        cell_size_meters = int(cell_size_meters)

        if cell_size_meters in self._resolution_grid:
            return self._resolution_grid[cell_size_meters]

        for band in range(1, 13):
            if self.band_cell_size(band) == cell_size_meters:
                affine = self.band_affine(band)
                rows, cols = self.band_dimensions(band)
                grid = RasterGrid.from_affine(affine, rows, cols, projection=self.crs)
                self._resolution_grid[cell_size_meters] = grid

                return grid

        affine10 = self.resolution_affine(10)
        affine = resize_affine(affine10, cell_size_meters)
        rows, cols = self.resolution_dimensions(cell_size_meters)
        grid = RasterGrid.from_affine(affine, rows, cols, projection=self.crs)
        self._resolution_grid[cell_size_meters] = grid

        return grid

    def band_cell_size(self, band: int) -> float:
        if self.band_affine(band) is None:
            return None

        return int(self.band_affine(band).a)

    @property
    def cloud_filename(self) -> str:
        return next((filename for filename in self.filenames if "CLOUDS" in filename))

    @property
    def cloud_gml(self) -> str:
        with ZipFile(self.filename, "r") as z:
            try:
                cloud_gml = z.read(self.cloud_filename).decode()
            except Exception as e:
                # self.logger.exception(e)
                raise IOError(
                    f"unable to load Sentinel cloud mask: {join(abspath(self.filename), self.cloud_filename)}")

        return cloud_gml

    @property
    def cloud_df(self) -> gpd.GeoDataFrame:
        if self._cloud_df is not None:
            return self._cloud_df

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with ZipFile(self.filename, "r") as z:
                try:
                    cloud_df = gpd.read_file(io.BytesIO(z.read(self.cloud_filename)))
                except Exception as e:
                    # self.logger.exception(e)
                    raise IOError(
                        f"unable to load Sentinel cloud mask: {join(abspath(self.filename), self.cloud_filename)}")

        self._cloud_df = cloud_df

        return cloud_df

    def band_cloud_mask(self, band: int) -> Raster:
        resolution = self.band_cell_size(band)

        if resolution not in self._cloud_masks:
            shape = self.band_dimensions(band)
            affine = self.band_affine(band)

            # when there are no clouds identified over the sentinel scene, the GML file is empty, and the fiona driver can't interpret it
            # interpreting an error in loading the cloud shape as meaning no clouds

            try:
                surface = np.full(shape, np.nan)
                shapes = [(shape, 1) for shape in self.cloud_df.geometry]
                rasterize(shapes=shapes, fill=0, out=surface, transform=affine)
                surface = np.where(surface == 1, 1, 0)
            except Exception as e:
                surface = np.full(shape, 0)

            geometry = self.band_grid(band)
            raster = Raster(surface, geometry=geometry)
            self._cloud_masks[resolution] = raster

        return self._cloud_masks[resolution]

    @property
    def metadata_filename(self) -> str:
        return next((filename for filename in self.filenames if "MTD_MSI" in filename))

    @property
    def metadata_xml(self) -> str:
        with ZipFile(self.filename, "r") as z:
            return z.read(self.metadata_filename).decode()

    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            self._metadata = xmltodict.parse(self.metadata_xml)

        return self._metadata

    @property
    def dn_nodata(self):
        if self._dn_nodata is None:
            i1 = next((key for key in self.metadata if "User_Product" in key))
            self._dn_nodata = int(
                self.metadata[i1]["n1:General_Info"]["Product_Image_Characteristics"][
                    "Special_Values"][0]["SPECIAL_VALUE_INDEX"])

        return self._dn_nodata

    @property
    def dn_saturation(self):
        if self._dn_saturation is None:
            i1 = next((key for key in self.metadata if "User_Product" in key))
            self._dn_saturation = int(
                self.metadata[i1]["n1:General_Info"]["Product_Image_Characteristics"][
                    "Special_Values"][1]["SPECIAL_VALUE_INDEX"])

        return self._dn_saturation

    @property
    def dn_scale_factor(self):
        if self._dn_scale_factor is None:
            i1 = next((key for key in self.metadata if "User_Product" in key))
            characteristics = self.metadata[i1]["n1:General_Info"]["Product_Image_Characteristics"]

            if "QUANTIFICATION_VALUE" in characteristics.keys():
                scale_factor = float(characteristics["QUANTIFICATION_VALUE"]["#text"])
            elif "QUANTIFICATION_VALUES_LIST" in characteristics.keys():
                scale_factor = float(characteristics["QUANTIFICATION_VALUES_LIST"]["BOA_QUANTIFICATION_VALUE"]["#text"])
            else:
                raise ValueError(f"scale factor not found in: {', '.join(characteristics.keys())}")

            self._dn_scale_factor = scale_factor

        return self._dn_scale_factor

    def product_directory(self, product):
        if self.products_directory is None:
            raise ValueError("no Sentinel product directory given")
        else:
            return join(self.products_directory, product, f"{self.date_UTC:%Y.%m.%d}")

    def product_filename(self, product: str, target_resolution: float = None):
        if target_resolution is None:
            target_resolution = self.target_resolution

        if target_resolution is None:
            raise ValueError("no target resolution")

        product_directory = self.product_directory(product)

        product_filename = join(
            product_directory,
            f"{self.filename_stem}_{product}_{int(target_resolution)}m.tif"
        )

        return product_filename

    def band_reflectance(
            self, band: int,
            target_resolution: float = None,
            apply_cloud_mask: bool = True) -> Raster:
        source_resolution = self.band_cell_size(band)
        self.logger.info(f"loading Sentinel band {colored_logging.val(band)} ({colored_logging.val(source_resolution)}m)")
        dn = self.band_DN(band)
        dn = rasters.where(dn == self.dn_nodata, np.nan, dn)

        if apply_cloud_mask:
            dn = rasters.where(self.band_cloud_mask(band), np.nan, dn)

        reflectance = dn / self.dn_scale_factor

        if target_resolution is None and self.target_resolution is not None:
            target_resolution = self.target_resolution

        if target_resolution is not None:
            target_grid = self.resolution_grid(target_resolution)

            if source_resolution < target_resolution:
                resampling = Resampling.average
            elif source_resolution > target_resolution:
                resampling = Resampling.cubic
            else:
                resampling = None

            if source_resolution != target_resolution:
                self.logger.info(f"resampling Sentinel band {colored_logging.val(band)} ({colored_logging.val(source_resolution)}m -> "
                                 f"{colored_logging.val(target_resolution)}m)")
                reflectance = reflectance.to_grid(target_grid, resampling=resampling)

        return reflectance

    def get_blue(self, target_resolution: float = None, apply_cloud_mask: bool = True) -> Raster:
        return self.band_reflectance(2, target_resolution=target_resolution, apply_cloud_mask=apply_cloud_mask)

    blue = property(get_blue)

    def get_green(self, target_resolution: float = None, apply_cloud_mask: bool = True) -> Raster:
        return self.band_reflectance(3, target_resolution=target_resolution, apply_cloud_mask=apply_cloud_mask)

    green = property(get_green)

    def get_red(self, target_resolution: float = None, apply_cloud_mask: bool = True) -> Raster:
        return self.band_reflectance(4, target_resolution=target_resolution, apply_cloud_mask=apply_cloud_mask)

    red = property(get_red)

    def get_NIR(self, target_resolution: float = None, apply_cloud_mask: bool = True) -> Raster:
        return self.band_reflectance(8, target_resolution=target_resolution, apply_cloud_mask=apply_cloud_mask)

    NIR = property(get_NIR)

    def get_SWIR1(self, target_resolution: float = None, apply_cloud_mask: bool = True) -> Raster:
        return self.band_reflectance(11, target_resolution=target_resolution, apply_cloud_mask=apply_cloud_mask)

    SWIR1 = property(get_SWIR1)

    def get_SWIR2(self, target_resolution: float = None, apply_cloud_mask: bool = True) -> Raster:
        return self.band_reflectance(12, target_resolution=target_resolution, apply_cloud_mask=apply_cloud_mask)

    SWIR2 = property(get_SWIR2)

    def get_NDVI(self, target_resolution: float = None, apply_cloud_mask: bool = True, save_data: bool = True, save_preview: bool = True, product_filename: str = None,
                 preview_filename: str = None, return_filename: bool = False) -> Union[Raster, Tuple[Raster, str]]:
        if target_resolution is None:
            target_resolution = self.target_resolution

        if target_resolution is None:
            target_resolution = max(self.band_cell_size(8), self.band_cell_size(4))

        if product_filename is None:
            product_filename = self.product_filename("NDVI", target_resolution=target_resolution)

        if preview_filename is None:
            preview_filename = product_filename.replace(".tif", ".jpeg")

        if exists(product_filename):
            self.logger.info(f"loading Sentinel NDVI: {colored_logging.file(product_filename)}")
            NDVI = Raster.open(product_filename)
        else:
            NIR = self.get_NIR(target_resolution=target_resolution, apply_cloud_mask=apply_cloud_mask)
            red = self.get_red(target_resolution=target_resolution, apply_cloud_mask=apply_cloud_mask)
            NDVI = np.clip((NIR - red) / (NIR + red), -1, 1)

        if save_data and not exists(product_filename):
            self.logger.info(f"saving Sentinel NDVI: {colored_logging.file(product_filename)}")
            NDVI.to_COG(product_filename)

            if save_preview:
                self.logger.info(f"saving Sentinel NDVI preview: {colored_logging.file(preview_filename)}")
                NDVI.percentilecut.to_geojpeg(preview_filename, quality=20, remove_XML=True)

        if return_filename:
            return NDVI, product_filename
        else:
            return NDVI

    NDVI = property(get_NDVI)

    # Band      λ (μm)	Δλ (μm)	Esun (W m−2)	ωbi
    # 1	        0.443	0.020	1893
    # 2	        0.490	0.065	1927	        0.1324
    # 3	        0.560	0.035	1846	        0.1269
    # 4	        0.665	0.030	1528	        0.1051
    # 5	        0.705	0.015	1413	        0.0971
    # 6	        0.740	0.015	1294	        0.0890
    # 7	        0.783	0.020	1190	        0.0818
    # 8	        0.842	0.115	1050	        0.0722
    # 8a	    0.865	0.020	970
    # 9	        0.945	0.020	831
    # 10	    1.375	0.030	360
    # 11	    1.610	0.090	242	            0.0167
    # 12	    2.190	0.180	3	            0.0002
    # https://www.sciencedirect.com/science/article/pii/S0034425718303134#t0015

    def get_albedo(self, target_resolution: float = None, apply_cloud_mask: bool = True, save_data: bool = True, save_preview: bool = True, product_filename: str = None,
                   preview_filename: str = None, geometry: RasterGeometry = None, return_filename: bool = False) -> Union[Raster, Tuple[Raster, str]]:

        if target_resolution is None:
            target_resolution = self.target_resolution

        if target_resolution is None:
            target_resolution = max(self.band_cell_size(2), self.band_cell_size(3), self.band_cell_size(4), self.band_cell_size(5), self.band_cell_size(6),
                                    self.band_cell_size(7), self.band_cell_size(8), self.band_cell_size(11), self.band_cell_size(12))

        if product_filename is None:
            product_filename = self.product_filename("albedo", target_resolution=target_resolution)

        if preview_filename is None:
            preview_filename = product_filename.replace(".tif", ".jpeg")

        if exists(product_filename):
            self.logger.info(f"loading Sentinel albedo: {colored_logging.file(product_filename)}")
            albedo = Raster.open(product_filename)
        else:
            b2 = self.band_reflectance(2, target_resolution=target_resolution, apply_cloud_mask=apply_cloud_mask)
            b3 = self.band_reflectance(3, target_resolution=target_resolution, apply_cloud_mask=apply_cloud_mask)
            b4 = self.band_reflectance(4, target_resolution=target_resolution, apply_cloud_mask=apply_cloud_mask)
            b5 = self.band_reflectance(5, target_resolution=target_resolution, apply_cloud_mask=apply_cloud_mask)
            b6 = self.band_reflectance(6, target_resolution=target_resolution, apply_cloud_mask=apply_cloud_mask)
            b7 = self.band_reflectance(7, target_resolution=target_resolution, apply_cloud_mask=apply_cloud_mask)
            b8 = self.band_reflectance(8, target_resolution=target_resolution, apply_cloud_mask=apply_cloud_mask)
            b11 = self.band_reflectance(11, target_resolution=target_resolution, apply_cloud_mask=apply_cloud_mask)
            b12 = self.band_reflectance(12, target_resolution=target_resolution, apply_cloud_mask=apply_cloud_mask)
            albedo = 0.1324 * b2 + 0.1269 * b3 + 0.1051 * b4 + 0.0971 * b5 + 0.0890 * b6 + 0.0818 * b7 + 0.0722 * b8 + 0.0167 * b11 + 0.0002 * b12

            albedo = np.clip(albedo, -1, 1)

        if save_data and not exists(product_filename):
            self.logger.info(f"saving Sentinel albedo: {colored_logging.file(product_filename)}")
            albedo.to_COG(product_filename)

            if save_preview:
                self.logger.info(f"saving Sentinel albedo preview: {colored_logging.file(preview_filename)}")
                albedo.percentilecut.to_geojpeg(preview_filename, quality=20, remove_XML=True)

        if geometry is not None:
            albedo = albedo.to_geometry(geometry)

        if return_filename:
            return albedo, product_filename
        else:
            return albedo

    albedo = property(get_albedo)

class MGRS(mgrs.MGRS):
    def bbox(self, tile: str) -> BBox:
        if len(tile) == 5:
            precision = 100000
        elif len(tile) == 7:
            precision = 10000
        elif len(tile) == 9:
            precision = 1000
        elif len(tile) == 11:
            precision = 100
        elif len(tile) == 13:
            precision = 10
        elif len(tile) == 15:
            precision = 1
        else:
            raise ValueError(f"unrecognized MGRS tile: {tile}")

        zone, hemisphere, xmin, ymin = self.MGRSToUTM(tile)
        crs = CRS(UTM_proj4_from_zone(f"{int(zone)}{str(hemisphere).upper()}"))
        xmax = xmin + precision
        ymax = ymin + precision

        bbox = BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, crs=crs)

        return bbox


class SentinelTileGrid(MGRS):
    def __init__(self, *args, target_resolution: float = 30, **kwargs):
        super(SentinelTileGrid, self).__init__()
        self.target_resolution = target_resolution
        self._sentinel_polygons = None

    def __repr__(self) -> str:
        return f"SentinelTileGrid(target_resolution={self.target_resolution})"

    @property
    def sentinel_polygons(self) -> gpd.GeoDataFrame:
        if self._sentinel_polygons is None:
            self._sentinel_polygons = gpd.read_file(SENTINEL_POLYGONS_FILENAME)

        return self._sentinel_polygons

    @property
    def crs(self) -> CRS:
        return CRS(self._sentinel_polygons.crs)

    def UTM_proj4(self, tile: str) -> str:
        zone, hemisphere, _, _ = self.MGRSToUTM(tile)
        proj4 = UTM_proj4_from_zone(f"{int(zone)}{str(hemisphere).upper()}")

        return proj4

    def footprint(self, tile: str, in_UTM: bool = False, round_UTM: bool = True, in_2d: bool = True) -> Polygon:
        try:
            polygon = Polygon(self.sentinel_polygons[self.sentinel_polygons.Name == tile].iloc[0]["geometry"], crs=self.crs)
        except Exception as e:
            raise ValueError(f"polygon for target {tile} not found")

        if in_2d:
            polygon = Polygon([xy[0:2] for xy in polygon.exterior.coords], crs=self.crs)

        if in_UTM:
            UTM_proj4 = self.UTM_proj4(tile)
            polygon = polygon.to_crs(UTM_proj4)

            if round_UTM:
                polygon = Polygon([[round(item) for item in xy] for xy in polygon.exterior.coords], crs=polygon.crs)

        return polygon

    def footprint_UTM(self, tile: str) -> Polygon:
        return self.footprint(tile=tile, in_UTM=True, round_UTM=True, in_2d=True)

    def bbox(self, tile: str, MGRS: bool = False) -> BBox:
        if len(tile) != 5 or MGRS:
            return super(SentinelTileGrid, self).bbox(tile=tile)

        polygon = self.footprint(tile=tile, in_UTM=True, round_UTM=True, in_2d=True)

        bbox = polygon.bbox

        return bbox

    def tiles(self, target_geometry: shapely.geometry.shape) -> Set[str]:
        if isinstance(target_geometry, str):
            target_geometry = shapely.wkt.loads(target_geometry)

        matches = self.sentinel_polygons[self.sentinel_polygons.intersects(target_geometry)]
        tiles = set(sorted(list(matches.apply(lambda row: row.Name, axis=1))))

        return tiles

    def tile_footprints(self, target_geometry: shapely.geometry.shape or gpd.GeoDataFrame, calculate_area: bool = False,
                        eliminate_redundancy: bool = False) -> gpd.GeoDataFrame:
        if isinstance(target_geometry, str):
            target_geometry = shapely.wkt.loads(target_geometry)

        if isinstance(target_geometry, BaseGeometry):
            target_geometry = gpd.GeoDataFrame(geometry=[target_geometry], crs="EPSG:4326")

        if not isinstance(target_geometry, gpd.GeoDataFrame):
            raise ValueError("invalid target geometry")

        matches = self.sentinel_polygons[self.sentinel_polygons.intersects(target_geometry.to_crs(self.sentinel_polygons.crs).unary_union)]
        matches.rename(columns={"Name": "tile"}, inplace=True)
        tiles = matches[["tile", "geometry"]]

        if calculate_area or eliminate_redundancy:
            centroid = target_geometry.to_crs("EPSG:4326").unary_union.centroid
            lon = centroid.x
            lat = centroid.y
            projection = UTM_proj4_from_latlon(lat, lon)
            tiles_UTM = tiles.to_crs(projection)
            target_UTM = target_geometry.to_crs(projection)
            tiles_UTM["area"] = gpd.overlay(tiles_UTM, target_UTM).geometry.area
            # overlap = gpd.overlay(tiles_UTM, target_UTM)
            # area = overlap.geometry.area

            if eliminate_redundancy:
                # tiles_UTM["area"] = np.array(area)
                tiles_UTM.sort_values(by="area", ascending=False, inplace=True)
                tiles_UTM.reset_index(inplace=True)
                tiles_UTM = tiles_UTM[["tile", "area", "geometry"]]
                remaining_target = target_UTM.unary_union
                remaining_target_area = remaining_target.area
                indices = []

                for i, (tile, area, geometry) in tiles_UTM.iterrows():
                    remaining_target = remaining_target - geometry
                    previous_area = remaining_target_area
                    remaining_target_area = remaining_target.area
                    change_in_area = remaining_target_area - previous_area

                    if change_in_area != 0:
                        indices.append(i)

                    if remaining_target_area == 0:
                        break

                tiles_UTM = tiles_UTM.iloc[indices, :]
                tiles = tiles_UTM.to_crs(tiles.crs)
                tiles.sort_values(by="tile", ascending=True, inplace=True)
                tiles = tiles[["tile", "area", "geometry"]]

            else:
                # tiles["area"] = np.array(area)
                tiles = tiles[["tile", "area", "geometry"]]

        return tiles

    def grid(self, tile: str, cell_size: float = None, buffer=0) -> RasterGrid:
        if cell_size is None:
            cell_size = self.target_resolution

        bbox = self.bbox(tile).buffer(buffer)
        projection = self.UTM_proj4(tile)
        grid = RasterGrid.from_bbox(bbox=bbox, cell_size=cell_size, crs=projection)
        # logger.info(f"tile {colored_logging.place(tile)} at resolution {colored_logging.val(cell_size)} {grid.shape}")

        return grid

    def centroid(self, tile: str) -> shapely.geometry.Point:
        return self.footprint(tile).centroid

    def tile_grids(self, target_geometry: shapely.geometry.shape or gpd.GeoDataFrame, eliminate_redundancy: bool = True) -> gpd.GeoDataFrame:
        tiles = self.tile_footprints(target_geometry=target_geometry, eliminate_redundancy=eliminate_redundancy)

        tiles["grid"] = tiles["tile"].apply(lambda tile: self.grid(tile))
        tiles = tiles[["tile", "area", "grid", "geometry"]]

        return tiles

sentinel_tile_grid = SentinelTileGrid()

class NoSentinelGranulesAvailable(Exception):
    pass


class Sentinel(SentinelAPI, SentinelTileGrid):
    logger = logging.getLogger(__name__)

    def __init__(self, username: str = None, password: str = None, working_directory: str = None, download_directory: str = None, products_directory: str = None,
                 target_resolution: float = None, *args, **kwargs):
        if username is None or password is None:
            credentials = get_credentials(filename="~/.sentinelhub", displayed=["username"], hidden=["password"],
                                          prompt="credentials for Sentinel Hub https://www.sentinel-hub.com", replace=True)

            username = credentials["username"]
            password = credentials["password"]

        SentinelAPI.__init__(self, username, password, *args, **kwargs)
        SentinelTileGrid.__init__(self, target_resolution=target_resolution)

        if working_directory is None:
            working_directory = DEFAULT_WORKING_DIRECTORY

        if working_directory.startswith("~"):
            working_directory = expanduser(working_directory)

        if download_directory is None:
            download_directory = join(working_directory, DEFAULT_DOWNLOAD_DIRECTORY)

        if download_directory.startswith("~"):
            download_directory = expanduser(download_directory)

        if products_directory is None:
            products_directory = join(working_directory, DEFAULT_PRODUCTS_DIRECTORY)

        if products_directory.startswith("~"):
            products_directory = expanduser(products_directory)

        self.working_directory = working_directory
        self.download_directory = download_directory
        self.products_directory = products_directory
        self.target_resolution = target_resolution
        self._sentinel_polygons = None

    def __repr__(self):
        display_dict = {"working_directory": self.working_directory, "download_directory": self.download_directory, "products_directory": self.products_directory}

        if self.target_resolution is not None:
            display_dict["target_resolution"] = self.target_resolution

        display_string = json.dumps(display_dict, indent=2)

        return display_string

    def source_tile_filenames(self, tile: str) -> List[str]:
        return sorted(glob(join(self.download_directory, "**", f"*_T{tile}_*.zip"), recursive=True), reverse=True)

    def find_example(self, tile: str) -> SentinelGranule:
        source_tile_filenames = self.source_tile_filenames(tile)

        if len(source_tile_filenames) == 0:
            raise IOError(f"no files found for target: {tile}")

        filename = source_tile_filenames[0]

        granule = SentinelGranule(filename=filename, working_directory=self.working_directory, products_directory=self.products_directory,
                                  target_resolution=self.target_resolution)

        return granule

    def polygon(self, tile: str) -> shapely.geometry.Polygon:
        try:
            polygon = self.sentinel_polygons[self.sentinel_polygons.Name == tile].iloc[0]["geometry"]
        except Exception as e:
            raise ValueError("polygon for target {target} not found")

        return polygon

    def centroid(self, tile: str) -> shapely.geometry.Point:
        return self.polygon(tile).centroid

    def search_geometry(self, footprint: shapely.geometry.Polygon, start_date: date or str, filename_pattern: str, end_date: date = None, cloud_min: float = None,
                        cloud_max: float = None, order_by: str = None, max_results: int = None) -> gpd.GeoDataFrame:
        DATE_FORMAT = "%Y%m%d"

        if isinstance(start_date, string_types):
            start_date = parser.parse(start_date).date().strftime(DATE_FORMAT)

        if isinstance(end_date, string_types):
            start_date = parser.parse(end_date).date().strftime(DATE_FORMAT)

        if end_date is None:
            end_date = (start_date - timedelta(DEFAULT_SEARCH_DAYS)).strftime(DATE_FORMAT)

        if cloud_min is None:
            cloud_min = DEFAULT_CLOUD_MIN

        if cloud_max is None:
            cloud_max = DEFAULT_CLOUD_MAX

        if order_by is None:
            order_by = DEFAULT_ORDER_BY

        if max_results is None:
            limit = None
        else:
            limit = max_results

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            listing = self.to_geodataframe(self.query(area=footprint, date=(f"{start_date:%Y%m%d}", f"{end_date:%Y%m%d}"), platformname='Sentinel-2',
                                                      filename=filename_pattern, cloudcoverpercentage=(cloud_min, cloud_max), order_by=order_by, limit=limit))

            listing = listing.reset_index()
            listing["UUID"] = listing["index"]
            listing.pop("index")

        return listing

    def URL_from_UUID(self, UUID: str) -> str:
        return self.get_product_odata(UUID)["url"]

    def sensor_from_ID(self, ID: str) -> str:
        return ID.split("_")[0]

    def tile_from_ID(self, ID: str) -> str:
        return ID.split("_")[5][1:]

    def date_from_ID(self, ID: str) -> date:
        return parser.parse(ID.split("_")[2]).date()

    def level_from_ID(self, ID: str) -> str:
        return ID.split("_")[1][3:]

    def search_pattern(self, pattern: str, start_date: date, end_date: date = None, tile: str = None, geometry: BaseGeometry or RasterGeometry = None,
                       cloud_min: float = None, cloud_max: float = None, order_by: str = None, max_results: int = None, keep_all_columns: bool = False):

        if isinstance(geometry, RasterGeometry):
            geometry = geometry.corner_polygon_latlon

        if isinstance(start_date, str):
            start_date = parser.parse(start_date).date()

        if isinstance(end_date, str):
            end_date = parser.parse(end_date).date()

        if end_date is None or start_date == end_date:
            end_date = start_date + timedelta(1)

        if geometry is None and isinstance(tile, str):
            geometry = self.centroid(tile)

        # self.logger.info(f"Sentinel search from {start:%Y-%m-%d} to {end:%Y-%m-%d}")

        listing = self.search_geometry(geometry, start_date, filename_pattern=pattern, end_date=end_date, cloud_min=cloud_min, cloud_max=cloud_max, order_by=order_by,
                                       max_results=max_results)

        if len(listing.columns) == 0 or len(listing) == 0:
            raise NoSentinelGranulesAvailable("empty listing")

        listing["ID"] = listing["title"]
        listing["filename"] = listing["ID"].apply(lambda ID: f"{ID}.zip")
        listing["URL"] = listing["UUID"].apply(lambda UUID: self.URL_from_UUID(UUID))
        listing["sensor"] = listing["ID"].apply(lambda ID: self.sensor_from_ID(ID))
        listing["tile"] = listing["ID"].apply(lambda ID: self.tile_from_ID(ID))
        listing["date"] = listing["ID"].apply(lambda ID: self.date_from_ID(ID))
        listing["level"] = listing["ID"].apply(lambda ID: self.level_from_ID(ID))

        if tile is not None:
            listing = listing[listing.tile == tile]

        if not keep_all_columns:
            listing = listing[["ID", "sensor", "level", "tile", "date", "UUID", "URL", "filename"]]

        listing = listing[listing.tile.apply(lambda t: t == tile)]
        listing = listing.sort_values(by="date")
        listing = listing.reset_index(drop=True)

        return listing

    def search_L2A(self, start_date: date, end_date: date = None, tile: str = None, geometry: BaseGeometry or RasterGeometry = None, cloud_min: float = None,
                   cloud_max: float = None, order_by: str = None, max_results: int = None, keep_all_columns: bool = False):

        return self.search_pattern(pattern="*L2A*", start_date=start_date, end_date=end_date, tile=tile, geometry=geometry, cloud_min=cloud_min, cloud_max=cloud_max,
                                   order_by=order_by, max_results=max_results, keep_all_columns=keep_all_columns)

    def search_L1C(self, start_date: date, end_date: date = None, tile: str = None, geometry: BaseGeometry or RasterGeometry = None, cloud_min: float = None,
                   cloud_max: float = None, order_by: str = None, max_results: int = None, keep_all_columns: bool = False):

        return self.search_pattern(pattern="*L1C*", start_date=start_date, end_date=end_date, geometry=geometry, tile=tile, cloud_min=cloud_min, cloud_max=cloud_max,
                                   order_by=order_by, max_results=max_results, keep_all_columns=keep_all_columns)

    @classmethod
    def filter_to_sensor(cls, listing: gpd.GeoDataFrame, sensor: str) -> gpd.GeoDataFrame:
        filtered_listing = listing[listing["sensor"] == sensor]
        filtered_listing = filtered_listing.sort_values(by="date", ascending=False).groupby(by="tile").first().sort_values(by="date", ascending=False)
        filtered_listing["tile"] = filtered_listing.index
        filtered_listing = filtered_listing.reset_index(drop=True)

        return filtered_listing

    def search(self, geometry: shapely.geometry.shape, start_date: date, end_date: date = None, cloud_min: float = None, cloud_max: float = None,
               max_results: int = None) -> gpd.GeoDataFrame:

        if isinstance(start_date, string_types):
            start_date = parser.parse(start_date).date()

        if isinstance(end_date, string_types):
            start_date = parser.parse(end_date).date()

        if start_date is None:
            start_date = end_date - timedelta(days=DEFAULT_SEARCH_DAYS)

        self.logger.info(f"searching Sentinel L2A between {colored_logging.time(start_date)} and {colored_logging.time(end_date)}")
        listing = self.search_L2A(geometry, start_date, end_date, cloud_min, cloud_max, max_results)

        if len(listing) == 0:
            self.logger.info(f"no L2A granules found between {start_date} and {end_date}, searching L1C instead")
            listing = self.search_L1C(geometry, start_date, end_date, cloud_min, cloud_max, max_results)

        # filter to most recent for each sensor at each target
        S2A_listing = self.filter_to_sensor(listing, "S2A")
        self.logger.info(f"{colored_logging.val(len(S2A_listing))} S2A granules found")
        S2B_listing = self.filter_to_sensor(listing, "S2B")
        self.logger.info(f"{colored_logging.val(len(S2B_listing))} S2B granules found")

        listing = pd.concat([S2A_listing, S2B_listing])

        return listing

    def download_path(self, acquisition_date):
        return join(self.download_directory, acquisition_date.strftime("%Y.%m.%d"))

    def download_path_from_ID(self, ID: str) -> str:
        return self.download_path(self.date_from_ID(ID))

    def output_path(self, product: str, level: str, sensor: str, acquisition_date: date) -> str:
        return join(self.download_directory, level.upper(), sensor.upper(), product, acquisition_date.strftime("%Y.%m.%d"))

    def source_filename(self, ID: str) -> str:
        download_path = self.download_path_from_ID(ID)
        source_filename = join(download_path, f"{ID}.zip")

        return source_filename

    def output_filename(self, product: str, acquisition_date: date, ID: str, sensor: str, level: str) -> str:
        output_path = self.output_path(product, level, sensor, acquisition_date)
        output_filename = join(output_path, f"{ID}.zip")

        return output_filename

    def retrieve_granule(self, ID: str, UUID: str):
        directory_path = self.download_path_from_ID(ID)

        if not exists(directory_path):
            makedirs(directory_path)

        download_filename = self.source_filename(ID)
        self.logger.info(f"downloading Sentinel granule: {colored_logging.name(ID)}")
        self.download(id=UUID, directory_path=directory_path)

        if exists(download_filename):
            self.logger.info(f"download successful: {colored_logging.file(download_filename)}")
        else:
            raise IOError(f"downloaded Sentinel file not found: {colored_logging.file(download_filename)}")

        granule = SentinelGranule(filename=download_filename, working_directory=self.working_directory, products_directory=self.products_directory,
                                  target_resolution=self.target_resolution)

        return granule

    def acquire(self, geometry: BaseGeometry, end_date: date, start_date: date = None, cloud_min: float = None, cloud_max: float = None, max_results: int = None):
        listing = self.search(geometry=geometry, end_date=end_date, start_date=start_date, cloud_min=cloud_min, cloud_max=cloud_max)

        if max_results is not None:
            listing = listing.iloc[:max_results]

        for i, row in listing.iterrows():
            ID = row["ID"]
            UUID = row["UUID"]

            self.retrieve_granule(ID, UUID)

    def processed_data_directory(self, name: str, acquisition_date: date):
        return join(self.products_directory)

    def most_recent(self, tile: str = None, geometry: BaseGeometry or RasterGeometry = None, target_date: date = None, cloud_min: float = None, cloud_max: float = None,
                    level: str = None) -> SentinelGranule:

        if isinstance(target_date, str):
            target_date = parser.parse(target_date).date()

        if target_date is None:
            target_date = datetime.now().date()

        if level is None:
            level = "L2A"

        start_date = target_date - timedelta(days=10)
        end_date = target_date

        if level == "L2A":
            listing = self.search_L2A(tile=tile, geometry=geometry, start_date=start_date, end_date=end_date, cloud_min=cloud_min, cloud_max=cloud_max,
                                      order_by="-beginposition", max_results=1)
        elif level == "L1C":
            listing = self.search_L1C(tile=tile, geometry=geometry, start_date=start_date, end_date=end_date, cloud_min=cloud_min, cloud_max=cloud_max,
                                      order_by="-beginposition", max_results=1)
        else:
            raise ValueError(f"invalid level: {level}")

        ID = listing.iloc[0].ID
        UUID = listing.iloc[0].UUID
        granule = self.retrieve_granule(ID, UUID)

        return granule

    def process_granule(self, granule_ID, sentinel_ID, tile, acquisition_date, filename, product_names: List[str] = None) -> dict:
        logger = logging.getLogger(__name__)

        product_filenames = {}

        if product_names is None:
            product_names = ["NDVI", "albedo"]

        download_directory = self.download_path(acquisition_date)
        makedirs(download_directory, exist_ok=True)
        sentinel_filename = self.download_filename(acquisition_date, filename)

        granule = SentinelGranule(filename=sentinel_filename, working_directory=self.working_directory, products_directory=self.products_directory,
                                  target_resolution=self.target_resolution)

        for product_name in product_names:
            product_filename = granule.product_filename(product_name)

            if exists(product_filename):
                self.logger.info(f"Sentinel {colored_logging.name(product_name)} already exists: {colored_logging.file(product_filename)}")

            if exists(product_filename):
                product_filenames[product_name] = product_filename
                continue

            logger.info(f"processing Sentinel granule: {colored_logging.name(granule_ID)} target: {colored_logging.place(tile)} date: "
                        f"{colored_logging.time(acquisition_date)}")
            start_time = perf_counter()
            logger.info(f"downloading Sentinel granule: {colored_logging.name(sentinel_ID)}")
            self.download(sentinel_ID, download_directory)
            end_time = perf_counter()
            duration_seconds = end_time - start_time
            logger.info("download completed in " + colored_logging.time(f"{duration_seconds:0.2f}") + " seconds")

            if exists(sentinel_filename):
                logger.info(f"download successful: {colored_logging.file(sentinel_filename)}")
            else:
                logger.info(f"file not found: {colored_logging.file(sentinel_filename)}")

            if not exists(product_filename):
                start_time = datetime.now()
                logger.info(f"downloading Sentinel granule: {colored_logging.name(sentinel_ID)}")
                self.download(sentinel_ID, download_directory)
                end_time = datetime.now()
                duration = end_time - start_time
                duration_seconds = duration.total_seconds()
                logger.info(f"download completed in {duration_seconds:0.2f} seconds")

                if exists(sentinel_filename):
                    logger.info(f"download successful: {colored_logging.file(sentinel_filename)}")
                else:
                    logger.info(f"file not found: {colored_logging.file(sentinel_filename)}")

                granule = SentinelGranule(filename=sentinel_filename, working_directory=self.working_directory, products_directory=self.products_directory,
                                          target_resolution=self.target_resolution)

                logger.info(f"generating Sentinel {colored_logging.name(product_name)} at {colored_logging.val(self.target_resolution)} m "
                            f"for tile {colored_logging.name(tile)} on " + colored_logging.time(f"{acquisition_date:%Y-%m-%d}"))
                start_time = datetime.now()

                if product_name == "NDVI":
                    product_image, product_filename = granule.get_NDVI(return_filename=True)
                elif product_name == "albedo":
                    product_image, product_filename = granule.get_albedo(return_filename=True)
                else:
                    raise ValueError(f"unrecognized product: {colored_logging.name(product_name)}")

                product_filenames[product_name] = product_filename

                end_time = datetime.now()
                duration = end_time - start_time
                duration_seconds = duration.total_seconds()
                logger.info(f"{colored_logging.name(product_name)} completed in " + colored_logging.time(f"{duration_seconds:0.2f}") + " seconds")

            if product_name not in product_filenames:
                raise ValueError(f"unable to produce Sentinel product: {product_name}")

        return product_filenames

    def download_filename(self, acquisition_date, filename):
        return join(self.download_path(acquisition_date), filename)

    def process(self, start_date: date or str, tile: str, product_names: List[str] = None, end_date: date or str = None, cloud_min: float = None, cloud_max: float = None,
                max_results: int = None) -> (pd.DataFrame, RasterGeometry):
        if product_names is None:
            product_names = ["NDVI", "albedo"]

        target_geometry = self.centroid(tile)

        if isinstance(start_date, str):
            start_date = parser.parse(start_date).date()

        if isinstance(end_date, str):
            end_date = parser.parse(end_date).date()

        if start_date == end_date:
            end_date = None

        if end_date is None:
            self.logger.info(f"searching Sentinel target {colored_logging.name(tile)} on " + colored_logging.time(f"{start_date:%Y-%m-%d}"))
        else:
            self.logger.info(f"searching Sentinel target {colored_logging.name(tile)} from " + colored_logging.time("f{start_date:%Y-%m-%d}") +
                             " to " + colored_logging.time("f{end_date:%Y-%m-%d}"))

        sentinel_listing = self.search_L2A(start_date=start_date, end_date=end_date, tile=tile, geometry=target_geometry, cloud_min=cloud_min, cloud_max=cloud_max,
                                           max_results=max_results)

        self.logger.info(f"found {colored_logging.val(len(sentinel_listing))} granules")

        for product_name in product_names:
            sentinel_listing[product_name] = pd.Series(dtype=str)

        for product_name in product_names:
            sentinel_listing[product_name] = sentinel_listing.apply(lambda row: self.process_granule(granule_ID=row["ID"], sentinel_ID=row["UUID"], tile=tile,
                                                                                                     acquisition_date=row["date"], filename=row["filename"],
                                                                                                     product_names=product_names)[product_name], axis=1)

        return sentinel_listing


def sentinel(tile: str, start: Union[date, str], end: Union[date, str]):
    logger = logging.getLogger(__name__)
    sentinel = Sentinel()
    listing = sentinel.search_L2A(tile=tile, start_date=start, end_date=end)

    listing = listing[["date", "ID"]]

    for i, (acquisition_date, ID) in listing.iterrows():
        logger.info("* " + colored_logging.time(f"{acquisition_date:%Y-%m-%d}") + ": " + colored_logging.val(ID))

    return listing


def main(argv=sys.argv):
    tile = str(argv[1])
    start = parser.parse(argv[2]).date()
    end = parser.parse(argv[3]).date()

    return sentinel(tile=tile, start=start, end=end)


if __name__ == "__main__":
    sys.exit(main(argv=sys.argv))
