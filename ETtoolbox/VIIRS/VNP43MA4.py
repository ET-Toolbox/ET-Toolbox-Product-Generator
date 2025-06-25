import logging
from datetime import datetime, date
from glob import glob
from os import makedirs
from os.path import exists, join
from typing import List, Union
from dateutil import parser
import h5py
import numpy as np
import pandas as pd
from modland import find_modland_tiles
from shapely.geometry import Point, Polygon

import colored_logging
import rasters as rt
from modland.indices import generate_modland_grid
from rasters import Raster, RasterGrid, RasterGeometry
from .VIIRSDataPool import VIIRSDataPool, VIIRSGranule

NDVI_COLORMAP = "jet_r"
ALBEDO_COLORMAP = "gray"

logger = logging.getLogger(__name__)


class VIIRSUnavailableError(Exception):
    pass


class VNP43MA4Granule(VIIRSGranule):
    def reflectance(self, band: int, geometry: RasterGeometry = None, save_data: bool = False, include_preview: bool = True, apply_QA: bool = True,
                    product_filename: str = None) -> Raster:

        if product_filename is None:
            product_filename = self.product_filename(f"M{band}")

        if product_filename is not None and exists(product_filename):
            logger.info(f"loading VNP43MA4 NBAR M{band}: {colored_logging.file(product_filename)}")
            image = Raster.open(product_filename)
        else:
            image = self.dataset(filename=self.filename, dataset_name=f"HDFEOS/GRIDS/VIIRS_Grid_BRDF/Data Fields/Nadir_Reflectance_M{int(band)}",
                                 fill_value=32767, scale_factor=0.0001)

            if apply_QA:
                QA = self.QA(band=band, geometry=geometry, save_data=save_data, include_preview=include_preview)

                image = rt.where(QA == 0, image, np.nan)

        if save_data and not exists(product_filename):
            logger.info(f"writing VNP43MA4 NBAR M{band}: {colored_logging.file(product_filename)}")
            image.to_geotiff(product_filename, include_preview=include_preview)

        if geometry is not None:
            image = image.to_geometry(geometry)

        return image

    def QA(self, band: int, geometry: RasterGeometry = None, save_data: bool = True, include_preview: bool = True, product_filename: str = None) -> Raster:

        if product_filename is None:
            product_filename = self.product_filename(f"VNP43MA4_QA_M{band}")

        if product_filename is not None and exists(product_filename):
            logger.info(f"loading VNP43MA4 QA M{band}: {colored_logging.file(product_filename)}")
            image = Raster.open(product_filename)
        else:
            dataset_name = f"HDFEOS/GRIDS/VIIRS_Grid_BRDF/Data Fields/BRDF_Albedo_Band_Mandatory_Quality_M{int(band)}"

            with h5py.File(self.filename, "r") as f:
                image = np.array(f[dataset_name])
                h, v = self.hv
                grid = generate_modland_grid(h, v, image.shape[0])

                logger.info("opening file: " + colored_logging.file(self.filename))
                logger.info(f"loading {colored_logging.val(dataset_name)} at " + colored_logging.val(f"{grid.cell_size:0.2f} m") + " resolution")

                image = Raster(image, geometry=grid)

        if save_data and not exists(product_filename):
            logger.info(f"writing VNP43MA4 QA M{band}: {colored_logging.file(product_filename)}")
            image.to_geotiff(product_filename, include_preview=include_preview)

        if geometry is not None:
            image = image.to_geometry(geometry)

        return image

    @property
    def red(self) -> Raster:
        return self.reflectance(5)
    
    @property
    def green(self) -> Raster:
        return self.reflectance(4)

    @property
    def blue(self) -> Raster:
        return self.reflectance(3)

    @property
    def NIR(self) -> Raster:
        return self.reflectance(7)

    @property
    def NDVI(self) -> Raster:
        NDVI = (self.NIR - self.red) / (self.NIR + self.red)
        NDVI = rt.clip(NDVI, -1, 1)

        return NDVI

    @property
    def SWIR1(self) -> Raster:
        return self.reflectance(10)
    
    @property
    def SWIR2(self) -> Raster:
        return self.reflectance(11)

    @property
    def albedo(self) -> Raster:
        M1 = self.reflectance(1)
        M2 = self.reflectance(2)
        M3 = self.reflectance(3)
        M4 = self.reflectance(4)
        M5 = self.reflectance(5)
        M7 = self.reflectance(7)
        M8 = self.reflectance(8)
        M10 = self.reflectance(10)
        M11 = self.reflectance(11)

        albedo = -0.0131 + (M1 * 0.2418) + (M2 * -0.201) + (M3 * 0.2093) + (M4 * 0.1146) + (M5 * 0.1348) + (M7 * 0.2251) + (M8 * 0.1123) + (M10 * 0.086) + (M11 * 0.0803)

        return albedo

    def product(self, product: str) -> Raster:
        if product == "red":
            return self.red
        elif product == "green":
            return self.green
        elif product == "blue":
            return self.blue
        elif product == "NIR":
            return self.NIR
        elif product == "NDVI":
            return self.NDVI
        elif product == "SWIR1":
            return self.SWIR1
        elif product == "SWIR2":
            return self.SWIR2
        elif product == "albedo":
            return self.albedo
        else:
            raise ValueError(f"unrecognized product: {product}")


class VNP43MA4(VIIRSDataPool):
    DEFAULT_DOWNLOAD_DIRECTORY = "VNP43MA4_download"
    DEFAULT_PRODUCTS_DIRECTORY = "VNP43MA4_products"
    DEFAULT_MOSAIC_DIRECTORY = "VNP43MA4_mosaics"

    def __init__(self, username: str = None, password: str = None, remote: str = None, working_directory: str = None, download_directory: str = None,
                 products_directory: str = None, mosaic_directory: str = None, *args, **kwargs):

        super(VNP43MA4, self).__init__(username=username, password=password, remote=remote, working_directory=working_directory, download_directory=download_directory,
                                       products_directory=products_directory, mosaic_directory=mosaic_directory, *args, **kwargs)

        logger.info(f"VNP43MA4 LP-DAAC URL: {colored_logging.URL(self.remote)}")
        logger.info(f"VNP43MA4 working directory: {colored_logging.dir(self.working_directory)}")
        logger.info(f"VNP43MA4 download directory: {colored_logging.dir(self.download_directory)}")
        logger.info(f"VNP43MA4 products directory: {colored_logging.dir(self.products_directory)}")

    def search(self, start_date: date or datetime or str, end_date: date or datetime or str = None, build: str = None, tiles: List[str] or str = None,
               target_geometry: Point or Polygon or RasterGrid = None, *args, **kwargs) -> pd.DataFrame:

        return super(VNP43MA4, self).search(product="VNP43MA4", start_date=start_date, end_date=end_date, build=build, tiles=tiles, target_geometry=target_geometry,
                                      *args, **kwargs)

    def granule(self, date_UTC: Union[date, str], tile: str, build: str = None) -> VNP43MA4Granule:
        # if download_location is None:
        download_location = join(self.download_directory, "VNP43MA4", f"{date_UTC:%Y.%m.%d}")

        makedirs(download_location, exist_ok=True)
        logger.info(f"download location: {colored_logging.dir(download_location)}")

        if exists(download_location):
            filenames = glob(join(download_location, f"VNP43MA4.A{date_UTC:%Y%j}.{tile}.*.h5"))

            if len(filenames) > 0:
                filename = sorted(filenames)[0]
                logger.info(f"found previously retrieved VNP09GA file: {filename}")

                granule = VNP43MA4Granule(filename=filename, products_directory=self.products_directory)

                return granule

        listing = self.search(start_date=date_UTC, end_date=date_UTC, build=build, tiles=[tile])

        if len(listing) > 0:
            URL = listing.iloc[0].URL

        filename = super(VNP43MA4, self).download_URL(URL=URL, download_location=download_location)

        granule = VNP43MA4Granule(filename=filename, products_directory=self.products_directory)

        return granule

    def product_filename(self, target: str, date_UTC: Union[date, str], product: str, resolution: int) -> str:
        if not isinstance(date_UTC, date):
            date_UTC = parser.parse(str(date_UTC)).date()

        timestamp = date_UTC.strftime("%Y.%m.%d")
        product_directory = join(self.products_directory, product, timestamp)
        product_filename_base = f"VNP43MA4_{target}_{timestamp}_{product}_{resolution}m.tif"
        product_filename = join(product_directory, product_filename_base)

        return product_filename

    def product(self, product: str, date_UTC: Union[date, str], geometry: RasterGeometry, target: str = None, filename: str = None, save_data: bool = True,
                resampling: str = None) -> Raster:

        if isinstance(date_UTC, str):
            date_UTC = parser.parse(date_UTC).date()

        if filename is None and target is not None:
            filename = self.product_filename(target=target, date_UTC=date_UTC, product=product, resolution=int(geometry.cell_size_meters))

        if filename is not None and exists(filename):
            return Raster.open(filename)

        tiles = sorted(find_modland_tiles(geometry.boundary_latlon))

        if len(tiles) == 0:
            raise ValueError("no VIIRS tiles found covering target geometry")

        composite = None

        for tile in tiles:
            granule = self.granule(date_UTC=date_UTC, tile=tile)
            granule_image = granule.product(product=product)
            producted_image = granule_image.to_geometry(geometry, resampling=resampling)

            if composite is None:
                composite = producted_image
            else:
                composite = rt.where(np.isnan(composite), producted_image, composite)

        if composite is None:
            raise ValueError("VIIRS composite did not generate")

        if save_data and filename is not None:
            logger.info("writing composite: " + colored_logging.file(filename))
            composite.to_geotiff(filename)

        return composite

    def red(self, date_UTC: Union[date, str], geometry: RasterGeometry, resampling: str = "cubic") -> Raster:
        return self.product(product="red", date_UTC=date_UTC, geometry=geometry, resampling=resampling)

    def NIR(self, date_UTC: Union[date, str], geometry: RasterGeometry, resampling: str = "cubic") -> Raster:
        return self.product(product="NIR", date_UTC=date_UTC, geometry=geometry, resampling=resampling)

    def NDVI(self, date_UTC: Union[date, str], geometry: RasterGeometry, resampling: str = "cubic") -> Raster:
        return self.product(product="NDVI", date_UTC=date_UTC, geometry=geometry, resampling=resampling)

    def SWIR1(self, date_UTC: Union[date, str], geometry: RasterGeometry, resampling: str = "cubic") -> Raster:
        return self.product(product="SWIR1", date_UTC=date_UTC, geometry=geometry, resampling=resampling)

    def albedo(self, date_UTC: Union[date, str], geometry: RasterGeometry, resampling: str = "cubic") -> Raster:
        return self.product(product="albedo", date_UTC=date_UTC, geometry=geometry, resampling=resampling)
