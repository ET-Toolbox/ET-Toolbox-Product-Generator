import logging
from datetime import datetime, date
from glob import glob
from os import makedirs
from os.path import exists, join
from typing import List

import h5py
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from ..daterange import date_range
from dateutil import parser
from shapely.geometry import Point, Polygon
from skimage.transform import resize

import colored_logging
import rasters
import rasters as rt
from modland.indices import parsehv, generate_modland_grid
from rasters import Raster, RasterGrid, RasterGeometry
from .VIIRSDataPool import VIIRSDataPool, parse_VIIRS_tile, find_modland_tiles, VIIRSGranule

NDVI_COLORMAP = LinearSegmentedColormap.from_list(
    name="NDVI",
    colors=[
        "#0000ff",
        "#000000",
        "#745d1a",
        "#e1dea2",
        "#45ff01",
        "#325e32"
    ]
)

ALBEDO_COLORMAP = "gray"

logger = logging.getLogger(__name__)

class VIIRSUnavailableError(Exception):
    pass


class VNP09GAGranule(VIIRSGranule):
    CLOUD_DATASET_NAME = "HDFEOS/GRIDS/VNP_Grid_1km_2D/Data Fields/SurfReflect_QF1_1"

    def get_cloud_mask(self, target_shape: tuple = None) -> Raster:
        h, v = self.hv

        if self._cloud_mask is None:
            with h5py.File(self.filename, "r") as f:
                QF1 = np.array(f[self.CLOUD_DATASET_NAME])
                cloud_levels = (QF1 >> 2) & 3
                cloud_mask = cloud_levels > 0
                self._cloud_mask = cloud_mask
        else:
            cloud_mask = self._cloud_mask

        if target_shape is not None:
            cloud_mask = resize(cloud_mask, target_shape, order=0).astype(bool)
            shape = target_shape
        else:
            shape = cloud_mask.shape

        geometry = generate_modland_grid(h, v, shape[0])
        cloud_mask = Raster(cloud_mask, geometry=geometry)

        return cloud_mask

    cloud_mask = property(get_cloud_mask)

    def dataset(self, filename: str, dataset_name: str, scale_factor: float, cloud_mask: Raster = None, apply_cloud_mask: bool = True, geometry: RasterGeometry = None,
                resampling: str = None) -> Raster:

        with h5py.File(filename, "r") as f:
            DN = np.array(f[dataset_name])
            h, v = self.hv
            grid = generate_modland_grid(h, v, DN.shape[0])
            logger.info(f"opening VIIRS file: {colored_logging.file(self.filename)}")
            logger.info(f"loading {colored_logging.val(dataset_name)} at {colored_logging.val(f'{grid.cell_size:0.2f} m')} resolution")
            DN = Raster(DN, geometry=grid)

        data = DN * scale_factor

        if apply_cloud_mask:
            if cloud_mask is None:
                cloud_mask = self.get_cloud_mask(target_shape=DN.shape)

            data = rt.where(cloud_mask, np.nan, data)

        if geometry is not None:
            data = data.to_geometry(geometry, resampling=resampling)

        return data

    @property
    def geometry_M(self) -> RasterGrid:
        return generate_modland_grid(*self.hv, 1200)

    @property
    def geometry_I(self) -> RasterGrid:
        return generate_modland_grid(*self.hv, 2400)

    def geometry(self, band: str) -> RasterGrid:
        try:
            band_letter = band[0]
        except Exception as e:
            raise ValueError(f"invalid band: {band}")

        if band_letter == "I":
            return self.geometry_I
        elif band_letter == "M":
            return self.geometry_M
        else:
            raise ValueError(f"invalid band: {band}")

    def get_sensor_zenith_M(self, geometry: RasterGeometry = None, save_data: bool = True, save_preview: bool = True, product_filename: str = None) -> Raster:
        if product_filename is None:
            product_filename = self.product_filename(f"sensor_zenith_M")

        if product_filename is not None and exists(product_filename):
            logger.info(f"loading VIIRS sensor zenith: {colored_logging.file(product_filename)}")
            image = Raster.open(product_filename)
        else:
            image = self.dataset(self.filename, f"HDFEOS/GRIDS/VNP_Grid_1km_2D/Data Fields/SensorZenith_1", 0.01, cloud_mask=None,
                                 apply_cloud_mask=False)

        if np.all(np.isnan(image)):
            raise ValueError("blank sensor zenith image")

        if save_data and not exists(product_filename):
            logger.info(f"writing VIIRS sensor zenith: {colored_logging.file(product_filename)} {colored_logging.val(image.shape)}")
            image.to_geotiff(product_filename)

            if save_preview:
                image.percentilecut.to_geojpeg(product_filename.replace(".tif", ".jpeg"), quality=20, remove_XML=True)

        if geometry is not None:
            image = image.to_geometry(geometry)

        return image

    sensor_zenith_M = property(get_sensor_zenith_M)

    def get_sensor_zenith_I(self, geometry: RasterGeometry = None, save_data: bool = True, save_preview: bool = True, product_filename: str = None) -> Raster:
        if product_filename is None:
            product_filename = self.product_filename(f"sensor_zenith_I")

        if product_filename is not None and exists(product_filename):
            logger.info(f"loading VIIRS sensor zenith: {colored_logging.file(product_filename)}")
            image = Raster.open(product_filename)
        else:
            h, v = self.hv
            grid_I = generate_modland_grid(h, v, 2400)

            image = self.dataset(self.filename, f"HDFEOS/GRIDS/VNP_Grid_1km_2D/Data Fields/SensorZenith_1", 0.01, cloud_mask=None,
                                 apply_cloud_mask=False, geometry=grid_I, resampling="cubic")

        if np.all(np.isnan(image)):
            raise ValueError("blank sensor zenith image")

        if save_data and not exists(product_filename):
            logger.info(f"writing VIIRS sensor zenith: {colored_logging.file(product_filename)} {colored_logging.val(image.shape)}")
            image.to_geotiff(product_filename)

            if save_preview:
                image.percentilecut.to_geojpeg(product_filename.replace(".tif", ".jpeg"), quality=20, remove_XML=True)

        if geometry is not None:
            image = image.to_geometry(geometry)

        return image

    sensor_zenith_I = property(get_sensor_zenith_I)

    def sensor_zenith(self, band: str, geometry: RasterGeometry = None, save_data: bool = True, save_preview: bool = True, product_filename: str = None) -> Raster:
        try:
            band_letter = band[0]
        except Exception as e:
            raise ValueError(f"invalid band: {band}")

        if band_letter == "I":
            return self.get_sensor_zenith_I(geometry=geometry, save_data=save_data, save_preview=save_preview, product_filename=product_filename)
        elif band_letter == "M":
            return self.get_sensor_zenith_M(geometry=geometry, save_data=save_data, save_preview=save_preview, product_filename=product_filename)
        else:
            raise ValueError(f"invalid band: {band}")

    def get_sensor_azimuth_M(self, geometry: RasterGeometry = None, save_data: bool = True, save_preview: bool = True, product_filename: str = None) -> Raster:
        if product_filename is None:
            product_filename = self.product_filename(f"sensor_azimuth_M")

        if product_filename is not None and exists(product_filename):
            logger.info(f"loading VIIRS sensor azimuth: {colored_logging.file(product_filename)}")
            image = Raster.open(product_filename)
        else:
            image = self.dataset(self.filename,f"HDFEOS/GRIDS/VNP_Grid_1km_2D/Data Fields/SensorAzimuth_1",0.01, cloud_mask=None,
                                 apply_cloud_mask=False)

        if np.all(np.isnan(image)):
            raise ValueError("blank sensor azimuth image")

        if save_data and not exists(product_filename):
            logger.info(f"writing VIIRS sensor azimuth: {colored_logging.file(product_filename)} {colored_logging.val(image.shape)}")
            image.to_geotiff(product_filename)

            if save_preview:
                image.percentilecut.to_geojpeg(product_filename.replace(".tif", ".jpeg"), quality=20, remove_XML=True)

        if geometry is not None:
            image = image.to_geometry(geometry)

        return image

    sensor_azimuth_M = property(get_sensor_azimuth_M)

    def get_sensor_azimuth_I(self, geometry: RasterGeometry = None, save_data: bool = True, save_preview: bool = True, product_filename: str = None) -> Raster:
        if product_filename is None:
            product_filename = self.product_filename(f"sensor_azimuth_I")

        if product_filename is not None and exists(product_filename):
            logger.info(f"loading VIIRS sensor azimuth: {colored_logging.file(product_filename)}")
            image = Raster.open(product_filename)
        else:
            h, v = self.hv
            grid_I = generate_modland_grid(h, v, 2400)

            image = self.dataset(self.filename, f"HDFEOS/GRIDS/VNP_Grid_1km_2D/Data Fields/SensorAzimuth_1", 0.01, cloud_mask=None,
                                 apply_cloud_mask=False, geometry=grid_I, resampling="cubic" )

        if np.all(np.isnan(image)):
            raise ValueError("blank sensor azimuth image")

        if save_data and not exists(product_filename):
            logger.info(f"writing VIIRS sensor azimuth: {colored_logging.file(product_filename)} {colored_logging.val(image.shape)}")
            image.to_geotiff(product_filename)

            if save_preview:
                image.percentilecut.to_geojpeg(product_filename.replace(".tif", ".jpeg"), quality=20, remove_XML=True)

        if geometry is not None:
            image = image.to_geometry(geometry)

        return image

    sensor_azimuth_I = property(get_sensor_azimuth_I)

    def sensor_azimuth(self, band: str, geometry: RasterGeometry = None, save_data: bool = True, save_preview: bool = True, product_filename: str = None) -> Raster:
        try:
            band_letter = band[0]
        except Exception as e:
            raise ValueError(f"invalid band: {band}")

        if band_letter == "I":
            return self.get_sensor_azimuth_I(geometry=geometry, save_data=save_data, save_preview=save_preview, product_filename=product_filename)
        elif band_letter == "M":
            return self.get_sensor_azimuth_M(geometry=geometry, save_data=save_data, save_preview=save_preview, product_filename=product_filename)
        else:
            raise ValueError(f"invalid band: {band}")

    def get_solar_zenith_M(self, geometry: RasterGeometry = None, save_data: bool = True, save_preview: bool = True, product_filename: str = None) -> Raster:
        if product_filename is None:
            product_filename = self.product_filename(f"solar_zenith_M")

        if product_filename is not None and exists(product_filename):
            logger.info(f"loading VIIRS solar zenith: {colored_logging.file(product_filename)}")
            image = Raster.open(product_filename)
        else:
            image = self.dataset(self.filename,f"HDFEOS/GRIDS/VNP_Grid_1km_2D/Data Fields/SolarZenith_1",0.01, cloud_mask=None,
                                 apply_cloud_mask=False)

        if np.all(np.isnan(image)):
            raise ValueError("blank solar zenith image")

        if save_data and not exists(product_filename):
            logger.info(f"writing VIIRS solar zenith: {colored_logging.file(product_filename)} {colored_logging.val(image.shape)}")
            image.to_geotiff(product_filename)

            if save_preview:
                image.percentilecut.to_geojpeg(product_filename.replace(".tif", ".jpeg"), quality=20, remove_XML=True)

        if geometry is not None:
            image = image.to_geometry(geometry)

        return image

    solar_zenith_M = property(get_solar_zenith_M)

    def get_solar_zenith_I(self, geometry: RasterGeometry = None, save_data: bool = True, save_preview: bool = True, product_filename: str = None) -> Raster:
        if product_filename is None:
            product_filename = self.product_filename("solar_zenith_I")

        if product_filename is not None and exists(product_filename):
            logger.info(f"loading VIIRS solar zenith: {colored_logging.file(product_filename)}")
            image = Raster.open(product_filename)
        else:
            h, v = self.hv
            grid_I = generate_modland_grid(h, v, 2400)

            image = self.dataset(self.filename, f"HDFEOS/GRIDS/VNP_Grid_1km_2D/Data Fields/SolarZenith_1", 0.01, cloud_mask=None,
                                 apply_cloud_mask=False, geometry=grid_I, resampling="cubic")

        if np.all(np.isnan(image)):
            raise ValueError("blank solar zenith image")

        if save_data and not exists(product_filename):
            logger.info(f"writing VIIRS solar zenith: {colored_logging.file(product_filename)} {colored_logging.val(image.shape)}")

            image.to_geotiff(product_filename)

            if save_preview:
                image.percentilecut.to_geojpeg(product_filename.replace(".tif", ".jpeg"), quality=20, remove_XML=True)

        if geometry is not None:
            image = image.to_geometry(geometry)

        return image

    solar_zenith_I = property(get_solar_zenith_I)

    def solar_zenith(self, band: str, geometry: RasterGeometry = None, save_data: bool = True, save_preview: bool = True, product_filename: str = None) -> Raster:
        try:
            band_letter = band[0]
        except Exception as e:
            raise ValueError(f"invalid band: {band}")

        if band_letter == "I":
            return self.get_solar_zenith_I(geometry=geometry, save_data=save_data, save_preview=save_preview, product_filename=product_filename)
        elif band_letter == "M":
            return self.get_solar_zenith_M(geometry=geometry, save_data=save_data, save_preview=save_preview, product_filename=product_filename)
        else:
            raise ValueError(f"invalid band: {band}")

    def get_solar_azimuth_M(self, geometry: RasterGeometry = None, save_data: bool = True, save_preview: bool = True, product_filename: str = None) -> Raster:
        if product_filename is None:
            product_filename = self.product_filename("solar_azimuth_M")

        if product_filename is not None and exists(product_filename):
            logger.info(f"loading VIIRS solar azimuth: {colored_logging.file(product_filename)}")
            image = Raster.open(product_filename)
        else:
            image = self.dataset(self.filename, f"HDFEOS/GRIDS/VNP_Grid_1km_2D/Data Fields/SolarAzimuth_1", 0.01, cloud_mask=None,
                                 apply_cloud_mask=False)

        if np.all(np.isnan(image)):
            raise ValueError("blank solar azimuth image")

        if save_data and not exists(product_filename):
            logger.info(f"writing VIIRS solar azimuth: {colored_logging.file(product_filename)} {colored_logging.val(image.shape)}")
            image.to_geotiff(product_filename)

            if save_preview:
                image.percentilecut.to_geojpeg(product_filename.replace(".tif", ".jpeg"), quality=20, remove_XML=True)

        if geometry is not None:
            image = image.to_geometry(geometry)

        return image

    solar_azimuth_M = property(get_solar_azimuth_M)

    def get_solar_azimuth_I(self, geometry: RasterGeometry = None, save_data: bool = True, save_preview: bool = True, product_filename: str = None) -> Raster:

        if product_filename is None:
            product_filename = self.product_filename("solar_azimuth_I")

        if product_filename is not None and exists(product_filename):
            logger.info(f"loading VIIRS solar azimuth: {colored_logging.file(product_filename)}")
            image = Raster.open(product_filename)
        else:
            h, v = self.hv
            grid_I = generate_modland_grid(h, v, 2400)

            image = self.dataset(self.filename, f"HDFEOS/GRIDS/VNP_Grid_1km_2D/Data Fields/SolarAzimuth_1", 0.01, cloud_mask=None,
                                 apply_cloud_mask=False, geometry=grid_I, resampling="cubic")

        if np.all(np.isnan(image)):
            raise ValueError("blank solar azimuth image")

        if save_data and not exists(product_filename):
            logger.info(f"writing VIIRS solar azimuth: {colored_logging.file(product_filename)} {colored_logging.val(image.shape)}")
            image.to_geotiff(product_filename)

            if save_preview:
                image.percentilecut.to_geojpeg(product_filename.replace(".tif", ".jpeg"), quality=20, remove_XML=True)

        if geometry is not None:
            image = image.to_geometry(geometry)

        return image

    solar_azimuth_I = property(get_solar_azimuth_I)

    def solar_azimuth(self, band: str, geometry: RasterGeometry = None, save_data: bool = True, save_preview: bool = True, product_filename: str = None) -> Raster:
        try:
            band_letter = band[0]
        except Exception as e:
            raise ValueError(f"invalid band: {band}")

        if band_letter == "I":
            return self.get_solar_azimuth_I(geometry=geometry, save_data=save_data, save_preview=save_preview, product_filename=product_filename)
        elif band_letter == "M":
            return self.get_solar_azimuth_M(geometry=geometry, save_data=save_data, save_preview=save_preview, product_filename=product_filename)
        else:
            raise ValueError(f"invalid band: {band}")

    def get_M_band(self, band: int, cloud_mask: Raster = None, apply_cloud_mask: bool = True, geometry: RasterGeometry = None, save_data: bool = True,
                   save_preview: bool = True, product_filename: str = None) -> Raster:
        if product_filename is None:
            product_filename = self.product_filename(f"M{band}")

        if product_filename is not None and exists(product_filename):
            logger.info(f"loading VIIRS M{band}: {colored_logging.file(product_filename)}")
            image = Raster.open(product_filename)
        else:
            image = self.dataset(self.filename, f"HDFEOS/GRIDS/VNP_Grid_1km_2D/Data Fields/SurfReflect_M{int(band)}_1", 0.0001,
                                 cloud_mask=cloud_mask, apply_cloud_mask=apply_cloud_mask)

        if save_data and not exists(product_filename):
            logger.info(f"writing VIIRS M{band}: {colored_logging.file(product_filename)}")
            image.to_geotiff(product_filename)

            if save_preview:
                image.percentilecut.to_geojpeg(product_filename.replace(".tif", ".jpeg"), quality=20, remove_XML=True)

        if geometry is not None:
            image = image.to_geometry(geometry)

        return image

    def get_I_band(self, band: int, cloud_mask: Raster = None, apply_cloud_mask: bool = True, geometry: RasterGeometry = None, save_data: bool = True,
                   save_preview: bool = True, product_filename: str = None) -> Raster:
        if product_filename is None:
            product_filename = self.product_filename(f"I{band}")

        if product_filename is not None and exists(product_filename):
            logger.info(f"loading VIIRS I{band}: {colored_logging.file(product_filename)}")
            image = Raster.open(product_filename)
        else:
            image = self.dataset(self.filename, f"HDFEOS/GRIDS/VNP_Grid_500m_2D/Data Fields/SurfReflect_I{int(band)}_1", 0.0001, cloud_mask=cloud_mask,
                                 apply_cloud_mask=apply_cloud_mask)

        if save_data and not exists(product_filename):
            logger.info(f"writing VIIRS I{band}: {colored_logging.file(product_filename)}")
            image.to_geotiff(product_filename)

            if save_preview:
                image.percentilecut.to_geojpeg(product_filename.replace(".tif", ".jpeg"), quality=20, remove_XML=True)

        if geometry is not None:
            image = image.to_geometry(geometry)

        return image

    def band(self, band: str, cloud_mask: Raster = None, apply_cloud_mask: bool = True, geometry: RasterGeometry = None, save_data: bool = True,
             save_preview: bool = True, product_filename: str = None) -> Raster:

        try:
            band_letter = band[0]
            band_number = int(band[1:])
        except Exception as e:
            raise ValueError(f"invalid band: {band}")

        if band_letter == "I":
            return self.get_I_band(band=band_number, cloud_mask=cloud_mask, apply_cloud_mask=apply_cloud_mask, geometry=geometry, save_data=save_data,
                                   save_preview=save_preview, product_filename=product_filename)
        elif band_letter == "M":
            return self.get_M_band(band=band_number, cloud_mask=cloud_mask, apply_cloud_mask=apply_cloud_mask, geometry=geometry, save_data=save_data,
                                   save_preview=save_preview, product_filename=product_filename)
        else:
            raise ValueError(f"invalid band: {band}")

    def get_red(self, cloud_mask: Raster = None, apply_cloud_mask: bool = True, geometry: RasterGeometry = None, save_data: bool = True, save_preview: bool = True,
                product_filename: str = None) -> Raster:

        return self.get_I_band(band=1, cloud_mask=cloud_mask, apply_cloud_mask=apply_cloud_mask, geometry=geometry, save_data=save_data, save_preview=save_preview,
                               product_filename=product_filename)

    red = property(get_red)

    def get_NIR(self, cloud_mask: Raster = None, apply_cloud_mask: bool = True, geometry: RasterGeometry = None, save_data: bool = True, save_preview: bool = True,
                product_filename: str = None) -> Raster:
        return self.get_I_band(band=2, cloud_mask=cloud_mask, apply_cloud_mask=apply_cloud_mask, geometry=geometry, save_data=save_data, save_preview=save_preview,
                               product_filename=product_filename)

    NIR = property(get_NIR)

    def get_NDVI(self, cloud_mask: Raster = None, apply_cloud_mask: bool = True, geometry: RasterGeometry = None, save_data: bool = True, save_preview: bool = True,
                 product_filename: str = None) -> Raster:
        if product_filename is None:
            product_filename = self.product_filename("NDVI")

        if product_filename is not None and exists(product_filename):
            logger.info(f"loading VIIRS NDVI: {colored_logging.file(product_filename)}")
            NDVI = Raster.open(product_filename)
        else:
            red = self.get_red(cloud_mask=cloud_mask, apply_cloud_mask=apply_cloud_mask, geometry=geometry, save_data=save_data, save_preview=save_preview)

            NIR = self.get_NIR(cloud_mask=cloud_mask, apply_cloud_mask=apply_cloud_mask, geometry=geometry, save_data=save_data, save_preview=save_preview)

            NDVI = np.clip((NIR - red) / (NIR + red), -1, 1)

        if save_data and not exists(product_filename):
            logger.info(f"writing VIIRS NDVI: {colored_logging.file(product_filename)}")
            NDVI.to_geotiff(product_filename)

            if save_preview:
                NDVI.percentilecut.to_geojpeg(product_filename.replace(".tif", ".jpeg"))

        if geometry is not None:
            NDVI = NDVI.to_geometry(geometry)

        NDVI.cmap = NDVI_COLORMAP

        return NDVI

    NDVI = property(get_NDVI)

    def get_albedo(self, cloud_mask: Raster = None, apply_cloud_mask: bool = True, geometry: RasterGeometry = None, save_data: bool = True, save_preview: bool = True,
                  product_filename: str = None) -> Raster:

        if product_filename is None:
            product_filename = self.product_filename("albedo")

        if product_filename is not None and exists(product_filename):
            logger.info(f"loading VIIRS albedo: {colored_logging.file(product_filename)}")
            albedo = Raster.open(product_filename)
        else:
            b1 = self.get_M_band(1, cloud_mask=cloud_mask, apply_cloud_mask=apply_cloud_mask, geometry=geometry, save_data=save_data, save_preview=save_preview)
            b2 = self.get_M_band(2, cloud_mask=cloud_mask, apply_cloud_mask=apply_cloud_mask, geometry=geometry, save_data=save_data, save_preview=save_preview)
            b3 = self.get_M_band(3, cloud_mask=cloud_mask, apply_cloud_mask=apply_cloud_mask, geometry=geometry, save_data=save_data, save_preview=save_preview)
            b4 = self.get_M_band(4, cloud_mask=cloud_mask, apply_cloud_mask=apply_cloud_mask, geometry=geometry, save_data=save_data, save_preview=save_preview)
            b5 = self.get_M_band(5, cloud_mask=cloud_mask, apply_cloud_mask=apply_cloud_mask, geometry=geometry, save_data=save_data, save_preview=save_preview)
            b7 = self.get_M_band(7, cloud_mask=cloud_mask, apply_cloud_mask=apply_cloud_mask, geometry=geometry, save_data=save_data, save_preview=save_preview)
            b8 = self.get_M_band(8, cloud_mask=cloud_mask, apply_cloud_mask=apply_cloud_mask, geometry=geometry, save_data=save_data, save_preview=save_preview)
            b10 = self.get_M_band(10,cloud_mask=cloud_mask, apply_cloud_mask=apply_cloud_mask, geometry=geometry, save_data=save_data, save_preview=save_preview)
            b11 = self.get_M_band(11, cloud_mask=cloud_mask, apply_cloud_mask=apply_cloud_mask, geometry=geometry, save_data=save_data, save_preview=save_preview)

            # https://lpdaac.usgs.gov/documents/194/VNP43_ATBD_V1.pdf
            albedo = 0.2418 * b1 - 0.201 * b2 + 0.2093 * b3 + 0.1146 * b4 + 0.1348 * b5 + 0.2251 * b7 + 0.1123 * b8 + 0.0860 * b10 + 0.0803 * b11 - 0.0131
            albedo = np.clip(albedo, 0, 1)

        if save_data and not exists(product_filename):
            logger.info(f"writing VIIRS albedo: {colored_logging.file(product_filename)}")
            albedo.to_geotiff(product_filename)

        if geometry is not None:
            logger.info(f"projecting VIIRS albedo from {colored_logging.val(albedo.geometry.cell_size)} to {colored_logging.val(geometry.cell_size)}")
            albedo = albedo.to_geometry(geometry)

        albedo.cmap = ALBEDO_COLORMAP

        return albedo

    albedo = property(get_albedo)


class VNP09GA(VIIRSDataPool):
    DEFAULT_DOWNLOAD_DIRECTORY = "VNP09GA_download"
    DEFAULT_PRODUCTS_DIRECTORY = "VNP09GA_products"
    DEFAULT_MOSAIC_DIRECTORY = "VNP09GA_mosaics"
    DEFAULT_RESAMPLING = "nearest"

    def __init__(self, *args, username: str = None, password: str = None, remote: str = None, working_directory: str = None, download_directory: str = None,
                 products_directory: str = None, mosaic_directory: str = None, resampling: str = None, **kwargs):
        super(VNP09GA, self).__init__(*args, username=username, password=password, remote=remote, working_directory=working_directory,
                                      download_directory=download_directory, products_directory=products_directory, mosaic_directory=mosaic_directory, **kwargs)

        if resampling is None:
            resampling = self.DEFAULT_RESAMPLING

        self.resampling = resampling

    def search(self, start_date: date or datetime or str, end_date: date or datetime or str = None, build: str = None, tiles: List[str] or str = None,
               target_geometry: Point or Polygon or RasterGrid = None, *args, **kwargs) -> pd.DataFrame:

        return super(VNP09GA, self).search(product="VNP09GA", start_date=start_date, end_date=end_date, build=build, tiles=tiles, target_geometry=target_geometry,
                                     *args, **kwargs)

    def granule(self, date_UTC: date or str, tile: str, build: str = None) -> VNP09GAGranule:
        if isinstance(date_UTC, str):
            date_UTC = parser.parse(date_UTC).date()

        download_location = join(self.download_directory, "VNP09GA", f"{date_UTC:%Y.%m.%d}")

        if exists(download_location):
            filenames = glob(join(download_location, f"VNP09GA.A{date_UTC:%Y%j}.{tile}.*.h5"))

            if len(filenames) > 0:
                filename = sorted(filenames)[0]
                logger.info(f"found previously retrieved VNP09GA file: {filename}")

                granule = VNP09GAGranule(filename=filename, products_directory=self.products_directory)

                return granule

        listing = self.search(start_date=date_UTC, end_date=date_UTC, build=build, tiles=[tile])

        if len(listing) > 0:
            URL = listing.iloc[0].URL

        makedirs(download_location, exist_ok=True)

        filename = super(VNP09GA, self).download_URL(URL=URL, download_location=download_location)

        granule = VNP09GAGranule(filename=filename, products_directory=self.products_directory)

        return granule

    def get_cloud_mask(self, target_shape: tuple = None) -> Raster:
        h, v = self.hv

        if self._cloud_mask is None:
            with h5py.File(self.filename, "r") as f:
                QF1 = np.array(f[self.CLOUD_DATASET_NAME])
                cloud_levels = (QF1 >> 2) & 3
                cloud_mask = cloud_levels > 0
                self._cloud_mask = cloud_mask
        else:
            cloud_mask = self._cloud_mask

        if target_shape is not None:
            cloud_mask = resize(cloud_mask, target_shape, order=0).astype(bool)

        geometry = generate_modland_grid(h, v, target_shape[0])
        cloud_mask = Raster(cloud_mask, geometry=geometry)

        return cloud_mask

    cloud_mask = property(get_cloud_mask)

    def dataset(self, filename: str, dataset_name: str, scale_factor: float, cloud_mask: Raster = None, apply_cloud_mask: bool = True) -> Raster:
        tile = parse_VIIRS_tile(filename)
        h, v = parsehv(tile)

        with h5py.File(filename, "r") as f:
            DN = np.array(f[dataset_name])
            grid = generate_modland_grid(h, v, DN.shape[0])
            logger.info(f"loading {colored_logging.val(dataset_name)} at {colored_logging.val(f'{grid.cell_size} m')} resolution from {colored_logging.file(filename)}")
            DN = Raster(DN, geometry=grid)

        data = DN * scale_factor

        if apply_cloud_mask:
            if cloud_mask is None:
                cloud_mask = self.get_cloud_mask(target_shape=DN.shape)

            data = rt.where(cloud_mask, np.nan, data)

        return data

    def NDVI(self, date_UTC: date or str, geometry: RasterGeometry, filename: str = None, resampling: str = None) -> Raster:

        if isinstance(date_UTC, str):
            date_UTC = parser.parse(date_UTC).date()

        if filename is not None and exists(filename):
            return Raster.open(filename, cmap=NDVI_COLORMAP)

        if resampling is None:
            resampling = self.resampling

        tiles = sorted(find_modland_tiles(geometry.boundary_latlon))

        if len(tiles) == 0:
            raise ValueError("no VIIRS tiles found covering target geometry")

        NDVI = None

        for tile in tiles:
            granule = self.granule(date_UTC=date_UTC, tile=tile)
            granule_NDVI = granule.NDVI
            projected_NDVI = granule_NDVI.to_geometry(geometry, resampling=resampling)

            if NDVI is None:
                NDVI = projected_NDVI
            else:
                NDVI = rasters.where(np.isnan(NDVI), projected_NDVI, NDVI)

        if NDVI is None:
            raise ValueError("VIIRS NDVI did not generate")

        NDVI.cmap = NDVI_COLORMAP

        if filename is not None:
            logger.info(f"writing NDVI mosaic: {colored_logging.file(filename)}")
            NDVI.to_geotiff(filename)

        return NDVI

    def albedo(self, acquisition_date: date or str, geometry: RasterGeometry, filename: str = None, resampling: str = None) -> Raster:
        if isinstance(acquisition_date, str):
            acquisition_date = parser.parse(acquisition_date).date()

        if filename is not None and exists(filename):
            return Raster.open(filename, cmap=ALBEDO_COLORMAP)

        if resampling is None:
            resampling = self.resampling

        tiles = sorted(find_modland_tiles(geometry.boundary_latlon))
        albedo = None

        for tile in tiles:
            granule = self.granule(date_UTC=acquisition_date, tile=tile)
            granule_albedo = granule.albedo
            source_cell_size = granule_albedo.geometry.cell_size
            dest_cell_size = geometry.cell_size
            logger.info(f"projecting VIIRS albedo from {colored_logging.val(f'{source_cell_size} m')} to {colored_logging.val(f'{dest_cell_size} m')}")
            projected_albedo = granule_albedo.to_geometry(geometry, resampling=resampling)

            if albedo is None:
                albedo = projected_albedo
            else:
                albedo = rasters.where(np.isnan(albedo), projected_albedo, albedo)

        albedo.cmap = ALBEDO_COLORMAP

        if filename is not None:
            logger.info(f"writing albedo mosaic: {colored_logging.file(filename)}")
            albedo.to_geotiff(filename)

        return albedo


    def process(self, start: date or str, target_geometry: RasterGeometry, target: str, end: date or str = None, product_names: List[str] = None) -> pd.DataFrame:
        if product_names is None:
            product_names = ["NDVI", "albedo"]

        if isinstance(start, str):
            start = parser.parse(start).date()

        if end is None:
            end = start

        if isinstance(end, str):
            end = parser.parse(end).date()

        if start == end:
            logger.info(f"processing VIIRS at {colored_logging.place(target)} on {colored_logging.time(start)}")
        else:
            logger.info(f"processing VIIRS at {colored_logging.place(target)} from {colored_logging.time(start)} to {colored_logging.time(end)}"
                             )

        if not isinstance(target_geometry, RasterGeometry):
            raise ValueError("invalid target geometry")

        if not isinstance(target, str):
            raise ValueError(f"invalid target name: {target}")

        rows = []

        for acquisition_date in date_range(start, end):
            row = [acquisition_date]

            for product in product_names:

                product_filename = join(self.mosaic_directory, product, f"{acquisition_date:%Y.%m.%d}",
                                        f"{acquisition_date:%Y.%m.%d}_{target}_{product}_{int(target_geometry.cell_size)}m.tif")

                if exists(product_filename):
                    logger.info(f"VIIRS {colored_logging.val(product)} already exists: {colored_logging.file(product_filename)}")
                else:
                    logger.info(f"generating VIIRS {colored_logging.val(product)} mosaic at " + colored_logging.place(target) + "on " +
                                colored_logging.time(f"{acquisition_date:%Y-%m-%d}"))

                    if product == "NDVI":
                        self.NDVI(date_UTC=acquisition_date, geometry=target_geometry, filename=product_filename)
                    elif product == "albedo":
                        self.albedo(acquisition_date=acquisition_date, geometry=target_geometry, filename=product_filename)
                    else:
                        raise ValueError(f"unrecognized product: {product}")

                row.append(product_filename)

            # rows.append([acquisition_date, product_filename, VIIRS_albedo_filename])
            rows.append(row)

        df = pd.DataFrame(rows, columns=["date"] + product_names)

        return df
