import logging
from datetime import datetime, date
from glob import glob
from os import makedirs
from os.path import exists, join
from typing import List

import h5py
import numpy as np
import pandas as pd
from dateutil import parser
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point, Polygon
from skimage.transform import resize

import colored_logging
import rasters
import rasters as rt
from modland.indices import parsehv, generate_modland_grid
from ..daterange import date_range
from rasters import Raster, RasterGrid, RasterGeometry
from .VIIRSDataPool import VIIRSDataPool, parse_VIIRS_tile, find_modland_tiles, VIIRSGranule

NDVI_COLORMAP = LinearSegmentedColormap.from_list(name="NDVI",
                                                  colors=["#0000ff", "#000000", "#745d1a", "#e1dea2", "#45ff01","#325e32"])

ALBEDO_COLORMAP = "gray"

logger = logging.getLogger(__name__)


class VIIRSUnavailableError(Exception):
    pass


class VNP21A1DGranule(VIIRSGranule):
    CLOUD_DATASET_NAME = "HDFEOS/GRIDS/VNP_Grid_1km_2D/Data Fields/SurfReflect_QF1_1"

    def get_QC(self, geometry: RasterGeometry = None, resampling: str = "nearest") -> Raster:
        with h5py.File(self.filename, "r") as f:
            dataset_name = "HDFEOS/GRIDS/VIIRS_Grid_Daily_1km_LST21/Data Fields/QC"
            QC = np.array(f[dataset_name])
            h, v = self.hv
            grid = generate_modland_grid(h, v, QC.shape[0])

            logger.info("opening VIIRS file: " + colored_logging.file(self.filename))

            logger.info(f"loading {colored_logging.val(dataset_name)} at " + colored_logging.val(f"{grid.cell_size:0.2f} m") + " resolution")

            QC = Raster(QC, geometry=grid)

        if geometry is not None:
            QC = QC.to_geometry(geometry, resampling=resampling)

        return QC

    QC = property(get_QC)

    def get_cloud_mask(self, target_shape: tuple = None) -> Raster:
        h, v = self.hv

        if self._cloud_mask is None:
            QC = self.QC
            cloud_mask = ((QC >> 4) & 3) > 0
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

    def dataset(self, filename: str, dataset_name: str, scale_factor: float = 1, offset: float = 0, fill: float = None, lower_range: float = None,
                upper_range: float = None, cloud_mask: Raster = None, apply_cloud_mask: bool = True, geometry: RasterGeometry = None, resampling: str = None) -> Raster:

        with h5py.File(filename, "r") as f:
            DN = np.array(f[dataset_name])
            h, v = self.hv
            grid = generate_modland_grid(h, v, DN.shape[0])

            logger.info("opening VIIRS file: " + colored_logging.file(self.filename))

            logger.info(f"loading {colored_logging.val(dataset_name)} at " + colored_logging.val(f"{grid.cell_size:0.2f} m") + " resolution")

            DN = Raster(DN, geometry=grid)

        data = DN

        if fill is not None:
            data = np.where(data == fill, np.nan, data)

        if lower_range is not None:
            data = np.where(data < lower_range, np.nan, data)

        if upper_range is not None:
            data = np.where(data > upper_range, np.nan, data)

        data = data * scale_factor + offset

        if apply_cloud_mask:
            if cloud_mask is None:
                cloud_mask = self.get_cloud_mask(target_shape=data.shape)

            data = rt.where(cloud_mask, np.nan, data)

        if geometry is not None:
            data = data.to_geometry(geometry, resampling=resampling)

        return data

    @property
    def geometry(self) -> RasterGrid:
        return generate_modland_grid(*self.hv, 1200)

    def get_Emis_14(self, geometry: RasterGeometry = None, save_data: bool = True, save_preview: bool = True, product_filename: str = None) -> Raster:
        if product_filename is None:
            product_filename = self.product_filename(f"Emis_14")

        if product_filename is not None and exists(product_filename):
            logger.info("loading VIIRS emissivity band 14: " + colored_logging.file(product_filename))
            image = Raster.open(product_filename)
        else:
            image = self.dataset(self.filename,f"HDFEOS/GRIDS/VIIRS_Grid_Daily_1km_LST21/Data Fields/Emis_14", scale_factor=0.002, offset=0.49,
                                 cloud_mask=None, apply_cloud_mask=False)

        if np.all(np.isnan(image)):
            raise ValueError("blank emissivity band 14 image")

        if save_data and not exists(product_filename):
            logger.info(f"writing VIIRS emissivity band 14: {colored_logging.file(product_filename)} {colored_logging.val(image.shape)}")
            image.to_geotiff(product_filename)

            if save_preview:
                image.percentilecut.to_geojpeg(product_filename.replace(".tif", ".jpeg"), quality=20, remove_XML=True)

        if geometry is not None:
            image = image.to_geometry(geometry)

        return image

    Emis_14 = property(get_Emis_14)

    def get_Emis_15(self, geometry: RasterGeometry = None, save_data: bool = True, save_preview: bool = True, product_filename: str = None) -> Raster:
        if product_filename is None:
            product_filename = self.product_filename(f"Emis_15")

        if product_filename is not None and exists(product_filename):
            logger.info("loading VIIRS emissivity band 15: " + colored_logging.file(product_filename))
            image = Raster.open(product_filename)
        else:
            image = self.dataset(self.filename,f"HDFEOS/GRIDS/VIIRS_Grid_Daily_1km_LST21/Data Fields/Emis_15", scale_factor=0.002, offset=0.49,
                                 cloud_mask=None, apply_cloud_mask=False)

        if np.all(np.isnan(image)):
            raise ValueError("blank emissivity band 15 image")

        if save_data and not exists(product_filename):
            logger.info(f"writing VIIRS emissivity band 15: {colored_logging.file(product_filename)} {colored_logging.val(image.shape)}")
            image.to_geotiff(product_filename)

            if save_preview:
                image.percentilecut.to_geojpeg(product_filename.replace(".tif", ".jpeg"), quality=20, remove_XML=True)

        if geometry is not None:
            image = image.to_geometry(geometry)

        return image

    Emis_15 = property(get_Emis_15)

    def get_Emis_16(self, geometry: RasterGeometry = None, save_data: bool = True, save_preview: bool = True, product_filename: str = None) -> Raster:
        if product_filename is None:
            product_filename = self.product_filename(f"Emis_16")

        if product_filename is not None and exists(product_filename):
            logger.info("loading VIIRS emissivity band 16: " + colored_logging.file(product_filename))
            image = Raster.open(product_filename)
        else:
            image = self.dataset(self.filename, f"HDFEOS/GRIDS/VIIRS_Grid_Daily_1km_LST21/Data Fields/Emis_16", scale_factor=0.002, offset=0.49,
                                 cloud_mask=None, apply_cloud_mask=False)

        if np.all(np.isnan(image)):
            raise ValueError("blank emissivity band 16 image")

        if save_data and not exists(product_filename):
            logger.info(f"writing VIIRS emissivity band 16: {colored_logging.file(product_filename)} {colored_logging.val(image.shape)}")
            image.to_geotiff(product_filename)

            if save_preview:
                image.percentilecut.to_geojpeg(product_filename.replace(".tif", ".jpeg"), quality=20, remove_XML=True)

        if geometry is not None:
            image = image.to_geometry(geometry)

        return image

    Emis_16 = property(get_Emis_16)

    def get_LST_1KM(self, geometry: RasterGeometry = None, save_data: bool = True, save_preview: bool = True, product_filename: str = None) -> Raster:
        if product_filename is None:
            product_filename = self.product_filename(f"LST_1KM")

        if product_filename is not None and exists(product_filename):
            logger.info("loading VIIRS LST 1km: " + colored_logging.file(product_filename))
            image = Raster.open(product_filename)
        else:
            image = self.dataset(self.filename, f"HDFEOS/GRIDS/VIIRS_Grid_Daily_1km_LST21/Data Fields/LST_1KM", scale_factor=0.02, offset=0.0, fill=0,
                                 lower_range=7500, upper_range=65535, cloud_mask=None, apply_cloud_mask=True)

        if np.all(np.isnan(image)):
            raise ValueError("blank LST 1km image")

        if save_data and not exists(product_filename):
            logger.info(f"writing VIIRS LST 1km: {colored_logging.file(product_filename)} {colored_logging.val(image.shape)}")
            image.to_geotiff(product_filename)

            if save_preview:
                image.percentilecut.to_geojpeg(product_filename.replace(".tif", ".jpeg"), quality=20, remove_XML=True)

        if geometry is not None:
            image = image.to_geometry(geometry)

        return image

    LST_1KM = property(get_LST_1KM)

    ST_K = LST_1KM

    @property
    def ST_C(self):
        return self.ST_K - 273.15

    def get_View_Angle(self, geometry: RasterGeometry = None, save_data: bool = True, save_preview: bool = True, product_filename: str = None) -> Raster:
        if product_filename is None:
            product_filename = self.product_filename(f"View_Angle")

        if product_filename is not None and exists(product_filename):
            logger.info("loading VIIRS view angle: " + colored_logging.file(product_filename))
            image = Raster.open(product_filename)
        else:
            image = self.dataset(self.filename, f"HDFEOS/GRIDS/VIIRS_Grid_Daily_1km_LST21/Data Fields/View_Angle", scale_factor=1.0, offset=-65.0,
                                 cloud_mask=None, apply_cloud_mask=False)

        if np.all(np.isnan(image)):
            raise ValueError("blank view angle image")

        if save_data and not exists(product_filename):
            logger.info(f"writing VIIRS view angle: {colored_logging.file(product_filename)} {colored_logging.val(image.shape)}")
            image.to_geotiff(product_filename)

            if save_preview:
                image.percentilecut.to_geojpeg(product_filename.replace(".tif", ".jpeg"), quality=20, remove_XML=True)

        if geometry is not None:
            image = image.to_geometry(geometry)

        return image

    View_Angle = property(get_View_Angle)


class VNP21A1D(VIIRSDataPool):
    DEFAULT_DOWNLOAD_DIRECTORY = "VNP21A1D_download"
    DEFAULT_PRODUCTS_DIRECTORY = "VNP21A1D_products"
    DEFAULT_MOSAIC_DIRECTORY = "VNP21A1D_mosaics"
    DEFAULT_RESAMPLING = "nearest"

    def __init__(self, *args, username: str = None, password: str = None, remote: str = None, working_directory: str = None, download_directory: str = None,
                 products_directory: str = None, mosaic_directory: str = None, resampling: str = None, **kwargs):
        super(VNP21A1D, self).__init__(*args, username=username, password=password, remote=remote, working_directory=working_directory,
                                       download_directory=download_directory, products_directory=products_directory, mosaic_directory=mosaic_directory, **kwargs)

        if resampling is None:
            resampling = self.DEFAULT_RESAMPLING

        self.resampling = resampling

    def search(self, start_date: date or datetime or str, end_date: date or datetime or str = None, build: str = None, tiles: List[str] or str = None,
               target_geometry: Point or Polygon or RasterGrid = None, *args, **kwargs) -> pd.DataFrame:
        return super(VNP21A1D, self).search(product="VNP21A1D", start_date=start_date, end_date=end_date, build=build, tiles=tiles, target_geometry=target_geometry,
                                      *args, **kwargs)

    def granule(self, date_UTC: date or str, tile: str, build: str = None) -> VNP21A1DGranule:
        if isinstance(date_UTC, str):
            date_UTC = parser.parse(date_UTC).date()

        download_location = join(self.download_directory, "VNP21A1D", f"{date_UTC:%Y.%m.%d}")

        if exists(download_location):
            filenames = glob(join(download_location, f"VNP21A1D.A{date_UTC:%Y%j}.{tile}.*.h5"))

            if len(filenames) > 0:
                filename = sorted(filenames)[0]
                logger.info(f"found previously retrieved VNP21A1D file: {filename}")

                granule = VNP21A1DGranule(filename=filename, products_directory=self.products_directory)

                return granule

        listing = self.search(start_date=date_UTC, end_date=date_UTC, build=build, tiles=[tile])

        if len(listing) > 0:
            URL = listing.iloc[0].URL
        else:
            raise VIIRSUnavailableError(f"VNP21A1D not available on {date_UTC}")

        makedirs(download_location, exist_ok=True)

        filename = self.download_URL(URL=URL, download_location=download_location)

        granule = VNP21A1DGranule(filename=filename, products_directory=self.products_directory)

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
            logger.info("loading " + colored_logging.val(dataset_name) + "at " + colored_logging.val(f"{grid.cell_size} m") + " resolution from " +
                        colored_logging.file(filename))

            DN = Raster(DN, geometry=grid)

        data = DN * scale_factor

        if apply_cloud_mask:
            if cloud_mask is None:
                cloud_mask = self.get_cloud_mask(target_shape=DN.shape)

            data = rt.where(cloud_mask, np.nan, data)

        return data

    def ST_C(self, date_UTC: date or str, geometry: RasterGeometry, filename: str = None, resampling: str = None) -> Raster:
        if isinstance(date_UTC, str):
            date_UTC = parser.parse(date_UTC).date()

        if filename is not None and exists(filename):
            return Raster.open(filename, cmap="jet")

        if resampling is None:
            resampling = self.resampling

        tiles = sorted(find_modland_tiles(geometry.boundary_latlon))

        if len(tiles) == 0:
            raise ValueError("no VIIRS tiles found covering target geometry")

        ST_C = None

        for tile in tiles:
            granule = self.granule(date_UTC=date_UTC, tile=tile)
            granule_ST_C = granule.ST_C
            projected_ST_C = granule_ST_C.to_geometry(geometry, resampling=resampling)

            if ST_C is None:
                ST_C = projected_ST_C
            else:
                ST_C = rasters.where(np.isnan(ST_C), projected_ST_C, ST_C)

        if ST_C is None:
            raise ValueError("VIIRS ST_C did not generate")

        ST_C.cmap = "jet"

        if filename is not None:
            logger.info("writing ST_C mosaic: " + colored_logging.file(filename))
            ST_C.to_geotiff(filename)

        return ST_C

    def process(self, start: date or str, target_geometry: RasterGeometry, target: str, end: date or str = None, product_names: List[str] = None) -> pd.DataFrame:

        if product_names is None:
            product_names = ["ST_C"]

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
                    logger.info(f"generating VIIRS {colored_logging.val(product)} mosaic at " + colored_logging.place(target) +
                                "on " + colored_logging.time(f"{acquisition_date:%Y-%m-%d}"))

                    if product == "ST_C":
                        self.ST_C(date_UTC=acquisition_date, geometry=target_geometry, filename=product_filename # return_raster=False
                                 )
                    else:
                        raise ValueError(f"unrecognized product: {product}")

                row.append(product_filename)

            rows.append(row)

        df = pd.DataFrame(rows, columns=["date"] + product_names)

        return df
