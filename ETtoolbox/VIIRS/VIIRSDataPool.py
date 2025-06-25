import json
import logging
import posixpath
from datetime import date
from datetime import datetime
from os.path import basename, splitext, abspath, expanduser
from os.path import join
from typing import List, Union

import h5py
import numpy as np
import pandas as pd
from dateutil import parser
from shapely.geometry import Point, Polygon

import colored_logging
import rasters as rt
from ETtoolbox.LPDAAC import LPDAACDataPool
from modland import find_modland_tiles
from modland.indices import generate_modland_grid, parsehv
from ..daterange import date_range
from rasters import RasterGrid, Raster

logger = logging.getLogger(__name__)


def parse_VIIRS_product(filename: str) -> str:
    return str(basename(filename).split(".")[0])


def parse_VIIRS_date(filename: str) -> date:
    return datetime.strptime(basename(filename).split(".")[1][1:], "%Y%j").date()


def parse_VIIRS_tile(filename: str) -> str:
    return str(basename(filename).split(".")[2])


def parse_VIIRS_build(filename: str) -> int:
    return int(basename(filename).split(".")[3])


DEFAULT_WORKING_DIRECTORY = "."
DEFAULT_PRODUCTS_DIRECTORY = "VIIRS_products"


class VIIRSGranule:
    CLOUD_DATASET_NAME = "HDFEOS/GRIDS/VNP_Grid_1km_2D/Data Fields/SurfReflect_QF1_1"

    def __init__(self, filename: str, working_directory: str = None, products_directory: str = None):
        if working_directory is None:
            working_directory = DEFAULT_WORKING_DIRECTORY

        working_directory = abspath(expanduser(working_directory))

        if products_directory is None:
            products_directory = join(working_directory, DEFAULT_PRODUCTS_DIRECTORY)

        products_directory = abspath(expanduser(products_directory))

        self._filename = abspath(filename)
        self._cloud_mask = None
        self.working_directory = working_directory
        self.products_directory = products_directory

    def __repr__(self):
        display_dict = {"filename": self.filename, "products_directory": self.products_directory}

        display_string = json.dumps(display_dict, indent=2)

        return display_string

    @property
    def filename(self):
        return self._filename

    @property
    def filename_base(self):
        return basename(self.filename)

    @property
    def filename_stem(self) -> str:
        return splitext(self.filename_base)[0]

    @property
    def tile(self):
        return parse_VIIRS_tile(self.filename)

    @property
    def hv(self):
        return parsehv(self.tile)

    @property
    def h(self):
        return self.hv[0]

    @property
    def v(self):
        return self.hv[1]

    @property
    def date_UTC(self):
        return datetime.strptime(self.filename_base.split(".")[1][1:], "%Y%j")

    @property
    def grids(self) -> List[str]:
        with h5py.File(self.filename, "r") as file:
            return list(file["HDFEOS/GRIDS/"].keys())

    def variables(self, grid: str) -> List[str]:
        with h5py.File(self.filename, "r") as file:
            return list(file[f"HDFEOS/GRIDS/{grid}/Data Fields/"].keys())

    def dataset(self, filename: str, dataset_name: str, fill_value: int, scale_factor: float) -> Raster:
        tile = parse_VIIRS_tile(filename)
        h, v = parsehv(tile)

        with h5py.File(filename, "r") as f:
            DN = np.array(f[dataset_name])
            grid = generate_modland_grid(h, v, DN.shape[0])

            logger.info("opening VIIRS file: " + colored_logging.file(self.filename))

            logger.info(f"loading {colored_logging.val(dataset_name)} at " + colored_logging.val(f"{grid.cell_size:0.2f} m") + " resolution")

            DN = Raster(DN, geometry=grid)

        data = rt.where(DN == fill_value, np.nan, DN * scale_factor)

        return data

    def product_directory(self, product) -> Union[str, None]:
        if self.products_directory is None:
            return None
        else:
            return join(self.products_directory, product, f"{self.date_UTC:%Y.%m.%d}")

    def product_filename(self, product: str) -> str:
        if self.product_directory(product) is None:
            raise ValueError("no product directory given")

        return join(self.product_directory(product), f"{self.filename_stem}_{product}.tif")


class VIIRSDataPool(LPDAACDataPool):
    DEFAULT_WORKING_DIRECTORY = "."
    DEFAULT_DOWNLOAD_DIRECTORY = "VIIRS_download"
    DEFAULT_PRODUCTS_DIRECTORY = "VIIRS_products"
    DEFAULT_MOSAIC_DIRECTORY = "VIIRS_mosaics"

    def __init__(self, username: str = None, password: str = None, remote: str = None, working_directory: str = None, download_directory: str = None,
                 products_directory: str = None, mosaic_directory: str = None, *args, **kwargs):

        super(VIIRSDataPool, self).__init__(username=username, password=password, remote=remote, *args, **kwargs)

        if working_directory is None:
            working_directory = self.DEFAULT_WORKING_DIRECTORY

        working_directory = abspath(expanduser(working_directory))

        if download_directory is None:
            download_directory = join(working_directory, self.DEFAULT_DOWNLOAD_DIRECTORY)

        download_directory = abspath(expanduser(download_directory))

        if products_directory is None:
            products_directory = join(working_directory, self.DEFAULT_PRODUCTS_DIRECTORY)

        products_directory = abspath(expanduser(products_directory))

        if mosaic_directory is None:
            mosaic_directory = join(working_directory, self.DEFAULT_MOSAIC_DIRECTORY)

        mosaic_directory = abspath(expanduser(mosaic_directory))

        self.working_directory = working_directory
        self.download_directory = download_directory
        self.products_directory = products_directory
        self.mosaic_directory = mosaic_directory

    def __repr__(self):
        display_dict = {"download_directory": self.download_directory, "products_directory": self.products_directory, "mosaic_directory": self.mosaic_directory}

        display_string = json.dumps(display_dict, indent=2)

        return display_string

    def search(self, product: str, start_date: date or datetime or str, end_date: date or datetime or str = None, build: str = None, tiles: List[str] or str = None,
               target_geometry: Point or Polygon or RasterGrid = None, *args, **kwargs) -> pd.DataFrame:

        if tiles is None and target_geometry is not None:
            tiles = find_modland_tiles(target_geometry)

        if isinstance(tiles, str):
            tiles = [tiles]

        if isinstance(start_date, str):
            start_date = parser.parse(start_date).date()

        if end_date is None:
            end_date = start_date
        elif isinstance(end_date, str):
            end_date = parser.parse(end_date).date()

        rows = []

        for acquisition_date in date_range(start_date, end_date):
            date_URL = self.date_URL("VIIRS", product, acquisition_date, build)

            if tiles is None:
                listing = self.get_HTTP_listing(date_URL, pattern="*.h5")
            else:
                listing = []

                for tile in tiles:
                    listing.extend(self.get_HTTP_listing(date_URL, pattern=f"*.{tile}.*.h5"))

            URLs = sorted([posixpath.join(date_URL, item) for item in listing])

            for URL in URLs:
                tile = parse_VIIRS_tile(posixpath.basename(URL))
                rows.append([acquisition_date, tile, URL])

        df = pd.DataFrame(rows, columns=["date", "tile", "URL"])

        return df

