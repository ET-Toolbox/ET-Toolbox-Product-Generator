import logging
import posixpath
from os import makedirs, system
from os.path import exists, dirname, join, expanduser, abspath
from shutil import move
from time import perf_counter
from typing import List

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import rasters as rt
from rasters import RasterGeometry, Raster

CANOPY_COLORMAP = LinearSegmentedColormap.from_list(
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

class GEDICanopyHeight:
    AUS_URL = "https://glad.geog.umd.edu/Potapov/Forest_height_2019/Forest_height_2019_AUS.tif"
    NAFR_URL = "https://glad.geog.umd.edu/Potapov/Forest_height_2019/Forest_height_2019_NAFR.tif"
    NAM_URL = "https://glad.geog.umd.edu/Potapov/Forest_height_2019/Forest_height_2019_NAM.tif"
    NASIA_URL = "https://glad.geog.umd.edu/Potapov/Forest_height_2019/Forest_height_2019_NASIA.tif"
    SAFR_URL = "https://glad.geog.umd.edu/Potapov/Forest_height_2019/Forest_height_2019_SAFR.tif"
    SAM_URL = "https://glad.geog.umd.edu/Potapov/Forest_height_2019/Forest_height_2019_SAM.tif"
    SASIA_URL = "https://glad.geog.umd.edu/Potapov/Forest_height_2019/Forest_height_2019_SASIA.tif"

    logger = logging.getLogger(__name__)

    def __init__(self, source_directory: str):
        self.source_directory = abspath(expanduser(source_directory))

    def __repr__(self) -> str:
        return f'GEDICanopyHeight(source_directory="{self.source_directory}")'

    def download_file(self, URL: str, filename: str) -> str:
        if exists(filename):
            self.logger.info(f"file already downloaded: {filename}")
            return filename

        self.logger.info(f"downloading: {URL} -> {filename}")
        directory = dirname(filename)
        makedirs(directory, exist_ok=True)
        partial_filename = f"{filename}.download"
        command = f'wget -c -O "{partial_filename}" "{URL}"'
        download_start = perf_counter()
        system(command)
        download_end = perf_counter()
        download_duration = download_end - download_start
        self.logger.info(f"completed download in {download_duration:0.2f} seconds: {filename}")

        if not exists(partial_filename):
            raise IOError(f"unable to download URL: {URL}")

        move(partial_filename, filename)

        return filename

    @property
    def source_URLs(self) -> dict:
        return {
            "AUS": self.AUS_URL,
            "NAFR": self.NAFR_URL,
            "NAM": self.NAM_URL,
            "NASIA": self.NASIA_URL,
            "SAFR": self.SAFR_URL,
            "SAM": self.SAM_URL,
            "SASIA": self.SASIA_URL
        }

    def source_URL(self, name: str) -> str:
        if name not in self.source_URLs:
            raise ValueError(f"unrecognized canopy height region: {name}")

        return self.source_URLs[name]

    def source_filename(self, name: str) -> str:
        URL = self.source_URL(name)
        filename_base = posixpath.basename(URL)
        filename = join(self.source_directory, filename_base)

        return filename

    def download_sources(self) -> List[str]:
        filenames = []

        for name, URL in self.source_URLs.items():
            filename = self.source_filename(name)
            self.download_file(URL, filename)

            if not exists(filename):
                raise IOError(f"unable to download {name} canopy height: {URL}")

            filenames.append(filename)

        return filenames

    @property
    def VRT_filename(self):
        return join(self.source_directory, "Forest_height_2019.vrt")

    @property
    def VRT(self) -> str:
        if exists(self.VRT_filename):
            return self.VRT_filename

        source_filenames = self.download_sources()
        command = f"gdalbuildvrt {self.VRT_filename} {' '.join(source_filenames)}"
        self.logger.info(command)
        system(command)

        if not exists(self.VRT_filename):
            raise IOError(f"unable to produce canopy height VRT: {self.VRT_filename}")

    def canopy_height_meters(self, geometry: RasterGeometry, resampling=None) -> Raster:
        return rt.Raster.open(
            filename=self.VRT,
            geometry=geometry,
            nodata=np.nan,
            remove=101,
            resampling=resampling,
            cmap=CANOPY_COLORMAP
        )
