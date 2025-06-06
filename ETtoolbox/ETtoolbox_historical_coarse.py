import logging
import sys
from datetime import datetime, timedelta, date
from os.path import join
from typing import List, Callable, Union

import colored_logging
import numpy as np
from dateutil import parser
from gedi_canopy_height import GEDICanopyHeight
from GEOS5FP import GEOS5FP
from MODISCI import MODISCI
from rasters import Raster, RasterGrid
from sentinel_tiles import sentinel_tiles
from soil_capacity_wilting import SoilGrids
from solar_apparent_time import solar_to_UTC

from LandsatL2C2 import LandsatL2C2
from PTJPL import PTJPL
from ETtoolbox.VIIRS_GEOS5FP import VIIRS_GEOS5FP, check_VIIRS_GEOS5FP_already_processed, VIIRS_DOWNLOAD_DIRECTORY, \
    VIIRS_PRODUCTS_DIRECTORY
from ETtoolbox.daterange import date_range

from check_distribution import check_distribution

from NASADEM import NASADEMConnection
from VNP09GA_002 import VNP09GA
from VNP21A1D_002 import VNP21A1D

import colored_logging as cl

from .constants import *

logger = logging.getLogger(__name__)


class BlankOutputError(ValueError):
    pass

def generate_landsat_ST_C_prior(
        date_UTC: Union[date, str],
        geometry: RasterGrid,
        target_name: str,
        landsat: LandsatL2C2 = None,
        working_directory: str = None,
        download_directory: str = None,
        landsat_initialization_days: int = LANDSAT_INITIALIZATION_DAYS) -> Raster:
    if isinstance(date_UTC, str):
        date_UTC = parser.parse(date_UTC).date()

    if landsat is None:
        landsat = LandsatL2C2(
            working_directory=working_directory,
            download_directory=download_directory
        )

    landsat_start = date_UTC - timedelta(days=landsat_initialization_days)
    landsat_end = date_UTC - timedelta(days=1)
    logger.info(f"generating Landsat temperature composite from {colored_logging.time(landsat_start)} to {colored_logging.time(landsat_end)}")
    landsat_listing = landsat.scene_search(start=landsat_start, end=landsat_end, target_geometry=geometry)
    landsat_composite_dates = sorted(set(landsat_listing.date_UTC))
    logger.info(f"found Landsat granules on dates: {', '.join([colored_logging.time(d) for d in landsat_composite_dates])}")

    ST_C_images = []

    for date_UTC in landsat_composite_dates:
        try:
            ST_C = landsat.product(acquisition_date=date_UTC, product="ST_C", geometry=geometry,
                                   target_name=target_name)
            ST_C_images.append(ST_C)
        except Exception as e:
            logger.warning(e)
            continue

    composite = Raster(np.nanmedian(np.stack(ST_C_images), axis=0), geometry=geometry)

    return composite


def ET_toolbox_historical_coarse_tile(
        tile: str,
        start_date: Union[date, str] = None,
        end_date: Union[date, str] = None,
        water: Raster = None,
        ET_model_name: str = ET_MODEL_NAME,
        SWin_model_name: str = SWIN_MODEL_NAME,
        Rn_model_name: str = RN_MODEL_NAME,
        working_directory: str = None,
        static_directory: str = None,
        VNP09GA_download_directory: str = None,
        VNP21A1D_download_directory: str = None,
        use_VIIRS_composite: bool = USE_VIIRS_COMPOSITE,
        VIIRS_composite_days: int = VIIRS_COMPOSITE_DAYS,
        VIIRS_GEOS5FP_output_directory: str = None,
        SRTM_connection: NASADEMConnection = None,
        SRTM_download: str = None,
        GEOS5FP_connection: GEOS5FP = None,
        GEOS5FP_download: str = None,
        GEOS5FP_products: str = None,
        GEDI_connection: GEDICanopyHeight = None,
        GEDI_download: str = None,
        ORNL_connection: MODISCI = None,
        CI_directory: str = None,
        soil_grids_connection: SoilGrids = None,
        soil_grids_download: str = None,
        intermediate_directory: str = None,
        preview_quality: int = PREVIEW_QUALITY,
        ANN_model: Callable = None,
        ANN_model_filename: str = None,
        resampling: str = RESAMPLING,
        M_geometry: RasterGrid = None,
        M_cell_size: float = M_CELL_SIZE,
        GEOS5FP_geometry: RasterGrid = None,
        GEOS5FP_cell_size: float = GEOS5FP_CELL_SIZE,
        save_intermediate: bool = SAVE_INTERMEDIATE,
        show_distribution: bool = SHOW_DISTRIBUTION,
        load_previous: bool = LOAD_PREVIOUS,
        target_variables: List[str] = None):
    if isinstance(start_date, str):
        start_date = parser.parse(start_date).date()

    if isinstance(end_date, str):
        end_date = parser.parse(end_date).date()

    logger.info(
        f"generating ET Toolbox hindcast and forecast at tile {colored_logging.place(tile)} from {colored_logging.time(start_date)} to {colored_logging.time(end_date)}")

    if M_geometry is None:
        logger.info(f"I-band cell size: {colored_logging.val(M_cell_size)}m")
        M_geometry = sentinel_tiles.grid(tile, cell_size=M_cell_size)

    if GEOS5FP_geometry is None:
        logger.info(f"GEOS-5 FP cell size: {colored_logging.val(GEOS5FP_cell_size)}m")
        GEOS5FP_geometry = sentinel_tiles.grid(tile, cell_size=GEOS5FP_cell_size)

    if target_variables is None:
        target_variables = TARGET_VARIABLES

    if GEOS5FP_connection is None:
        GEOS5FP_connection = GEOS5FP(
            working_directory=working_directory,
            download_directory=GEOS5FP_download,
            products_directory=GEOS5FP_products
        )

    if SRTM_connection is None:
        SRTM_connection = NASADEMConnection(
            working_directory=working_directory,
            download_directory=SRTM_download,
            offline_ok=True
        )

    water_M = SRTM_connection.swb(M_geometry)

    if VNP09GA_download_directory is None:
        VNP09GA_download_directory = VNP09GA_DOWNLOAD_DIRECTORY

    logger.info(f"VNP09GA download directory: {cl.dir(VNP09GA_download_directory)}")

    VNP21A1D_connection = VNP21A1D(
        download_directory=VNP21A1D_download_directory,
        )
    
    if VNP21A1D_download_directory is None:
        VNP21A1D_download_directory = VNP21A1D_DOWNLOAD_DIRECTORY

    logger.info(f"VNP21A1D download directory: {cl.dir(VNP21A1D_download_directory)}")

    VNP21A1D_connection = VNP21A1D(
        download_directory=VNP21A1D_download_directory,
        )

    if VIIRS_GEOS5FP_output_directory is None:
        VIIRS_GEOS5FP_output_directory = join(working_directory, VIIRS_GEOS5FP_OUTPUT_DIRECTORY)

    logger.info(f"VIIRS GEOS-5 FP output directory: {colored_logging.dir(VIIRS_GEOS5FP_output_directory)}")

    VIIRS_dates_processed = set()

    for target_date in date_range(start_date, end_date):
        logger.info(f"ET Toolbox historical fine target date: {colored_logging.time(target_date)}")
        time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
        logger.info(f"ET Toolbox historical fine time solar: {colored_logging.time(time_solar)}")
        time_UTC = solar_to_UTC(time_solar, M_geometry.centroid.latlon.x)

        VIIRS_already_processed = check_VIIRS_GEOS5FP_already_processed(
            VIIRS_GEOS5FP_output_directory=VIIRS_GEOS5FP_output_directory,
            target_date=target_date,
            time_UTC=time_UTC,
            target=tile,
            products=target_variables
        )

        if VIIRS_already_processed:
            logger.info(f"VIIRS GEOS-5 FP already processed at tile {colored_logging.place(tile)} for date {target_date}")
            VIIRS_dates_processed |= {target_date}
            continue
        else:
            VIIRS_not_processed = False

    missing_dates = []

    for target_date in date_range(start_date, end_date):
        logger.info(f"VIIRS GEOS-5 FP target date: {colored_logging.time(target_date)}")

        time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
        logger.info(f"VIIRS target time solar: {colored_logging.time(time_solar)}")
        time_UTC = solar_to_UTC(time_solar, M_geometry.centroid.latlon.x)

        try:
            VIIRS_already_processed = check_VIIRS_GEOS5FP_already_processed(
                VIIRS_GEOS5FP_output_directory=VIIRS_GEOS5FP_output_directory,
                target_date=target_date,
                time_UTC=time_UTC,
                target=tile,
                products=target_variables
            )

            if VIIRS_already_processed:
                logger.info(f"VIIRS GEOS-5 FP already processed at tile {colored_logging.place(tile)} for date {target_date}")
                continue

            VIIRS_GEOS5FP(
                target_date=target_date,
                geometry=M_geometry,
                target=tile,
                working_directory=working_directory,
                static_directory=static_directory,
                SRTM_connection=SRTM_connection,
                SRTM_download=SRTM_download,
                GEDI_connection=GEDI_connection,
                GEDI_download=GEDI_download,
                ORNL_connection=ORNL_connection,
                CI_directory=CI_directory,
                soil_grids_connection=soil_grids_connection,
                soil_grids_download=soil_grids_download,
                VNP09GA_download_directory=VNP09GA_download_directory,
                VNP21A1D_download_directory=VNP21A1D_download_directory,`
                VIIRS_GEOS5FP_output_directory=VIIRS_GEOS5FP_output_directory,
                intermediate_directory=intermediate_directory,
                preview_quality=preview_quality,
                ANN_model=ANN_model,
                ANN_model_filename=ANN_model_filename,
                ET_model_name=ET_model_name,
                SWin=SWin_model_name,
                Rn=Rn_model_name,
                water=water_M,
                coarse_cell_size=GEOS5FP_cell_size,
                target_variables=target_variables,
                downscale_air=DOWNSCALE_AIR,
                downscale_humidity=DOWNSCALE_HUMIDITY,
                downscale_moisture=DOWNSCALE_MOISTURE,
                floor_Topt=FLOOR_TOPT,
                resampling=resampling,
                show_distribution=show_distribution,
                load_previous=load_previous,
                save_intermediate=save_intermediate
            )

            VIIRS_dates_processed |= {target_date}

        except Exception as e:
            logger.exception(e)
            logger.warning(f"VIIRS GEOS-5 FP cannot be processed for date: {target_date}")
            missing_dates.append(target_date)
            continue

    logger.info("missing VIIRS GEOS-5 FP dates: " + ", ".join(colored_logging.time(d) for d in missing_dates))


def main(argv=sys.argv):
    tile = argv[1]
    start_date = parser.parse(argv[2]).date()
    end_date = parser.parse(argv[3]).date()

    if "--working" in argv:
        working_directory = argv[argv.index("--working") + 1]
    else:
        working_directory = "."

    if "--static" in argv:
        static_directory = argv[argv.index("--static") + 1]
    else:
        static_directory = join(working_directory, "PTJPL_static")

    if "--SRTM" in argv:
        SRTM_download = argv[argv.index("--SRTM") + 1]
    else:
        SRTM_download = join(working_directory, "SRTM_download_directory")

    if "--VIIRS" in argv:
        VIIRS_download_directory = argv[argv.index("--VIIRS") + 1]
    else:
        VIIRS_download_directory = join(working_directory, "VIIRS_download")

    if "--GEOS5FP" in argv:
        GEOS5FP_download = argv[argv.index("--GEOS5FP") + 1]
    else:
        GEOS5FP_download = join(working_directory, "GEOS5FP_download_directory")

    ET_toolbox_historical_coarse_tile(
        tile=tile,
        start_date=start_date,
        end_date=end_date,
        working_directory=working_directory,
        static_directory=static_directory,
        SRTM_download=SRTM_download,
        VIIRS_download_directory=VIIRS_download_directory,
        GEOS5FP_download=GEOS5FP_download,
    )
