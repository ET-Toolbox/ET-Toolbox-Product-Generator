import logging
import sys
from datetime import datetime, timedelta, date
from os.path import join
from typing import List, Callable, Union

import colored_logging
import numpy as np
from dateutil import parser
from gedi_canopy_height import GEDICanopyHeight
from geos5fp import GEOS5FP
from modisci import MODISCI
from rasters import Raster, RasterGrid
from sentinel_tiles import sentinel_tiles
from soil_capacity_wilting import SoilGrids
from solar_apparent_time import solar_to_UTC

from ETtoolbox.LandsatL2C2 import LandsatL2C2
from ETtoolbox.PTJPLSM import PTJPLSM, DEFAULT_PREVIEW_QUALITY, DEFAULT_RESAMPLING
from ETtoolbox.SRTM import SRTM
from ETtoolbox.VIIRS import VNP43MA4
from ETtoolbox.VIIRS.VNP09GA import VNP09GA
from ETtoolbox.VIIRS.VNP21A1D import VNP21A1D
from ETtoolbox.VIIRS_GEOS5FP import VIIRS_GEOS5FP, check_VIIRS_GEOS5FP_already_processed, VIIRS_DOWNLOAD_DIRECTORY, \
    VIIRS_PRODUCTS_DIRECTORY
from ETtoolbox.daterange import date_range

logger = logging.getLogger(__name__)

ET_MODEL_NAME = "PTJPLSM"
SWIN_MODEL_NAME = "GEOS5FP"
RN_MODEL_NAME = "Verma"

DOWNSCALE_AIR = False
DOWNSCALE_HUMIDITY = False
DOWNSCALE_MOISTURE = False
FLOOR_TOPT = True

USE_VIIRS_COMPOSITE = True
VIIRS_COMPOSITE_DAYS = 8

HLS_CELL_SIZE = 30
I_CELL_SIZE = 500
M_CELL_SIZE = 1000
GEOS5FP_CELL_SIZE = 27375
# GFS_CELL_SIZE = 54750
GFS_CELL_SIZE = 27375
LANDSAT_INITIALIZATION_DAYS = 16
HLS_INITIALIZATION_DAYS = 10

SAVE_INTERMEDIATE = False
SHOW_DISTRIBUTION = True
LOAD_PREVIOUS = False

VIIRS_GEOS5FP_OUTPUT_DIRECTORY = "VIIRS_GEOS5FP_VermaRn_PTJPLSMboostedToptET_1km_output"

TARGET_VARIABLES = [
    "Rn",
    "LE",
    "ET",
    "ESI",
    "SM",
    "ST",
    "Ta",
    "RH",
    "SWin",
    "VPD",
    "fg",
    "fM",
    "fSM",
    "fT",
    "fREW",
    "fTREW",
    "fTRM",
    "Topt"
]


class BlankOutputError(ValueError):
    pass


def check_distribution(
        image: Raster,
        variable: str,
        date_UTC: date or str,
        target: str):
    unique = np.unique(image)
    nan_proportion = np.count_nonzero(np.isnan(image)) / np.size(image)

    if len(unique) < 10:
        logger.info(
            "variable " + colored_logging.name(variable) + " on " + colored_logging.time(f"{date_UTC:%Y-%m-%d}") + " at " + colored_logging.place(
                target))

        for value in unique:
            count = np.count_nonzero(image == value)

            if value == 0:
                logger.info(f"* {colored_logging.colored(value, 'red')}: {colored_logging.colored(count, 'red')}")
            else:
                logger.info(f"* {colored_logging.val(value)}: {colored_logging.val(count)}")
    else:
        minimum = np.nanmin(image)

        if minimum < 0:
            minimum_string = colored_logging.colored(f"{minimum:0.3f}", "red")
        else:
            minimum_string = colored_logging.val(f"{minimum:0.3f}")

        maximum = np.nanmax(image)

        if maximum <= 0:
            maximum_string = colored_logging.colored(f"{maximum:0.3f}", "red")
        else:
            maximum_string = colored_logging.val(f"{maximum:0.3f}")

        if nan_proportion > 0.5:
            nan_proportion_string = colored_logging.colored(f"{(nan_proportion * 100):0.2f}%", "yellow")
        elif nan_proportion == 1:
            nan_proportion_string = colored_logging.colored(f"{(nan_proportion * 100):0.2f}%", "red")
        else:
            nan_proportion_string = colored_logging.val(f"{(nan_proportion * 100):0.2f}%")

        message = "variable " + colored_logging.name(variable) + \
                  " on " + colored_logging.time(f"{date_UTC:%Y-%m-%d}") + \
                  " at " + colored_logging.place(target) + \
                  " min: " + minimum_string + \
                  " mean: " + colored_logging.val(f"{np.nanmean(image):0.3f}") + \
                  " max: " + maximum_string + \
                  " nan: " + nan_proportion_string + f" ({colored_logging.val(image.nodata)})"

        if np.all(image == 0):
            message += " all zeros"
            logger.warning(message)
        else:
            logger.info(message)

    if nan_proportion == 1:
        raise BlankOutputError(f"variable {variable} on {date_UTC:%Y-%m-%d} at {target} is a blank image")


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
        model: PTJPLSM = None,
        ET_model_name: str = ET_MODEL_NAME,
        SWin_model_name: str = SWIN_MODEL_NAME,
        Rn_model_name: str = RN_MODEL_NAME,
        working_directory: str = None,
        static_directory: str = None,
        VIIRS_download_directory: str = None,
        VIIRS_products_directory: str = None,
        use_VIIRS_composite: bool = USE_VIIRS_COMPOSITE,
        VIIRS_composite_days: int = VIIRS_COMPOSITE_DAYS,
        VIIRS_GEOS5FP_output_directory: str = None,
        VIIRS_shortwave_source: Union[VNP09GA, VNP43MA4] = None,
        SRTM_connection: SRTM = None,
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
        preview_quality: int = DEFAULT_PREVIEW_QUALITY,
        ANN_model: Callable = None,
        ANN_model_filename: str = None,
        resampling: str = DEFAULT_RESAMPLING,
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
        # FIXME fix handling of credentials here
        SRTM_connection = SRTM(
            working_directory=working_directory,
            download_directory=SRTM_download,
            offline_ok=True
        )

    water_M = SRTM_connection.swb(M_geometry)

    if VIIRS_download_directory is None:
        VIIRS_download_directory = join(working_directory, VIIRS_DOWNLOAD_DIRECTORY)

    logger.info(f"VIIRS download directory: {colored_logging.dir(VIIRS_download_directory)}")

    if VIIRS_products_directory is None:
        VIIRS_products_directory = join(working_directory, VIIRS_PRODUCTS_DIRECTORY)

    logger.info(f"VIIRS products directory: {colored_logging.dir(VIIRS_products_directory)}")

    vnp21 = VNP21A1D(
        working_directory=working_directory,
        download_directory=VIIRS_download_directory,
        products_directory=VIIRS_products_directory
    )

    if VIIRS_shortwave_source is None:
        VIIRS_shortwave_source = VNP43MA4(
            working_directory=working_directory,
            download_directory=VIIRS_download_directory,
            products_directory=VIIRS_products_directory
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
                VIIRS_download_directory=VIIRS_download_directory,
                VIIRS_GEOS5FP_output_directory=VIIRS_GEOS5FP_output_directory,
                VIIRS_shortwave_source=VIIRS_shortwave_source,
                intermediate_directory=intermediate_directory,
                preview_quality=preview_quality,
                ANN_model=ANN_model,
                ANN_model_filename=ANN_model_filename,
                model=model,
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
        # except (VIIRSNotAvailableError, GEOS5FPNotAvailableError) as e:
        #     logger.warning(e)
        #     logger.warning(f"VIIRS GEOS-5 FP cannot be processed for date: {target_date}")
        #     missing_dates.append(target_date)
        #     continue
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
