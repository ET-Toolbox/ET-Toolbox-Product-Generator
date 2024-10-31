import logging
import sys
from datetime import datetime, timedelta, date
from os.path import join
from typing import List, Callable, Union

import numpy as np
from dateutil import parser

import colored_logging
import rasters as rt
from gedi_canopy_height import GEDICanopyHeight
from geos5fp import GEOS5FP
from solar_apparent_time import solar_to_UTC
from harmonized_landsat_sentinel import HLS2Connection
from ETtoolbox.LandsatL2C2 import LandsatL2C2
from modisci import MODISCI
from ETtoolbox.PTJPLSM import PTJPLSM, DEFAULT_PREVIEW_QUALITY, DEFAULT_RESAMPLING
from ETtoolbox.SRTM import SRTM
from soil_capacity_wilting import SoilGrids
from ETtoolbox.VIIRS.VNP09GA import VNP09GA
from ETtoolbox.VIIRS.VNP21A1D import VNP21A1D
from ETtoolbox.VIIRS import VNP43MA4
from ETtoolbox.VIIRS_GEOS5FP import VIIRS_GEOS5FP, check_VIIRS_GEOS5FP_already_processed, VIIRS_DOWNLOAD_DIRECTORY, \
    VIIRS_PRODUCTS_DIRECTORY, VIIRS_GEOS5FP_OUTPUT_DIRECTORY
from ETtoolbox.daterange import date_range
from geos5fp.downscaling import bias_correct, downscale_soil_moisture, downscale_air_temperature, \
    downscale_vapor_pressure_deficit, downscale_relative_humidity
from rasters import Raster, RasterGrid
from sentinel_tiles import sentinel_tiles

logger = logging.getLogger(__name__)

ET_MODEL_NAME = "PTJPLSM"
SWIN_MODEL_NAME = "GEOS5FP"
RN_MODEL_NAME = "Verma"

FLOOR_TOPT = True

DOWNSCALE_AIR = False
DOWNSCALE_HUMIDITY = False
DOWNSCALE_MOISTURE = False

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

TARGET_VARIABLES = [
    "Rn",
    "LE",
    "ET",
    "ESI",
    "SM",
    "ST",
    "Ta",
    "RH",
    "SWin"
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
            ST_C = landsat.product(acquisition_date=date_UTC, product="ST_C", geometry=geometry, target_name=target_name)
            ST_C_images.append(ST_C)
        except Exception as e:
            logger.warning(e)
            continue

    composite = Raster(np.nanmedian(np.stack(ST_C_images), axis=0), geometry=geometry)

    return composite


def ET_toolbox_historical_fine_tile(
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
        HLS_download: str = None,
        HLS_initialization_days: int = HLS_INITIALIZATION_DAYS,
        landsat_download: str = None,
        landsat_initialization_days: int = LANDSAT_INITIALIZATION_DAYS,
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
        HLS_geometry: RasterGrid = None,
        HLS_cell_size: float = HLS_CELL_SIZE,
        I_geometry: RasterGrid = None,
        I_cell_size: float = I_CELL_SIZE,
        M_geometry: RasterGrid = None,
        M_cell_size: float = M_CELL_SIZE,
        GEOS5FP_geometry: RasterGrid = None,
        GEOS5FP_cell_size: float = GEOS5FP_CELL_SIZE,
        GFS_geometry: RasterGrid = None,
        GFS_cell_size: float = GFS_CELL_SIZE,
        downscale_air: bool = DOWNSCALE_AIR,
        downscale_humidity: bool = DOWNSCALE_HUMIDITY,
        downscale_moisture: bool = DOWNSCALE_MOISTURE,
        apply_GEOS5FP_GFS_bias_correction: bool = True,
        save_intermediate: bool = False,
        show_distribution: bool = False,
        load_previous: bool = True,
        target_variables: List[str] = None):
    if isinstance(start_date, str):
        start_date = parser.parse(start_date).date()

    if isinstance(end_date, str):
        end_date = parser.parse(end_date).date()

    logger.info(
        f"generating ET Toolbox hindcast and forecast at tile {colored_logging.place(tile)} from {colored_logging.time(start_date)} to {colored_logging.time(end_date)}")

    if HLS_geometry is None:
        logger.info(f"HLS cell size: {colored_logging.val(HLS_cell_size)}m")
        HLS_geometry = sentinel_tiles.grid(tile, cell_size=HLS_cell_size)

    if I_geometry is None:
        logger.info(f"I-band cell size: {colored_logging.val(I_cell_size)}m")
        I_geometry = sentinel_tiles.grid(tile, cell_size=I_cell_size)

    if M_geometry is None:
        logger.info(f"I-band cell size: {colored_logging.val(M_cell_size)}m")
        M_geometry = sentinel_tiles.grid(tile, cell_size=M_cell_size)

    if GEOS5FP_geometry is None:
        logger.info(f"GEOS-5 FP cell size: {colored_logging.val(GEOS5FP_cell_size)}m")
        GEOS5FP_geometry = sentinel_tiles.grid(tile, cell_size=GEOS5FP_cell_size)

    if GFS_geometry is None:
        logger.info(f"GFS cell size: {colored_logging.val(GFS_cell_size)}m")
        GFS_geometry = sentinel_tiles.grid(tile, cell_size=GFS_cell_size)

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

    HLS = HLS2Connection(
        working_directory=working_directory,
        download_directory=HLS_download,
        target_resolution=int(HLS_cell_size)
    )

    water_HLS = SRTM_connection.swb(HLS_geometry)
    water_M = SRTM_connection.swb(M_geometry)
    water_I = SRTM_connection.swb(I_geometry)

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

        # VIIRS_shortwave_source = VNP09GA(
        #     working_directory=working_directory,
        #     download_directory=VIIRS_download_directory,
        #     products_directory=VIIRS_products_directory
        # )

    if VIIRS_GEOS5FP_output_directory is None:
        VIIRS_GEOS5FP_output_directory = join(working_directory, VIIRS_GEOS5FP_OUTPUT_DIRECTORY)

    logger.info(f"VIIRS GEOS-5 FP output directory: {colored_logging.dir(VIIRS_GEOS5FP_output_directory)}")

    VIIRS_dates_processed = set()

    for target_date in date_range(start_date, end_date):
        logger.info(f"ET Toolbox historical fine target date: {colored_logging.time(target_date)}")
        time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
        logger.info(f"ET Toolbox historical fine time solar: {colored_logging.time(time_solar)}")
        time_UTC = solar_to_UTC(time_solar, HLS_geometry.centroid.latlon.x)

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

    HLS_start = start_date - timedelta(days=HLS_initialization_days)
    HLS_end = start_date - timedelta(days=1)

    logger.info(f"forming HLS NDVI composite from {colored_logging.time(HLS_start)} to {colored_logging.time(HLS_end)}")

    missing_dates = []
    NDVI_images = []

    for HLS_date in date_range(HLS_start, HLS_end):
        try:
            NDVI_images.append(HLS.NDVI(tile=tile, date_UTC=HLS_date).to_geometry(HLS_geometry))
        except Exception as e:
            logger.warning(e)
            continue

    NDVI_HLS_initial = Raster(np.nanmedian(np.stack(NDVI_images), axis=0), geometry=HLS_geometry)
    NDVI_HLS_initial = rt.where(water_HLS, np.nan, NDVI_HLS_initial)
    NDVI_HLS_prior = NDVI_HLS_initial

    albedo_images = []

    for HLS_date in date_range(HLS_start, HLS_end):
        try:
            albedo_images.append(HLS.albedo(tile=tile, date_UTC=HLS_date).to_geometry(HLS_geometry))
        except Exception as e:
            logger.warning(e)
            continue

    albedo_HLS_initial = Raster(np.nanmedian(np.stack(albedo_images), axis=0), geometry=HLS_geometry)
    albedo_HLS_prior = albedo_HLS_initial

    landsat = LandsatL2C2(
        working_directory=working_directory,
        download_directory=landsat_download
    )

    # landsat_start = start_date - timedelta(days=landsat_initialization_days)
    # landsat_end = start_date - timedelta(days=1)
    #
    # logger.info(f"generating Landsat temperature composite from {landsat_start} to {landsat_end}")
    #
    # landsat_listing = landsat.scene_search(start=landsat_start, end=landsat_end, target_geometry=HLS_geometry)
    # landsat_composite_dates = sorted(set(landsat_listing.date_UTC))
    # logger.info(f"found Landsat granules on dates: {', '.join(landsat_composite_dates)}")
    #
    # ST_C_images = []
    #
    # for date_UTC in landsat_composite_dates:
    #     try:
    #         ST_C = landsat.product(acquisition_date=date_UTC, product="ST_C", geometry=HLS_geometry, target_name=tile)
    #         ST_C_images.append(ST_C)
    #     except Exception as e:
    #         logger.warning(e)
    #         continue
    #
    # landsat_ST_C_initial = Raster(np.nanmedian(np.stack(ST_C_images), axis=0), geometry=HLS_geometry)
    # landsat_ST_C_prior = landsat_ST_C_initial

    landsat_ST_C_prior = generate_landsat_ST_C_prior(
        date_UTC=start_date,
        geometry=HLS_geometry,
        target_name=tile,
        landsat=landsat,
        landsat_initialization_days=landsat_initialization_days
    )

    for target_date in date_range(start_date, end_date):
        logger.info(f"VIIRS GEOS-5 FP target date: {colored_logging.time(target_date)}")

        time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
        logger.info(f"VIIRS target time solar: {colored_logging.time(time_solar)}")
        time_UTC = solar_to_UTC(time_solar, HLS_geometry.centroid.latlon.x)

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

            try:
                new_landsat_ST_C = landsat.product(
                    acquisition_date=target_date,
                    product="ST_C",
                    geometry=HLS_geometry,
                    target_name=tile
                )

                landsat_ST_C = rt.where(np.isnan(new_landsat_ST_C), landsat_ST_C_prior, new_landsat_ST_C)
            except Exception as e:
                landsat_ST_C = landsat_ST_C_prior

            landsat_ST_C_prior = landsat_ST_C

            logger.info(
                f"retrieving VIIRS VNP21 ST for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} at {colored_logging.val(M_cell_size)}m resolution")
            # ST_K_M = retrieve_VNP21NRT_ST(
            #     geometry=M_geometry,
            #     date_solar=target_date,
            #     directory=VIIRS_download, resampling="cubic"
            # )
            ST_C_M = vnp21.ST_C(
                date_UTC=target_date,
                geometry=M_geometry,
                resampling="cubic"
            )

            if use_VIIRS_composite:
                for days_back in range(1, VIIRS_composite_days):
                    fill_date = target_date - timedelta(days_back)
                    logger.info(
                        f"gap-filling {colored_logging.name('VNP21A1D')} {colored_logging.name('ST_C')} from VIIRS on {colored_logging.time(fill_date)} for {colored_logging.time(target_date)}")
                    ST_C_M_fill = vnp21.ST_C(date_UTC=target_date, geometry=M_geometry, resampling="cubic")
                    ST_C_M = rt.where(np.isnan(ST_C_M), ST_C_M_fill, ST_C_M)

            # for i in range(1, 16):
            #     ST_C_M_fill = vnp21.ST_C(
            #         date_UTC=target_date - timedelta(days=i),
            #         geometry=M_geometry,
            #         resampling="cubic"
            #     )
            #
            #     ST_C_M = rt.where(np.isnan(ST_C_M), ST_C_M_fill, ST_C_M)

            ST_C_M_smooth = GEOS5FP_connection.Ts_K(time_UTC=time_UTC, geometry=M_geometry,
                                                    resampling="cubic") - 273.15
            ST_C_M = rt.where(np.isnan(ST_C_M), ST_C_M_smooth, ST_C_M)

            try:
                NDVI_HLS = HLS.NDVI(tile=tile, date_UTC=target_date).to_geometry(HLS_geometry)
                NDVI_HLS = rt.where(np.isnan(NDVI_HLS), NDVI_HLS_prior, NDVI_HLS)
            except Exception as e:
                NDVI_HLS = NDVI_HLS_prior

            NDVI_HLS_prior = NDVI_HLS

            # logger.info(
            #     f"retrieving VIIRS VIIRS M-band NDVI for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} at {colored_logging.val(M_cell_size)}m resolution")
            # NDVI_M = retrieve_VNP43MA4N(
            #     geometry=M_geometry,
            #     date_UTC=target_date,
            #     variable="NDVI",
            #     directory=VIIRS_download,
            #     resampling="cubic"
            # )
            #
            # NDVI_M_smooth = GEOS5FP_connection.NDVI(time_UTC=time_UTC, geometry=M_geometry, resampling="cubic")
            # NDVI_M = rt.where(np.isnan(NDVI_M), NDVI_M_smooth, NDVI_M)
            # NDVI_M = rt.where(water_M, np.nan, NDVI_M)

            logger.info(
                f"retrieving VIIRS I-band NDVI for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} at {colored_logging.val(I_cell_size)}m resolution")
            # NDVI_I = retrieve_VNP43IA4N(
            #     geometry=I_geometry,
            #     date_UTC=target_date,
            #     variable="NDVI",
            #     directory=VIIRS_download,
            #     resampling="cubic"
            # )
            NDVI_I = VIIRS_shortwave_source.NDVI(
                date_UTC=target_date,
                geometry=I_geometry,
                resampling="cubic"
            )

            if use_VIIRS_composite:
                for days_back in range(1, VIIRS_composite_days):
                    fill_date = target_date - timedelta(days_back)
                    logger.info(
                        f"gap-filling {colored_logging.name('VNP09GA')} {colored_logging.name('NDVI')} from VIIRS on {colored_logging.time(fill_date)} for {colored_logging.time(target_date)}")
                    NDVI_I_fill = VIIRS_shortwave_source.NDVI(date_UTC=target_date, geometry=I_geometry, resampling="cubic")
                    NDVI_I = rt.where(np.isnan(NDVI_I), NDVI_I_fill, NDVI_I)

            NDVI_I_smooth = GEOS5FP_connection.NDVI(time_UTC=time_UTC, geometry=I_geometry, resampling="cubic")
            NDVI_I = rt.where(np.isnan(NDVI_I), NDVI_I_smooth, NDVI_I)
            NDVI_I = rt.where(water_I, np.nan, NDVI_I)

            logger.info(
                f"down-scaling I-band NDVI to HLS composite for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} from {colored_logging.val(I_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")
            NDVI = bias_correct(
                coarse_image=NDVI_I,
                fine_image=NDVI_HLS
            )

            NDVI_smooth = GEOS5FP_connection.NDVI(time_UTC=time_UTC, geometry=HLS_geometry, resampling="cubic")
            NDVI = rt.where(np.isnan(NDVI), NDVI_smooth, NDVI)
            NDVI = rt.where(water_HLS, np.nan, NDVI)
            NDVI = rt.clip(NDVI, 0, 1)
            check_distribution(NDVI, "NDVI", target_date, tile)

            logger.info(
                f"retrieving VIIRS VNP21 emissivity for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} at {colored_logging.val(M_cell_size)}m resolution")
            # emissivity_M = retrieve_VNP21NRT_emissivity(
            #     geometry=M_geometry,
            #     date_solar=target_date,
            #     directory=VIIRS_download,
            #     resampling="cubic"
            # )
            #
            # emissivity_M = rt.where(water_M, 0.96, emissivity_M)
            # emissivity_M = rt.where(np.isnan(emissivity_M), 1.0094 + 0.047 * np.log(rt.clip(NDVI_M, 0.01, 1)),
            #                         emissivity_M)
            # emissivity_estimate = 1.0094 + 0.047 * np.log(rt.clip(NDVI, 0.01, 1))


            # logger.info(
            #     f"down-scaling VNP21 emissivity to HLS composite for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} from {colored_logging.val(M_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")
            # emissivity = bias_correct(
            #     coarse_image=emissivity_M,
            #     fine_image=emissivity_estimate
            # )
            #
            # emissivity = rt.where(water_HLS, 0.96, emissivity)
            # emissivity = rt.clip(emissivity, 0, 1)
            emissivity = 1.0094 + 0.047 * np.log(NDVI)
            emissivity = rt.where(water, 0.96, emissivity)
            check_distribution(emissivity, "emissivity", target_date, tile)

            try:
                albedo_HLS = HLS.albedo(tile=tile, date_UTC=target_date).to_geometry(HLS_geometry)
                albedo_HLS = rt.where(np.isnan(albedo_HLS), albedo_HLS_prior, albedo_HLS)
            except Exception as e:
                albedo_HLS = albedo_HLS_prior

            albedo_HLS_prior = albedo_HLS

            logger.info(
                f"retrieving VIIRS M-band albedo for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} at {colored_logging.val(M_cell_size)}m resolution")
            # albedo_M = retrieve_VNP43MA4N(
            #     geometry=M_geometry,
            #     date_UTC=target_date,
            #     variable="albedo",
            #     directory=VIIRS_download,
            #     resampling="cubic"
            # )
            albedo_M = VIIRS_shortwave_source.albedo(date_UTC=target_date, geometry=M_geometry, resampling="cubic")

            if use_VIIRS_composite:
                for days_back in range(1, VIIRS_composite_days):
                    fill_date = target_date - timedelta(days_back)
                    logger.info(
                        f"gap-filling {colored_logging.name('VNP09GA')} {colored_logging.name('albedo')} from VIIRS on {colored_logging.time(fill_date)} for {colored_logging.time(target_date)}")
                    albedo_M_fill = VIIRS_shortwave_source.albedo(date_UTC=target_date, geometry=M_geometry, resampling="cubic")
                    albedo_M = rt.where(np.isnan(albedo_M), albedo_M_fill, albedo_M)

            albedo_M_smooth = GEOS5FP_connection.ALBEDO(time_UTC=time_UTC, geometry=M_geometry, resampling="cubic")
            albedo_M = rt.where(np.isnan(albedo_M), albedo_M_smooth, albedo_M)

            logger.info(
                f"down-scaling M-band albedo to HLS composite for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} from {colored_logging.val(M_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")
            albedo = bias_correct(
                coarse_image=albedo_M,
                fine_image=albedo_HLS
            )

            albedo_smooth = GEOS5FP_connection.ALBEDO(time_UTC=time_UTC, geometry=HLS_geometry, resampling="cubic")
            albedo = rt.where(np.isnan(albedo), albedo_smooth, albedo)
            albedo = rt.clip(albedo, 0, 1)
            check_distribution(albedo, "albedo", target_date, tile)
            # most_recent["albedo"] = albedo

            logger.info(
                f"down-scaling VNP21 ST to Landsat 8/9 for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} from {colored_logging.val(M_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")

            ST_C = bias_correct(
                coarse_image=ST_C_M,
                fine_image=landsat_ST_C
            )

            ST_C_smooth = GEOS5FP_connection.Ts_K(time_UTC=time_UTC, geometry=HLS_geometry,
                                                  resampling="cubic") - 273.15
            ST_C = rt.where(np.isnan(ST_C), ST_C_smooth, ST_C)
            ST_K = ST_C + 273.15
            check_distribution(ST_C, "ST_C", target_date, tile)
            # most_recent["ST_C"] = ST_C

            SM = None

            if downscale_moisture:
                logger.info(
                    f"down-scaling GEOS-5 FP soil moisture to HLS composite for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} from {colored_logging.val(GEOS5FP_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")

                SM_coarse_GEOS5FP = GEOS5FP_connection.SFMC(time_UTC=time_UTC, geometry=GEOS5FP_geometry,
                                                            resampling="cubic")
                SM_smooth_GEOS5FP = GEOS5FP_connection.SFMC(time_UTC=time_UTC, geometry=HLS_geometry,
                                                            resampling="cubic")
                SM = downscale_soil_moisture(
                    time_UTC=time_UTC,
                    fine_geometry=HLS_geometry,
                    coarse_geometry=GEOS5FP_geometry,
                    SM_coarse=SM_coarse_GEOS5FP,
                    SM_resampled=SM_smooth_GEOS5FP,
                    ST_fine=ST_K,
                    NDVI_fine=NDVI,
                    water=water_HLS
                )
            else:
                logger.info(
                    f"down-sampling GEOS-5 FP soil moisture for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} from {colored_logging.val(GEOS5FP_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")
                SM = GEOS5FP_connection.SFMC(time_UTC=time_UTC, geometry=HLS_geometry, resampling="cubic")

            check_distribution(SM, "SM", target_date, tile)
            # most_recent["SM"] = SM

            Ta_K = None

            if downscale_air:
                logger.info(
                    f"down-scaling GEOS-5 FP air temperature to HLS composite for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} from {colored_logging.val(GEOS5FP_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")
                Ta_K_coarse = GEOS5FP_connection.Ta_K(time_UTC=time_UTC, geometry=GEOS5FP_geometry, resampling="cubic")
                Ta_K_smooth = GEOS5FP_connection.Ta_K(time_UTC=time_UTC, geometry=HLS_geometry, resampling="cubic")
                Ta_K = downscale_air_temperature(
                    time_UTC=time_UTC,
                    Ta_K_coarse=Ta_K_coarse,
                    ST_K=ST_K,
                    fine_geometry=HLS_geometry,
                    coarse_geometry=GEOS5FP_geometry
                )

                Ta_K = rt.where(np.isnan(Ta_K), Ta_K_smooth, Ta_K)
            else:
                logger.info(
                    f"down-sampling GEOS-5 FP air temperature for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} from {colored_logging.val(GEOS5FP_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")
                Ta_K = GEOS5FP_connection.Ta_K(time_UTC=time_UTC, geometry=HLS_geometry, resampling="cubic")

            Ta_C = Ta_K - 273.15
            check_distribution(Ta_C, "Ta_C", target_date, tile)
            # most_recent["Ta_C"] = Ta_C

            RH = None

            if downscale_humidity:
                logger.info(
                    f"down-scaling GEOS-5 FP humidity to HLS composite for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} from {colored_logging.val(GEOS5FP_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")

                VPD_Pa_coarse = GEOS5FP_connection.VPD_Pa(time_UTC=time_UTC, geometry=GEOS5FP_geometry,
                                                          resampling="cubic")
                VPD_Pa_smooth = GEOS5FP_connection.VPD_Pa(time_UTC=time_UTC, geometry=HLS_geometry, resampling="cubic")

                VPD_Pa = downscale_vapor_pressure_deficit(
                    time_UTC=time_UTC,
                    VPD_Pa_coarse=VPD_Pa_coarse,
                    ST_K=ST_K,
                    fine_geometry=HLS_geometry,
                    coarse_geometry=GEOS5FP_geometry
                )

                VPD_Pa = rt.where(np.isnan(VPD_Pa), VPD_Pa_smooth, VPD_Pa)

                VPD_kPa = VPD_Pa / 1000

                RH_coarse = GEOS5FP_connection.RH(time_UTC=time_UTC, geometry=GEOS5FP_geometry, resampling="cubic")
                RH_smooth = GEOS5FP_connection.RH(time_UTC=time_UTC, geometry=HLS_geometry, resampling="cubic")

                RH = downscale_relative_humidity(
                    time_UTC=time_UTC,
                    RH_coarse=RH_coarse,
                    SM=SM,
                    ST_K=ST_K,
                    VPD_kPa=VPD_kPa,
                    water=water_HLS,
                    fine_geometry=HLS_geometry,
                    coarse_geometry=GEOS5FP_geometry
                )

                RH = rt.where(np.isnan(RH), RH_smooth, RH)
            else:
                logger.info(
                    f"down-sampling GEOS-5 FP relative humidity for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} from {colored_logging.val(GEOS5FP_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")
                RH = GEOS5FP_connection.RH(time_UTC=time_UTC, geometry=HLS_geometry, resampling="cubic")

            check_distribution(RH, "RH", target_date, tile)

            VIIRS_GEOS5FP(
                target_date=target_date,
                geometry=HLS_geometry,
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
                ST_C=ST_C,
                emissivity=emissivity,
                NDVI=NDVI,
                albedo=albedo,
                SM=SM,
                Ta_C=Ta_C,
                RH=RH,
                SWin=SWin_model_name,
                Rn=Rn_model_name,
                water=water_HLS,
                coarse_cell_size=GEOS5FP_cell_size,
                target_variables=target_variables,
                downscale_air=downscale_air,
                downscale_humidity=downscale_humidity,
                downscale_moisture=downscale_moisture,
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

    ET_toolbox_historical_fine_tile(
        tile=tile,
        start_date=start_date,
        end_date=end_date,
        working_directory=working_directory,
        static_directory=static_directory,
        SRTM_download=SRTM_download,
        VIIRS_download_directory=VIIRS_download_directory,
        GEOS5FP_download=GEOS5FP_download,
    )
