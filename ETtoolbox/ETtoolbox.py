import logging
from datetime import datetime, timedelta, date
from os.path import join, abspath, expanduser
from typing import List, Callable, Union

import numpy as np

import colored_logging
import rasters as rt
from gedi_canopy_height import GEDICanopyHeight
from geos5fp import GEOS5FP
from ETtoolbox.GFS import forecast_Ta_C, forecast_RH, get_GFS_listing, forecast_SWin
from harmonized_landsat_sentinel import HLS2Connection
from ETtoolbox.LANCE import retrieve_VNP43MA4N, retrieve_VNP43IA4N, retrieve_VNP21NRT_emissivity, available_LANCE_dates
from ETtoolbox.LANCE_GEOS5FP_NRT import LANCE_GEOS5FP_NRT, LANCENotAvailableError, GEOS5FPNotAvailableError, retrieve_VNP21NRT_ST, \
    check_LANCE_already_processed, DEFAULT_LANCE_OUTPUT_DIRECTORY, load_LANCE
from ETtoolbox.LANCE_GFS_forecast import LANCE_GFS_forecast
from ETtoolbox.LandsatL2C2 import LandsatL2C2
from modisci import MODISCI
from ETtoolbox.PTJPLSM import PTJPLSM, DEFAULT_PREVIEW_QUALITY, DEFAULT_RESAMPLING
from ETtoolbox.SRTM import SRTM
from soil_capacity_wilting import SoilGrids
from solar_apparent_time import solar_to_UTC
from ETtoolbox.daterange import date_range
from geos5fp.downscaling import bias_correct, downscale_soil_moisture, downscale_air_temperature, \
    downscale_vapor_pressure_deficit, downscale_relative_humidity
from rasters import Raster, RasterGrid
from sentinel_tiles import sentinel_tiles

logger = logging.getLogger(__name__)

ET_MODEL_NAME = "PTJPLSM"
SWIN_MODEL_NAME = "GEOS5FP"
RN_MODEL_NAME = "Verma"

DOWNSCALE_AIR = True
DOWNSCALE_HUMIDITY = True
DOWNSCALE_MOISTURE = True
FLOOR_TOPT = True

STATIC_DIRECTORY = "PTJPL_static"
SRTM_DIRECTORY = "SRTM_download_directory"
LANCE_DIRECTORY = "LANCE_download_directory"
GEOS5FP_DIRECTORY = "GEOS5FP_download_directory"
GFS_DIRECTORY = "GFS_download_directory"

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


def ET_toolbox_hindcast_forecast_tile(
        tile: str,
        present_date: Union[date, str] = None,
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
        GFS_download_directory: str = None,
        LANCE_download_directory: str = None,
        LANCE_output_directory: str = None,
        SRTM_connection: SRTM = None,
        SRTM_download_directory: str = None,
        GEOS5FP_connection: GEOS5FP = None,
        GEOS5FP_download_directory: str = None,
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
        floor_Topt: bool = FLOOR_TOPT,
        apply_GEOS5FP_GFS_bias_correction: bool = True,
        save_intermediate: bool = False,
        show_distribution: bool = False,
        load_previous: bool = True,
        target_variables: List[str] = None):
    if present_date is None:
        present_date = datetime.utcnow().date()

    logger.info(
        f"generating ET Toolbox hindcast and forecast at tile {colored_logging.place(tile)} centered on present date: {colored_logging.time(present_date)}")

    if working_directory is None:
        working_directory = "."

    working_directory = abspath(expanduser(working_directory))
    logger.info(f"working directory: {working_directory}")

    if static_directory is None:
        static_directory = join(working_directory, STATIC_DIRECTORY)

    logger.info(f"static directory: {static_directory}")

    if SRTM_download_directory is None:
        SRTM_download_directory = join(working_directory, SRTM_DIRECTORY)

    logger.info(f"SRTM directory: {SRTM_download_directory}")

    if LANCE_download_directory is None:
        LANCE_download_directory = join(working_directory, LANCE_DIRECTORY)

    logger.info(f"LANCE directory: {LANCE_download_directory}")

    if GEOS5FP_download_directory is None:
        GEOS5FP_download_directory = join(working_directory, GEOS5FP_DIRECTORY)

    logger.info(f"GEOS-5 FP directory: {GEOS5FP_download_directory}")

    if GFS_download_directory is None:
        GFS_download_directory = join(working_directory, GFS_DIRECTORY)

    if HLS_geometry is None:
        logger.info(f"HLS cell size: {colored_logging.val(HLS_cell_size)}m")
        HLS_geometry = sentinel_tiles.grid(tile, cell_size=HLS_cell_size)

    HLS_polygon_latlon = HLS_geometry.boundary_latlon.geometry

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
            download_directory=GEOS5FP_download_directory,
            products_directory=GEOS5FP_products
        )

    if SRTM_connection is None:
        # FIXME fix handling of credentials here
        SRTM_connection = SRTM(
            working_directory=working_directory,
            download_directory=SRTM_download_directory,
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

    if LANCE_output_directory is None:
        LANCE_output_directory = join(working_directory, DEFAULT_LANCE_OUTPUT_DIRECTORY)

    logger.info("listing available LANCE dates")
    LANCE_dates = available_LANCE_dates("VNP43MA4N", archive="5000")
    earliest_LANCE_date = LANCE_dates[0]
    latest_LANCE_date = LANCE_dates[-1]
    logger.info(f"LANCE is available from {colored_logging.time(earliest_LANCE_date)} to {colored_logging.time(latest_LANCE_date)}")

    LANCE_dates_processed = set()
    LANCE_not_processed = True

    for relative_days in range(-7, 0):
        target_date = present_date + timedelta(days=relative_days)
        logger.info(f"LANCE GEOS-5 FP target date: {colored_logging.time(target_date)} ({colored_logging.time(relative_days)} days)")

        time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
        logger.info(f"LANCE target time solar: {colored_logging.time(time_solar)}")
        time_UTC = solar_to_UTC(time_solar, HLS_geometry.centroid.latlon.x)

        if target_date < earliest_LANCE_date:
            logger.info(
                f"target date {target_date} is before earliest available LANCE {earliest_LANCE_date}")
            continue

        if target_date > latest_LANCE_date:
            logger.info(
                f"target date {target_date} is after latest available LANCE {latest_LANCE_date}")
            continue

        LANCE_already_processed = check_LANCE_already_processed(
            LANCE_output_directory=LANCE_output_directory,
            target_date=target_date,
            time_UTC=time_UTC,
            target=tile,
            products=target_variables
        )

        if LANCE_already_processed:
            logger.info(f"LANCE GEOS-5 FP already processed at tile {colored_logging.place(tile)} for date {target_date}")
            LANCE_dates_processed |= {target_date}
            continue
        else:
            LANCE_not_processed = False

    HLS_start = earliest_LANCE_date - timedelta(days=HLS_initialization_days)
    HLS_end = earliest_LANCE_date - timedelta(days=1)

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

    landsat_start = earliest_LANCE_date - timedelta(days=landsat_initialization_days)
    landsat_end = earliest_LANCE_date - timedelta(days=1)
    landsat_listing = landsat.scene_search(start=landsat_start, end=landsat_end, target_geometry=HLS_polygon_latlon)
    ST_C_images = []

    for date_UTC in sorted(set(landsat_listing.date_UTC)):
        try:
            ST_C = landsat.product(acquisition_date=date_UTC, product="ST_C", geometry=HLS_geometry, target_name=tile)
            ST_C_images.append(ST_C)
        except Exception as e:
            logger.warning(e)
            continue

    landsat_ST_C_initial = Raster(np.nanmedian(np.stack(ST_C_images), axis=0), geometry=HLS_geometry)
    landsat_ST_C_prior = landsat_ST_C_initial

    logger.info("getting GFS listing")
    GFS_listing = get_GFS_listing()

    for relative_days in range(-7, 0):
        target_date = present_date + timedelta(days=relative_days)
        logger.info(f"LANCE GEOS-5 FP target date: {colored_logging.time(target_date)} ({colored_logging.time(relative_days)} days)")

        time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
        logger.info(f"LANCE target time solar: {colored_logging.time(time_solar)}")
        time_UTC = solar_to_UTC(time_solar, HLS_geometry.centroid.latlon.x)

        if target_date < earliest_LANCE_date:
            logger.info(
                f"target date {target_date} is before earliest available LANCE {earliest_LANCE_date}")
            continue

        if target_date > latest_LANCE_date:
            logger.info(
                f"target date {target_date} is after latest available LANCE {latest_LANCE_date}")
            continue

        try:
            LANCE_already_processed = check_LANCE_already_processed(
                LANCE_output_directory=LANCE_output_directory,
                target_date=target_date,
                time_UTC=time_UTC,
                target=tile,
                products=target_variables
            )

            if LANCE_already_processed:
                logger.info(f"LANCE GEOS-5 FP already processed at tile {colored_logging.place(tile)} for date {target_date}")
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
                f"retrieving LANCE VNP21 ST for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} at {colored_logging.val(M_cell_size)}m resolution")
            ST_K_M = retrieve_VNP21NRT_ST(
                geometry=M_geometry,
                date_solar=target_date,
                directory=LANCE_download_directory, resampling="cubic"
            )

            ST_C_M = ST_K_M - 273.15
            ST_C_M_smooth = GEOS5FP_connection.Ts_K(time_UTC=time_UTC, geometry=M_geometry,
                                                    resampling="cubic") - 273.15
            ST_C_M = rt.where(np.isnan(ST_C_M), ST_C_M_smooth, ST_C_M)

            try:
                NDVI_HLS = HLS.NDVI(tile=tile, date_UTC=target_date).to_geometry(HLS_geometry)
                NDVI_HLS = rt.where(np.isnan(NDVI_HLS), NDVI_HLS_prior, NDVI_HLS)
            except Exception as e:
                NDVI_HLS = NDVI_HLS_prior

            NDVI_HLS_prior = NDVI_HLS

            logger.info(
                f"retrieving LANCE VIIRS M-band NDVI for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} at {colored_logging.val(M_cell_size)}m resolution")
            NDVI_M = retrieve_VNP43MA4N(
                geometry=M_geometry,
                date_UTC=target_date,
                variable="NDVI",
                directory=LANCE_download_directory,
                resampling="cubic"
            )

            NDVI_M_smooth = GEOS5FP_connection.NDVI(time_UTC=time_UTC, geometry=M_geometry, resampling="cubic")
            NDVI_M = rt.where(np.isnan(NDVI_M), NDVI_M_smooth, NDVI_M)
            NDVI_M = rt.where(water_M, np.nan, NDVI_M)

            logger.info(
                f"retrieving LANCE VIIRS I-band NDVI for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} at {colored_logging.val(I_cell_size)}m resolution")
            NDVI_I = retrieve_VNP43IA4N(
                geometry=I_geometry,
                date_UTC=target_date,
                variable="NDVI",
                directory=LANCE_download_directory,
                resampling="cubic"
            )

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
                f"retrieving LANCE VNP21 emissivity for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} at {colored_logging.val(M_cell_size)}m resolution")
            emissivity_M = retrieve_VNP21NRT_emissivity(
                geometry=M_geometry,
                date_solar=target_date,
                directory=LANCE_download_directory,
                resampling="cubic"
            )

            emissivity_M = rt.where(water_M, 0.96, emissivity_M)
            emissivity_M = rt.where(np.isnan(emissivity_M), 1.0094 + 0.047 * np.log(rt.clip(NDVI_M, 0.01, 1)),
                                    emissivity_M)
            emissivity_estimate = 1.0094 + 0.047 * np.log(rt.clip(NDVI, 0.01, 1))

            logger.info(
                f"down-scaling VNP21 emissivity to HLS composite for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} from {colored_logging.val(M_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")
            emissivity = bias_correct(
                coarse_image=emissivity_M,
                fine_image=emissivity_estimate
            )

            emissivity = rt.where(water_HLS, 0.96, emissivity)
            emissivity = rt.clip(emissivity, 0, 1)
            check_distribution(emissivity, "emissivity", target_date, tile)

            try:
                albedo_HLS = HLS.albedo(tile=tile, date_UTC=target_date).to_geometry(HLS_geometry)
                albedo_HLS = rt.where(np.isnan(albedo_HLS), albedo_HLS_prior, albedo_HLS)
            except Exception as e:
                albedo_HLS = albedo_HLS_prior

            albedo_HLS_prior = albedo_HLS

            logger.info(
                f"retrieving LANCE VIIRS M-band albedo for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} at {colored_logging.val(M_cell_size)}m resolution")
            albedo_M = retrieve_VNP43MA4N(
                geometry=M_geometry,
                date_UTC=target_date,
                variable="albedo",
                directory=LANCE_download_directory,
                resampling="cubic"
            )

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

            LANCE_GEOS5FP_NRT(
                target_date=target_date,
                geometry=HLS_geometry,
                target=tile,
                working_directory=working_directory,
                static_directory=static_directory,
                SRTM_connection=SRTM_connection,
                SRTM_download=SRTM_download_directory,
                GEDI_connection=GEDI_connection,
                GEDI_download=GEDI_download,
                ORNL_connection=ORNL_connection,
                CI_directory=CI_directory,
                soil_grids_connection=soil_grids_connection,
                soil_grids_download=soil_grids_download,
                LANCE_download_directory=LANCE_download_directory,
                LANCE_output_directory=LANCE_output_directory,
                intermediate_directory=intermediate_directory,
                preview_quality=preview_quality,
                ANN_model=ANN_model,
                ANN_model_filename=ANN_model_filename,
                model=model,
                model_name=ET_model_name,
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
                floor_Topt=floor_Topt,
                resampling=resampling,
                show_distribution=show_distribution,
                load_previous=load_previous,
                save_intermediate=save_intermediate
            )

            LANCE_dates_processed |= {target_date}
        except (LANCENotAvailableError, GEOS5FPNotAvailableError) as e:
            logger.warning(e)
            logger.warning(f"LANCE GEOS-5 FP cannot be processed for date: {target_date}")
            missing_dates.append(target_date)
            continue
        except Exception as e:
            logger.exception(e)
            logger.warning(f"LANCE GEOS-5 FP cannot be processed for date: {target_date}")
            missing_dates.append(target_date)
            continue

    logger.info("missing LANCE GEOS-5 FP dates: " + ", ".join(colored_logging.time(d) for d in missing_dates))

    forecast_dates = missing_dates + [present_date + timedelta(days=d) for d in range(8)]

    earliest_LANCE_date = min(LANCE_dates_processed)
    latest_LANCE_date = max(LANCE_dates_processed)

    for target_date in forecast_dates:
        relative_days = target_date - present_date
        logger.info(f"GFS LANCE target date: {colored_logging.time(target_date)} ({colored_logging.time(relative_days)} days)")

        time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
        logger.info(f"LANCE target time solar: {colored_logging.time(time_solar)}")
        time_UTC = solar_to_UTC(time_solar, HLS_geometry.centroid.latlon.x)

        if target_date < earliest_LANCE_date:
            # raise ValueError(f"target date {target_date} is before earliest available LANCE {earliest_LANCE_date}")
            logger.warning(f"target date {target_date} is before earliest available LANCE {earliest_LANCE_date}")
            continue

        if target_date <= latest_LANCE_date:
            logger.warning(
                f"target date {colored_logging.time(target_date)} is within LANCE date range from {colored_logging.time(earliest_LANCE_date)} to {colored_logging.time(latest_LANCE_date)}")
            LANCE_processing_date = target_date
        else:
            LANCE_processing_date = latest_LANCE_date
            logger.info(f"processing LANCE on latest date available: {colored_logging.time(LANCE_processing_date)}")

        LANCE_processing_datetime_solar = datetime(LANCE_processing_date.year, LANCE_processing_date.month,
                                                   LANCE_processing_date.day, 13, 30)
        logger.info(f"LANCE processing date/time solar: {colored_logging.time(LANCE_processing_datetime_solar)}")
        LANCE_processing_datetime_UTC = solar_to_UTC(LANCE_processing_datetime_solar, HLS_geometry.centroid.latlon.x)
        logger.info(f"LANCE processing date/time UTC: {colored_logging.time(LANCE_processing_datetime_UTC)}")

        most_recent = load_LANCE(
            LANCE_output_directory=LANCE_output_directory,
            target_date=LANCE_processing_date,
            target=tile
        )

        ST_C = most_recent["ST"]
        emissivity = most_recent["emissivity"]
        NDVI = most_recent["NDVI"]
        albedo = most_recent["albedo"]
        SM = most_recent["SM"]

        logger.info(
            f"down-scaling GFS solar radiation to HLS composite for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} from {colored_logging.val(GFS_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")

        SWin_prior = most_recent["SWin"]
        SWin_GFS = forecast_SWin(time_UTC=time_UTC, geometry=GFS_geometry, directory=GFS_download_directory, listing=GFS_listing)

        if apply_GEOS5FP_GFS_bias_correction:
            matching_SWin_GFS = forecast_SWin(
                time_UTC=LANCE_processing_datetime_UTC,
                geometry=GFS_geometry,
                directory=GFS_download_directory,
                resampling="cubic",
                listing=GFS_listing
            )

            matching_SWin_GEOS5FP = GEOS5FP_connection.SWin(
                time_UTC=LANCE_processing_datetime_UTC,
                geometry=GFS_geometry,
                resampling="cubic"
            )

            SWin_GFS_bias = matching_SWin_GFS - matching_SWin_GEOS5FP
            SWin_GFS = SWin_GFS - SWin_GFS_bias

        SWin = bias_correct(
            coarse_image=SWin_GFS,
            fine_image=SWin_prior
        )

        logger.info(
            f"down-scaling GFS air temperature to HLS composite for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} from {colored_logging.val(GFS_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")

        Ta_C_prior = most_recent["Ta"]
        Ta_C_GFS = forecast_Ta_C(time_UTC=time_UTC, geometry=GFS_geometry, directory=GFS_download_directory, listing=GFS_listing)

        if apply_GEOS5FP_GFS_bias_correction:
            matching_Ta_C_GFS = forecast_Ta_C(time_UTC=LANCE_processing_datetime_UTC, geometry=GFS_geometry,
                                              directory=GFS_download_directory, resampling="cubic", listing=GFS_listing)
            matching_Ta_C_GEOS5FP = GEOS5FP_connection.Ta_C(time_UTC=LANCE_processing_datetime_UTC,
                                                            geometry=GFS_geometry,
                                                            resampling="cubic")
            Ta_C_GFS_bias = matching_Ta_C_GFS - matching_Ta_C_GEOS5FP
            Ta_C_GFS = Ta_C_GFS - Ta_C_GFS_bias

        Ta_C = bias_correct(
            coarse_image=Ta_C_GFS,
            fine_image=Ta_C_prior
        )

        logger.info(
            f"down-scaling GFS humidity to HLS composite for tile {colored_logging.place(tile)} on date {colored_logging.time(target_date)} from {colored_logging.val(GFS_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")

        RH_prior = most_recent["RH"]
        RH_GFS = forecast_RH(time_UTC=time_UTC, geometry=GFS_geometry, directory=GFS_download_directory, listing=GFS_listing)

        if apply_GEOS5FP_GFS_bias_correction:
            matching_RH_GFS = forecast_RH(time_UTC=LANCE_processing_datetime_UTC, geometry=GFS_geometry,
                                          directory=GFS_download_directory, resampling="cubic", listing=GFS_listing)
            matching_RH_GEOS5FP = GEOS5FP_connection.RH(time_UTC=LANCE_processing_datetime_UTC, geometry=GFS_geometry,
                                                        resampling="cubic")
            RH_GFS_bias = matching_RH_GFS - matching_RH_GEOS5FP
            RH_GFS = RH_GFS - RH_GFS_bias

        RH = bias_correct(
            coarse_image=RH_GFS,
            fine_image=RH_prior
        )

        try:
            LANCE_GFS_forecast(
                target_date=target_date,
                geometry=HLS_geometry,
                coarse_geometry=GFS_geometry,
                coarse_cell_size=GFS_cell_size,
                target=tile,
                working_directory=working_directory,
                static_directory=static_directory,
                SRTM_connection=SRTM_connection,
                SRTM_download=SRTM_download_directory,
                GEDI_connection=GEDI_connection,
                GEDI_download=GEDI_download,
                ORNL_connection=ORNL_connection,
                CI_directory=CI_directory,
                soil_grids_connection=soil_grids_connection,
                soil_grids_download=soil_grids_download,
                LANCE_download_directory=LANCE_download_directory,
                intermediate_directory=intermediate_directory,
                preview_quality=preview_quality,
                ANN_model=ANN_model,
                ANN_model_filename=ANN_model_filename,
                model=model,
                model_name=ET_model_name,
                ST_C=ST_C,
                emissivity=emissivity,
                NDVI=NDVI,
                albedo=albedo,
                SM=SM,
                Ta_C=Ta_C,
                RH=RH,
                SWin=SWin,
                water=water_HLS,
                GFS_listing=GFS_listing,
                target_variables=target_variables,
                downscale_air=downscale_air,
                downscale_humidity=downscale_humidity,
                downscale_moisture=downscale_moisture,
                apply_GEOS5FP_GFS_bias_correction=apply_GEOS5FP_GFS_bias_correction,
                LANCE_processing_date=LANCE_processing_date,
                resampling=resampling,
                show_distribution=show_distribution,
                load_previous=load_previous,
                save_intermediate=save_intermediate
            )
        except Exception as e:
            logger.exception(e)
            logger.warning(f"LANCE GFS cannot be processed for date: {target_date}")
            continue
