from typing import Union, List
from datetime import date, datetime
import VNP09GA_002
import VNP21A1D_002
from dateutil import parser
import logging
from os.path import join, exists, splitext, basename, abspath, expanduser
import numpy as np
import rasters as rt
from glob import glob
from typing import Dict, Callable
from rasters import Raster, RasterGrid
from gedi_canopy_height import GEDICanopyHeight
from GEOS5FP import GEOS5FP
from global_forecasting_system import *
from ETtoolbox.VIIRS import *
from MODISCI import MODISCI
from PTJPL import PTJPL
from PTJPLSM import PTJPLSM
from ETtoolbox.SRTM import SRTM
from soil_capacity_wilting import SoilGrids
from GEOS5FP.downscaling import downscale_air_temperature, downscale_soil_moisture, bias_correct
from sentinel_tiles import sentinel_tiles
from solar_apparent_time import solar_to_UTC
import colored_logging as cl
from verma_net_radiation import process_verma_net_radiation

DEFAULT_GFS_DOWNLOAD_DIRECTORY = "GFS_download_directory"
DEFAULT_GFS_OUTPUT_DIRECTORY = "GFS_output"
DEFAULT_VIIRS_DOWNLOAD_DIRECTORY = "VIIRS_download_directory"
DEFAULT_RESAMPLING = "cubic"
DEFAULT_PREVIEW_QUALITY = 20
GFS_CELL_SIZE = 27375
DEFAULT_TARGET_VARIABLES = ["LE", "ET", "ESI"]

DEFAULT_DOWNSCALE_AIR = False
DEFAULT_DOWNSCALE_HUMIDITY = False
DEFAULT_DOWNSCALE_MOISTURE = False

logger = logging.getLogger(__name__)


def generate_GFS_output_directory(
        GFS_output_directory: str,
        target_date: Union[date, str],
        target: str):
    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()

    directory = join(
        abspath(expanduser(GFS_output_directory)),
        f"{target_date:%Y-%m-%d}",
        f"GFS_{target_date:%Y-%m-%d}_{target}",
    )

    return directory


def generate_GFS_output_filename(
        GFS_output_directory: str,
        target_date: Union[date, str],
        time_UTC: Union[datetime, str],
        target: str,
        product: str):
    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()

    if isinstance(time_UTC, str):
        time_UTC = parser.parse(time_UTC)

    directory = generate_GFS_output_directory(
        GFS_output_directory=GFS_output_directory,
        target_date=target_date,
        target=target
    )

    filename = join(directory, f"GFS_{time_UTC:%Y.%m.%d.%H.%M.%S}_{target}_{product}.tif")

    return filename


def check_GFS_already_processed(
        GFS_output_directory: str,
        target_date: Union[date, str],
        time_UTC: Union[datetime, str],
        target: str,
        products: List[str]):
    already_processed = True
    logger.info(f"checking if GFS VIIRS has previously been processed at {colored_logging.place(target)} on {colored_logging.time(target_date)}")

    for product in products:
        filename = generate_GFS_output_filename(
            GFS_output_directory=GFS_output_directory,
            target_date=target_date,
            time_UTC=time_UTC,
            target=target,
            product=product
        )

        if exists(filename):
            logger.info(
                f"found previous GFS VIIRS {colored_logging.name(product)} at {colored_logging.place(target)} on {colored_logging.time(target_date)}: {colored_logging.file(filename)}")
        else:
            logger.info(
                f"did not find previous GFS VIIRS {colored_logging.name(product)} at {colored_logging.place(target)} on {colored_logging.time(target_date)}")
            already_processed = False

    return already_processed


def load_GFS(GFS_output_directory: str, target_date: Union[date, str], target: str, products: List[str] = None):
    dataset = {}

    directory = generate_GFS_output_directory(
        GFS_output_directory=GFS_output_directory,
        target_date=target_date,
        target=target
    )

    pattern = join(directory, "*.tif")
    logger.info(f"searching for GFS products: {colored_logging.val(pattern)}")
    filenames = glob(pattern)

    for filename in filenames:
        logger.info(f"loading GFS VIIRS file: {colored_logging.file(filename)}")
        product = splitext(basename(filename))[0].split("_")[-1]

        if products is not None and product not in products:
            continue

        image = Raster.open(filename)
        dataset[product] = image

    return dataset

def VIIRS_GFS_forecast(
        target_date: Union[date, str],
        geometry: RasterGrid,
        target: str,
        coarse_geometry: RasterGrid = None,
        coarse_cell_size: float = GFS_CELL_SIZE,
        ST_C: Raster = None,
        emissivity: Raster = None,
        NDVI: Raster = None,
        albedo: Raster = None,
        SWin: Raster = None,
        Rn: Raster = None,
        SM: Raster = None,
        wind_speed: Raster = None,
        Ta_C: Raster = None,
        RH: Raster = None,
        water: Raster = None,
        model: PTJPLSM = None,
        working_directory: str = None,
        static_directory: str = None,
        GFS_download: str = None,
        GFS_output_directory: str = None,
        VIIRS_download_directory: str = None,
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
        intermediate_directory=None,
        model_name: str = "PTJPL",
        preview_quality: int = DEFAULT_PREVIEW_QUALITY,
        ANN_model: Callable = None,
        ANN_model_filename: str = None,
        spacetrack_credentials_filename: str = None,
        ERS_credentials_filename: str = None,
        resampling: str = DEFAULT_RESAMPLING,
        downscale_air: bool = DEFAULT_DOWNSCALE_AIR,
        downscale_humidity: bool = DEFAULT_DOWNSCALE_HUMIDITY,
        downscale_moisture: bool = DEFAULT_DOWNSCALE_MOISTURE,
        apply_GEOS5FP_GFS_bias_correction: bool = True,
        VIIRS_processing_date: Union[date, str] = None,
        GFS_listing: pd.DataFrame = None,
        save_intermediate: bool = False,
        include_preview: bool = True,
        show_distribution: bool = True,
        load_previous: bool = True,
        target_variables: List[str] = DEFAULT_TARGET_VARIABLES) -> Dict[str, Raster]:
    results = {}

    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()

    logger.info(f"GFS-VIIRS target date: {colored_logging.time(target_date)}")
    time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
    logger.info(f"GFS-VIIRS target time solar: {colored_logging.time(time_solar)}")
    time_UTC = solar_to_UTC(time_solar, geometry.centroid.latlon.x)
    logger.info(f"GFS-VIIRS target time UTC: {colored_logging.time(time_UTC)}")
    date_UTC= time_UTC.date()

    if isinstance(VIIRS_processing_date, str):
        VIIRS_processing_date = parser.parse(VIIRS_processing_date).date()

    if working_directory is None:
        working_directory = "."

    working_directory = abspath(expanduser(working_directory))

    logger.info(f"GFS-VIIRS working directory: {colored_logging.dir(working_directory)}")

    if GFS_download is None:
        GFS_download = join(working_directory, DEFAULT_GFS_DOWNLOAD_DIRECTORY)

    logger.info(f"GFS download directory: {colored_logging.dir(GFS_download)}")

    if GFS_output_directory is None:
        GFS_output_directory = join(working_directory, DEFAULT_GFS_OUTPUT_DIRECTORY)

    logger.info(f"GFS output directory: {colored_logging.dir(GFS_output_directory)}")

    if SRTM_connection is None:
        SRTM_connection = SRTM(
            working_directory=working_directory,
            download_directory=SRTM_download,
            offline_ok=True
        )

    if water is None:
        water = SRTM_connection.swb(geometry)

    if GEOS5FP_connection is None:
        try:
            logger.info(f"connecting to GEOS-5 FP")
            GEOS5FP_connection = GEOS5FP(
                working_directory=working_directory,
                download_directory=GEOS5FP_download
            )
        except Exception as e:
            logger.exception(e)
            raise Exception("unable to connect to GEOS-5 FP")

    GFS_already_processed = check_GFS_already_processed(
        GFS_output_directory=GFS_output_directory,
        target_date=target_date,
        time_UTC=time_UTC,
        target=target,
        products=target_variables
    )

    if GFS_already_processed:
        if load_previous:
            logger.info("loading previously generated VIIRS GEOS-5 FP output")
            return load_GFS(
                GFS_output_directory=GFS_output_directory,
                target_date=target_date,
                target=target
            )
        else:
            return

    if VIIRS_download_directory is None:
        VIIRS_download_directory = join(working_directory, DEFAULT_VIIRS_DOWNLOAD_DIRECTORY)

    VIIRS_processing_date = target_date
    VIIRS_processing_time = time_UTC

    VIIRS_processing_datetime_solar = datetime(VIIRS_processing_date.year, VIIRS_processing_date.month,
                                               VIIRS_processing_date.day, 13, 30)
    logger.info(f"VIIRS processing date/time solar: {colored_logging.time(VIIRS_processing_datetime_solar)}")
    VIIRS_processing_datetime_UTC = solar_to_UTC(VIIRS_processing_datetime_solar, geometry.centroid.latlon.x)
    logger.info(f"VIIRS processing date/time UTC: {colored_logging.time(VIIRS_processing_datetime_UTC)}")

    forecast_distance_days = (target_date - VIIRS_processing_date).days

    if forecast_distance_days > 0:
        logger.info(
            f"target date {colored_logging.time(target_date)} is {colored_logging.val(forecast_distance_days)} days past VIIRS processing date {colored_logging.time(VIIRS_processing_date)}")

    VNP21_connection = VNP21A1D_002.VNP21A1D(download_directory=VIIRS_download_directory)
    VNP09_connection = VNP09GA_002.VNP09GA(download_directory=VIIRS_download_directory)

    if ST_C is None:
        logger.info(
            f"retrieving {cl.name('VNP21A1D')} {cl.name('ST_C')} from VIIRS on {cl.time(VIIRS_processing_date)}")

        ST_K = VNP21_connection.ST_K(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )

        ST_C = ST_K - 273.15
        
        ST_C_smooth = GEOS5FP_connection.Ts_K(time_UTC=time_UTC, geometry=geometry, resampling="cubic") - 273.15
        ST_C = rt.where(np.isnan(ST_C), ST_C_smooth, ST_C)

    results["ST"] = ST_C

    if NDVI is None:
        logger.info(
            f"retrieving {cl.name('VNP09GA')} {cl.name('NDVI')} from VIIRS on {cl.time(VIIRS_processing_date)}")

        NDVI = VNP09_connection.NDVI(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )

    results["NDVI"] = NDVI

    if emissivity is None:
        logger.info(
            f"retrieving {cl.name('VNP21A1D')} {cl.name('emissivity')} from VIIRS on {cl.time(VIIRS_processing_date)}")
        emissivity = VNP21_connection.Emis_14(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )

    emissivity = rt.where(water, 0.96, emissivity)
    emissivity = rt.where(np.isnan(emissivity), 1.0094 + 0.047 * np.log(NDVI), emissivity)

    results["emissivity"] = emissivity

    if albedo is None:
        logger.info(
            f"retrieving {cl.name('VNP09GA')} {cl.name('albedo')} from VIIRS on {cl.time(VIIRS_processing_date)}")

        albedo = VNP09_connection.albedo(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )

    results["albedo"] = albedo

    if SRTM_connection is None:
        logger.info("connecting to SRTM")
        SRTM_connection = SRTM(
            working_directory=static_directory,
            download_directory=SRTM_download,
            offline_ok=True
        )

    logger.info("retrieving water mask from SRTM")
    water = SRTM_connection.swb(geometry)
    logger.info(f"running PT-JPL-SM ET model forecast at {colored_logging.time(time_UTC)}")

    if coarse_geometry is None:
        coarse_geometry = sentinel_tiles.grid(target, coarse_cell_size)

    if Ta_C is None:
        logger.info(f"retrieving GFS {colored_logging.name('Ta')} forecast at {colored_logging.time(time_UTC)}")
        if downscale_air:
            Ta_K_coarse = forecast_Ta_K(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic", listing=GFS_listing)

            if apply_GEOS5FP_GFS_bias_correction:
                matching_Ta_K_GFS = forecast_Ta_K(
                    time_UTC=VIIRS_processing_datetime_UTC,
                    geometry=coarse_geometry,
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )

                matching_Ta_K_GEOS5FP = GEOS5FP_connection.Ta_K(
                    time_UTC=VIIRS_processing_datetime_UTC,
                    geometry=coarse_geometry,
                    resampling="cubic"
                )

                Ta_K_GFS_bias = matching_Ta_K_GFS - matching_Ta_K_GEOS5FP
                Ta_K_coarse = Ta_K_coarse - Ta_K_GFS_bias

            ST_K = ST_C + 273.15

            Ta_K = downscale_air_temperature(
                time_UTC=time_UTC,
                Ta_K_coarse=Ta_K_coarse,
                ST_K=ST_K,
                fine_geometry=geometry,
                coarse_geometry=coarse_geometry
            )

            Ta_C = Ta_K - 273.15
        else:
            if apply_GEOS5FP_GFS_bias_correction:
                matching_Ta_C_GFS = forecast_Ta_C(
                    time_UTC=VIIRS_processing_datetime_UTC,
                    geometry=coarse_geometry,
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )

                matching_Ta_C_GEOS5FP = GEOS5FP_connection.Ta_C(
                    time_UTC=VIIRS_processing_datetime_UTC,
                    geometry=coarse_geometry,
                    resampling="cubic"
                )

                Ta_C_GFS_bias = matching_Ta_C_GFS - matching_Ta_C_GEOS5FP

                Ta_C_coarse = forecast_Ta_C(
                    time_UTC=time_UTC,
                    geometry=coarse_geometry,
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )

                Ta_C_coarse = Ta_C_coarse - Ta_C_GFS_bias
                Ta_C = Ta_C_coarse.to_geometry(geometry, resampling="cubic")
            else:
                Ta_C = forecast_Ta_C(
                    time_UTC=time_UTC,
                    geometry=geometry,
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )

    results["Ta"] = Ta_C

    if SM is None and model_name == "PTJPLSM":
        logger.info(f"retrieving GFS {colored_logging.name('SM')} forecast at {colored_logging.time(time_UTC)}")

        if downscale_moisture:
            SM_coarse = forecast_SM(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic")

            if apply_GEOS5FP_GFS_bias_correction:
                matching_SM_GFS = forecast_SM(
                    time_UTC=VIIRS_processing_datetime_UTC,
                    geometry=coarse_geometry,
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )

                matching_SM_GEOS5FP = GEOS5FP_connection.SFMC(
                    time_UTC=VIIRS_processing_datetime_UTC,
                    geometry=coarse_geometry,
                    resampling="cubic"
                )

                SM_GFS_bias = matching_SM_GFS - matching_SM_GEOS5FP

                SM_coarse = forecast_SM(
                    time_UTC=time_UTC,
                    geometry=coarse_geometry,
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )

                SM_coarse = SM_coarse - SM_GFS_bias

            SM_smooth = forecast_SM(time_UTC=time_UTC, geometry=geometry, resampling="cubic", listing=GFS_listing)
            ST_K = ST_C + 273.15

            SM = downscale_soil_moisture(
                time_UTC=time_UTC,
                fine_geometry=geometry,
                coarse_geometry=coarse_geometry,
                SM_coarse=SM_coarse,
                SM_resampled=SM_smooth,
                ST_fine=ST_K,
                NDVI_fine=NDVI,
                water=water
            )
        else:
            if apply_GEOS5FP_GFS_bias_correction:
                SM_coarse = forecast_SM(
                    time_UTC=time_UTC,
                    geometry=coarse_geometry,
                    resampling="cubic",
                    listing=GFS_listing
                )

                if apply_GEOS5FP_GFS_bias_correction:
                    matching_SM_GFS = forecast_SM(
                        time_UTC=VIIRS_processing_datetime_UTC,
                        geometry=coarse_geometry,
                        directory=GFS_download,
                        resampling="cubic",
                        listing=GFS_listing
                    )

                    matching_SM_GEOS5FP = GEOS5FP_connection.SFMC(
                        time_UTC=VIIRS_processing_datetime_UTC,
                        geometry=coarse_geometry,
                        resampling="cubic"
                    )

                    SM_GFS_bias = matching_SM_GFS - matching_SM_GEOS5FP

                    SM_coarse = forecast_SM(
                        time_UTC=time_UTC,
                        geometry=coarse_geometry,
                        directory=GFS_download,
                        resampling="cubic",
                        listing=GFS_listing
                    )

                    SM_coarse = SM_coarse - SM_GFS_bias

                SM = SM_coarse.to_geometry(geometry, resampling="cubic")
            else:
                SM = forecast_SM(
                    time_UTC=time_UTC,
                    geometry=geometry,
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )

    results["SM"] = SM

    if RH is None:
        logger.info(f"retrieving GFS {colored_logging.name('RH')} forecast at {colored_logging.time(time_UTC)}")

        if downscale_humidity:
            SVP_Pa = 0.6108 * np.exp((17.27 * Ta_C) / (Ta_C + 237.3)) * 1000  # [Pa]
            RH_smooth = forecast_RH(time_UTC=time_UTC, geometry=geometry, resampling="cubic", listing=GFS_listing)
            Ea_Pa_estimate = RH_smooth * SVP_Pa
            VPD_Pa_estimate = SVP_Pa - Ea_Pa_estimate
            VPD_kPa_estimate = VPD_Pa_estimate / 1000
            RH_estimate = SM ** (1 / VPD_kPa_estimate)
            RH_coarse = forecast_RH(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic", listing=GFS_listing)

            if apply_GEOS5FP_GFS_bias_correction:
                matching_RH_GFS = forecast_RH(
                    time_UTC=VIIRS_processing_datetime_UTC,
                    geometry=coarse_geometry,
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )

                matching_RH_GEOS5FP = GEOS5FP_connection.RH(
                    time_UTC=VIIRS_processing_datetime_UTC,
                    geometry=coarse_geometry,
                    resampling="cubic"
                )

                RH_GFS_bias = matching_RH_GFS - matching_RH_GEOS5FP
                RH_coarse = RH_coarse - RH_GFS_bias

            RH = bias_correct(
                coarse_image=RH_coarse,
                fine_image=RH_estimate,
                upsampling="average",
                downsampling="linear",
                return_bias=False
            )
        else:
            if apply_GEOS5FP_GFS_bias_correction:
                matching_RH_GFS = forecast_RH(
                    time_UTC=VIIRS_processing_datetime_UTC,
                    geometry=coarse_geometry,
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )

                matching_RH_GEOS5FP = GEOS5FP_connection.RH(
                    time_UTC=VIIRS_processing_datetime_UTC,
                    geometry=coarse_geometry,
                    resampling="cubic"
                )

                RH_GFS_bias = matching_RH_GFS - matching_RH_GEOS5FP

                RH_coarse = forecast_RH(
                    time_UTC=time_UTC,
                    geometry=coarse_geometry,
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )

                RH_coarse = RH_coarse - RH_GFS_bias

                RH = RH_coarse.to_geometry(geometry, resampling="cubic")
            else:
                RH = forecast_RH(time_UTC=time_UTC, geometry=geometry, directory=GFS_download, listing=GFS_listing)

    model.check_distribution(RH, "RH", date_UTC=date_UTC, target=target)
    results["RH"] = RH

    if wind_speed is None:
        logger.info(f"retrieving GFS {colored_logging.name('wind_speed')} forecast at {colored_logging.time(time_UTC)}")

        if apply_GEOS5FP_GFS_bias_correction:
            matching_wind_speed_GFS = forecast_wind(
                time_UTC=VIIRS_processing_datetime_UTC,
                geometry=coarse_geometry,
                directory=GFS_download,
                resampling="cubic",
                listing=GFS_listing
            )

            matching_wind_speed_GEOS5FP = GEOS5FP_connection.wind_speed(
                time_UTC=VIIRS_processing_datetime_UTC,
                geometry=coarse_geometry,
                resampling="cubic"
            )

            wind_speed_GFS_bias = matching_wind_speed_GFS - matching_wind_speed_GEOS5FP

            wind_speed_coarse = forecast_wind(
                time_UTC=time_UTC,
                geometry=coarse_geometry,
                directory=GFS_download,
                resampling="cubic",
                listing=GFS_listing
            )

            wind_speed_coarse = wind_speed_coarse - wind_speed_GFS_bias

            wind_speed = wind_speed_coarse.to_geometry(geometry, resampling="cubic")
        else:
            wind_speed = forecast_wind(
                time_UTC=time_UTC,
                geometry=geometry,
                directory=GFS_download,
                resampling="cubic",
                listing=GFS_listing
            )

    results["wind_speed"] = wind_speed

    if SWin is None:
        logger.info(f"retrieving GFS {colored_logging.name('SWin')} forecast at {colored_logging.time(time_UTC)}")

        if apply_GEOS5FP_GFS_bias_correction:
            matching_SWin_GFS = forecast_SWin(
                time_UTC=VIIRS_processing_datetime_UTC,
                geometry=coarse_geometry,
                directory=GFS_download,
                resampling="cubic",
                listing=GFS_listing
            )

            matching_SWin_GEOS5FP = GEOS5FP_connection.SWin(
                time_UTC=VIIRS_processing_datetime_UTC,
                geometry=coarse_geometry,
                resampling="cubic"
            )

            SWin_GFS_bias = matching_SWin_GFS - matching_SWin_GEOS5FP

            SWin_coarse = forecast_SWin(
                time_UTC=time_UTC,
                geometry=coarse_geometry,
                directory=GFS_download,
                resampling="cubic",
                listing=GFS_listing
            )

            SWin_coarse = SWin_coarse - SWin_GFS_bias

            SWin = SWin_coarse.to_geometry(geometry, resampling="cubic")
        else:
            SWin = forecast_SWin(time_UTC=time_UTC, geometry=geometry, directory=GFS_download, listing=GFS_listing)

    results["SWin"] = SWin

    
    verma_results = process_verma_net_radiation(
        SWin=SWin,
        albedo=albedo,
        ST_C=ST_C,
        emissivity=emissivity,
        Ta_C=Ta_C,
        RH=RH
    )

    Rn = verma_results["Rn"]

    logger.info(f"running PT-JPL ET model hindcast at {cl.time(time_UTC)}")

    PTJPL_results = PTJPL(
        NDVI=NDVI,
        ST_C=ST_C,
        emissivity=emissivity,
        albedo=albedo,
        Rn=Rn,
        Ta_C=Ta_C,
        RH=RH
    )

    for k, v in PTJPL_results.items():
        results[k] = v

    for product, image in results.items():
        filename = generate_GFS_output_filename(
            GFS_output_directory=GFS_output_directory,
            target_date=target_date,
            time_UTC=time_UTC,
            target=target,
            product=product
        )

        if image is None:
            logger.warning(f"no image result for {product}")
            continue

        logger.info(
            f"writing VIIRS GFS {colored_logging.name(product)} at {colored_logging.place(target)} at {colored_logging.time(time_UTC)} to file: {colored_logging.file(filename)}")
        image.to_geotiff(filename)

    return results
