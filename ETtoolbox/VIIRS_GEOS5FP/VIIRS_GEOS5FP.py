from glob import glob
from os.path import splitext
from typing import Dict, Callable

from gedi_canopy_height import GEDICanopyHeight
from geos5fp import GEOS5FP
from ETtoolbox.LANCE import *
from modisci import MODISCI
from ETtoolbox.PTJPL import PTJPL
from ETtoolbox.PTJPLSM import PTJPLSM
from ETtoolbox.SRTM import SRTM
from soil_capacity_wilting import SoilGrids
from ETtoolbox.VIIRS.VNP09GA import VNP09GA
from ETtoolbox.VIIRS.VNP21A1D import VNP21A1D
from ETtoolbox.VIIRS.VNP43MA4 import VNP43MA4
from geos5fp.downscaling import downscale_air_temperature, downscale_soil_moisture, downscale_vapor_pressure_deficit, \
    downscale_relative_humidity, bias_correct
from ETtoolbox.PTJPL import FLOOR_TOPT

ET_MODEL_NAME = "PTJPLSM"

VIIRS_DOWNLOAD_DIRECTORY = "VIIRS_download"
VIIRS_PRODUCTS_DIRECTORY = "VIIRS_products"
VIIRS_GEOS5FP_OUTPUT_DIRECTORY = "VIIRS_GEOS5FP_output"

USE_VIIRS_COMPOSITE = True
VIIRS_COMPOSITE_DAYS = 0

DEFAULT_RESAMPLING = "cubic"
DEFAULT_PREVIEW_QUALITY = 20
DEFAULT_DOWNSCALE_AIR = False
DEFAULT_DOWNSCALE_HUMIDITY = False
DEFAULT_DOWNSCALE_MOISTURE = False
DEFAULT_COARSE_CELL_SIZE = 27375
DEFAULT_TARGET_VARIABLES = ["LE", "ET", "ESI"]

logger = logging.getLogger(__name__)


class GEOS5FPNotAvailableError(Exception):
    pass


def generate_VIIRS_GEOS5FP_output_directory(VIIRS_GEOS5FP_output_directory: str, target_date: Union[date, str], target: str):
    if VIIRS_GEOS5FP_output_directory is None:
        raise ValueError("no VIIRS GEOS-5 FP output directory given")

    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()

    directory = join(abspath(expanduser(VIIRS_GEOS5FP_output_directory)), f"{target_date:%Y-%m-%d}", f"VIIRS-GEOS5FP_{target_date:%Y-%m-%d}_{target}",)

    return directory


def generate_VIIRS_GEOS5FP_output_filename(VIIRS_GEOS5FP_output_directory: str, target_date: Union[date, str], time_UTC: Union[datetime, str], target: str,
                                           product: str):
    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()

    if isinstance(time_UTC, str):
        time_UTC = parser.parse(time_UTC)

    directory = generate_VIIRS_GEOS5FP_output_directory(VIIRS_GEOS5FP_output_directory=VIIRS_GEOS5FP_output_directory, target_date=target_date, target=target)

    filename = join(directory, f"VIIRS-GEOS5FP_{time_UTC:%Y.%m.%d.%H.%M.%S}_{target}_{product}.tif")

    return filename


def check_VIIRS_GEOS5FP_already_processed(VIIRS_GEOS5FP_output_directory: str, target_date: Union[date, str], time_UTC: Union[datetime, str], target: str,
                                          products: List[str]):
    already_processed = True
    logger.info(f"checking if VIIRS GEOS-5 FP has previously been processed at {colored_logging.place(target)} on {colored_logging.time(target_date)}")

    for product in products:
        filename = generate_VIIRS_GEOS5FP_output_filename(VIIRS_GEOS5FP_output_directory=VIIRS_GEOS5FP_output_directory, target_date=target_date, time_UTC=time_UTC,
                                                          target=target, product=product)

        if exists(filename):
            logger.info(f"found previous VIIRS GEOS-5 FP {colored_logging.name(product)} at {colored_logging.place(target)} on {colored_logging.time(target_date)}: "
                        f"{colored_logging.file(filename)}")
        else:
            logger.info(f"did not find previous VIIRS GEOS-5 FP {colored_logging.name(product)} at {colored_logging.place(target)} on {colored_logging.time(target_date)}")
            already_processed = False

    return already_processed


def load_VIIRS_GEOS5FP(VIIRS_GEOS5FP_output_directory: str, target_date: Union[date, str], target: str, products: List[str] = None):
    logger.info(f"loading VIIRS GEOS-5 FP products for {colored_logging.place(target)} on {colored_logging.time(target_date)}")

    dataset = {}

    directory = generate_VIIRS_GEOS5FP_output_directory(VIIRS_GEOS5FP_output_directory=VIIRS_GEOS5FP_output_directory, target_date=target_date, target=target)

    pattern = join(directory, "*.tif")
    logger.info(f"searching for VIIRS GEOS-5 FP product: {colored_logging.val(pattern)}")
    filenames = glob(pattern)
    logger.info(f"found {colored_logging.val(len(filenames))} VIIRS GEOS-5 FP files")

    for filename in filenames:
        product = splitext(basename(filename))[0].split("_")[-1]

        if products is not None and product not in products:
            continue

        logger.info(f"loading VIIRS GEOS-5 FP file: {colored_logging.file(filename)}")
        image = rt.Raster.open(filename)
        dataset[product] = image

    return dataset


def VIIRS_GEOS5FP(target_date: Union[date, str], geometry: RasterGrid, target: str,
                  ST_C: rt.Raster = None, emissivity: rt.Raster = None, NDVI: rt.Raster = None, albedo: rt.Raster = None, SWin: Union[rt.Raster, str] = None,
                  Rn: Union[rt.Raster, str] = None, SM: Union[rt.Raster, str] = None, wind_speed: rt.Raster = None, Ta_C: Union[rt.Raster, str] = None,
                  RH: Union[rt.Raster, str] = None, water: rt.Raster = None, elevation_km: rt.Raster = None,
                  model: PTJPLSM = None, ET_model_name: str = ET_MODEL_NAME,
                  working_directory: str = None, static_directory: str = None,
                  VIIRS_download_directory: str = None, VIIRS_products_directory: str = None, VIIRS_shortwave_source: Union[VNP09GA, VNP43MA4] = None,
                  use_VIIRS_composite: bool = USE_VIIRS_COMPOSITE, VIIRS_composite_days: int = VIIRS_COMPOSITE_DAYS, VIIRS_GEOS5FP_output_directory: str = None,
                  SRTM_connection: SRTM = None, SRTM_download: str = None,
                  GEOS5FP_connection: GEOS5FP = None, GEOS5FP_download: str = None, GEOS5FP_products: str = None, GEOS5FP_offline_processing: bool = True,
                  GEDI_connection: GEDICanopyHeight = None, GEDI_download: str = None,
                  ORNL_connection: MODISCI = None,
                  CI_directory: str = None,
                  soil_grids_connection: SoilGrids = None, soil_grids_download: str = None,
                  intermediate_directory: str = None, preview_quality: int = DEFAULT_PREVIEW_QUALITY,
                  ANN_model: Callable = None, ANN_model_filename: str = None,
                  resampling: str = DEFAULT_RESAMPLING, coarse_cell_size: float = DEFAULT_COARSE_CELL_SIZE, downscale_air: bool = DEFAULT_DOWNSCALE_AIR,
                  downscale_humidity: bool = DEFAULT_DOWNSCALE_HUMIDITY, downscale_moisture: bool = DEFAULT_DOWNSCALE_MOISTURE,
                  floor_Topt: bool = FLOOR_TOPT,
                  save_intermediate: bool = False, include_preview: bool = True, show_distribution: bool = True, load_previous: bool = True,
                  target_variables: List[str] = DEFAULT_TARGET_VARIABLES) -> Dict[str, rt.Raster]:
    results = {}

    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()

    logger.info(f"VIIRS GEOS-5 FP target date: {colored_logging.time(target_date)}")
    time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
    logger.info(f"VIIRS GEOS-5 FP target time solar: {colored_logging.time(time_solar)}")
    time_UTC = solar_to_UTC(time_solar, geometry.centroid.latlon.x)
    logger.info(f"VIIRS GEOS-5 FP target time UTC: {colored_logging.time(time_UTC)}")

    if working_directory is None:
        working_directory = "."

    working_directory = abspath(expanduser(working_directory))

    if SRTM_connection is None:
        SRTM_connection = SRTM(working_directory=static_directory, download_directory=SRTM_download, offline_ok=True)

    if water is None:
        water = SRTM_connection.swb(geometry)

    if elevation_km is None:
        elevation_km = SRTM_connection.elevation_km(geometry)

    logger.info(f"VIIRS GEOS-5 FP working directory: {colored_logging.dir(working_directory)}")

    if VIIRS_download_directory is None:
        VIIRS_download_directory = join(working_directory, VIIRS_DOWNLOAD_DIRECTORY)

    logger.info(f"VIIRS download directory: {colored_logging.dir(VIIRS_download_directory)}")

    if VIIRS_products_directory is None:
        VIIRS_products_directory = join(working_directory, VIIRS_PRODUCTS_DIRECTORY)

    logger.info(f"VIIRS products directory: {colored_logging.dir(VIIRS_products_directory)}")

    vnp21 = VNP21A1D(working_directory=working_directory, download_directory=VIIRS_download_directory, products_directory=VIIRS_products_directory)

    if VIIRS_shortwave_source is None:
        VIIRS_shortwave_source = VNP43MA4(working_directory=working_directory, download_directory=VIIRS_download_directory, products_directory=VIIRS_products_directory)

        # VIIRS_shortwave_source = VNP09GA(
        #     working_directory=working_directory,
        #     download_directory=VIIRS_download_directory,
        #     products_directory=VIIRS_products_directory
        # )

    if VIIRS_GEOS5FP_output_directory is None:
        VIIRS_GEOS5FP_output_directory = join(working_directory, VIIRS_GEOS5FP_OUTPUT_DIRECTORY)

    logger.info(f"VIIRS GEOS-5 FP output directory: {colored_logging.dir(VIIRS_GEOS5FP_output_directory)}")

    VIIRS_GEOS5FP_already_processed = check_VIIRS_GEOS5FP_already_processed(VIIRS_GEOS5FP_output_directory=VIIRS_GEOS5FP_output_directory, target_date=target_date,
                                                                            time_UTC=time_UTC, target=target, products=target_variables)

    if VIIRS_GEOS5FP_already_processed:
        if load_previous:
            logger.info("loading previously generated VIIRS GEOS-5 FP output")
            return load_VIIRS_GEOS5FP(VIIRS_GEOS5FP_output_directory=VIIRS_GEOS5FP_output_directory, target_date=target_date, target=target)
        else:
            return

    if GEOS5FP_connection is None:
        try:
            logger.info(f"connecting to GEOS-5 FP")
            GEOS5FP_connection = GEOS5FP(working_directory=working_directory, download_directory=GEOS5FP_download, products_directory=GEOS5FP_products)
        except Exception as e:
            logger.exception(e)
            raise GEOS5FPNotAvailableError("unable to connect to GEOS-5 FP")

    if not GEOS5FP_offline_processing:
        latest_GEOS5FP_time = GEOS5FP_connection.latest_time_available
        logger.info(f"latest GEOS-5 FP time available: {latest_GEOS5FP_time}")
        logger.info(f"processing time: {time_UTC}")

        if time_UTC.strftime("%Y-%m-%d %H:%M:%S") > latest_GEOS5FP_time.strftime("%Y-%m-%d %H:%M:%S"):
            raise GEOS5FPNotAvailableError(
                f"VIIRS GEOS-5 FP target time {time_UTC} is past latest available GEOS-5 FP time {latest_GEOS5FP_time}")

    if ST_C is None:
        logger.info(
            f"retrieving {colored_logging.name('VNP21A1D')} {colored_logging.name('ST_C')} from VIIRS on {colored_logging.time(target_date)}")
        # ST_C = retrieve_VNP21NRT_ST(geometry=geometry, date_solar=target_date,
        #                             directory=VIIRS_download_directory, resampling="cubic") - 273.15
        ST_C = vnp21.ST_C(date_UTC=target_date, geometry=geometry, resampling="cubic")

        if use_VIIRS_composite:
            for days_back in range(1, VIIRS_composite_days):
                fill_date = target_date - timedelta(days_back)
                logger.info(f"gap-filling {colored_logging.name('VNP21A1D')} {colored_logging.name('ST_C')} from VIIRS on {colored_logging.time(fill_date)} for "
                            f"{colored_logging.time(target_date)}")
                ST_C_fill = vnp21.ST_C(date_UTC=target_date, geometry=geometry, resampling="cubic")
                ST_C = rt.where(np.isnan(ST_C), ST_C_fill, ST_C)

        ST_C_smooth = GEOS5FP_connection.Ts_K(time_UTC=time_UTC, geometry=geometry, resampling="cubic") - 273.15
        ST_C = rt.where(np.isnan(ST_C), ST_C_smooth, ST_C)

    results["ST"] = ST_C

    if NDVI is None:
        logger.info(f"retrieving {colored_logging.name('VNP09GA')} {colored_logging.name('NDVI')} from LANCE on {colored_logging.time(target_date)}")

        # NDVI = retrieve_VNP43IA4N(
        #     geometry=geometry,
        #     date_UTC=target_date,
        #     variable="NDVI",
        #     directory=VIIRS_download_directory,
        #     resampling="cubic"
        # )
        NDVI = VIIRS_shortwave_source.NDVI(date_UTC=target_date, geometry=geometry, resampling="cubic")

        if use_VIIRS_composite:
            for days_back in range(1, VIIRS_composite_days):
                fill_date = target_date - timedelta(days_back)
                logger.info(f"gap-filling {colored_logging.name('VNP09GA')} {colored_logging.name('NDVI')} from VIIRS on {colored_logging.time(fill_date)} for "
                            f"{colored_logging.time(target_date)}")
                NDVI_fill = VIIRS_shortwave_source.NDVI(date_UTC=target_date, geometry=geometry, resampling="cubic")
                NDVI = rt.where(np.isnan(NDVI), NDVI_fill, NDVI)

    results["NDVI"] = NDVI

    if emissivity is None:
        # logger.info(
        #     f"retrieving {colored_logging.name('VNP21_NRT')} {colored_logging.name('emissivity')} from LANCE on {colored_logging.time(target_date)}")
        # emissivity = retrieve_VNP21NRT_emissivity(geometry=geometry, date_solar=target_date,
        #                                           directory=VIIRS_download_directory, resampling="cubic")

        emissivity = 1.0094 + 0.047 * np.log(NDVI)
        emissivity = rt.where(water, 0.96, emissivity)

    results["emissivity"] = emissivity

    if albedo is None:
        logger.info(
            f"retrieving {colored_logging.name('VNP09GA')} {colored_logging.name('albedo')} from LANCE on {colored_logging.time(target_date)}")

        # albedo = retrieve_VNP43MA4N(
        #     geometry=geometry,
        #     date_UTC=target_date,
        #     variable="albedo",
        #     directory=VIIRS_download_directory,
        #     resampling="cubic"
        # )
        albedo = VIIRS_shortwave_source.albedo(date_UTC=target_date, geometry=geometry, resampling="cubic")

        if use_VIIRS_composite:
            for days_back in range(1, VIIRS_composite_days):
                fill_date = target_date - timedelta(days_back)
                logger.info(
                    f"gap-filling {colored_logging.name('VNP09GA')} {colored_logging.name('albedo')} from VIIRS on {colored_logging.time(fill_date)} for {colored_logging.time(target_date)}")
                albedo_fill = VIIRS_shortwave_source.albedo(date_UTC=target_date, geometry=geometry, resampling="cubic")
                albedo = rt.where(np.isnan(albedo), albedo_fill, albedo)

    results["albedo"] = albedo

    if model is None:
        if ET_model_name == "PTJPLSM":
            model = PTJPLSM(working_directory=working_directory, static_directory=static_directory,
                            SRTM_connection=SRTM_connection, SRTM_download=SRTM_download,
                            GEOS5FP_connection=GEOS5FP_connection, GEOS5FP_download=GEOS5FP_download, GEOS5FP_products=GEOS5FP_products,
                            GEDI_connection=GEDI_connection, GEDI_download=GEDI_download,
                            ORNL_connection=ORNL_connection,
                            CI_directory=CI_directory,
                            soil_grids_connection=soil_grids_connection, soil_grids_download=soil_grids_download,
                            intermediate_directory=intermediate_directory, preview_quality=preview_quality,
                            ANN_model=ANN_model, ANN_model_filename=ANN_model_filename,
                            resampling=resampling, downscale_air=downscale_air, downscale_humidity=downscale_humidity, downscale_moisture=downscale_moisture,
                            floor_Topt=floor_Topt, save_intermediate=save_intermediate, include_preview=include_preview, show_distribution=show_distribution)
        elif ET_model_name == "PTJPL":
            model = PTJPL(working_directory=working_directory, static_directory=static_directory,
                          SRTM_connection=SRTM_connection, SRTM_download=SRTM_download,
                          GEOS5FP_connection=GEOS5FP_connection, GEOS5FP_download=GEOS5FP_download, GEOS5FP_products=GEOS5FP_products,
                          GEDI_connection=GEDI_connection, GEDI_download=GEDI_download,
                          ORNL_connection=ORNL_connection,
                          CI_directory=CI_directory, intermediate_directory=intermediate_directory, preview_quality=preview_quality,
                          ANN_model=ANN_model, ANN_model_filename=ANN_model_filename,
                          resampling=resampling, downscale_air=downscale_air, downscale_humidity=downscale_humidity, downscale_moisture=downscale_moisture,
                          floor_Topt=floor_Topt, save_intermediate=save_intermediate, include_preview=include_preview, show_distribution=show_distribution,)
        else:
            raise ValueError(f"unrecognized model: {ET_model_name}")

    coarse_geometry = geometry.rescale(coarse_cell_size)

    if Ta_C is None:
        if downscale_air:
            ST_K = ST_C + 273.15
            Ta_K_coarse = GEOS5FP_connection.Ta_K(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic")
            Ta_K = downscale_air_temperature(time_UTC=time_UTC, Ta_K_coarse=Ta_K_coarse, ST_K=ST_K, fine_geometry=geometry, coarse_geometry=coarse_geometry)

            Ta_C = Ta_K - 273.15
        else:
            Ta_C = GEOS5FP_connection.Ta_C(time_UTC=time_UTC, geometry=geometry, resampling="cubic")

    results["Ta"] = Ta_C

    if water is None:
        water = model.SRTM_connection.swb(geometry)

    results["water"] = water

    if SM is None:
        if downscale_moisture:
            ST_K = ST_C + 273.15
            SM_coarse = GEOS5FP_connection.SFMC(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic")
            SM_smooth = GEOS5FP_connection.SFMC(time_UTC=time_UTC, geometry=geometry, resampling="cubic")

            SM = downscale_soil_moisture(time_UTC=time_UTC, fine_geometry=geometry, coarse_geometry=coarse_geometry, SM_coarse=SM_coarse, SM_resampled=SM_smooth,
                                         ST_fine=ST_K, NDVI_fine=NDVI, water=water)

        else:
            SM = GEOS5FP_connection.SFMC(time_UTC=time_UTC, geometry=geometry, resampling="cubic")

    results["SM"] = SM

    if RH is None:
        if downscale_humidity:
            ST_K = ST_C + 273.15
            VPD_Pa_coarse = GEOS5FP_connection.VPD_Pa(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic")

            VPD_Pa = downscale_vapor_pressure_deficit(time_UTC=time_UTC, VPD_Pa_coarse=VPD_Pa_coarse, ST_K=ST_K, fine_geometry=geometry, coarse_geometry=coarse_geometry)

            VPD_kPa = VPD_Pa / 1000

            RH_coarse = GEOS5FP_connection.RH(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic")

            RH = downscale_relative_humidity(time_UTC=time_UTC, RH_coarse=RH_coarse, SM=SM, ST_K=ST_K, VPD_kPa=VPD_kPa, water=water, fine_geometry=geometry,
                                             coarse_geometry=coarse_geometry)
        else:
            RH = GEOS5FP_connection.RH(time_UTC=time_UTC, geometry=geometry, resampling="cubic")

    results["RH"] = RH

    Ra = None
    Rg = None
    UV = None
    VIS = None
    NIR = None
    VISdiff = None
    NIRdiff = None
    VISdir = None
    NIRdir = None

    if SWin is None or isinstance(SWin, str):
        if SWin == "FLiES":
            logger.info("generating solar radiation using the Forest Light Environmental Simulator")
            Ra, Rg, UV, VIS, NIR, VISdiff, NIRdiff, VISdir, NIRdir = model.FLiES(geometry=geometry, target=target, time_UTC=time_UTC, albedo=albedo)

            SWin = Rg

        if SWin == "FLiES-GEOS5FP":
            logger.info("generating solar radiation using Forest Light Environmental Simulator bias-corrected with GEOS-5 FP")
            Ra, Rg, UV, VIS, NIR, VISdiff, NIRdiff, VISdir, NIRdir = model.FLiES(geometry=geometry, target=target, time_UTC=time_UTC, albedo=albedo)

            SWin_coarse = GEOS5FP_connection.SWin(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic")
            SWin = bias_correct(coarse_image=SWin_coarse, fine_image=Rg)
        elif SWin == "GEOS5FP" or SWin is None:
            logger.info("generating solar radiation using GEOS-5 FP")
            SWin = GEOS5FP_connection.SWin(time_UTC=time_UTC, geometry=geometry, resampling="cubic")

    if Rn is None or isinstance(Rn, str):
        if Rn == "BESS":
            logger.info(f"generating net radiation using Breathing Earth System Simulator for {colored_logging.place(target)} at {colored_logging.time(time_UTC)} UTC")

            ST_K = ST_C + 273.15
            Ta_K = Ta_C + 273.15

            BESS_results = model.BESS(geometry=geometry, target=target, time_UTC=time_UTC, ST_K=ST_K, Ta_K=Ta_K, RH=RH, elevation_km=elevation_km, NDVI=NDVI,
                                      albedo=albedo, Rg=SWin, VISdiff=VISdiff, VISdir=VISdir, NIRdiff=NIRdiff, NIRdir=NIRdir, UV=UV, water=water,
                                      output_variables=["Rn", "LE", "GPP"])

            Rn = BESS_results["Rn"]
        if Rn == "Verma":
            Rn = None

    if ET_model_name == "PTJPLSM":
        logger.info(f"running PT-JPL-SM ET model at {colored_logging.time(time_UTC)}")

        PTJPL_results = model.PTJPL(geometry=geometry, target=target, time_UTC=time_UTC, ST_C=ST_C, emissivity=emissivity, NDVI=NDVI, albedo=albedo, SWin=SWin, SM=SM,
                                    wind_speed=wind_speed, Ta_C=Ta_C, RH=RH, Rn=Rn, water=water, output_variables=target_variables,)
    elif ET_model_name == "PTJPL":
        logger.info(f"running PT-JPL ET model at {colored_logging.time(time_UTC)}")

        PTJPL_results = model.PTJPL(geometry=geometry, target=target, time_UTC=time_UTC, ST_C=ST_C, emissivity=emissivity, NDVI=NDVI, albedo=albedo, SWin=SWin,
                                    wind_speed=wind_speed, Ta_C=Ta_C, RH=RH, Rn=Rn, water=water, output_variables=target_variables)
    else:
        raise ValueError(f"unrecognized model: {ET_model_name}")

    for k, v in PTJPL_results.items():
        results[k] = v

    for product, image in results.items():
        filename = generate_VIIRS_GEOS5FP_output_filename(VIIRS_GEOS5FP_output_directory=VIIRS_GEOS5FP_output_directory, target_date=target_date, time_UTC=time_UTC,
                                                          target=target, product=product)

        if image is None:
            logger.warning(f"no image result for {product}")
            continue

        logger.info(
            f"writing VIIRS GEOS-5 FP {colored_logging.name(product)} at {colored_logging.place(target)} at {colored_logging.time(time_UTC)} to file: {colored_logging.file(filename)}")
        image.to_geotiff(filename)

    return results
