from glob import glob
from os.path import splitext
from typing import Dict, Callable
import boto3
import rasters
from GEDI import GEDICanopyHeight
from geos5fp import GEOS5FP
from LANCE import *
from LANCE import LANCENotAvailableError
from modisci import MODISCI
from PTJPL import PTJPL
from PTJPLSM import PTJPLSM
from SRTM import SRTM
from soil_grids import SoilGrids
from geos5fp.downscaling import downscale_air_temperature, downscale_soil_moisture, downscale_vapor_pressure_deficit, \
    downscale_relative_humidity, bias_correct

from LANCE import ARCHIVE

ET_MODEL_NAME = "PTJPL"
DEFAULT_LANCE_DOWNLOAD_DIRECTORY = "LANCE_download_directory"
DEFAULT_LANCE_OUTPUT_DIRECTORY = "LANCE_output"
DEFAULT_RESAMPLING = "cubic"
DEFAULT_PREVIEW_QUALITY = 20
DEFAULT_DOWNSCALE_AIR = False
DEFAULT_DOWNSCALE_HUMIDITY = False
DEFAULT_DOWNSCALE_MOISTURE = False
DEFAULT_COARSE_CELL_SIZE = 27375
DEFAULT_TARGET_VARIABLES = ["LE", "ET", "ESI"]
FLOOR_TOPT = True

logger = logging.getLogger(__name__)


class GEOS5FPNotAvailableError(Exception):
    pass


def generate_LANCE_output_directory(
        LANCE_output_directory: str,
        target_date: Union[date, str],
        target: str):
    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()

    directory = join(
        abspath(expanduser(LANCE_output_directory)),
        f"{target_date:%Y-%m-%d}",
        f"LANCE_{target_date:%Y-%m-%d}_{target}",
    )

    return directory


def generate_LANCE_output_filename(
        LANCE_output_directory: str,
        target_date: Union[date, str],
        time_UTC: Union[datetime, str],
        target: str,
        product: str):
    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()

    if isinstance(time_UTC, str):
        time_UTC = parser.parse(time_UTC)

    directory = generate_LANCE_output_directory(
        LANCE_output_directory=LANCE_output_directory,
        target_date=target_date,
        target=target
    )

    filename = join(directory, f"LANCE_{time_UTC:%Y.%m.%d.%H.%M.%S}_{target}_{product}.tif")

    return filename


def check_LANCE_already_processed(
        LANCE_output_directory: str,
        target_date: Union[date, str],
        time_UTC: Union[datetime, str],
        target: str,
        products: List[str]):
    already_processed = True
    logger.info(
        f"checking if LANCE GEOS-5 FP has previously been processed at {cl.place(target)} on {cl.time(target_date)}")

    for product in products:
        filename = generate_LANCE_output_filename(
            LANCE_output_directory=LANCE_output_directory,
            target_date=target_date,
            time_UTC=time_UTC,
            target=target,
            product=product
        )

        if exists(filename):
            logger.info(
                f"found previous LANCE GEOS-5 FP {cl.name(product)} at {cl.place(target)} on {cl.time(target_date)}: {cl.file(filename)}")
        else:
            logger.info(
                f"did not find previous LANCE GEOS-5 FP {cl.name(product)} at {cl.place(target)} on {cl.time(target_date)}")
            already_processed = False

    return already_processed


def load_LANCE(LANCE_output_directory: str, target_date: Union[date, str], target: str, products: List[str] = None):
    logger.info(f"loading LANCE GEOS-5 FP products for {cl.place(target)} on {cl.time(target_date)}")

    dataset = {}

    directory = generate_LANCE_output_directory(
        LANCE_output_directory=LANCE_output_directory,
        target_date=target_date,
        target=target
    )

    pattern = join(directory, "*.tif")
    logger.info(f"searching for LANCE product: {cl.val(pattern)}")
    filenames = glob(pattern)
    logger.info(f"found {cl.val(len(filenames))} LANCE files")

    for filename in filenames:
        product = splitext(basename(filename))[0].split("_")[-1]

        if products is not None and product not in products:
            continue

        logger.info(f"loading LANCE GEOS-5 FP file: {cl.file(filename)}")
        image = rt.Raster.open(filename)
        dataset[product] = image

    return dataset


def LANCE_GEOS5FP_NRT(
        target_date: Union[date, str],
        geometry: RasterGrid,
        target: str,
        ST_C: rt.Raster = None,
        emissivity: rt.Raster = None,
        NDVI: rt.Raster = None,
        albedo: rt.Raster = None,
        SWin: Union[rt.Raster, str] = None,
        Rn: Union[rt.Raster, str] = None,
        SM: rt.Raster = None,
        wind_speed: rt.Raster = None,
        Ta_C: rt.Raster = None,
        RH: rt.Raster = None,
        water: rt.Raster = None,
        elevation_km: rt.Raster = None,
        model: PTJPLSM = None,
        model_name: str = ET_MODEL_NAME,
        working_directory: str = None,
        static_directory: str = None,
        LANCE_download_directory: str = None,
        LANCE_output_directory: str = None,
        output_bucket_name: str = None,
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
        spacetrack_credentials_filename: str = None,
        preview_quality: int = DEFAULT_PREVIEW_QUALITY,
        ANN_model: Callable = None,
        ANN_model_filename: str = None,
        resampling: str = DEFAULT_RESAMPLING,
        coarse_cell_size: float = DEFAULT_COARSE_CELL_SIZE,
        downscale_air: bool = DEFAULT_DOWNSCALE_AIR,
        downscale_humidity: bool = DEFAULT_DOWNSCALE_HUMIDITY,
        downscale_moisture: bool = DEFAULT_DOWNSCALE_MOISTURE,
        floor_Topt: bool = FLOOR_TOPT,
        save_intermediate: bool = False,
        include_preview: bool = True,
        show_distribution: bool = True,
        load_previous: bool = True,
        target_variables: List[str] = DEFAULT_TARGET_VARIABLES) -> Dict[str, rt.Raster]:
    results = {}

    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()

    logger.info(f"LANCE target date: {cl.time(target_date)}")
    time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
    logger.info(f"LANCE target time solar: {cl.time(time_solar)}")
    time_UTC = solar_to_UTC(time_solar, geometry.centroid.latlon.x)
    logger.info(f"LANCE target time UTC: {cl.time(time_UTC)}")

    if working_directory is None:
        working_directory = "."

    working_directory = abspath(expanduser(working_directory))

    if SRTM_connection is None:
        SRTM_connection = SRTM(
            working_directory=static_directory,
            download_directory=SRTM_download,
            offline_ok=True
        )

    if water is None:
        water = SRTM_connection.swb(geometry)

    if elevation_km is None:
        elevation_km = SRTM_connection.elevation_km(geometry)

    logger.info(f"LANCE working directory: {cl.dir(working_directory)}")

    if LANCE_download_directory is None:
        LANCE_download_directory = join(working_directory, DEFAULT_LANCE_DOWNLOAD_DIRECTORY)

    logger.info(f"LANCE download directory: {cl.dir(LANCE_download_directory)}")

    if LANCE_output_directory is None:
        LANCE_output_directory = join(working_directory, DEFAULT_LANCE_OUTPUT_DIRECTORY)

    logger.info(f"LANCE output directory: {cl.dir(LANCE_output_directory)}")

    if output_bucket_name is not None:
        logger.info(f"output S3 bucket: {output_bucket_name}")
        session = boto3.Session()
        s3 = session.resource("s3")
        output_bucket = s3.Bucket(output_bucket_name)

    LANCE_already_processed = check_LANCE_already_processed(
        LANCE_output_directory=LANCE_output_directory,
        target_date=target_date,
        time_UTC=time_UTC,
        target=target,
        products=target_variables
    )

    if LANCE_already_processed:
        if load_previous:
            logger.info("loading previously generated LANCE GEOS-5 FP output")
            return load_LANCE(
                LANCE_output_directory=LANCE_output_directory,
                target_date=target_date,
                target=target
            )
        else:
            return

    LANCE_dates = available_LANCE_dates("VNP43MA4N", archive=ARCHIVE)
    earliest_LANCE_date = LANCE_dates[0]
    latest_LANCE_date = LANCE_dates[-1]
    logger.info(f"LANCE is available from {cl.time(earliest_LANCE_date)} to {cl.time(latest_LANCE_date)}")

    if target_date < earliest_LANCE_date:
        raise LANCENotAvailableError(
            f"target date {target_date} is before earliest available LANCE {earliest_LANCE_date}")

    if GEOS5FP_connection is None:
        try:
            logger.info(f"connecting to GEOS-5 FP")
            GEOS5FP_connection = GEOS5FP(
                working_directory=working_directory,
                download_directory=GEOS5FP_download,
                products_directory=GEOS5FP_products
            )
        except Exception as e:
            logger.exception(e)
            raise GEOS5FPNotAvailableError("unable to connect to GEOS-5 FP")

    # GEOS5FP_connection = GEOS5FP_connection
    latest_GEOS5FP_time = GEOS5FP_connection.latest_time_available
    logger.info(f"latest GEOS-5 FP time available: {latest_GEOS5FP_time}")
    logger.info(f"processing time: {time_UTC}")

    if time_UTC.strftime("%Y-%m-%d %H:%M:%S") > latest_GEOS5FP_time.strftime("%Y-%m-%d %H:%M:%S"):
        raise GEOS5FPNotAvailableError(
            f"LANCE target time {time_UTC} is past latest available GEOS-5 FP time {latest_GEOS5FP_time}")

    LANCE_processing_date = target_date
    LANCE_processing_time = time_UTC

    if ST_C is None:
        logger.info(
            f"retrieving {cl.name('VNP21_NRT')} {cl.name('ST_C')} from LANCE on {cl.time(LANCE_processing_date)}")

        ST_K = retrieve_VNP21NRT_ST(
            geometry=geometry,
            date_solar=LANCE_processing_date,
            directory=LANCE_download_directory,
            resampling="cubic",
            spacetrack_credentials_filename=spacetrack_credentials_filename
        )

        ST_C = ST_K - 273.15
        ST_C_smooth = GEOS5FP_connection.Ts_K(time_UTC=time_UTC, geometry=geometry, resampling="cubic") - 273.15
        ST_C = rt.where(np.isnan(ST_C), ST_C_smooth, ST_C)

    results["ST"] = ST_C

    if NDVI is None:
        logger.info(
            f"retrieving {cl.name('VNP43IA4N')} {cl.name('NDVI')} from LANCE on {cl.time(LANCE_processing_date)}")

        NDVI = retrieve_VNP43IA4N(
            geometry=geometry,
            date_UTC=LANCE_processing_date,
            variable="NDVI",
            directory=LANCE_download_directory,
            resampling="cubic"
        )

    results["NDVI"] = NDVI

    if emissivity is None:
        logger.info(
            f"retrieving {cl.name('VNP21_NRT')} {cl.name('emissivity')} from LANCE on {cl.time(LANCE_processing_date)}")
        emissivity = retrieve_VNP21NRT_emissivity(geometry=geometry, date_solar=LANCE_processing_date,
                                                  directory=LANCE_download_directory, resampling="cubic")

    emissivity = rt.where(water, 0.96, emissivity)
    emissivity = rt.where(np.isnan(emissivity), 1.0094 + 0.047 * np.log(NDVI), emissivity)

    results["emissivity"] = emissivity

    if albedo is None:
        logger.info(
            f"retrieving {cl.name('VNP43MA4N')} {cl.name('albedo')} from LANCE on {cl.time(LANCE_processing_date)}")

        albedo = retrieve_VNP43MA4N(
            geometry=geometry,
            date_UTC=LANCE_processing_date,
            variable="albedo",
            directory=LANCE_download_directory,
            resampling="cubic"
        )

    results["albedo"] = albedo

    if model is None:
        if model_name == "PTJPLSM":
            model = PTJPLSM(
                working_directory=working_directory,
                static_directory=static_directory,
                SRTM_connection=SRTM_connection,
                SRTM_download=SRTM_download,
                GEOS5FP_connection=GEOS5FP_connection,
                GEOS5FP_download=GEOS5FP_download,
                GEOS5FP_products=GEOS5FP_products,
                GEDI_connection=GEDI_connection,
                GEDI_download=GEDI_download,
                ORNL_connection=ORNL_connection,
                CI_directory=CI_directory,
                soil_grids_connection=soil_grids_connection,
                soil_grids_download=soil_grids_download,
                intermediate_directory=intermediate_directory,
                preview_quality=preview_quality,
                ANN_model=ANN_model,
                ANN_model_filename=ANN_model_filename,
                resampling=resampling,
                downscale_air=downscale_air,
                downscale_humidity=downscale_humidity,
                downscale_moisture=downscale_moisture,
                floor_Topt=floor_Topt,
                save_intermediate=save_intermediate,
                include_preview=include_preview,
                show_distribution=show_distribution
            )
        elif model_name == "PTJPL":
            model = PTJPL(
                working_directory=working_directory,
                static_directory=static_directory,
                SRTM_connection=SRTM_connection,
                SRTM_download=SRTM_download,
                GEOS5FP_connection=GEOS5FP_connection,
                GEOS5FP_download=GEOS5FP_download,
                GEOS5FP_products=GEOS5FP_products,
                GEDI_connection=GEDI_connection,
                GEDI_download=GEDI_download,
                ORNL_connection=ORNL_connection,
                CI_directory=CI_directory,
                intermediate_directory=intermediate_directory,
                preview_quality=preview_quality,
                ANN_model=ANN_model,
                ANN_model_filename=ANN_model_filename,
                resampling=resampling,
                downscale_air=downscale_air,
                downscale_humidity=downscale_humidity,
                downscale_moisture=downscale_moisture,
                floor_Topt=floor_Topt,
                save_intermediate=save_intermediate,
                include_preview=include_preview,
                show_distribution=show_distribution
            )
        else:
            raise ValueError(f"unrecognized model: {model_name}")

    coarse_geometry = geometry.rescale(coarse_cell_size)

    if Ta_C is None:
        if downscale_air:
            ST_K = ST_C + 273.15
            Ta_K_coarse = GEOS5FP_connection.Ta_K(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic")
            Ta_K = downscale_air_temperature(
                time_UTC=time_UTC,
                Ta_K_coarse=Ta_K_coarse,
                ST_K=ST_K,
                fine_geometry=geometry,
                coarse_geometry=coarse_geometry
            )

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
            SM = GEOS5FP_connection.SFMC(time_UTC=time_UTC, geometry=geometry, resampling="cubic")

    results["SM"] = SM

    if RH is None:
        if downscale_humidity:
            ST_K = ST_C + 273.15
            VPD_Pa_coarse = GEOS5FP_connection.VPD_Pa(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic")

            VPD_Pa = downscale_vapor_pressure_deficit(
                time_UTC=time_UTC,
                VPD_Pa_coarse=VPD_Pa_coarse,
                ST_K=ST_K,
                fine_geometry=geometry,
                coarse_geometry=coarse_geometry
            )

            VPD_kPa = VPD_Pa / 1000

            RH_coarse = GEOS5FP_connection.RH(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic")

            RH = downscale_relative_humidity(
                time_UTC=time_UTC,
                RH_coarse=RH_coarse,
                SM=SM,
                ST_K=ST_K,
                VPD_kPa=VPD_kPa,
                water=water,
                fine_geometry=geometry,
                coarse_geometry=coarse_geometry
            )
        else:
            RH = GEOS5FP_connection.RH(time_UTC=time_UTC, geometry=geometry, resampling="cubic")

    results["RH"] = RH

    # if SWin is None:
    #     Ra, Rg, UV, VIS, NIR, VISdiff, NIRdiff, VISdir, NIRdir = model.FLiES(
    #         geometry=geometry,
    #         target=target,
    #         time_UTC=time_UTC,
    #         albedo=albedo
    #     )
    #
    #     SWin_coarse = GEOS5FP_connection.SWin(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic")
    #     SWin = bias_correct(
    #         coarse_image=SWin_coarse,
    #         fine_image=Rg
    #     )

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
            Ra, Rg, UV, VIS, NIR, VISdiff, NIRdiff, VISdir, NIRdir = model.FLiES(
                geometry=geometry,
                target=target,
                time_UTC=time_UTC,
                albedo=albedo
            )

            SWin = Rg

        if SWin == "FLiES-GEOS5FP":
            logger.info(
                "generating solar radiation using Forest Light Environmental Simulator bias-corrected with GEOS-5 FP")
            Ra, Rg, UV, VIS, NIR, VISdiff, NIRdiff, VISdir, NIRdir = model.FLiES(
                geometry=geometry,
                target=target,
                time_UTC=time_UTC,
                albedo=albedo
            )

            SWin_coarse = GEOS5FP_connection.SWin(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic")
            SWin = bias_correct(
                coarse_image=SWin_coarse,
                fine_image=Rg
            )
        elif SWin == "GEOS5FP" or SWin is None:
            logger.info("generating solar radiation using GEOS-5 FP")
            SWin = GEOS5FP_connection.SWin(time_UTC=time_UTC, geometry=geometry, resampling="cubic")

    if Rn is None or isinstance(Rn, str):
        if Rn == "BESS":
            logger.info(
                f"generating net radiation using Breathing Earth System Simulator for {cl.place(target)} at {cl.time(time_UTC)} UTC")

            ST_K = ST_C + 273.15
            Ta_K = Ta_C + 273.15

            BESS_results = model.BESS(
                geometry=geometry,
                target=target,
                time_UTC=time_UTC,
                ST_K=ST_K,
                Ta_K=Ta_K,
                RH=RH,
                elevation_km=elevation_km,
                NDVI=NDVI,
                albedo=albedo,
                Rg=SWin,
                VISdiff=VISdiff,
                VISdir=VISdir,
                NIRdiff=NIRdiff,
                NIRdir=NIRdir,
                UV=UV,
                water=water,
                output_variables=["Rn", "LE", "GPP"]
            )

            Rn = BESS_results["Rn"]
        if Rn == "Verma":
            Rn = None

    logger.info(f"running PT-JPL-SM ET model hindcast at {cl.time(time_UTC)}")

    if model_name == "PTJPLSM":
        PTJPL_results = model.PTJPL(
            geometry=geometry,
            target=target,
            time_UTC=time_UTC,
            ST_C=ST_C,
            emissivity=emissivity,
            NDVI=NDVI,
            albedo=albedo,
            SWin=SWin,
            SM=SM,
            wind_speed=wind_speed,
            Ta_C=Ta_C,
            RH=RH,
            Rn=Rn,
            water=water,
            output_variables=target_variables,
        )
    elif model_name == "PTJPL":
        PTJPL_results = model.PTJPL(
            geometry=geometry,
            target=target,
            time_UTC=time_UTC,
            ST_C=ST_C,
            emissivity=emissivity,
            NDVI=NDVI,
            albedo=albedo,
            SWin=SWin,
            wind_speed=wind_speed,
            Ta_C=Ta_C,
            RH=RH,
            Rn=Rn,
            water=water,
            output_variables=target_variables
        )
    else:
        raise ValueError(f"unrecognized model: {model_name}")

    for k, v in PTJPL_results.items():
        results[k] = v

    for product, image in results.items():
        filename = generate_LANCE_output_filename(
            LANCE_output_directory=LANCE_output_directory,
            target_date=target_date,
            time_UTC=time_UTC,
            target=target,
            product=product
        )

        if image is None:
            logger.warning(f"no image result for {product}")
            continue

        logger.info(
            f"writing LANCE GEOS-5 FP {cl.name(product)} at {cl.place(target)} at {cl.time(time_UTC)} to file: {cl.file(filename)}")
        image.to_geotiff(filename)

        ## TODO upload to S3 bucket
        if output_bucket_name is not None:
            filename_base = basename(filename)
            logger.info(f"uploading {filename} to bucket {output_bucket_name}")
            output_bucket.upload_file(filename, filename_base)

    return results
