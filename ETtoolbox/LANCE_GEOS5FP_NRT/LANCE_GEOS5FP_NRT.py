from glob import glob
from os.path import splitext
from typing import Dict, Callable
import boto3
from rasters import RasterGrid

from gedi_canopy_height import GEDICanopyHeight
from geos5fp import GEOS5FP
from ETtoolbox.LANCE import *
from ETtoolbox.LANCE import LANCENotAvailableError
from modisci import MODISCI
from ETtoolbox.PTJPL import PTJPL
from ETtoolbox.PTJPLSM import PTJPLSM
from ETtoolbox.SRTM import SRTM
from soil_capacity_wilting import SoilGrids
from geos5fp.downscaling import downscale_air_temperature, downscale_soil_moisture, downscale_vapor_pressure_deficit, \
    downscale_relative_humidity, bias_correct

from ETtoolbox.LANCE import ARCHIVE

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


def generate_LANCE_output_directory(LANCE_output_directory: str, target_date: Union[date, str], target: str):
    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()

    directory = join(abspath(expanduser(LANCE_output_directory)), f"{target_date:%Y-%m-%d}", f"LANCE_{target_date:%Y-%m-%d}_{target}")

    return directory


def generate_LANCE_output_filename(LANCE_output_directory: str, target_date: Union[date, str], time_UTC: Union[datetime, str], target: str, product: str):
    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()

    if isinstance(time_UTC, str):
        time_UTC = parser.parse(time_UTC)

    directory = generate_LANCE_output_directory(LANCE_output_directory=LANCE_output_directory, target_date=target_date, target=target)

    filename = join(directory, f"LANCE_{time_UTC:%Y.%m.%d.%H.%M.%S}_{target}_{product}.tif")

    return filename


def check_LANCE_already_processed(LANCE_output_directory: str, target_date: Union[date, str], time_UTC: Union[datetime, str], target: str, products: List[str]):
    already_processed = True
    logger.info(
        f"checking if LANCE GEOS-5 FP has previously been processed at {colored_logging.place(target)} on {colored_logging.time(target_date)}")

    for product in products:
        filename = generate_LANCE_output_filename( LANCE_output_directory=LANCE_output_directory, target_date=target_date, time_UTC=time_UTC, target=target, product=product)

        if exists(filename):
            logger.info(f"found previous LANCE GEOS-5 FP {colored_logging.name(product)} at {colored_logging.place(target)} on {colored_logging.time(target_date)}: "
                        f"{colored_logging.file(filename)}")
        else:
            logger.info(f"did not find previous LANCE GEOS-5 FP {colored_logging.name(product)} at {colored_logging.place(target)} on {colored_logging.time(target_date)}")
            already_processed = False

    return already_processed


def load_LANCE(LANCE_output_directory: str, target_date: Union[date, str], target: str, products: List[str] = None):
    logger.info(f"loading LANCE GEOS-5 FP products for {colored_logging.place(target)} on {colored_logging.time(target_date)}")

    dataset = {}

    directory = generate_LANCE_output_directory(LANCE_output_directory=LANCE_output_directory, target_date=target_date, target=target)

    pattern = join(directory, "*.tif")
    logger.info(f"searching for LANCE product: {colored_logging.val(pattern)}")
    filenames = glob(pattern)
    logger.info(f"found {colored_logging.val(len(filenames))} LANCE files")

    for filename in filenames:
        product = splitext(basename(filename))[0].split("_")[-1]

        if products is not None and product not in products:
            continue

        logger.info(f"loading LANCE GEOS-5 FP file: {colored_logging.file(filename)}")
        image = rt.Raster.open(filename)
        dataset[product] = image

    return dataset


def LANCE_GEOS5FP_NRT(
        target_date: Union[date, str],
        o_geometry: RasterGrid,
        target: str,
        o_st_band_celsius: rt.Raster = None,
        o_emissivity: rt.Raster = None,
        o_ndvi: rt.Raster = None,
        o_albedo: rt.Raster = None,
        SWin: Union[rt.Raster, str] = None,
        Rn: Union[rt.Raster, str] = None,
        SM: rt.Raster = None,
        wind_speed: rt.Raster = None,
        o_air_temperature_celsius: rt.Raster = None,
        RH: rt.Raster = None,
        o_water: rt.Raster = None,
        elevation_km: rt.Raster = None,
        o_model: PTJPLSM = None,
        s_model_name: str = ET_MODEL_NAME,
        working_directory: str = None,
        static_directory: str = None,
        s_lance_download_directory: str = None,
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
    c_results = {}

    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()

    logger.info(f"LANCE target date: {colored_logging.time(target_date)}")
    time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
    logger.info(f"LANCE target time solar: {colored_logging.time(time_solar)}")
    time_UTC = solar_to_UTC(time_solar, o_geometry.centroid.latlon.x)
    logger.info(f"LANCE target time UTC: {colored_logging.time(time_UTC)}")

    if working_directory is None:
        working_directory = "."

    working_directory = abspath(expanduser(working_directory))

    if SRTM_connection is None:
        SRTM_connection = SRTM(working_directory=static_directory, download_directory=SRTM_download, offline_ok=True)

    if o_water is None:
        o_water = SRTM_connection.swb(o_geometry)

    if elevation_km is None:
        elevation_km = SRTM_connection.elevation_km(o_geometry)

    logger.info(f"LANCE working directory: {colored_logging.dir(working_directory)}")

    if s_lance_download_directory is None:
        s_lance_download_directory = join(working_directory, DEFAULT_LANCE_DOWNLOAD_DIRECTORY)

    logger.info(f"LANCE download directory: {colored_logging.dir(s_lance_download_directory)}")

    if LANCE_output_directory is None:
        LANCE_output_directory = join(working_directory, DEFAULT_LANCE_OUTPUT_DIRECTORY)

    logger.info(f"LANCE output directory: {colored_logging.dir(LANCE_output_directory)}")

    if output_bucket_name is not None:
        logger.info(f"output S3 bucket: {output_bucket_name}")
        session = boto3.Session()
        s3 = session.resource("s3")
        output_bucket = s3.Bucket(output_bucket_name)

    LANCE_already_processed = check_LANCE_already_processed(LANCE_output_directory=LANCE_output_directory, target_date=target_date, time_UTC=time_UTC,
                                                            target=target, products=target_variables)

    if LANCE_already_processed:
        if load_previous:
            logger.info("loading previously generated LANCE GEOS-5 FP output")
            return load_LANCE(LANCE_output_directory=LANCE_output_directory, target_date=target_date, target=target)
        else:
            return

    LANCE_dates = available_LANCE_dates("VNP43MA4N", archive=ARCHIVE)
    earliest_LANCE_date = LANCE_dates[0]
    latest_LANCE_date = LANCE_dates[-1]
    logger.info(f"LANCE is available from {colored_logging.time(earliest_LANCE_date)} to {colored_logging.time(latest_LANCE_date)}")

    if target_date < earliest_LANCE_date:
        raise LANCENotAvailableError(f"target date {target_date} is before earliest available LANCE {earliest_LANCE_date}")

    if GEOS5FP_connection is None:
        try:
            logger.info(f"connecting to GEOS-5 FP")
            GEOS5FP_connection = GEOS5FP(working_directory=working_directory, download_directory=GEOS5FP_download, products_directory=GEOS5FP_products)
        except Exception as e:
            logger.exception(e)
            raise GEOS5FPNotAvailableError("unable to connect to GEOS-5 FP")

    # GEOS5FP_connection = GEOS5FP_connection
    latest_GEOS5FP_time = GEOS5FP_connection.latest_time_available
    logger.info(f"latest GEOS-5 FP time available: {latest_GEOS5FP_time}")
    logger.info(f"processing time: {time_UTC}")

    if time_UTC.strftime("%Y-%m-%d %H:%M:%S") > latest_GEOS5FP_time.strftime("%Y-%m-%d %H:%M:%S"):
        raise GEOS5FPNotAvailableError(f"LANCE target time {time_UTC} is past latest available GEOS-5 FP time {latest_GEOS5FP_time}")

    o_lance_processing_date = target_date
    LANCE_processing_time = time_UTC

    ### Process ST_C ###
    # Request the ST_C variable if not provided
    if o_st_band_celsius is None:
        # Log the attempt
        logger.info(f"retrieving {colored_logging.name('VNP21_NRT')} {colored_logging.name('ST_C')} from LANCE on {colored_logging.time(o_lance_processing_date)}")

        # Retrieve the data from the server
        o_st_band_kelvin, o_valid_date_utc = retrieve_vnp21nrt_st(o_geometry=o_geometry, o_date_solar=o_lance_processing_date, s_directory=s_lance_download_directory,
                                                                  s_resampling="cubic", s_spacetrack_credentials_filename=spacetrack_credentials_filename)

        # If the retrieval is successful, perform the conversion
        o_st_band_celsius = None
        if o_st_band_kelvin is not None:
            o_st_band_celsius = o_st_band_kelvin - 273.15
            o_st_band_celsius_smooth = GEOS5FP_connection.Ts_K(time_UTC=time_UTC, geometry=o_geometry, resampling="cubic") - 273.15
            o_st_band_celsius = rt.where(np.isnan(o_st_band_celsius), o_st_band_celsius_smooth, o_st_band_celsius)

    # Store the value into the results dictionary
    c_results["ST"] = o_st_band_celsius

    ### Proces NDVI values ###
    # Request NDVI if not provided
    if o_ndvi is None:
        # Log the attempt
        logger.info(f"retrieving {colored_logging.name('VNP43IA4N')} {colored_logging.name('NDVI')} from LANCE on {colored_logging.time(o_lance_processing_date)}")

        # Make the request
        o_ndvi = retrieve_vnp43ia4n(o_geometry=o_geometry, o_date_utc=o_lance_processing_date, s_variable="NDVI", s_directory=s_lance_download_directory, s_resampling="cubic")

    # Store the value into the dictionary
    c_results["NDVI"] = o_ndvi

    ### Process emissivity data ###
    # Request the data if not available
    if o_emissivity is None:
        # Log the attempt
        logger.info(f"retrieving {colored_logging.name('VNP21_NRT')} {colored_logging.name('emissivity')} from LANCE on {colored_logging.time(o_lance_processing_date)}")

        # Make the request
        o_emissivity, o_valid_date = retrieve_vnp21nrt_emissivity(o_geometry=o_geometry, o_date_solar=o_lance_processing_date, s_directory=s_lance_download_directory,
                                                                  s_resampling="cubic")

    # If data exists, mask the values to keep only feasible values
    if o_emissivity is not None:
        o_emissivity = rt.where(o_water, 0.96, o_emissivity)
        o_emissivity = rt.where(np.isnan(o_emissivity), 1.0094 + 0.047 * np.log(o_ndvi), o_emissivity)

    # Store the value into the results dictionary
    c_results["emissivity"] = o_emissivity

    ### Process the albedo data ##
    # Request albedo if not provided
    if o_albedo is None:
        # Log the attempt
        logger.info(f"retrieving {colored_logging.name('VNP43MA4N')} {colored_logging.name('albedo')} from LANCE on {colored_logging.time(o_lance_processing_date)}")

        # Make the request
        o_albedo = retrieve_vnp43ma4n(o_geometry=o_geometry, o_date_utc=o_lance_processing_date, s_variable="albedo", s_directory=s_lance_download_directory, s_resampling="cubic")

    # Store the value in the results dictionary
    c_results["albedo"] = o_albedo

    ### Run the model ###
    # Run the model only if the model is provided and the data is available to do so
    b_valid_data = all([x != None for x in list(c_results.values())])
    if o_model is None and b_valid_data:
        # Determine which model to run
        if s_model_name == "PTJPLSM":
            # Run the PTJPLSM model
            o_model = PTJPLSM(working_directory=working_directory, static_directory=static_directory, SRTM_connection=SRTM_connection, SRTM_download=SRTM_download,
                              GEOS5FP_connection=GEOS5FP_connection, GEOS5FP_download=GEOS5FP_download, GEOS5FP_products=GEOS5FP_products,
                              GEDI_connection=GEDI_connection, GEDI_download=GEDI_download,
                              ORNL_connection=ORNL_connection,
                              CI_directory=CI_directory,
                              soil_grids_connection=soil_grids_connection, soil_grids_download=soil_grids_download,
                              intermediate_directory=intermediate_directory, preview_quality=preview_quality,
                              ANN_model=ANN_model, ANN_model_filename=ANN_model_filename,
                              resampling=resampling, downscale_air=downscale_air, downscale_humidity=downscale_humidity, downscale_moisture=downscale_moisture,
                              floor_Topt=floor_Topt, save_intermediate=save_intermediate, include_preview=include_preview, show_distribution=show_distribution)

        elif s_model_name == "PTJPL":
            # RUN the PTJPL model
            o_model = PTJPL(working_directory=working_directory, static_directory=static_directory, SRTM_connection=SRTM_connection, SRTM_download=SRTM_download,
                            GEOS5FP_connection=GEOS5FP_connection, GEOS5FP_download=GEOS5FP_download, GEOS5FP_products=GEOS5FP_products,
                            GEDI_connection=GEDI_connection, GEDI_download=GEDI_download,
                            ORNL_connection=ORNL_connection,
                            CI_directory=CI_directory, intermediate_directory=intermediate_directory, preview_quality=preview_quality,
                            ANN_model=ANN_model, ANN_model_filename=ANN_model_filename,
                            resampling=resampling, downscale_air=downscale_air, downscale_humidity=downscale_humidity, downscale_moisture=downscale_moisture,
                            floor_Topt=floor_Topt, save_intermediate=save_intermediate, include_preview=include_preview, show_distribution=show_distribution)

        else:
            # Model is not understood. Take actions to handle
            logger.error(f"unable to run LANCE GEOS5FP NRT model due to missing model  {s_model_name}")

            # Set the model to None
            o_model = None

    else:
        # Model is not provided or data is missing.
        # Log the issue
        logger.error(f"unable to run LANCE GEOS5FP NRT model due to missing model or missing data")

        # Set the model to indicate a run failure
        o_model = None

    ### Rescale geometry ###
    # Coarsen the geometry for subsequent calcualtions
    o_coarse_geometry = o_geometry.rescale(coarse_cell_size)

    ### Process air temperature ###
    # Request air temperature if not provided
    if o_air_temperature_celsius is None:
        # Split the workflow based on whether resampled or full resolution is requested
        if downscale_air:
            # Resampled data is requested. Reformat and resample
            # Convert back to kelvin from celsius
            o_st_band_kelvin = o_st_band_celsius + 273.15

            # Create the connection and downscale
            o_air_temperature_kelvin_coarse = GEOS5FP_connection.Ta_K(time_UTC=time_UTC, geometry=o_coarse_geometry, resampling="cubic")
            o_air_temperature_kelvin = downscale_air_temperature(time_UTC=time_UTC, Ta_K_coarse=o_air_temperature_kelvin_coarse, ST_K=o_st_band_kelvin,
                                                                 fine_geometry=o_geometry, coarse_geometry=o_coarse_geometry)

            # Convert back to celsius
            o_air_temperature_celsius = o_air_temperature_kelvin - 273.15

        else:
            # Resampled data is not requested. Request at the full resolution.
            o_air_temperature_celsius = GEOS5FP_connection.Ta_C(time_UTC=time_UTC, geometry=o_geometry, resampling="cubic")

    # Store the value in the dictionary
    c_results["Ta"] = o_air_temperature_celsius

    ### Process modeled datasets ###
    # Check if model is available. Process if available, otherwise skip and fill fields with None values
    if o_model is not None:
        # Model is available. Attempt processing with it.
        # Solve for soil water balance
        if o_water is None:
            # No data is provided. Request the data.
            o_water = o_model.SRTM_connection.swb(o_geometry)

        # Set value into the results dictionary
        c_results["water"] = o_water

        # Solve for the soil moisture
        if SM is None:
            # Soil moisture is not provided. Determine how to calculate it.
            if downscale_moisture:
                # Calculate using downscaling
                o_st_band_kelvin = o_st_band_celsius + 273.15
                SM_coarse = GEOS5FP_connection.SFMC(time_UTC=time_UTC, geometry=o_coarse_geometry, resampling="cubic")
                SM_smooth = GEOS5FP_connection.SFMC(time_UTC=time_UTC, geometry=o_geometry, resampling="cubic")

                SM = downscale_soil_moisture(time_UTC=time_UTC, fine_geometry=o_geometry, coarse_geometry=o_coarse_geometry, SM_coarse=SM_coarse, SM_resampled=SM_smooth,
                                             ST_fine=o_st_band_kelvin, NDVI_fine=o_ndvi, water=o_water)

            else:
                # Calculate it at the standard resolution
                SM = GEOS5FP_connection.SFMC(time_UTC=time_UTC, geometry=o_geometry, resampling="cubic")

        # Set the value into the results dictionary
        c_results["SM"] = SM

        if RH is None:
            if downscale_humidity:
                o_st_band_kelvin = o_st_band_celsius + 273.15
                VPD_Pa_coarse = GEOS5FP_connection.VPD_Pa(time_UTC=time_UTC, geometry=o_coarse_geometry, resampling="cubic")

                VPD_Pa = downscale_vapor_pressure_deficit(time_UTC=time_UTC, VPD_Pa_coarse=VPD_Pa_coarse, ST_K=o_st_band_kelvin, fine_geometry=o_geometry,
                                                          coarse_geometry=o_coarse_geometry)

                VPD_kPa = VPD_Pa / 1000

                RH_coarse = GEOS5FP_connection.RH(time_UTC=time_UTC, geometry=o_coarse_geometry, resampling="cubic")

                RH = downscale_relative_humidity(time_UTC=time_UTC, RH_coarse=RH_coarse, SM=SM, ST_K=o_st_band_kelvin, VPD_kPa=VPD_kPa, water=o_water,
                                                 fine_geometry=o_geometry, coarse_geometry=o_coarse_geometry)
            else:
                RH = GEOS5FP_connection.RH(time_UTC=time_UTC, geometry=o_geometry, resampling="cubic")

        c_results["RH"] = RH

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
                Ra, Rg, UV, VIS, NIR, VISdiff, NIRdiff, VISdir, NIRdir = o_model.FLiES(geometry=o_geometry, target=target, time_UTC=time_UTC, albedo=o_albedo)

                SWin = Rg

            if SWin == "FLiES-GEOS5FP":
                logger.info("generating solar radiation using Forest Light Environmental Simulator bias-corrected with GEOS-5 FP")
                Ra, Rg, UV, VIS, NIR, VISdiff, NIRdiff, VISdir, NIRdir = o_model.FLiES(geometry=o_geometry, target=target, time_UTC=time_UTC, albedo=o_albedo)

                SWin_coarse = GEOS5FP_connection.SWin(time_UTC=time_UTC, geometry=o_coarse_geometry, resampling="cubic")
                SWin = bias_correct(coarse_image=SWin_coarse, fine_image=Rg)

            elif SWin == "GEOS5FP" or SWin is None:
                logger.info("generating solar radiation using GEOS-5 FP")
                SWin = GEOS5FP_connection.SWin(time_UTC=time_UTC, geometry=o_geometry, resampling="cubic")

        if Rn is None or isinstance(Rn, str):
            if Rn == "BESS":
                logger.info(f"generating net radiation using Breathing Earth System Simulator for {colored_logging.place(target)} at {colored_logging.time(time_UTC)} UTC")

                o_st_band_kelvin = o_st_band_celsius + 273.15
                o_air_temperature_kelvin = o_air_temperature_celsius + 273.15

                BESS_results = o_model.BESS(geometry=o_geometry, target=target, time_UTC=time_UTC, ST_K=o_st_band_kelvin, Ta_K=o_air_temperature_kelvin, RH=RH, elevation_km=elevation_km, NDVI=o_ndvi, albedo=o_albedo,
                                            Rg=SWin, VISdiff=VISdiff, VISdir=VISdir, NIRdiff=NIRdiff, NIRdir=NIRdir, UV=UV, water=o_water, output_variables=["Rn", "LE", "GPP"])

                Rn = BESS_results["Rn"]

            if Rn == "Verma":
                Rn = None

        logger.info(f"running PT-JPL-SM ET model hindcast at {colored_logging.time(time_UTC)}")

        if s_model_name == "PTJPLSM":
            PTJPL_results = o_model.PTJPL(geometry=o_geometry, target=target, time_UTC=time_UTC, ST_C=o_st_band_celsius, emissivity=o_emissivity, NDVI=o_ndvi, albedo=o_albedo, SWin=SWin, SM=SM,
                                          wind_speed=wind_speed, Ta_C=o_air_temperature_celsius, RH=RH, Rn=Rn, water=o_water, output_variables=target_variables)

        elif s_model_name == "PTJPL":
            PTJPL_results = o_model.PTJPL(geometry=o_geometry, target=target, time_UTC=time_UTC, ST_C=o_st_band_celsius, emissivity=o_emissivity, NDVI=o_ndvi, albedo=o_albedo, SWin=SWin,
                                          wind_speed=wind_speed, Ta_C=o_air_temperature_celsius, RH=RH, Rn=Rn, water=o_water, output_variables=target_variables)

        else:
            raise ValueError(f"unrecognized model: {s_model_name}")

        for k, v in PTJPL_results.items():
            c_results[k] = v

        for product, image in c_results.items():
            filename = generate_LANCE_output_filename( LANCE_output_directory=LANCE_output_directory, target_date=target_date, time_UTC=time_UTC, target=target, product=product)

            if image is None:
                logger.warning(f"no image result for {product}")
                continue

            logger.info(f"writing LANCE GEOS-5 FP {colored_logging.name(product)} at {colored_logging.place(target)} at {colored_logging.time(time_UTC)} to file: "
                        f"{colored_logging.file(filename)}")
            image.to_geotiff(filename)

    else:
        # Model is not available. Set results to None an continue processing.
        c_results["water"] = None
        c_results["SM"] = None
        c_results["RH"] = None

    # Return to the calling function
    return c_results
