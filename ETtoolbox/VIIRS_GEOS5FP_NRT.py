from typing import Union, List
from datetime import date, datetime
import VNP09GA_002
import VNP21A1D_002
from dateutil import parser
from os.path import join, abspath, expanduser, basename, exists
import numpy as np
import rasters as rt
from glob import glob
from os.path import splitext
from typing import Dict, Callable
import logging
import colored_logging as cl
import rasters
from rasters import RasterGrid
from gedi_canopy_height import GEDICanopyHeight
from GEOS5FP import GEOS5FP
from MODISCI import MODISCI
from PTJPL import PTJPL
from verma_net_radiation import process_verma_net_radiation
from PTJPL import PTJPL
import NASADEM
from soil_capacity_wilting import SoilGrids
from GEOS5FP.downscaling import downscale_air_temperature, downscale_soil_moisture, downscale_vapor_pressure_deficit, \
    downscale_relative_humidity, bias_correct
from solar_apparent_time import solar_to_UTC

from .constants import *
from .generate_VIIRS_output_directory import generate_VIIRS_output_directory
from .generate_VIIRS_output_filename import generate_VIIRS_output_filename
from .check_VIIRS_already_processed import check_VIIRS_already_processed
from .load_VIIRS import load_VIIRS

logger = logging.getLogger(__name__)


class GEOS5FPNotAvailableError(Exception):
    """Custom exception for when GEOS-5 FP data is not available."""
    pass


def VIIRS_GEOS5FP_NRT(
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
        model: PTJPL = None,
        model_name: str = ET_MODEL_NAME,
        working_directory: str = None,
        static_directory: str = None,
        VIIRS_download_directory: str = None,
        VIIRS_output_directory: str = None,
        output_bucket_name: str = None,
        SRTM_connection: NASADEM = None,
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
        ERS_credentials_filename: str = None,
        preview_quality: int = PREVIEW_QUALITY,
        ANN_model: Callable = None,
        ANN_model_filename: str = None,
        resampling: str = RESAMPLING,
        coarse_cell_size: float = COARSE_CELL_SIZE,
        downscale_air: bool = DOWNSCALE_AIR,
        downscale_humidity: bool = DOWNSCALE_HUMIDITY,
        downscale_moisture: bool = DOWNSCALE_MOISTURE,
        floor_Topt: bool = FLOOR_TOPT,
        save_intermediate: bool = False,
        include_preview: bool = True,
        show_distribution: bool = True,
        load_previous: bool = True,
        target_variables: List[str] = TARGET_VARIABLES) -> Dict[str, rt.Raster]:
    """
    Processes VIIRS and GEOS-5 FP data to generate various environmental products,
    including land surface temperature, NDVI, emissivity, albedo, and
    evapotranspiration components using the PT-JPL model. It handles data
    retrieval, downscaling of atmospheric variables, and saving of results.

    Args:
        target_date (Union[date, str]): The target date for processing.
        geometry (RasterGrid): The spatial extent and resolution for the output rasters.
        target (str): A string identifier for the target area or region.
        ST_C (rt.Raster, optional): Pre-computed surface temperature in Celsius. Defaults to None.
        emissivity (rt.Raster, optional): Pre-computed emissivity. Defaults to None.
        NDVI (rt.Raster, optional): Pre-computed Normalized Difference Vegetation Index. Defaults to None.
        albedo (rt.Raster, optional): Pre-computed albedo. Defaults to None.
        SWin (Union[rt.Raster, str], optional): Pre-computed incoming shortwave radiation. Defaults to None.
        Rn (Union[rt.Raster, str], optional): Pre-computed net radiation. Defaults to None.
        SM (rt.Raster, optional): Pre-computed soil moisture. Defaults to None.
        wind_speed (rt.Raster, optional): Pre-computed wind speed. Defaults to None.
        Ta_C (rt.Raster, optional): Pre-computed air temperature in Celsius. Defaults to None.
        RH (rt.Raster, optional): Pre-computed relative humidity. Defaults to None.
        water (rt.Raster, optional): Pre-computed water mask. Defaults to None.
        elevation_km (rt.Raster, optional): Pre-computed elevation in kilometers. Defaults to None.
        model (PTJPL, optional): An instance of the PTJPL model. Defaults to None.
        model_name (str, optional): Name of the ET model. Defaults to ET_MODEL_NAME.
        working_directory (str, optional): Base directory for data processing. Defaults to None.
        static_directory (str, optional): Directory for static data. Defaults to None.
        VIIRS_download_directory (str, optional): Directory for downloading VIIRS data. Defaults to None.
        VIIRS_output_directory (str, optional): Directory for saving VIIRS processed outputs. Defaults to None.
        output_bucket_name (str, optional): S3 bucket name for outputs. Defaults to None.
        SRTM_connection (NASADEM, optional): Connection object for NASADEM. Defaults to None.
        SRTM_download (str, optional): Directory for downloading SRTM data. Defaults to None.
        GEOS5FP_connection (GEOS5FP, optional): Connection object for GEOS5FP. Defaults to None.
        GEOS5FP_download (str, optional): Directory for downloading GEOS5FP data. Defaults to None.
        GEOS5FP_products (str, optional): Specific GEOS5FP products to retrieve. Defaults to None.
        GEDI_connection (GEDICanopyHeight, optional): Connection object for GEDI Canopy Height. Defaults to None.
        GEDI_download (str, optional): Directory for downloading GEDI data. Defaults to None.
        ORNL_connection (MODISCI, optional): Connection object for MODISCI. Defaults to None.
        CI_directory (str, optional): Directory for CI data. Defaults to None.
        soil_grids_connection (SoilGrids, optional): Connection object for SoilGrids. Defaults to None.
        soil_grids_download (str, optional): Directory for downloading SoilGrids data. Defaults to None.
        intermediate_directory (str, optional): Directory for saving intermediate processing files. Defaults to None.
        spacetrack_credentials_filename (str, optional): Filename for Space-Track credentials. Defaults to None.
        ERS_credentials_filename (str, optional): Filename for ERS credentials. Defaults to None.
        preview_quality (int, optional): Quality setting for preview images. Defaults to PREVIEW_QUALITY.
        ANN_model (Callable, optional): Pre-trained ANN model. Defaults to None.
        ANN_model_filename (str, optional): Filename of the ANN model. Defaults to None.
        resampling (str, optional): Resampling method. Defaults to RESAMPLING.
        coarse_cell_size (float, optional): Cell size for coarse resolution data. Defaults to COARSE_CELL_SIZE.
        downscale_air (bool, optional): Whether to downscale air temperature. Defaults to DOWNSCALE_AIR.
        downscale_humidity (bool, optional): Whether to downscale humidity. Defaults to DOWNSCALE_HUMIDITY.
        downscale_moisture (bool, optional): Whether to downscale soil moisture. Defaults to DOWNSCALE_MOISTURE.
        floor_Topt (bool, optional): Whether to floor optimal temperature. Defaults to FLOOR_TOPT.
        save_intermediate (bool, optional): Whether to save intermediate processing files. Defaults to False.
        include_preview (bool, optional): Whether to include preview images in output. Defaults to True.
        show_distribution (bool, optional): Whether to show distribution plots. Defaults to True.
        load_previous (bool, optional): Whether to load previously processed results if available. Defaults to True.
        target_variables (List[str], optional): List of target variables to process. Defaults to TARGET_VARIABLES.

    Returns:
        Dict[str, rt.Raster]: A dictionary containing the processed raster outputs.

    Raises:
        GEOS5FPNotAvailableError: If GEOS-5 FP data is not available for the target time.
    """
    results = {}

    # Parse target_date if it's a string
    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()

    logger.info(f"VIIRS target date: {cl.time(target_date)}")

    # Define solar and UTC times for processing
    time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
    logger.info(f"VIIRS target time solar: {cl.time(time_solar)}")
    time_UTC = solar_to_UTC(time_solar, geometry.centroid.latlon.x)
    logger.info(f"VIIRS target time UTC: {cl.time(time_UTC)}")

    # Set up working directory
    if working_directory is None:
        working_directory = "~/data/ETtoolbox"
    working_directory = abspath(expanduser(working_directory))
    logger.info(f"VIIRS working directory: {cl.dir(working_directory)}")

    # Initialize SRTM connection and retrieve water mask and elevation if not provided
    if SRTM_connection is None:
        SRTM_connection = NASADEM.NASADEMConnection(
            download_directory=SRTM_download,
        )

    if water is None:
        water = SRTM_connection.swb(geometry)
    results["water"] = water # Add water mask to results early for potential use

    if elevation_km is None:
        elevation_km = SRTM_connection.elevation_km(geometry)

    # Set up VIIRS download and output directories
    if VIIRS_download_directory is None:
        VIIRS_download_directory = join(working_directory, VIIRS_DOWNLOAD_DIRECTORY)
    logger.info(f"VIIRS download directory: {cl.dir(VIIRS_download_directory)}")

    if VIIRS_output_directory is None:
        VIIRS_output_directory = join(working_directory, VIIRS_OUTPUT_DIRECTORY)
    logger.info(f"VIIRS output directory: {cl.dir(VIIRS_output_directory)}")

    # Check if VIIRS data for the target date and products has already been processed
    VIIRS_already_processed = check_VIIRS_already_processed(
        VIIRS_output_directory=VIIRS_output_directory,
        target_date=target_date,
        time_UTC=time_UTC,
        target=target,
        products=target_variables
    )

    if VIIRS_already_processed:
        if load_previous:
            logger.info("Loading previously generated VIIRS GEOS-5 FP output.")
            return load_VIIRS(
                VIIRS_output_directory=VIIRS_output_directory,
                target_date=target_date,
                target=target
            )
        else:
            logger.info("VIIRS GEOS-5 FP output already exists and 'load_previous' is False. Skipping processing.")
            return

    # Initialize GEOS-5 FP connection
    if GEOS5FP_connection is None:
        try:
            logger.info(f"Connecting to GEOS-5 FP.")
            GEOS5FP_connection = GEOS5FP(
                working_directory=working_directory,
                download_directory=GEOS5FP_download
            )
        except Exception as e:
            logger.exception(e)
            raise GEOS5FPNotAvailableError("Unable to connect to GEOS-5 FP.")

    # Check GEOS-5 FP data availability
    latest_GEOS5FP_time = GEOS5FP_connection.latest_time_available
    logger.info(f"Latest GEOS-5 FP time available: {latest_GEOS5FP_time}")
    logger.info(f"Processing time: {time_UTC}")

    if time_UTC.strftime("%Y-%m-%d %H:%M:%S") > latest_GEOS5FP_time.strftime("%Y-%m-%d %H:%M:%S"):
        raise GEOS5FPNotAvailableError(
            f"VIIRS target time {time_UTC} is past latest available GEOS-5 FP time {latest_GEOS5FP_time}. "
            "Cannot proceed with NRT processing."
        )

    # Set processing date and time
    VIIRS_processing_date = target_date
    VIIRS_processing_time = time_UTC

    # Initialize VIIRS data connections
    VNP21_connection = VNP21A1D_002.VNP21A1D(download_directory=VIIRS_download_directory)
    VNP09_connection = VNP09GA_002.VNP09GA(download_directory=VIIRS_download_directory)

    # Retrieve or calculate Surface Temperature (ST_C)
    if ST_C is None:
        logger.info(f"Retrieving {cl.name('VNP21A1D')} {cl.name('ST_C')} from VIIRS on {cl.time(VIIRS_processing_date)}.")
        ST_K = VNP21_connection.ST_K(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )
        ST_C = ST_K - 273.15
        # Fill in missing ST_C values with smooth GEOS-5 FP surface temperature
        ST_C_smooth = GEOS5FP_connection.Ts_K(time_UTC=time_UTC, geometry=geometry, resampling="cubic") - 273.15
        ST_C = rt.where(np.isnan(ST_C), ST_C_smooth, ST_C)
    results["ST"] = ST_C

    # Retrieve or calculate NDVI
    if NDVI is None:
        logger.info(f"Retrieving {cl.name('VNP09GA')} {cl.name('NDVI')} from VIIRS on {cl.time(VIIRS_processing_date)}.")
        NDVI = VNP09_connection.NDVI(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )
    results["NDVI"] = NDVI

    # Retrieve or calculate Emissivity
    if emissivity is None:
        logger.info(f"Retrieving {cl.name('VNP21A1D')} {cl.name('emissivity')} from VIIRS on {cl.time(VIIRS_processing_date)}.")
        emissivity = VNP21_connection.Emis_14(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )
    # Adjust emissivity for water bodies and fill missing values using NDVI-based empirical relationship
    emissivity = rt.where(water, 0.96, emissivity)
    emissivity = rt.where(np.isnan(emissivity), 1.0094 + 0.047 * np.log(NDVI), emissivity)
    results["emissivity"] = emissivity

    # Retrieve or calculate Albedo
    if albedo is None:
        logger.info(f"Retrieving {cl.name('VNP09GA')} {cl.name('albedo')} from VIIRS on {cl.time(VIIRS_processing_date)}.")
        albedo = VNP09_connection.albedo(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )
    results["albedo"] = albedo

    # Define coarse geometry for downscaling
    coarse_geometry = geometry.rescale(coarse_cell_size)

    # Retrieve or calculate Air Temperature (Ta_C)
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

    # Retrieve or calculate Soil Moisture (SM)
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

    # Retrieve or calculate Relative Humidity (RH)
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

    # Retrieve or calculate Incoming Shortwave Radiation (SWin)
    if SWin is None:
        logger.info("Generating solar radiation using GEOS-5 FP.")
        SWin = GEOS5FP_connection.SWin(time_UTC=time_UTC, geometry=geometry, resampling="cubic")
    results["SWin"] = SWin # Add SWin to results

    # Calculate Net Radiation (Rn) using Verma's model if not provided
    if Rn is None:
        logger.info("Calculating net radiation using Verma's model.")
        verma_results = process_verma_net_radiation(
            SWin=SWin,
            albedo=albedo,
            ST_C=ST_C,
            emissivity=emissivity,
            Ta_C=Ta_C,
            RH=RH
        )
        Rn = verma_results["Rn"]
    results["Rn"] = Rn

    # Run the PT-JPL ET model
    logger.info(f"Running PT-JPL ET model hindcast at {cl.time(time_UTC)}.")
    PTJPL_results = PTJPL(
        NDVI=NDVI,
        ST_C=ST_C,
        emissivity=emissivity,
        albedo=albedo,
        Rn=Rn,
        Ta_C=Ta_C,
        RH=RH
    )

    # Add PT-JPL results to the main results dictionary
    for k, v in PTJPL_results.items():
        results[k] = v

    # Save all processed raster outputs
    for product, image in results.items():
        filename = generate_VIIRS_output_filename(
            VIIRS_output_directory=VIIRS_output_directory,
            target_date=target_date,
            time_UTC=time_UTC,
            target=target,
            product=product
        )

        if image is None:
            logger.warning(f"No image result for {product}. Skipping save operation.")
            continue

        logger.info(
            f"Writing VIIRS GEOS-5 FP {cl.name(product)} for {cl.place(target)} at {cl.time(time_UTC)} to file: {cl.file(filename)}."
        )
        image.to_geotiff(filename)

    return results
