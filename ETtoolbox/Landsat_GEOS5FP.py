from LandsatL2C2 import LandsatL2C2
from PTJPL import PTJPL
from datetime import datetime, date, timedelta
from typing import Union, List
from os.path import join, abspath, expanduser, splitext, exists, basename
import logging
from dateutil import parser
from typing import Dict, Callable
from gedi_canopy_height import GEDICanopyHeight
from GEOS5FP import GEOS5FP
from MODISCI import MODISCI
from soil_capacity_wilting import SoilGrids
from glob import glob
import numpy as np
import rasters as rt
import colored_logging
from solar_apparent_time import UTC_to_solar, solar_to_UTC
from NASADEM import NASADEMConnection

from .constants import *
from .generate_Landsat_output_directory import generate_Landsat_output_directory
from .generate_Landsat_output_filename import generate_Landsat_output_filename
from .check_Landsat_already_processed import check_Landsat_already_processed
from .load_Landsat import load_Landsat

logger = logging.getLogger(__name__)

class LandsatNotAvailableError(Exception):
    """
    Exception raised when Landsat data is not available for the specified criteria.
    """
    pass

class GEOS5FPNotAvailableError(Exception):
    """
    Exception raised when GEOS-5 FP data is not available or connection fails.
    """
    pass

def Landsat_GEOS5FP(
        target_date: Union[date, str],
        geometry: rt.RasterGeometry,
        target: str,
        ST_C: rt.Raster = None,
        emissivity: rt.Raster = None,
        NDVI: rt.Raster = None,
        albedo: rt.Raster = None,
        SWin: rt.Raster = None,
        Rn: rt.Raster = None,
        SM: rt.Raster = None,
        wind_speed: rt.Raster = None,
        Ta_C: rt.Raster = None,
        RH: rt.Raster = None,
        water: rt.Raster = None,
        model: PTJPL = None,
        working_directory: str = None,
        static_directory: str = None,
        Landsat_download_directory: str = None,
        Landsat_output_directory: str = None,
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
        intermediate_directory=None,
        preview_quality: int = PREVIEW_QUALITY,
        ANN_model: Callable = None,
        ANN_model_filename: str = None,
        resampling: str = RESAMPLING,
        downscale_air: bool = True,
        downscale_humidity: bool = True,
        save_intermediate: bool = False,
        include_preview: bool = True,
        show_distribution: bool = True,
        load_previous: bool = True,
        target_variables: List[str] = TARGET_VARIABLES) -> Dict[str, rt.Raster]:
    """
    Processes Landsat and GEOS-5 FP data to generate various environmental products,
    including those derived from the PT-JPL Evapotranspiration model.

    Args:
        target_date (Union[date, str]): The target date for data acquisition.
        geometry (rt.RasterGeometry): The raster geometry defining the area of interest.
        target (str): A descriptive name for the target area.
        ST_C (rt.Raster, optional): Surface temperature in Celsius. If None, it will be retrieved from Landsat.
        emissivity (rt.Raster, optional): Surface emissivity. If None, it will be retrieved from Landsat.
        NDVI (rt.Raster, optional): Normalized Difference Vegetation Index. If None, it will be retrieved from Landsat.
        albedo (rt.Raster, optional): Surface albedo. If None, it will be retrieved from Landsat.
        SWin (rt.Raster, optional): Incoming shortwave radiation.
        Rn (rt.Raster, optional): Net radiation.
        SM (rt.Raster, optional): Soil moisture.
        wind_speed (rt.Raster, optional): Wind speed.
        Ta_C (rt.Raster, optional): Air temperature in Celsius.
        RH (rt.Raster, optional): Relative humidity.
        water (rt.Raster, optional): Water mask.
        model (PTJPL, optional): An instance of the PTJPL model. Defaults to None.
        working_directory (str, optional): The main working directory. Defaults to current directory.
        static_directory (str, optional): Directory for static data. Defaults to None.
        Landsat_download_directory (str, optional): Directory to download Landsat data.
            Defaults to a subdirectory within the working directory.
        Landsat_output_directory (str, optional): Directory to save Landsat output products.
            Defaults to a subdirectory within the working directory.
        SRTM_connection (NASADEMConnection, optional): An instance of the NASADEMConnection. Defaults to None.
        SRTM_download (str, optional): Directory for SRTM data downloads. Defaults to None.
        GEOS5FP_connection (GEOS5FP, optional): An instance of the GEOS5FP connection. Defaults to None.
        GEOS5FP_download (str, optional): Directory for GEOS-5 FP data downloads. Defaults to None.
        GEOS5FP_products (str, optional): Directory for GEOS-5 FP products. Defaults to None.
        GEDI_connection (GEDICanopyHeight, optional): An instance of the GEDICanopyHeight connection. Defaults to None.
        GEDI_download (str, optional): Directory for GEDI data downloads. Defaults to None.
        ORNL_connection (MODISCI, optional): An instance of the MODISCI connection. Defaults to None.
        CI_directory (str, optional): Directory for MODIS CI data. Defaults to None.
        soil_grids_connection (SoilGrids, optional): An instance of the SoilGrids connection. Defaults to None.
        soil_grids_download (str, optional): Directory for SoilGrids data downloads. Defaults to None.
        intermediate_directory ([type], optional): Directory for intermediate files. Defaults to None.
        preview_quality (int, optional): Quality setting for previews. Defaults to PREVIEW_QUALITY.
        ANN_model (Callable, optional): Artificial Neural Network model. Defaults to None.
        ANN_model_filename (str, optional): Filename for the ANN model. Defaults to None.
        resampling (str, optional): Resampling method. Defaults to RESAMPLING.
        downscale_air (bool, optional): Whether to downscale air temperature. Defaults to True.
        downscale_humidity (bool, optional): Whether to downscale humidity. Defaults to True.
        save_intermediate (bool, optional): Whether to save intermediate files. Defaults to False.
        include_preview (bool, optional): Whether to include preview images. Defaults to True.
        show_distribution (bool, optional): Whether to show distribution plots. Defaults to True.
        load_previous (bool, optional): Whether to load previously processed results if available. Defaults to True.
        target_variables (List[str], optional): List of target variables to process. Defaults to TARGET_VARIABLES.

    Returns:
        Dict[str, rt.Raster]: A dictionary containing the processed raster products.

    Raises:
        GEOS5FPNotAvailableError: If GEOS-5 FP data is not available or cannot be connected to.
    """
    results = {}

    logger.info(f"processing Landsat GOES-5 FP for target {colored_logging.name(target)} ({colored_logging.val(geometry.cell_width)}m, {colored_logging.val(geometry.rows)} rows {colored_logging.val(geometry.cols)} cols)")

    # Parse target_date if it's a string
    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()

    logger.info(f"Landsat target date: {colored_logging.time(target_date)}")
    
    # Define target time in solar time (13:30 local solar time)
    time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
    logger.info(f"Landsat target time solar: {colored_logging.time(time_solar)}")
    
    # Convert solar time to UTC based on geometry centroid longitude
    time_UTC = solar_to_UTC(time_solar, geometry.centroid.latlon.x)
    logger.info(f"Landsat target time UTC: {colored_logging.time(time_UTC)}")

    # Set up working directory
    if working_directory is None:
        working_directory = "."
    working_directory = abspath(expanduser(working_directory))
    logger.info(f"Landsat working directory: {colored_logging.dir(working_directory)}")

    # Set up Landsat download directory
    if Landsat_download_directory is None:
        Landsat_download_directory = join(working_directory, DEFAULT_Landsat_DOWNLOAD_DIRECTORY)
    logger.info(f"Landsat download directory: {colored_logging.dir(Landsat_download_directory)}")

    # Set up Landsat output directory
    if Landsat_output_directory is None:
        Landsat_output_directory = join(working_directory, DEFAULT_Landsat_OUTPUT_DIRECTORY)
    logger.info(f"Landsat output directory: {colored_logging.dir(Landsat_output_directory)}")

    # Check if Landsat data has already been processed for the target
    Landsat_already_processed = check_Landsat_already_processed(
        Landsat_output_directory=Landsat_output_directory,
        target_date=target_date,
        time_UTC=time_UTC,
        target=target,
        products=target_variables
    )

    if Landsat_already_processed:
        if load_previous:
            logger.info("loading previously generated Landsat GEOS-5 FP output")
            # Load and return previously processed data
            return load_Landsat(
                Landsat_output_directory=Landsat_output_directory,
                target_date=target_date,
                target=target
            )
        else:
            return None # Return None if not loading previous and already processed

    # Initialize GEOS5FP connection if not provided
    if GEOS5FP_connection is None:
        try:
            logger.info(f"connecting to GEOS-5 FP")
            GEOS5FP_connection = GEOS5FP(
                working_directory=working_directory,
                download_directory=GEOS5FP_download,
                products_directory=GEOS5FP_products,
            )
        except Exception as e:
            logger.exception(e)
            raise GEOS5FPNotAvailableError("unable to connect to GEOS-5 FP")

    # Get the latest available GEOS-5 FP time and check if target time is within range
    latest_GEOS5FP_time = GEOS5FP_connection.latest_time_available
    if time_UTC.strftime("%Y-%m-%d %H:%M:%S") > latest_GEOS5FP_time.strftime("%Y-%m-%d %H:%M:%S"):
        raise GEOS5FPNotAvailableError(f"Landsat target time {time_UTC} is past latest available GEOS-5 FP time {latest_GEOS5FP_time}")

    Landsat_processing_date = target_date
    Landsat_processing_time = time_UTC

    # Initialize LandsatL2C2 connection
    landsat = LandsatL2C2(
        working_directory=working_directory,
        download_directory=Landsat_download_directory
    )

    # Retrieve ST_C (Surface Temperature in Celsius) from Landsat if not provided
    if ST_C is None:
        logger.info(f"retrieving {colored_logging.name('ST_C')} from Landsat on {colored_logging.time(Landsat_processing_date)}")
        ST_C = landsat.product(
            acquisition_date=target_date,
            product="ST_C",
            geometry=geometry,
            target_name=target
        )
    results["ST_C"] = ST_C

    # Retrieve emissivity from Landsat if not provided
    if emissivity is None:
        logger.info(f"retrieving {colored_logging.name('emissivity')} from Landsat on {colored_logging.time(Landsat_processing_date)}")
        emissivity = landsat.product(
            acquisition_date=target_date,
            product="emissivity",
            geometry=geometry,
            target_name=target
        )
    results["emissivity"] = emissivity

    # Retrieve NDVI from Landsat if not provided
    if NDVI is None:
        logger.info(f"retrieving {colored_logging.name('NDVI')} from Landsat on {colored_logging.time(Landsat_processing_date)}")
        NDVI = landsat.product(
            acquisition_date=target_date,
            product="NDVI",
            geometry=geometry,
            target_name=target
        )
    results["NDVI"] = NDVI

    # Retrieve albedo from Landsat if not provided
    if albedo is None:
        logger.info(f"retrieving {colored_logging.name('VNP43MA4N')} {colored_logging.name('albedo')} from Landsat on {colored_logging.time(Landsat_processing_date)}")
        albedo = landsat.product(
            acquisition_date=target_date,
            product="albedo",
            geometry=geometry,
            target_name=target
        )
    results["albedo"] = albedo

    logger.info(f"running PT-JPL ET model forecast at {colored_logging.time(time_UTC)}")
    
    # Run the PT-JPL Evapotranspiration model
    PTJPL_results = PTJPL(
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
        water=water
    )

    # Add PT-JPL results to the main results dictionary
    for k, v in PTJPL_results.items():
        results[k] = v

    # Generate output directory for Landsat products if it doesn't exist
    generate_Landsat_output_directory(
        Landsat_output_directory=Landsat_output_directory,
        target_date=target_date,
        target=target
    )

    # Save each processed product to a GeoTIFF file
    for product, image in results.items():
        filename = generate_Landsat_output_filename(
            Landsat_output_directory=Landsat_output_directory,
            target_date=target_date,
            time_UTC=time_UTC,
            target=target,
            product=product
        )

        logger.info(f"writing Landsat GEOS-5 FP {colored_logging.name(product)} at {colored_logging.place(target)} at {colored_logging.time(time_UTC)} to file: {colored_logging.file(filename)}")
        image.to_geotiff(filename)

    return results