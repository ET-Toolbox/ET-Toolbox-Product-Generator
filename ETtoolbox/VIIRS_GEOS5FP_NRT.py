from typing import Union, List, Dict, Callable
from datetime import date, datetime # For handling date and time objects
from dateutil import parser # Library for robust parsing of date strings
from os.path import join, abspath, expanduser, basename, exists, splitext # Utilities for path manipulation
import numpy as np # Numerical computing library, essential for array operations and NaN handling
import rasters as rt # Custom or external library for handling raster (geospatial image) data
from rasters import RasterGrid # Specific class for defining a spatial grid for rasters

from glob import glob # For finding pathnames matching a specified pattern

import logging # Standard Python logging library
import colored_logging as cl # Custom logging utility for colored console output

# Import specific connection and processing modules for various datasets
import VNP09GA_002 # VIIRS VNP09GA (surface reflectance) data access
import VNP21A1D_002 # VIIRS VNP21A1D (land surface temperature) data access
from gedi_canopy_height import GEDICanopyHeight # GEDI Canopy Height data access
from GEOS5FP import GEOS5FP # GEOS-5 FP (atmospheric reanalysis) data access
from MODISCI import MODISCI # MODIS CI (clumping Index) data access
from PTJPL import PTJPL # PT-JPL evapotranspiration model implementation
from verma_net_radiation import process_verma_net_radiation # Function to calculate net radiation using the Verma model
import NASADEM # NASADEM (digital elevation model) data access, likely NASADEMConnection within it
from soil_capacity_wilting import SoilGrids # SoilGrids data access

# Import downscaling and bias correction functions specifically for GEOS-5 FP data
from GEOS5FP.downscaling import downscale_air_temperature, downscale_soil_moisture, \
    downscale_vapor_pressure_deficit, downscale_relative_humidity, bias_correct
from solar_apparent_time import solar_to_UTC # Function to convert local solar time to Coordinated Universal Time (UTC)

# Import constants and helper functions from the local package
from .constants import * # Global constants used across the package (e.g., directory names, flags)
from .generate_VIIRS_output_directory import generate_VIIRS_output_directory # Helper to create VIIRS-related output directories
from .generate_VIIRS_output_filename import generate_VIIRS_output_filename # Helper to generate standard VIIRS output filenames
from .check_VIIRS_already_processed import check_VIIRS_already_processed # Helper to check if VIIRS data for a date is already processed
from .load_VIIRS import load_VIIRS # Helper to load previously processed VIIRS data from disk

# Set up logging for this specific module
logger = logging.getLogger(__name__)

class GEOS5FPNotAvailableError(Exception):
    """
    Custom exception raised when GEOS-5 FP (Goddard Earth Observing System, Version 5,
    Forward Processing) data cannot be accessed or if the target time is beyond
    the latest available data for Near Real-Time (NRT) processing.
    """
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
        wind_speed: rt.Raster = None, # Not currently used in the processing, reserved for future extensions.
        Ta_C: rt.Raster = None,
        RH: rt.Raster = None,
        water: rt.Raster = None,
        elevation_km: rt.Raster = None,
        model: PTJPL = None, # An instance of the PTJPL model. If None, a default one will be used internally by the PTJPL module.
        model_name: str = ET_MODEL_NAME, # Name of the ET model to use. Currently not actively used for model selection.
        working_directory: str = None,
        static_directory: str = None,
        VIIRS_download_directory: str = None,
        VIIRS_output_directory: str = None,
        output_bucket_name: str = None, # S3 bucket name for cloud storage outputs. Not directly used here.
        SRTM_connection: NASADEM = None, # Connection object for NASADEM. If None, a new one will be initialized.
        SRTM_download: str = None, # Directory for downloading SRTM data.
        GEOS5FP_connection: GEOS5FP = None, # Connection object for GEOS5FP. If None, a new one will be initialized.
        GEOS5FP_download: str = None, # Directory for downloading GEOS5FP data.
        GEOS5FP_products: str = None, # Specific GEOS5FP products to retrieve. Not actively used here.
        GEDI_connection: GEDICanopyHeight = None, # Connection object for GEDI Canopy Height. Not currently used.
        GEDI_download: str = None, # Directory for downloading GEDI data. Not currently used.
        ORNL_connection: MODISCI = None, # Connection object for MODISCI. Not currently used.
        CI_directory: str = None, # Directory for CI data. Not currently used.
        soil_grids_connection: SoilGrids = None, # Connection object for SoilGrids. Not currently used.
        soil_grids_download: str = None, # Directory for downloading SoilGrids data. Not currently used.
        intermediate_directory: str = None, # Directory for saving intermediate processing files. Not currently used.
        spacetrack_credentials_filename: str = None, # Filename for Space-Track credentials. Not directly used.
        ERS_credentials_filename: str = None, # Filename for ERS credentials. Not directly used.
        preview_quality: int = PREVIEW_QUALITY, # Quality setting for preview images. Not currently used.
        ANN_model: Callable = None, # Pre-trained Artificial Neural Network model. Not currently used.
        ANN_model_filename: str = None, # Filename of the ANN model. Not currently used.
        resampling: str = RESAMPLING, # Default resampling method.
        coarse_cell_size: float = COARSE_CELL_SIZE, # Cell size for coarse resolution data used in downscaling.
        downscale_air: bool = DOWNSCALE_AIR, # Whether to downscale air temperature.
        downscale_humidity: bool = DOWNSCALE_HUMIDITY, # Whether to downscale humidity.
        downscale_moisture: bool = DOWNSCALE_MOISTURE, # Whether to downscale soil moisture.
        floor_Topt: bool = FLOOR_TOPT, # Whether to floor optimal temperature (Topt) in PT-JPL.
        save_intermediate: bool = False, # Whether to save intermediate processing files. Not currently used.
        include_preview: bool = True, # Whether to include preview images in output. Not currently used.
        show_distribution: bool = True, # Whether to show distribution plots. Not currently used.
        load_previous: bool = True, # Whether to load previously processed results if available.
        target_variables: List[str] = TARGET_VARIABLES) -> Dict[str, rt.Raster]:
    """
    Processes VIIRS and GEOS-5 FP data to generate various environmental products
    in a Near Real-Time (NRT) context. This includes deriving land surface temperature,
    NDVI, emissivity, albedo, and evapotranspiration components using the PT-JPL model.
    The function handles data retrieval, downscaling of atmospheric variables,
    and saving of results, with an emphasis on timely data availability.

    Args:
        target_date (Union[date, str]): The target date for which to process data.
            Can be a `datetime.date` object or a string parseable by `dateutil.parser`.
        geometry (RasterGrid): The spatial extent and resolution for all output rasters.
            All derived products will be aligned to this grid.
        target (str): A string identifier for the specific area or region being processed
            (e.g., 'site_A', 'basin_X'), used for file naming and logging.
        ST_C (rt.Raster, optional): Pre-computed land surface temperature in Celsius.
            If provided, this value will be used; otherwise, it will be retrieved from VNP21A1D.
        emissivity (rt.Raster, optional): Pre-computed surface emissivity.
            If provided, this value will be used; otherwise, it will be retrieved from VNP21A1D.
        NDVI (rt.Raster, optional): Pre-computed Normalized Difference Vegetation Index.
            If provided, this value will be used; otherwise, it will be retrieved from VNP09GA.
        albedo (rt.Raster, optional): Pre-computed surface albedo.
            If provided, this value will be used; otherwise, it will be retrieved from VNP09GA.
        SWin (Union[rt.Raster, str], optional): Pre-computed incoming shortwave radiation.
            If None or 'GEOS5FP', it will be retrieved from GEOS-5 FP.
        Rn (Union[rt.Raster, str], optional): Pre-computed net radiation.
            If None, it will be calculated using the Verma net radiation model.
        SM (rt.Raster, optional): Pre-computed soil moisture.
            If None, it will be retrieved and potentially downscaled from GEOS-5 FP.
        wind_speed (rt.Raster, optional): Pre-computed wind speed data.
            This parameter is not currently used in the processing pipeline but is
            included for potential future extensions. Defaults to None.
        Ta_C (rt.Raster, optional): Pre-computed air temperature in Celsius.
            If None, it will be retrieved and potentially downscaled from GEOS-5 FP.
        RH (rt.Raster, optional): Pre-computed relative humidity.
            If None, it will be retrieved and potentially downscaled from GEOS-5 FP.
        water (rt.Raster, optional): A binary raster mask indicating water bodies (1) or land (0).
            If None, it will be derived from SRTM data.
        elevation_km (rt.Raster, optional): Pre-computed elevation in kilometers.
            If None, it will be retrieved from SRTM data.
        model (PTJPL, optional): An existing instance of the PTJPL model.
            If None, a new one will be initialized internally by the PTJPL module as needed.
        model_name (str, optional): The name of the evapotranspiration model to use.
            Currently, the function primarily uses PT-JPL, so this parameter is not
            actively used for model selection but is reserved for future extensions.
        working_directory (str, optional): The base directory for all data processing and outputs.
            Defaults to "~/data/ETtoolbox".
        static_directory (str, optional): Directory for static datasets like SRTM.
            Defaults to None, implying a default from connection objects or the working directory.
        VIIRS_download_directory (str, optional): Local directory for downloading VIIRS data.
            Defaults to a subdirectory within the `working_directory`.
        VIIRS_output_directory (str, optional): Local directory for saving final processed VIIRS outputs.
            Defaults to a subdirectory within the `working_directory`.
        output_bucket_name (str, optional): Name of an S3 bucket for cloud storage outputs.
            This parameter is not directly used in the current file saving logic but
            might be used by external storage utilities. Defaults to None.
        SRTM_connection (NASADEM, optional): An existing connection object for NASADEM data.
            If None, a new `NASADEMConnection` will be initialized.
        SRTM_download (str, optional): Local directory for downloading SRTM data.
            Defaults to None, which lets the `NASADEMConnection` use its default.
        GEOS5FP_connection (GEOS5FP, optional): An existing connection object for GEOS-5 FP data.
            If None, a new `GEOS5FP` connection will be initialized.
        GEOS5FP_download (str, optional): Local directory for downloading GEOS-5 FP data.
            Defaults to None, which lets the `GEOS5FP` connection use its default.
        GEOS5FP_products (str, optional): A string specifying which GEOS-5 FP products to retrieve.
            This parameter is not directly used for product selection within this function
            but could be used in a more granular GEOS-5 FP retrieval setup. Defaults to None.
        GEDI_connection (GEDICanopyHeight, optional): An existing GEDI connection object.
            This parameter is not currently used but is reserved for future GEDI data integration. Defaults to None.
        GEDI_download (str, optional): Local directory for GEDI data downloads. Not currently used. Defaults to None.
        ORNL_connection (MODISCI, optional): An existing ORNL MODIS CI connection object.
            This parameter is not currently used but is reserved for future MODIS CI data integration. Defaults to None.
        CI_directory (str, optional): Local directory for CI data. Not currently used. Defaults to None.
        soil_grids_connection (SoilGrids, optional): An existing SoilGrids connection object.
            This parameter is not currently used but is reserved for future SoilGrids data integration. Defaults to None.
        soil_grids_download (str, optional): Local directory for SoilGrids data downloads. Not currently used. Defaults to None.
        intermediate_directory (str, optional): Directory for saving intermediate processing files.
            This parameter is not currently used. Defaults to None.
        spacetrack_credentials_filename (str, optional): Filename for Space-Track credentials.
            Not directly used in this function. Defaults to None.
        ERS_credentials_filename (str, optional): Filename for ERS credentials.
            Not directly used in this function. Defaults to None.
        preview_quality (int, optional): Quality setting for generating preview images.
            This parameter is not currently used. Defaults to PREVIEW_QUALITY.
        ANN_model (Callable, optional): A pre-trained Artificial Neural Network model.
            This parameter is not currently used. Defaults to None.
        ANN_model_filename (str, optional): Filename of the ANN model. Not currently used. Defaults to None.
        resampling (str, optional): The default resampling method to use for spatial
            operations when reprojecting or resizing rasters. Defaults to RESAMPLING.
        coarse_cell_size (float, optional): The target cell size (in degrees) for
            coarse resolution data, typically used as an input for downscaling algorithms.
            This defines the approximate resolution of GEOS-5 FP inputs before downscaling.
        downscale_air (bool, optional): If True, applies downscaling techniques to GEOS-5 FP
            air temperature data to match the finer `geometry` resolution.
        downscale_humidity (bool, optional): If True, applies downscaling techniques to GEOS-5 FP
            relative humidity data.
        downscale_moisture (bool, optional): If True, applies downscaling techniques to GEOS-5 FP
            soil moisture data.
        floor_Topt (bool, optional): Whether to enforce a minimum (floor) value for
            optimal temperature (Topt) in the PT-JPL model. This parameter is passed to PTJPL.
        save_intermediate (bool, optional): If True, saves intermediate products generated
            during the workflow. This parameter is not currently used. Defaults to False.
        include_preview (bool, optional): If True, includes preview images in the output.
            This parameter is not currently used. Defaults to True.
        show_distribution (bool, optional): If True, shows distribution plots of
            derived variables. This parameter is not currently used. Defaults to True.
        load_previous (bool, optional): If True, the function will first check if all
            required products for the `target_date` have already been processed and saved locally.
            If so, it will load them instead of re-processing, speeding up subsequent runs.
        target_variables (List[str], optional): A list of specific output variable names
            (e.g., "ST", "NDVI", "ET_total") that should be processed and saved.

    Returns:
        Dict[str, rt.Raster]: A dictionary where keys are variable names (e.g.,
            "ST", "NDVI", "ET_total") and values are the corresponding
            raster objects (geospatial images) that have been processed.

    Raises:
        GEOS5FPNotAvailableError: If GEOS-5 FP data is not available for the specified
            `target_time`, which is critical for NRT (Near Real-Time) processing.
    """
    # Initialize an empty dictionary to store all processed raster results
    results = {}

    # --- Section: Date and Time Setup ---
    # Convert the target_date to a datetime.date object if it's provided as a string.
    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()
    logger.info(f"VIIRS target date: {cl.time(target_date)}")

    # Define the local solar time for processing. For NRT, this is often set to a fixed
    # time like 13:30 local solar time, which is typically around local solar noon,
    # suitable for capturing peak daytime conditions for ET.
    time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
    logger.info(f"VIIRS target time solar: {cl.time(time_solar)}")

    # Convert the local solar time to Coordinated Universal Time (UTC) based on the
    # longitude of the geometry's centroid. This is essential for accurately querying
    # global datasets like GEOS-5 FP and VIIRS, which are indexed by UTC.
    time_UTC = solar_to_UTC(time_solar, geometry.centroid.latlon.x)
    logger.info(f"VIIRS target time UTC: {cl.time(time_UTC)}")

    # --- Section: Directory Setup ---
    # Set up the main working directory. If not provided, use a default path.
    if working_directory is None:
        working_directory = "~/data/ETtoolbox"
    # Resolve to an absolute path and expand user directory (e.g., '~' to '/home/user').
    working_directory = abspath(expanduser(working_directory))
    logger.info(f"VIIRS working directory: {cl.dir(working_directory)}")

    # --- Section: SRTM (Elevation Data) Handling ---
    # Initialize a NASADEMConnection object if one is not already provided.
    # This connection is used to retrieve elevation and water mask data from SRTM.
    if SRTM_connection is None:
        SRTM_connection = NASADEM.NASADEMConnection(
            download_directory=SRTM_download, # Specific download directory for SRTM data
            # offline_ok=True # This parameter was present in the previous version, but not in this one.
                              # Its absence here implies it might always attempt online connection.
        )

    # Retrieve the water mask if it was not provided as input.
    # The water mask (Surface Water Body) is derived from SRTM data.
    if water is None:
        logger.info(f"Retrieving {cl.name('water mask')} from SRTM.")
        water = SRTM_connection.swb(geometry)
    # Add the water mask to the results dictionary.
    results["water"] = water

    # Retrieve elevation data in kilometers if not provided as input.
    # Elevation is also derived from SRTM data.
    if elevation_km is None:
        logger.info(f"Retrieving {cl.name('elevation')} from SRTM.")
        elevation_km = SRTM_connection.elevation_km(geometry)

    # --- Section: VIIRS Data Directory Setup ---
    # Set up the base download directory for VIIRS data (both VNP09GA and VNP21A1D).
    if VIIRS_download_directory is None:
        VIIRS_download_directory = join(working_directory, VIIRS_DOWNLOAD_DIRECTORY)
    logger.info(f"VIIRS download directory: {cl.dir(VIIRS_download_directory)}")

    # Set up the output directory for processed VIIRS products.
    if VIIRS_output_directory is None:
        VIIRS_output_directory = join(working_directory, VIIRS_OUTPUT_DIRECTORY)
    logger.info(f"VIIRS output directory: {cl.dir(VIIRS_output_directory)}")

    # --- Section: Check for Previously Processed Data (for NRT context) ---
    # Check if the data for the target date and desired products has already been processed and saved.
    # This is crucial for NRT to avoid redundant computations if a run was already completed.
    VIIRS_already_processed = check_VIIRS_already_processed(
        VIIRS_output_directory=VIIRS_output_directory,
        target_date=target_date,
        time_UTC=time_UTC,
        target=target,
        products=target_variables # Check for the specific variables that will be targeted for output
    )

    # If data is already processed and 'load_previous' is True, load and return it.
    if VIIRS_already_processed:
        if load_previous:
            logger.info("Loading previously generated VIIRS GEOS-5 FP output.")
            return load_VIIRS(
                VIIRS_output_directory=VIIRS_output_directory,
                target_date=target_date,
                target=target
            )
        else:
            # If data is found but not configured to be loaded, log and return without processing.
            logger.info("VIIRS GEOS-5 FP output already exists and 'load_previous' is False. Skipping processing.")
            return {} # Return empty dictionary as current behavior if not loading

    # --- Section: GEOS-5 FP Data Connection and Availability Check ---
    # Initialize a GEOS5FP connection object if one is not already provided.
    if GEOS5FP_connection is None:
        try:
            logger.info(f"Connecting to GEOS-5 FP.")
            GEOS5FP_connection = GEOS5FP(
                working_directory=working_directory,
                download_directory=GEOS5FP_download
            )
        except Exception as e:
            # If connection to GEOS-5 FP fails, log the error and raise a custom exception.
            logger.exception(e)
            raise GEOS5FPNotAvailableError("Unable to connect to GEOS-5 FP.")

    # In an NRT context, it's critical to check if GEOS-5 FP data is available for the target time.
    # GEOS-5 FP typically has a short latency (e.g., a few hours to a day).
    latest_GEOS5FP_time = GEOS5FP_connection.latest_time_available
    logger.info(f"Latest GEOS-5 FP time available: {latest_GEOS5FP_time}")
    logger.info(f"Processing time: {time_UTC}")

    # Compare the target UTC time with the latest available GEOS-5 FP time.
    # If the target time is *after* the latest available data, NRT processing cannot proceed.
    # Using string formatting for robust comparison of datetime objects at specific precision.
    if time_UTC.strftime("%Y-%m-%d %H:%M:%S") > latest_GEOS5FP_time.strftime("%Y-%m-%d %H:%M:%S"):
        raise GEOS5FPNotAvailableError(
            f"VIIRS target time {time_UTC} is past latest available GEOS-5 FP time {latest_GEOS5FP_time}. "
            "Cannot proceed with NRT processing as data is not yet available."
        )

    # Define the actual processing date and time variables, which are the same as target_date/time_UTC here.
    VIIRS_processing_date = target_date
    VIIRS_processing_time = time_UTC

    # Initialize specific VIIRS data connections for VNP21A1D (LST) and VNP09GA (reflectance).
    VNP21_connection = VNP21A1D_002.VNP21A1D(download_directory=VIIRS_download_directory)
    VNP09_connection = VNP09GA_002.VNP09GA(download_directory=VIIRS_download_directory)

    # --- Section: Land Surface Temperature (ST_C) Processing ---
    # Retrieve or calculate Land Surface Temperature (ST_C) if not provided as input.
    if ST_C is None:
        logger.info(f"Retrieving {cl.name('VNP21A1D')} {cl.name('ST_C')} from VIIRS on {cl.time(VIIRS_processing_date)}.")
        # Retrieve Surface Temperature in Kelvin from VNP21A1D.
        ST_K = VNP21_connection.ST_K(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )
        ST_C = ST_K - 273.15 # Convert to Celsius.
        # Fill in missing ST_C values (e.g., due to clouds) with spatially smoothed GEOS-5 FP surface temperature.
        logger.info(f"Gap-filling missing {cl.name('ST_C')} with GEOS-5 FP surface temperature.")
        ST_C_smooth = GEOS5FP_connection.Ts_K(time_UTC=time_UTC, geometry=geometry, resampling="cubic") - 273.15
        ST_C = rt.where(np.isnan(ST_C), ST_C_smooth, ST_C) # Replace NaNs in ST_C with values from ST_C_smooth.
    # Store the processed Land Surface Temperature in the results dictionary.
    results["ST"] = ST_C

    # --- Section: Normalized Difference Vegetation Index (NDVI) Processing ---
    # Retrieve or calculate NDVI if not provided as input.
    if NDVI is None:
        logger.info(f"Retrieving {cl.name('VNP09GA')} {cl.name('NDVI')} from VIIRS on {cl.time(VIIRS_processing_date)}.")
        # Retrieve NDVI from VNP09GA.
        NDVI = VNP09_connection.NDVI(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )
    # Store the processed NDVI in the results dictionary.
    results["NDVI"] = NDVI

    # --- Section: Emissivity Processing ---
    # Retrieve or calculate Emissivity if not provided as input.
    if emissivity is None:
        logger.info(f"Retrieving {cl.name('VNP21A1D')} {cl.name('emissivity')} from VIIRS on {cl.time(VIIRS_processing_date)}.")
        # Retrieve Emissivity band 14 from VNP21A1D.
        emissivity = VNP21_connection.Emis_14(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )
    # Adjust emissivity for water bodies: set to a typical water value (0.96) where water mask is True.
    emissivity = rt.where(water, 0.96, emissivity)
    # Fill any remaining missing emissivity values using an empirical relationship with NDVI.
    logger.info(f"Gap-filling missing {cl.name('emissivity')} with NDVI-based empirical relationship.")
    emissivity = rt.where(np.isnan(emissivity), 1.0094 + 0.047 * np.log(NDVI), emissivity)
    # Store the processed Emissivity in the results dictionary.
    results["emissivity"] = emissivity

    # --- Section: Albedo Processing ---
    # Retrieve or calculate Albedo if not provided as input.
    if albedo is None:
        logger.info(f"Retrieving {cl.name('VNP09GA')} {cl.name('albedo')} from VIIRS on {cl.time(VIIRS_processing_date)}.")
        # Retrieve Albedo from VNP09GA.
        albedo = VNP09_connection.albedo(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )
    # Store the processed Albedo in the results dictionary.
    results["albedo"] = albedo

    # --- Section: Downscaling Preparations ---
    # Define a coarse geometry for downscaling operations. This new grid
    # will have a larger cell size (coarse_cell_size) compared to the target geometry,
    # approximating the resolution of the original GEOS-5 FP data.
    coarse_geometry = geometry.rescale(coarse_cell_size)

    # --- Section: Air Temperature (Ta_C) Processing with Downscaling ---
    # Retrieve or calculate Air Temperature (Ta_C) if not provided.
    if Ta_C is None:
        if downscale_air:
            logger.info(f"Downscaling {cl.name('air temperature')} using GEOS-5 FP and VIIRS ST.")
            ST_K = ST_C + 273.15 # Convert surface temperature to Kelvin for downscaling models.
            # Retrieve coarse resolution air temperature from GEOS-5 FP.
            Ta_K_coarse = GEOS5FP_connection.Ta_K(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic")
            # Apply downscaling to air temperature using the fine-resolution ST_K as a guide.
            Ta_K = downscale_air_temperature(
                time_UTC=time_UTC,
                Ta_K_coarse=Ta_K_coarse,
                ST_K=ST_K, # Fine-resolution surface temperature is a key input for thermal downscaling
                fine_geometry=geometry,
                coarse_geometry=coarse_geometry
            )
            Ta_C = Ta_K - 273.15 # Convert the downscaled air temperature back to Celsius.
        else:
            logger.info(f"Retrieving {cl.name('air temperature')} directly from GEOS-5 FP.")
            # If downscaling is not enabled, retrieve Ta_C directly from GEOS-5 FP at the target resolution.
            Ta_C = GEOS5FP_connection.Ta_C(time_UTC=time_UTC, geometry=geometry, resampling="cubic")
    # Store the processed Air Temperature in the results dictionary.
    results["Ta"] = Ta_C

    # --- Section: Soil Moisture (SM) Processing with Downscaling ---
    # Retrieve or calculate Soil Moisture (SM) if not provided.
    if SM is None:
        if downscale_moisture:
            logger.info(f"Downscaling {cl.name('soil moisture')} using GEOS-5 FP, VIIRS ST and NDVI.")
            ST_K = ST_C + 273.15 # Convert ST_C to Kelvin.
            # Retrieve coarse resolution soil moisture from GEOS-5 FP.
            SM_coarse = GEOS5FP_connection.SFMC(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic")
            # Also retrieve resampled GEOS-5 FP soil moisture at fine resolution; used as a reference/smooth background.
            SM_smooth = GEOS5FP_connection.SFMC(time_UTC=time_UTC, geometry=geometry, resampling="cubic")

            # Apply downscaling to soil moisture, incorporating fine-resolution ST and NDVI.
            SM = downscale_soil_moisture(
                time_UTC=time_UTC,
                fine_geometry=geometry,
                coarse_geometry=coarse_geometry,
                SM_coarse=SM_coarse,
                SM_resampled=SM_smooth, # Used as a smoothed reference
                ST_fine=ST_K, # Fine-resolution surface temperature
                NDVI_fine=NDVI, # Fine-resolution NDVI
                water=water # Water mask to exclude water bodies from soil moisture calculations
            )
        else:
            logger.info(f"Retrieving {cl.name('soil moisture')} directly from GEOS-5 FP.")
            # If downscaling is not enabled, retrieve SM directly from GEOS-5 FP at the target resolution.
            SM = GEOS5FP_connection.SFMC(time_UTC=time_UTC, geometry=geometry, resampling="cubic")
    # Store the processed Soil Moisture in the results dictionary.
    results["SM"] = SM

    # --- Section: Relative Humidity (RH) Processing with Downscaling ---
    # Retrieve or calculate Relative Humidity (RH) if not provided.
    if RH is None:
        if downscale_humidity:
            logger.info(f"Downscaling {cl.name('relative humidity')} using GEOS-5 FP, VIIRS ST and downscaled SM.")
            ST_K = ST_C + 273.15 # Convert ST_C to Kelvin.
            # Retrieve coarse resolution Vapor Pressure Deficit (VPD) from GEOS-5 FP.
            VPD_Pa_coarse = GEOS5FP_connection.VPD_Pa(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic")

            # Downscale Vapor Pressure Deficit (VPD) from coarse to fine resolution.
            VPD_Pa = downscale_vapor_pressure_deficit(
                time_UTC=time_UTC,
                VPD_Pa_coarse=VPD_Pa_coarse,
                ST_K=ST_K, # Fine-resolution ST used as a guide
                fine_geometry=geometry,
                coarse_geometry=coarse_geometry
            )
            VPD_kPa = VPD_Pa / 1000 # Convert VPD from Pascals (Pa) to kilopascals (kPa).

            # Retrieve coarse resolution Relative Humidity from GEOS-5 FP.
            RH_coarse = GEOS5FP_connection.RH(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic")

            # Apply downscaling to Relative Humidity, incorporating downscaled SM, ST_K, and VPD.
            RH = downscale_relative_humidity(
                time_UTC=time_UTC,
                RH_coarse=RH_coarse,
                SM=SM, # Downscaled soil moisture is a key input for RH downscaling
                ST_K=ST_K,
                VPD_kPa=VPD_kPa,
                water=water,
                fine_geometry=geometry,
                coarse_geometry=coarse_geometry
            )
        else:
            logger.info(f"Retrieving {cl.name('relative humidity')} directly from GEOS-5 FP.")
            # If downscaling is not enabled, retrieve RH directly from GEOS-5 FP at the target resolution.
            RH = GEOS5FP_connection.RH(time_UTC=time_UTC, geometry=geometry, resampling="cubic")
    # Store the processed Relative Humidity in the results dictionary.
    results["RH"] = RH

    # --- Section: Incoming Shortwave Radiation (SWin) and Net Radiation (Rn) ---
    # Retrieve or generate Incoming Shortwave Radiation (SWin).
    # In this NRT context, SWin is always derived from GEOS-5 FP if not provided.
    if SWin is None:
        logger.info("Generating solar radiation using GEOS-5 FP.")
        SWin = GEOS5FP_connection.SWin(time_UTC=time_UTC, geometry=geometry, resampling="cubic")
    # Store SWin in the results dictionary.
    results["SWin"] = SWin

    # Calculate Net Radiation (Rn) if not provided as input.
    # The Verma net radiation model is used for this calculation, requiring SWin, albedo, ST_C, emissivity, Ta_C, and RH.
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
        Rn = verma_results["Rn"] # Extract the net radiation component from the returned dictionary.
    # Store the calculated Net Radiation in the results dictionary.
    results["Rn"] = Rn

    # --- Section: Evapotranspiration (ET) Modeling (PT-JPL) ---
    # Run the PT-JPL (Priestley-Taylor Jet Propulsion Laboratory) Evapotranspiration model.
    logger.info(f"Running PT-JPL ET model hindcast at {cl.time(time_UTC)}.")
    PTJPL_results = PTJPL(
        NDVI=NDVI,
        ST_C=ST_C,
        emissivity=emissivity,
        albedo=albedo,
        Rn=Rn,
        Ta_C=Ta_C,
        RH=RH,
        # floor_Topt=floor_Topt # This parameter could be passed here if the PTJPL constructor/call supports it
    )

    # Add all calculated ET components from the PT-JPL model results to the main results dictionary.
    for k, v in PTJPL_results.items():
        results[k] = v

    # --- Section: Saving Processed Outputs ---
    # Iterate through all processed raster products stored in the 'results' dictionary.
    for product, image in results.items():
        # Generate the standardized output filename for each product based on target date, time, and site.
        filename = generate_VIIRS_output_filename(
            VIIRS_output_directory=VIIRS_output_directory,
            target_date=target_date,
            time_UTC=time_UTC,
            target=target,
            product=product # The specific product name (e.g., "ST", "NDVI", "ET_total")
        )

        # If a specific image result is None (i.e., data for that product couldn't be generated),
        # log a warning and skip saving for that product.
        if image is None:
            logger.warning(f"No image result for {cl.name(product)}. Skipping save operation.")
            continue

        # Log the action of writing the processed product to a file.
        logger.info(
            f"Writing VIIRS GEOS-5 FP {cl.name(product)} for {cl.place(target)} at {cl.time(time_UTC)} to file: {cl.file(filename)}."
        )
        # Save the raster object to a GeoTIFF file.
        image.to_geotiff(filename)

    # Return the dictionary containing all processed raster objects.
    return results