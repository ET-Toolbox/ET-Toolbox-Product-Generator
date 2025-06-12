from typing import Union, List, Dict, Callable
from datetime import date, datetime # For handling date and time objects
from dateutil import parser # Library for robust parsing of date strings
import logging # Standard Python logging library
from os.path import join, exists, splitext, basename, abspath, expanduser # Utilities for path manipulation
import numpy as np # Numerical computing library, essential for array operations and NaN handling
import rasters as rt # Custom or external library for handling raster (geospatial image) data
from rasters import Raster, RasterGrid # Specific classes for raster data and spatial grids

from glob import glob # For finding pathnames matching a specified pattern
import pandas as pd # Data manipulation library, used for GFS listing

# Import specific connection and processing modules for various datasets
import VNP09GA_002 # VIIRS VNP09GA (surface reflectance) data access
import VNP21A1D_002 # VIIRS VNP21A1D (land surface temperature) data access
from gedi_canopy_height import GEDICanopyHeight # GEDI Canopy Height data access (currently not used)
from GEOS5FP import GEOS5FP # GEOS-5 FP (atmospheric reanalysis) data access, used for bias correction
from global_forecasting_system import * # GFS (Global Forecast System) data access and forecast functions
from MODISCI import MODISCI # MODIS CI (Clumping Index) data access (currently not used)
from PTJPL import PTJPL # PT-JPL evapotranspiration model implementation
from soil_capacity_wilting import SoilGrids # SoilGrids data access (currently not used)

# Import downscaling and bias correction functions
from GEOS5FP.downscaling import downscale_air_temperature, downscale_soil_moisture, bias_correct
from sentinel_tiles import sentinel_tiles # Utility for defining spatial grids based on Sentinel tiles
from solar_apparent_time import solar_to_UTC # Function to convert local solar time to Coordinated Universal Time (UTC)
import colored_logging as cl # Custom logging utility for colored console output
from verma_net_radiation import process_verma_net_radiation # Function to calculate net radiation using the Verma model
from NASADEM import NASADEMConnection # NASADEM (digital elevation model) data access
from check_distribution import check_distribution # Utility to check and display data distributions (for debugging/QA)

# Import constants and helper functions from the local package
from .constants import * # Global constants (e.g., directory names, default values)
from .generate_GFS_output_directory import generate_GFS_output_directory # Helper to create GFS-related output directories
from .generate_GFS_output_filename import generate_GFS_output_filename # Helper to generate standard GFS output filenames
from .check_GFS_already_processed import check_GFS_already_processed # Helper to check if GFS data for a date is already processed
from .load_GFS import load_GFS # Helper to load previously processed GFS data from disk

# Set up logging for this specific module
logger = logging.getLogger(__name__)


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
        model: PTJPL = None, # Not directly used as PTJPL is called as a function.
        working_directory: str = None,
        static_directory: str = None,
        GFS_download: str = None,
        GFS_output_directory: str = None,
        VNP21A1D_download_directory: str = VNP21A1D_DOWNLOAD_DIRECTORY,
        VNP09GA_download_directory: str = VNP09GA_DOWNLOAD_DIRECTORY,
        SRTM_connection: NASADEMConnection = None,
        SRTM_download: str = None,
        GEOS5FP_connection: GEOS5FP = None,
        GEOS5FP_download: str = None,
        GEOS5FP_products: str = None, # Not used in the current implementation.
        GEDI_connection: GEDICanopyHeight = None, # Not used in the current implementation.
        GEDI_download: str = None, # Not used in the current implementation.
        ORNL_connection: MODISCI = None, # Not used in the current implementation.
        CI_directory: str = None, # Not used in the current implementation.
        soil_grids_connection: SoilGrids = None, # Not used in the current implementation.
        soil_grids_download: str = None, # Not used in the current implementation.
        intermediate_directory=None, # Not used in the current implementation.
        model_name: str = "PTJPL", # The name of the ET model to use. Currently only "PTJPL" is supported.
        preview_quality: int = PREVIEW_QUALITY, # Not directly used for saving previews.
        ANN_model: Callable = None, # Not used in the current implementation.
        ANN_model_filename: str = None, # Not used in the current implementation.
        spacetrack_credentials_filename: str = None, # Not used in the current implementation.
        ERS_credentials_filename: str = None, # Not used in the current implementation.
        resampling: str = RESAMPLING, # Resampling method to use for spatial operations.
        downscale_air: bool = DOWNSCALE_AIR, # Whether to downscale air temperature.
        downscale_humidity: bool = DOWNSCALE_HUMIDITY, # Whether to downscale humidity.
        downscale_moisture: bool = DOWNSCALE_MOISTURE, # Whether to downscale soil moisture.
        apply_GEOS5FP_GFS_bias_correction: bool = True, # Whether to apply bias correction to GFS using GEOS-5 FP as reference.
        VIIRS_processing_date: Union[date, str] = None, # The date to process VIIRS data. If None, it defaults to `target_date`.
        GFS_listing: pd.DataFrame = None, # A pre-loaded GFS listing (DataFrame). If None, it will be retrieved using `get_GFS_listing()`.
        save_intermediate: bool = False, # Whether to save intermediate products. Not fully implemented for all.
        include_preview: bool = True, # Whether to include previews. Not fully implemented for all.
        show_distribution: bool = True, # Whether to show distribution plots using `check_distribution`.
        load_previous: bool = True, # Whether to load previously processed GFS output if available.
        target_variables: List[str] = TARGET_VARIABLES) -> Dict[str, Raster]:
    """
    Generates a VIIRS-based forecast using GFS (Global Forecast System) meteorological data.

    This function integrates land surface parameters from VIIRS (Visible Infrared Imaging Radiometer Suite)
    with forecasted meteorological variables from GFS to produce various biophysical products,
    including land surface temperature, NDVI, emissivity, albedo, and forecasted meteorological
    variables (air temperature, relative humidity, wind speed, shortwave incoming radiation, and soil moisture).

    It can perform downscaling of GFS data to a finer resolution and apply bias correction
    using historical GEOS-5 FP data as a reference to improve forecast accuracy.
    The primary biophysical model used for evapotranspiration estimation is PT-JPL.

    Args:
        target_date (Union[date, str]): The target date for which the forecast is to be generated.
            Can be a `datetime.date` object or a string parseable by `dateutil.parser`.
        geometry (RasterGrid): The desired spatial extent and resolution for all output rasters.
        target (str): A string identifier for the target area or region (e.g., a flux tower site name),
            used for naming output files and logging.
        coarse_geometry (RasterGrid, optional): A `RasterGrid` object defining the coarser resolution
            grid for GFS data if downscaling is enabled. If None, it's derived from `target` and `coarse_cell_size`.
        coarse_cell_size (float, optional): The cell size (in degrees) for the `coarse_geometry` if it's
            not provided. Defaults to `GFS_CELL_SIZE` from constants.
        ST_C (Raster, optional): Pre-computed land surface temperature in Celsius. If None, it will be
            retrieved from VNP21A1D for `VIIRS_processing_date`.
        emissivity (Raster, optional): Pre-computed surface emissivity. If None, it will be retrieved
            from VNP21A1D for `VIIRS_processing_date`.
        NDVI (Raster, optional): Pre-computed Normalized Difference Vegetation Index. If None, it will be
            retrieved from VNP09GA for `VIIRS_processing_date`.
        albedo (Raster, optional): Pre-computed surface albedo. If None, it will be retrieved from VNP09GA
            for `VIIRS_processing_date`.
        SWin (Raster, optional): Pre-computed shortwave incoming radiation. If None, it will be
            retrieved as a forecast from GFS.
        Rn (Raster, optional): Pre-computed net radiation. If None, it will be calculated using
            `process_verma_net_radiation`.
        SM (Raster, optional): Pre-computed soil moisture. If None, it will be retrieved as a forecast from GFS.
        wind_speed (Raster, optional): Pre-computed wind speed. If None, it will be retrieved as a forecast from GFS.
        Ta_C (Raster, optional): Pre-computed air temperature in Celsius. If None, it will be retrieved
            as a forecast from GFS.
        RH (Raster, optional): Pre-computed relative humidity. If None, it will be retrieved as a forecast from GFS.
        water (Raster, optional): Pre-computed binary water mask (1 for water, 0 for land). If None, it will be
            retrieved from SRTM data.
        model (PTJPL, optional): An instance of the PTJPL model. This parameter is not directly used in the
            current implementation, as the `PTJPL` class is called as a function directly.
        working_directory (str, optional): The main working directory for temporary files and final outputs.
            Defaults to the current directory (`.`).
        static_directory (str, optional): Directory for static datasets like SRTM Digital Elevation Models.
        GFS_download (str, optional): Local directory where GFS forecast data will be downloaded.
            Defaults to `GFS_DOWNLOAD_DIRECTORY` constant.
        GFS_output_directory (str, optional): Directory to save final processed GFS-VIIRS outputs.
            Defaults to a subdirectory within `working_directory`.
        VNP21A1D_download_directory (str, optional): Local directory for downloading VIIRS VNP21A1D (LST) data.
            Defaults to `VNP21A1D_DOWNLOAD_DIRECTORY` constant.
        VNP09GA_download_directory (str, optional): Local directory for downloading VIIRS VNP09GA (reflectance) data.
            Defaults to `VNP09GA_DOWNLOAD_DIRECTORY` constant.
        SRTM_connection (NASADEMConnection, optional): An existing `NASADEMConnection` object. If None, a new one
            is created.
        SRTM_download (str, optional): Local directory for downloading SRTM data.
        GEOS5FP_connection (GEOS5FP, optional): An existing `GEOS5FP` connection object. If None, a new one is created.
            This connection is primarily used for bias correction of GFS data against historical GEOS-5 FP.
        GEOS5FP_download (str, optional): Local directory for downloading GEOS-5 FP data.
        GEOS5FP_products (str, optional): Specifies GEOS-5 FP products to download. Not used in this function.
        GEDI_connection (GEDICanopyHeight, optional): An existing `GEDICanopyHeight` connection object. Not used.
        GEDI_download (str, optional): Local directory for GEDI data downloads. Not used.
        ORNL_connection (MODISCI, optional): An existing `MODISCI` connection object. Not used.
        CI_directory (str, optional): Local directory for CI data. Not used.
        soil_grids_connection (SoilGrids, optional): An existing `SoilGrids` connection object. Not used.
        soil_grids_download (str, optional): Local directory for SoilGrids data downloads. Not used.
        intermediate_directory (str, optional): Directory for saving intermediate processing files. Not used.
        model_name (str, optional): The name of the ET model to use. Currently only "PTJPL" is explicitly
            supported and selected by specific logic within the function (e.g., for SM retrieval).
        preview_quality (int, optional): Quality setting for generating preview images. Not directly used.
        ANN_model (Callable, optional): An Artificial Neural Network model. Not used.
        ANN_model_filename (str, optional): Filename for the ANN model. Not used.
        spacetrack_credentials_filename (str, optional): Credentials for Space-Track. Not used.
        ERS_credentials_filename (str, optional): Credentials for ERS. Not used.
        resampling (str, optional): The default resampling method to use for spatial
            operations (e.g., "cubic"). Defaults to `RESAMPLING` constant.
        downscale_air (bool, optional): If True, applies downscaling to GFS air temperature data.
        downscale_humidity (bool, optional): If True, applies downscaling to GFS relative humidity data.
        downscale_moisture (bool, optional): If True, applies downscaling to GFS soil moisture data.
        apply_GEOS5FP_GFS_bias_correction (bool, optional): If True, applies a bias correction to
            GFS meteorological variables (Ta, SM, RH, wind_speed, SWin) using a difference
            between GFS and GEOS-5 FP at a historical reference time. Defaults to True.
        VIIRS_processing_date (Union[date, str], optional): The specific date to retrieve VIIRS data.
            This is useful if you want to use historical VIIRS data for a forecast period
            (e.g., using VIIRS from yesterday for a forecast today). If None, it defaults to `target_date`.
        GFS_listing (pd.DataFrame, optional): A pre-loaded pandas DataFrame containing the listing
            of available GFS forecast files. If None, the function will call `get_GFS_listing()`
            to retrieve it.
        save_intermediate (bool, optional): Whether to save intermediate products generated
            during the workflow. Not fully implemented for all intermediate steps. Defaults to False.
        include_preview (bool, optional): Whether to include preview images in the output. Not fully implemented. Defaults to True.
        show_distribution (bool, optional): Whether to show distribution plots of
            derived variables using `check_distribution`. Defaults to True.
        load_previous (bool, optional): If True, the function will first check if all
            required GFS products for the `target_date` have already been processed and saved locally.
            If so, it will load them instead of re-processing, speeding up subsequent runs.
        target_variables (List[str], optional): A list of specific output variable names
            (e.g., "ST", "NDVI", "ET_total") that should be processed and saved.
            Defaults to `TARGET_VARIABLES` constant.

    Returns:
        Dict[str, Raster]: A dictionary where keys are variable names (e.g.,
            "ST", "NDVI", "ET_total") and values are the corresponding
            Raster objects (geospatial images) that have been processed.

    Raises:
        ValueError: If the `target_date` for the forecast is before the earliest available GFS date.
        Exception: If unable to connect to GEOS-5 FP (required for bias correction).
    """
    results = {}

    # --- Section: Date and Time Setup ---
    # Parse the target_date to a datetime.date object if it's provided as a string.
    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()
    logger.info(f"GFS-VIIRS target date: {cl.time(target_date)}")

    # Define the local solar time for the forecast. It's often set to a fixed
    # time like 13:30 local solar time, typically around local solar noon,
    # suitable for capturing peak daytime conditions.
    time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
    logger.info(f"GFS-VIIRS target time solar: {cl.time(time_solar)}")

    # Convert the local solar time to Coordinated Universal Time (UTC) based on the
    # longitude of the geometry's centroid. This is essential for accurately querying
    # global datasets like GFS and VIIRS, which are indexed by UTC.
    time_UTC = solar_to_UTC(time_solar, geometry.centroid.latlon.x)
    logger.info(f"GFS-VIIRS target time UTC: {cl.time(time_UTC)}")
    date_UTC = time_UTC.date() # Extract just the date part of the UTC time.

    # Parse `VIIRS_processing_date` if it's a string. This date indicates when VIIRS data
    # (which are observational) should be retrieved. For a forecast, this might be a past date.
    if isinstance(VIIRS_processing_date, str):
        VIIRS_processing_date = parser.parse(VIIRS_processing_date).date()

    # --- Section: Directory Setup ---
    # Set up the main working directory. If not provided, use the current directory.
    if working_directory is None:
        working_directory = "."
    # Resolve to an absolute path and expand user directory (e.g., '~' to '/home/user').
    working_directory = abspath(expanduser(working_directory))
    logger.info(f"GFS-VIIRS working directory: {cl.dir(working_directory)}")

    # Set up the GFS data download directory.
    if GFS_download is None:
        GFS_download = GFS_DOWNLOAD_DIRECTORY # Use a constant from the local package.
    logger.info(f"GFS download directory: {cl.dir(GFS_download)}")

    # Set up the GFS forecast output directory.
    if GFS_output_directory is None:
        GFS_output_directory = join(working_directory, GFS_OUTPUT_DIRECTORY) # Use a constant from the local package.
    logger.info(f"GFS output directory: {cl.dir(GFS_output_directory)}")

    # --- Section: External Data Connections Initialization ---
    # Initialize a NASADEMConnection object for SRTM data if one is not already provided.
    # SRTM (Shuttle Radar Topography Mission) data is used for elevation and water masks.
    if SRTM_connection is None:
        SRTM_connection = NASADEMConnection(
            download_directory=SRTM_download,
            # offline_ok=True # This parameter might be relevant here depending on NASADEMConnection's implementation.
        )
    # Retrieve water mask if not provided.
    # Note: This is re-called later. A more efficient approach might be to store `water` in `results` and only call once.
    if water is None:
        logger.info(f"Retrieving water mask from SRTM.")
        water = SRTM_connection.swb(geometry)

    # Initialize a GEOS5FP connection object if one is not already provided.
    # GEOS-5 FP data is used as a historical reference for bias correction of GFS forecasts.
    if GEOS5FP_connection is None:
        try:
            logger.info(f"Connecting to GEOS-5 FP for bias correction reference.")
            GEOS5FP_connection = GEOS5FP(
                working_directory=working_directory,
                download_directory=GEOS5FP_download
            )
        except Exception as e:
            logger.exception(e)
            raise Exception("Unable to connect to GEOS-5 FP. This is required for GFS bias correction.")

    # --- Section: Check for Previously Processed GFS Forecasts ---
    # Check if the GFS forecast for the target date and products has already been processed and saved.
    GFS_already_processed = check_GFS_already_processed(
        GFS_output_directory=GFS_output_directory,
        target_date=target_date,
        time_UTC=time_UTC,
        target=target,
        products=target_variables
    )

    # If the forecast is already processed and `load_previous` is True, load and return the results.
    if GFS_already_processed:
        if load_previous:
            logger.info("Loading previously generated VIIRS GFS forecast output.")
            return load_GFS(
                GFS_output_directory=GFS_output_directory,
                target_date=target_date,
                target=target
            )
        else:
            logger.info("Previous GFS forecast output found, but 'load_previous' is False. Skipping reprocessing.")
            return {}  # Return empty dictionary if not loading previous

    # --- Section: GFS Data Availability Check ---
    # Pull the GFS listing to determine the range of available forecast dates.
    logger.info("Pulling GFS listing to check forecast availability.")
    if GFS_listing is None:
        GFS_listing = get_GFS_listing() # This function retrieves a DataFrame of GFS forecast availability.
    earliest_GFS_date = global_forecasting_system.earliest_date_UTC(listing=GFS_listing)

    # Validate the target date against the earliest available GFS forecast.
    # A forecast cannot be generated for a date before GFS data starts.
    if target_date < earliest_GFS_date:
        raise ValueError(
            f"Target date {cl.time(target_date)} is before earliest GFS date {cl.time(earliest_GFS_date)}. "
            f"Cannot generate forecast for this date."
        )

    # --- Section: VIIRS Processing Date and Forecast Distance ---
    # Set the actual date for VIIRS data retrieval. If not specified, it defaults to the `target_date`.
    if VIIRS_processing_date is None:
        VIIRS_processing_date = target_date
    
    # Define the solar and UTC time for VIIRS data acquisition on the VIIRS processing date.
    VIIRS_processing_datetime_solar = datetime(VIIRS_processing_date.year, VIIRS_processing_date.month,
                                               VIIRS_processing_date.day, 13, 30)
    logger.info(f"VIIRS processing date/time solar: {cl.time(VIIRS_processing_datetime_solar)}")
    VIIRS_processing_datetime_UTC = solar_to_UTC(VIIRS_processing_datetime_solar, geometry.centroid.latlon.x)
    logger.info(f"VIIRS processing date/time UTC: {cl.time(VIIRS_processing_datetime_UTC)}")

    # Calculate the forecast distance (how many days into the future `target_date` is from `VIIRS_processing_date`).
    forecast_distance_days = (target_date - VIIRS_processing_date).days
    if forecast_distance_days > 0:
        logger.info(
            f"Target forecast date {cl.time(target_date)} is {cl.val(forecast_distance_days)} days past "
            f"VIIRS observation date {cl.time(VIIRS_processing_date)}. Using VIIRS for hindcast."
        )

    # --- Section: VIIRS Data Connections ---
    # Initialize VNP21A1D (Land Surface Temperature) and VNP09GA (Surface Reflectance) connections.
    VNP21_connection = VNP21A1D_002.VNP21A1D(download_directory=VNP21A1D_download_directory)
    VNP09_connection = VNP09GA_002.VNP09GA(download_directory=VNP09GA_download_directory)

    # --- Section: VIIRS Parameter Retrieval (ST_C, NDVI, Emissivity, Albedo) ---
    # Retrieve or use provided Land Surface Temperature (ST_C) from VIIRS.
    if ST_C is None:
        logger.info(
            f"Retrieving {cl.name('VNP21A1D')} {cl.name('ST_C')} from VIIRS on {cl.time(VIIRS_processing_date)}."
        )
        ST_K = VNP21_connection.ST_K(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )
        ST_C = ST_K - 273.15 # Convert from Kelvin to Celsius.

        # Fill any remaining NaN values in ST_C with a smoothed GEOS-5 FP surface temperature.
        # This provides a broad fill if VIIRS data is entirely missing.
        logger.info(f"Gap-filling missing {cl.name('ST_C')} with GEOS-5 FP surface temperature.")
        ST_C_smooth = GEOS5FP_connection.Ts_K(time_UTC=VIIRS_processing_datetime_UTC, geometry=geometry, resampling="cubic") - 273.15
        ST_C = rt.where(np.isnan(ST_C), ST_C_smooth, ST_C)
    results["ST"] = ST_C

    # Retrieve or use provided NDVI from VIIRS.
    if NDVI is None:
        logger.info(
            f"Retrieving {cl.name('VNP09GA')} {cl.name('NDVI')} from VIIRS on {cl.time(VIIRS_processing_date)}."
        )
        NDVI = VNP09_connection.NDVI(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )
    results["NDVI"] = NDVI

    # Retrieve or use provided Emissivity from VIIRS.
    if emissivity is None:
        logger.info(
            f"Retrieving {cl.name('VNP21A1D')} {cl.name('emissivity')} from VIIRS on {cl.time(VIIRS_processing_date)}."
        )
        emissivity = VNP21_connection.Emis_14(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )
    # Adjust emissivity for water bodies (set to a typical water value 0.96) and fill NaNs
    # using an empirical relationship with NDVI.
    emissivity = rt.where(water, 0.96, emissivity)
    emissivity = rt.where(np.isnan(emissivity), 1.0094 + 0.047 * np.log(NDVI), emissivity)
    results["emissivity"] = emissivity

    # Retrieve or use provided Albedo from VIIRS.
    if albedo is None:
        logger.info(
            f"Retrieving {cl.name('VNP09GA')} {cl.name('albedo')} from VIIRS on {cl.time(VIIRS_processing_date)}."
        )
        albedo = VNP09_connection.albedo(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )
    results["albedo"] = albedo

    # --- Section: SRTM Data (Water Mask) Re-check and Coarse Geometry Definition ---
    # Re-initialize SRTM connection if not provided. (This check is redundant if done at the top, but harmless.)
    if SRTM_connection is None:
        logger.info("Connecting to SRTM (redundant check).")
        SRTM_connection = NASADEMConnection(
            working_directory=static_directory,
            download_directory=SRTM_download,
            offline_ok=True
        )
    logger.info("Retrieving water mask from SRTM (ensuring availability).")
    # Retrieve water mask from SRTM. (This call ensures `water` is indeed a RasterGrid at this point,
    # even if it was pre-loaded or retrieved earlier but not assigned to the correct object.)
    water = SRTM_connection.swb(geometry)
    results["water"] = water # Ensure water mask is part of the results dictionary.

    logger.info(f"Running PT-JPL-SM ET model forecast at {cl.time(time_UTC)}") # This log message seems to be placed prematurely.

    # Define coarse geometry if not provided. This grid is used for retrieving raw GFS data
    # before downscaling to the fine `geometry`.
    if coarse_geometry is None:
        coarse_geometry = sentinel_tiles.grid(target, coarse_cell_size) # Use a helper to define a grid.

    # --- Section: GFS Meteorological Variable Retrieval and Processing ---

    # Retrieve or use provided Air Temperature (Ta_C) from GFS.
    if Ta_C is None:
        logger.info(f"Retrieving GFS {cl.name('Ta')} forecast at {cl.time(time_UTC)}.")
        # Determine if downscaling is required for Ta.
        if downscale_air:
            Ta_K_coarse = forecast_Ta_K(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic",
                                        listing=GFS_listing)

            # Apply GEOS-5 FP GFS bias correction for air temperature.
            # This aims to correct systematic differences between GFS forecasts and a more
            # accurate reanalysis product (GEOS-5 FP) using historical data.
            if apply_GEOS5FP_GFS_bias_correction:
                logger.info("Applying GEOS-5 FP bias correction for air temperature.")
                # Get GFS data at the VIIRS processing date/time (reference time).
                matching_Ta_K_GFS = forecast_Ta_K(
                    time_UTC=VIIRS_processing_datetime_UTC,
                    geometry=coarse_geometry,
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )
                # Get GEOS-5 FP data at the same reference time.
                matching_Ta_K_GEOS5FP = GEOS5FP_connection.Ta_K(
                    time_UTC=VIIRS_processing_datetime_UTC,
                    geometry=coarse_geometry,
                    resampling="cubic"
                )
                # Calculate the bias: GFS - GEOS-5 FP.
                Ta_K_GFS_bias = matching_Ta_K_GFS - matching_Ta_K_GEOS5FP
                # Apply the bias to the current GFS forecast.
                Ta_K_coarse = Ta_K_coarse - Ta_K_GFS_bias

            ST_K = ST_C + 273.15 # Convert ST_C to Kelvin for downscaling input.

            # Downscale air temperature from coarse GFS to fine target geometry using ST_K.
            Ta_K = downscale_air_temperature(
                time_UTC=time_UTC,
                Ta_K_coarse=Ta_K_coarse,
                ST_K=ST_K,
                fine_geometry=geometry,
                coarse_geometry=coarse_geometry
            )
            Ta_C = Ta_K - 273.15 # Convert back to Celsius.
        else:
            # If downscaling is not enabled.
            if apply_GEOS5FP_GFS_bias_correction:
                logger.info("Applying GEOS-5 FP bias correction for air temperature (no downscaling).")
                # Get GFS data at the VIIRS processing date/time (reference time).
                matching_Ta_C_GFS = forecast_Ta_C(
                    time_UTC=VIIRS_processing_datetime_UTC,
                    geometry=coarse_geometry, # Use coarse geometry for bias calculation
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )
                # Get GEOS-5 FP data at the same reference time.
                matching_Ta_C_GEOS5FP = GEOS5FP_connection.Ta_C(
                    time_UTC=VIIRS_processing_datetime_UTC,
                    geometry=coarse_geometry,
                    resampling="cubic"
                )
                # Calculate bias.
                Ta_C_GFS_bias = matching_Ta_C_GFS - matching_Ta_C_GEOS5FP

                # Retrieve current GFS forecast at coarse resolution.
                Ta_C_coarse = forecast_Ta_C(
                    time_UTC=time_UTC,
                    geometry=coarse_geometry, # Retrieve at coarse res
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )
                # Apply bias and then resample to fine geometry.
                Ta_C_coarse = Ta_C_coarse - Ta_C_GFS_bias
                Ta_C = Ta_C_coarse.to_geometry(geometry, resampling="cubic")
            else:
                # Retrieve Ta_C directly from GFS at the target resolution without bias correction or downscaling.
                logger.info(f"Retrieving {cl.name('Ta')} directly from GFS.")
                Ta_C = forecast_Ta_C(
                    time_UTC=time_UTC,
                    geometry=geometry,
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )
    results["Ta"] = Ta_C

    # Retrieve or use provided Soil Moisture (SM) from GFS (only if model_name is "PTJPL").
    # The `model_name` check implies that SM might only be needed for this specific ET model.
    if SM is None and model_name == "PTJPL":
        logger.info(f"Retrieving GFS {cl.name('SM')} forecast at {cl.time(time_UTC)} for PTJPL.")

        if downscale_moisture:
            SM_coarse = forecast_SM(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic", listing=GFS_listing)

            # Apply GEOS-5 FP GFS bias correction for soil moisture.
            if apply_GEOS5FP_GFS_bias_correction:
                logger.info("Applying GEOS-5 FP bias correction for soil moisture.")
                # Get GFS data at reference time.
                matching_SM_GFS = forecast_SM(
                    time_UTC=VIIRS_processing_datetime_UTC,
                    geometry=coarse_geometry,
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )
                # Get GEOS-5 FP data at reference time.
                matching_SM_GEOS5FP = GEOS5FP_connection.SFMC(
                    time_UTC=VIIRS_processing_datetime_UTC,
                    geometry=coarse_geometry,
                    resampling="cubic"
                )
                # Calculate bias.
                SM_GFS_bias = matching_SM_GFS - matching_SM_GEOS5FP

                # Apply bias to the current GFS forecast at coarse resolution.
                SM_coarse = forecast_SM(
                    time_UTC=time_UTC,
                    geometry=coarse_geometry,
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )
                SM_coarse = SM_coarse - SM_GFS_bias

            ST_K = ST_C + 273.15
            # Downscale soil moisture using the fine-resolution ST and NDVI.
            SM = downscale_soil_moisture(
                time_UTC=time_UTC,
                fine_geometry=geometry,
                coarse_geometry=coarse_geometry,
                SM_coarse=SM_coarse,
                # SM_resampled: This is a smooth, resampled version of GFS SM at the fine resolution.
                SM_resampled=forecast_SM(time_UTC=time_UTC, geometry=geometry, resampling="cubic", listing=GFS_listing),
                ST_fine=ST_K,
                NDVI_fine=NDVI,
                water=water
            )
        else:
            # If downscaling is not enabled.
            if apply_GEOS5FP_GFS_bias_correction:
                logger.info("Applying GEOS-5 FP bias correction for soil moisture (no downscaling).")
                # Get GFS data at reference time.
                matching_SM_GFS = forecast_SM(
                    time_UTC=VIIRS_processing_datetime_UTC,
                    geometry=coarse_geometry,
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )
                # Get GEOS-5 FP data at reference time.
                matching_SM_GEOS5FP = GEOS5FP_connection.SFMC(
                    time_UTC=VIIRS_processing_datetime_UTC,
                    geometry=coarse_geometry,
                    resampling="cubic"
                )
                # Calculate bias.
                SM_GFS_bias = matching_SM_GFS - matching_SM_GEOS5FP

                # Retrieve current GFS forecast at coarse resolution.
                SM_coarse = forecast_SM(
                    time_UTC=time_UTC,
                    geometry=coarse_geometry,
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )
                # Apply bias and then resample to fine geometry.
                SM_coarse = SM_coarse - SM_GFS_bias
                SM = SM_coarse.to_geometry(geometry, resampling="cubic")
            else:
                # Retrieve SM directly from GFS without bias correction or downscaling.
                logger.info(f"Retrieving {cl.name('SM')} directly from GFS.")
                SM = forecast_SM(
                    time_UTC=time_UTC,
                    geometry=geometry,
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )
    results["SM"] = SM

    # Retrieve or use provided Relative Humidity (RH) from GFS.
    if RH is None:
        logger.info(f"Retrieving GFS {cl.name('RH')} forecast at {cl.time(time_UTC)}.")

        if downscale_humidity:
            # Calculate saturation vapor pressure (SVP) in Pascals based on Ta_C.
            SVP_Pa = 0.6108 * np.exp((17.27 * Ta_C) / (Ta_C + 237.3)) * 1000  # [Pa]
            # Retrieve a smoothed RH from GFS at fine resolution.
            RH_smooth = forecast_RH(time_UTC=time_UTC, geometry=geometry, resampling="cubic", listing=GFS_listing)
            # Estimate actual vapor pressure and vapor pressure deficit.
            Ea_Pa_estimate = RH_smooth * SVP_Pa
            VPD_Pa_estimate = SVP_Pa - Ea_Pa_estimate
            VPD_kPa_estimate = VPD_Pa_estimate / 1000
            # Custom RH estimation: This line `RH_estimate = SM ** (1 / VPD_kPa_estimate)` seems like a custom
            # empirical or simplified physical relationship for RH based on SM and VPD, potentially for downscaling.
            RH_estimate = SM ** (1 / VPD_kPa_estimate) # This line might need review if it's a specific model equation.

            # Retrieve coarse RH from GFS.
            RH_coarse = forecast_RH(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic",
                                    listing=GFS_listing)

            # Apply GEOS-5 FP GFS bias correction for humidity.
            if apply_GEOS5FP_GFS_bias_correction:
                logger.info("Applying GEOS-5 FP bias correction for relative humidity.")
                # Get GFS data at reference time.
                matching_RH_GFS = forecast_RH(
                    time_UTC=VIIRS_processing_datetime_UTC,
                    geometry=coarse_geometry,
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )
                # Get GEOS-5 FP data at reference time.
                matching_RH_GEOS5FP = GEOS5FP_connection.RH(
                    time_UTC=VIIRS_processing_datetime_UTC,
                    geometry=coarse_geometry,
                    resampling="cubic"
                )
                # Calculate bias.
                RH_GFS_bias = matching_RH_GFS - matching_RH_GEOS5FP
                # Apply bias to the coarse RH.
                RH_coarse = RH_coarse - RH_GFS_bias

            # Bias correct humidity using a more general `bias_correct` function,
            # which likely combines coarse and fine-scale information.
            RH = bias_correct(
                coarse_image=RH_coarse,
                fine_image=RH_estimate,
                upsampling="average", # Method for upsampling during bias correction
                downsampling="linear", # Method for downsampling during bias correction
                return_bias=False
            )
        else:
            # If downscaling is not enabled.
            if apply_GEOS5FP_GFS_bias_correction:
                logger.info("Applying GEOS-5 FP bias correction for relative humidity (no downscaling).")
                # Get GFS data at reference time.
                matching_RH_GFS = forecast_RH(
                    time_UTC=VIIRS_processing_datetime_UTC,
                    geometry=coarse_geometry,
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )
                # Get GEOS-5 FP data at reference time.
                matching_RH_GEOS5FP = GEOS5FP_connection.RH(
                    time_UTC=VIIRS_processing_datetime_UTC,
                    geometry=coarse_geometry,
                    resampling="cubic"
                )
                # Calculate bias.
                RH_GFS_bias = matching_RH_GFS - matching_RH_GEOS5FP

                # Retrieve current GFS forecast at coarse resolution.
                RH_coarse = forecast_RH(
                    time_UTC=time_UTC,
                    geometry=coarse_geometry,
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )
                # Apply bias and then resample to fine geometry.
                RH_coarse = RH_coarse - RH_GFS_bias
                RH = RH_coarse.to_geometry(geometry, resampling="cubic")
            else:
                # Retrieve RH directly from GFS without bias correction or downscaling.
                logger.info(f"Retrieving {cl.name('RH')} directly from GFS.")
                RH = forecast_RH(time_UTC=time_UTC, geometry=geometry, directory=GFS_download, listing=GFS_listing)

    # Check and display distribution of RH (for quality assurance/debugging).
    if show_distribution:
        check_distribution(RH, "RH", date_UTC=date_UTC, target=target)
    results["RH"] = RH

    # Retrieve or use provided Wind Speed from GFS.
    if wind_speed is None:
        logger.info(f"Retrieving GFS {cl.name('wind_speed')} forecast at {cl.time(time_UTC)}.")

        # Apply GEOS-5 FP GFS bias correction for wind speed.
        if apply_GEOS5FP_GFS_bias_correction:
            logger.info("Applying GEOS-5 FP bias correction for wind speed.")
            # Get GFS data at reference time.
            matching_wind_speed_GFS = forecast_wind(
                time_UTC=VIIRS_processing_datetime_UTC,
                geometry=coarse_geometry,
                directory=GFS_download,
                resampling="cubic",
                listing=GFS_listing
            )
            # Get GEOS-5 FP data at reference time.
            matching_wind_speed_GEOS5FP = GEOS5FP_connection.wind_speed(
                time_UTC=VIIRS_processing_datetime_UTC,
                geometry=coarse_geometry,
                resampling="cubic"
            )
            # Calculate bias.
            wind_speed_GFS_bias = matching_wind_speed_GFS - matching_wind_speed_GEOS5FP

            # Retrieve current GFS forecast at coarse resolution.
            wind_speed_coarse = forecast_wind(
                time_UTC=time_UTC,
                geometry=coarse_geometry,
                directory=GFS_download,
                resampling="cubic",
                listing=GFS_listing
            )
            # Apply bias and then resample to fine geometry.
            wind_speed_coarse = wind_speed_coarse - wind_speed_GFS_bias
            wind_speed = wind_speed_coarse.to_geometry(geometry, resampling="cubic")
        else:
            # Retrieve wind speed directly from GFS without bias correction.
            logger.info(f"Retrieving {cl.name('wind_speed')} directly from GFS.")
            wind_speed = forecast_wind(
                time_UTC=time_UTC,
                geometry=geometry,
                directory=GFS_download,
                resampling="cubic",
                listing=GFS_listing
            )
    results["wind_speed"] = wind_speed

    # Retrieve or use provided Shortwave Incoming Radiation (SWin) from GFS.
    if SWin is None:
        logger.info(f"Retrieving GFS {cl.name('SWin')} forecast at {cl.time(time_UTC)}.")

        # Apply GEOS-5 FP GFS bias correction for SWin.
        if apply_GEOS5FP_GFS_bias_correction:
            logger.info("Applying GEOS-5 FP bias correction for Shortwave Incoming Radiation.")
            # Get GFS data at reference time.
            matching_SWin_GFS = forecast_SWin(
                time_UTC=VIIRS_processing_datetime_UTC,
                geometry=coarse_geometry,
                directory=GFS_download,
                resampling="cubic",
                listing=GFS_listing
            )
            # Get GEOS-5 FP data at reference time.
            matching_SWin_GEOS5FP = GEOS5FP_connection.SWin(
                time_UTC=VIIRS_processing_datetime_UTC,
                geometry=coarse_geometry,
                resampling="cubic"
            )
            # Calculate bias.
            SWin_GFS_bias = matching_SWin_GFS - matching_SWin_GEOS5FP

            # Retrieve current GFS forecast at coarse resolution.
            SWin_coarse = forecast_SWin(
                time_UTC=time_UTC,
                geometry=coarse_geometry,
                directory=GFS_download,
                resampling="cubic",
                listing=GFS_listing
            )
            # Apply bias and then resample to fine geometry.
            SWin_coarse = SWin_coarse - SWin_GFS_bias
            SWin = SWin_coarse.to_geometry(geometry, resampling="cubic")
        else:
            # Retrieve SWin directly from GFS without bias correction.
            logger.info(f"Retrieving {cl.name('SWin')} directly from GFS.")
            SWin = forecast_SWin(time_UTC=time_UTC, geometry=geometry, directory=GFS_download, listing=GFS_listing)
    results["SWin"] = SWin

    # --- Section: Net Radiation (Rn) Calculation ---
    # Process Net Radiation using the Verma model. Rn is derived from SWin, albedo, ST_C, emissivity, Ta_C, and RH.
    logger.info(f"Calculating {cl.name('net radiation')} using Verma model.")
    verma_results = process_verma_net_radiation(
        SWin=SWin,
        albedo=albedo,
        ST_C=ST_C,
        emissivity=emissivity,
        Ta_C=Ta_C,
        RH=RH
    )
    Rn = verma_results["Rn"] # Assign Rn from Verma results to the local variable.
    results["Rn"] = Rn # Store Rn in the results dictionary.

    # --- Section: Evapotranspiration (ET) Modeling (PT-JPL) ---
    logger.info(f"Running PT-JPL ET model forecast at {cl.time(time_UTC)}.")

    # Run the PT-JPL ET model using the derived and forecasted inputs.
    PTJPL_results = PTJPL(
        NDVI=NDVI,
        ST_C=ST_C,
        emissivity=emissivity,
        albedo=albedo,
        Rn=Rn,
        Ta_C=Ta_C,
        RH=RH,
        # Potentially pass floor_Topt here if PTJPL supports it: floor_Topt=floor_Topt
    )

    # Add all calculated PT-JPL results (e.g., various ET components) to the main results dictionary.
    for k, v in PTJPL_results.items():
        results[k] = v

    # --- Section: Saving Processed Outputs ---
    # Save all processed raster results to GeoTIFF files.
    for product, image in results.items():
        # Generate the standardized output filename for each product based on target date, time, and site.
        filename = generate_GFS_output_filename(
            GFS_output_directory=GFS_output_directory,
            target_date=target_date,
            time_UTC=time_UTC,
            target=target,
            product=product
        )

        # If a specific image result is None, log a warning and skip saving for that product.
        if image is None:
            logger.warning(f"No image result for {cl.name(product)}. Skipping save operation.")
            continue

        # Log the action of writing the processed product to a file.
        logger.info(
            f"Writing VIIRS GFS {cl.name(product)} at {cl.place(target)} at {cl.time(time_UTC)} to file: {cl.file(filename)}."
        )
        # Save the raster object to a GeoTIFF file.
        image.to_geotiff(filename)

    # Return the dictionary containing all processed raster objects.
    return results