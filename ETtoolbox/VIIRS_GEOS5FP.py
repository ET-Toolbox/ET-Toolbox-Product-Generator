from typing import Union, List, Dict, Callable
from datetime import date, datetime, timedelta
from dateutil import parser # Library for parsing dates from various string formats
from os.path import join, exists, basename, abspath, expanduser, splitext # Path manipulation utilities
import rasters as rt # Custom or external library for handling raster (geospatial image) data
import numpy as np # Numerical computing library, essential for array operations and NaN handling

from glob import glob # For finding pathnames matching a specified pattern

# Import specific connection and processing modules for various datasets
from gedi_canopy_height import GEDICanopyHeight # GEDI Canopy Height data access
from GEOS5FP import GEOS5FP # GEOS-5 FP (atmospheric reanalysis) data access

from MODISCI import MODISCI # MODIS CI (clumping Index) data access
from PTJPL import PTJPL # PT-JPL evapotranspiration model implementation
from soil_capacity_wilting import SoilGrids # SoilGrids data access

# Import downscaling and bias correction functions specifically for GEOS-5 FP data
from GEOS5FP.downscaling import downscale_air_temperature, downscale_soil_moisture, \
    downscale_vapor_pressure_deficit, downscale_relative_humidity, bias_correct
from PTJPL import FLOOR_TOPT # A constant from the PT-JPL module, likely related to an optimization temperature floor

import logging # Standard Python logging library for tracking events
import colored_logging as cl # Custom logging utility for colored console output, enhancing readability

from solar_apparent_time import solar_to_UTC # Function to convert local solar time to Coordinated Universal Time (UTC)

from verma_net_radiation import process_verma_net_radiation # Function to calculate net radiation using the Verma model

# Import specific connection modules for VIIRS products
from VNP09GA_002 import VNP09GA # VIIRS VNP09GA (surface reflectance) data access
from VNP21A1D_002 import VNP21A1D # VIIRS VNP21A1D (land surface temperature) data access

from NASADEM import NASADEMConnection # NASADEM (digital elevation model) data access

# Import constants and helper functions from the local package
from .constants import * # Global constants used across the package
from .generate_VIIRS_GEOS5FP_output_directory import generate_VIIRS_GEOS5FP_output_directory # Helper to create output directories
from .generate_VIIRS_GEOS5FP_output_filename import generate_VIIRS_GEOS5FP_output_filename # Helper to generate output filenames
from .check_VIIRS_GEOS5FP_already_processed import check_VIIRS_GEOS5FP_already_processed # Helper to check if data is already processed
from .load_VIIRS_GEOS5FP import load_VIIRS_GEOS5FP # Helper to load previously processed data

# Set up logging for this specific module
logger = logging.getLogger(__name__)

class GEOS5FPNotAvailableError(Exception):
    """
    Custom exception raised when GEOS-5 FP data cannot be accessed
    or if the target time is beyond the latest available data.
    """
    pass

def VIIRS_GEOS5FP(
        target_date: Union[date, str],
        geometry: rt.RasterGrid,
        target: str,
        ST_C: rt.Raster = None,
        emissivity: rt.Raster = None,
        NDVI: rt.Raster = None,
        albedo: rt.Raster = None,
        SWin: Union[rt.Raster, str] = None,
        Rn: Union[rt.Raster, str] = None,
        SM: Union[rt.Raster, str] = None,
        wind_speed: rt.Raster = None, # Not currently used, reserved for future extensions
        Ta_C: Union[rt.Raster, str] = None,
        RH: Union[rt.Raster, str] = None,
        water: rt.Raster = None,
        elevation_km: rt.Raster = None,
        ET_model_name: str = ET_MODEL_NAME, # Not currently used, likely for future model integration
        working_directory: str = None,
        static_directory: str = None,
        VNP09GA_download_directory: str = None,
        VNP21A1D_download_directory: str = None,
        use_VIIRS_composite: bool = USE_VIIRS_COMPOSITE,
        VIIRS_composite_days: int = VIIRS_COMPOSITE_DAYS,
        VIIRS_GEOS5FP_output_directory: str = None,
        SRTM_connection: NASADEMConnection = None,
        SRTM_download: str = None,
        GEOS5FP_connection: GEOS5FP = None,
        GEOS5FP_download: str = None,
        GEOS5FP_products: str = None, # Not currently used, reserved for future product selection
        GEOS5FP_offline_processing: bool = True,
        GEDI_connection: GEDICanopyHeight = None, # Not currently used, reserved for future GEDI integration
        GEDI_download: str = None, # Not currently used
        ORNL_connection: MODISCI = None, # Not currently used, reserved for future MODIS CI integration
        CI_directory: str = None, # Not currently used
        soil_grids_connection: SoilGrids = None, # Not currently used, reserved for future SoilGrids integration
        soil_grids_download: str = None, # Not currently used
        intermediate_directory: str = None, # Not currently used, reserved for saving intermediate outputs
        preview_quality: int = PREVIEW_QUALITY, # Not currently used, likely for preview image generation
        ANN_model: Callable = None, # Not currently used, reserved for future ANN model integration
        ANN_model_filename: str = None, # Not currently used
        resampling: str = RESAMPLING, # Not currently used for all resampling operations, but can be a general setting
        coarse_cell_size: float = COARSE_CELL_SIZE,
        downscale_air: bool = DOWNSCALE_AIR,
        downscale_humidity: bool = DOWNSCALE_HUMIDITY,
        downscale_moisture: bool = DOWNSCALE_MOISTURE,
        floor_Topt: bool = FLOOR_TOPT, # Not currently used directly here, likely passed to PT-JPL internally
        save_intermediate: bool = False, # Not currently used, reserved for saving intermediate steps
        include_preview: bool = True, # Not currently used, likely for preview image generation
        show_distribution: bool = True, # Not currently used, likely for data distribution visualization
        load_previous: bool = True,
        target_variables: List[str] = TARGET_VARIABLES) -> Dict[str, rt.Raster]:
    """
    Processes VIIRS and GEOS-5 FP data to derive various biophysical parameters,
    including land surface temperature, NDVI, emissivity, albedo, air temperature,
    soil moisture, relative humidity, and evapotranspiration components.

    The function handles data retrieval, gap-filling, downscaling, and output
    generation, providing a comprehensive workflow for integrating VIIRS and
    GEOS-5 FP datasets for a specific target date and spatial extent.

    Args:
        target_date (Union[date, str]): The target date for processing. Can be a
            datetime.date object or a string parseable by dateutil.parser.
        geometry (rt.RasterGrid): The spatial extent and resolution for the output
            rasters. All generated rasters will conform to this grid.
        target (str): A string identifying the target area or site (e.g., a flux tower site name),
            used for naming output files and logging.
        ST_C (rt.Raster, optional): Pre-computed land surface temperature in
            Celsius. If provided, the function will use this instead of retrieving
            from VNP21A1D. Defaults to None.
        emissivity (rt.Raster, optional): Pre-computed surface emissivity. If provided,
            it will be used directly. Otherwise, it will be derived from NDVI. Defaults to None.
        NDVI (rt.Raster, optional): Pre-computed Normalized Difference Vegetation
            Index. If provided, it will be used directly. Otherwise, it will be retrieved
            from VNP09GA. Defaults to None.
        albedo (rt.Raster, optional): Pre-computed surface albedo. If provided, it
            will be used directly. Otherwise, it will be retrieved from VNP09GA. Defaults to None.
        SWin (Union[rt.Raster, str], optional): Pre-computed incoming shortwave
            radiation. If None or the string 'GEOS5FP', it will be retrieved from GEOS-5 FP.
            Defaults to None.
        Rn (Union[rt.Raster, str], optional): Pre-computed net radiation. If None,
            it will be calculated using the Verma net radiation model. Defaults to None.
        SM (Union[rt.Raster, str], optional): Pre-computed soil moisture. If None,
            it will be retrieved and potentially downscaled from GEOS-5 FP.
            Defaults to None.
        wind_speed (rt.Raster, optional): Wind speed data. This parameter is currently
            not used in the processing, but is kept for potential future extensions. Defaults to None.
        Ta_C (Union[rt.Raster, str], optional): Pre-computed air temperature in
            Celsius. If None, it will be retrieved and potentially downscaled
            from GEOS-5 FP. Defaults to None.
        RH (Union[rt.Raster, str], optional): Pre-computed relative humidity. If None,
            it will be retrieved and potentially downscaled from GEOS-5 FP.
            Defaults to None.
        water (rt.Raster, optional): A binary raster indicating water bodies (1)
            or land (0). If None, it will be derived from SRTM data. Defaults to None.
        elevation_km (rt.Raster, optional): Pre-computed elevation in kilometers.
            If None, it will be retrieved from SRTM data. Defaults to None.
        ET_model_name (str, optional): Name of the evapotranspiration model to use.
            Currently, only PT-JPL is implemented. This parameter is not
            currently used to select models but is reserved for future expansion. Defaults to ET_MODEL_NAME.
        working_directory (str, optional): The main working directory for data
            processing and output files. Defaults to the current directory.
        static_directory (str, optional): Directory for static datasets like SRTM DEMs.
            Defaults to None, which will typically use the working directory or a predefined static path.
        VNP09GA_download_directory (str, optional): Local directory for downloading
            VIIRS VNP09GA data. Defaults to VNP09GA_DOWNLOAD_DIRECTORY constant.
        VNP21A1D_download_directory (str, optional): Local directory for downloading
            VIIRS VNP21A1D data. Defaults to VNP21A1D_DOWNLOAD_DIRECTORY constant.
        use_VIIRS_composite (bool, optional): Whether to use VIIRS composite
            data (i.e., data from previous days) for gap-filling missing values. Defaults to USE_VIIRS_COMPOSITE constant.
        VIIRS_composite_days (int, optional): Number of days to look back for
            VIIRS compositing (for gap-filling). Defaults to VIIRS_COMPOSITE_DAYS constant.
        VIIRS_GEOS5FP_output_directory (str, optional): Directory to save final
            VIIRS-GEOS5FP derived products. Defaults to a subdirectory within the
            working_directory.
        SRTM_connection (NASADEMConnection, optional): An existing NASADEMConnection
            object. If None, a new one will be initialized. Defaults to None.
        SRTM_download (str, optional): Local directory for downloading SRTM data.
            Defaults to None, implying a default path from NASADEMConnection.
        GEOS5FP_connection (GEOS5FP, optional): An existing GEOS5FP connection
            object. If None, a new one will be initialized. Defaults to None.
        GEOS5FP_download (str, optional): Local directory for downloading GEOS-5 FP data.
            Defaults to None, implying a default path from GEOS5FP.
        GEOS5FP_products (str, optional): Specifies GEOS-5 FP products to download. This parameter is not
            currently used directly for product selection within this function. Defaults to None.
        GEOS5FP_offline_processing (bool, optional): Whether to allow offline
            processing for GEOS-5 FP (i.e., use local data without attempting to
            check for the absolute latest availability online). Defaults to True.
        GEDI_connection (GEDICanopyHeight, optional): GEDI connection object. This parameter is not
            currently used but is reserved for future GEDI data integration. Defaults to None.
        GEDI_download (str, optional): GEDI download directory. Not currently used. Defaults to None.
        ORNL_connection (MODISCI, optional): ORNL MODIS CI connection object. This parameter is not
            currently used but is reserved for future MODIS CI data integration. Defaults to None.
        CI_directory (str, optional): CI data directory. Not currently used. Defaults to None.
        soil_grids_connection (SoilGrids, optional): SoilGrids connection object. This parameter is not
            currently used but is reserved for future SoilGrids data integration. Defaults to None.
        soil_grids_download (str, optional): SoilGrids download directory. Not currently used. Defaults to None.
        intermediate_directory (str, optional): Directory for saving intermediate
            products during processing. Not currently used. Defaults to None.
        preview_quality (int, optional): Quality setting for generating preview images. This parameter is not
            currently used. Defaults to PREVIEW_QUALITY constant.
        ANN_model (Callable, optional): An Artificial Neural Network model. This parameter is not
            currently used. Defaults to None.
        ANN_model_filename (str, optional): Filename for the ANN model. Not currently
            used. Defaults to None.
        resampling (str, optional): Default resampling method to use for spatial
            operations. This parameter is not universally applied to all resampling
            operations within the function but can be a general setting. Defaults to RESAMPLING constant.
        coarse_cell_size (float, optional): The target cell size (in degrees) for
            coarse resolution data, typically used as an input for downscaling algorithms.
            Defaults to COARSE_CELL_SIZE constant.
        downscale_air (bool, optional): Whether to apply downscaling to GEOS-5 FP
            air temperature data to match the fine geometry. Defaults to DOWNSCALE_AIR constant.
        downscale_humidity (bool, optional): Whether to apply downscaling to GEOS-5 FP
            relative humidity data. Defaults to DOWNSCALE_HUMIDITY constant.
        downscale_moisture (bool, optional): Whether to apply downscaling to GEOS-5 FP
            soil moisture data. Defaults to DOWNSCALE_MOISTURE constant.
        floor_Topt (bool, optional): Whether to enforce a minimum (floor) value for
            optimal temperature (Topt) in the PT-JPL model. This parameter is not
            currently used directly in this function but is passed to PT-JPL. Defaults to FLOOR_TOPT constant.
        save_intermediate (bool, optional): Whether to save intermediate products generated
            during the workflow. This parameter is not currently used. Defaults to False.
        include_preview (bool, optional): Whether to include preview images in the output.
            This parameter is not currently used. Defaults to True.
        show_distribution (bool, optional): Whether to show distribution plots of
            derived variables. This parameter is not currently used. Defaults to True.
        load_previous (bool, optional): If True, the function will check if products
            for the target date have already been processed and saved. If so, it will
            load them from disk instead of re-processing. Defaults to True.
        target_variables (List[str], optional): A list of specific output variable names
            to process and save. Defaults to TARGET_VARIABLES constant.

    Returns:
        Dict[str, rt.Raster]: A dictionary where keys are variable names (e.g.,
            "ST", "NDVI", "ET_total") and values are the corresponding
            raster objects (geospatial images).

    Raises:
        GEOS5FPNotAvailableError: If unable to connect to GEOS-5 FP or if the
            target time is beyond the latest available GEOS-5 FP data.
    """
    # Initialize a dictionary to store all processed raster results
    results = {}

    # --- Section: Date and Time Setup ---
    # Convert the target_date to a datetime.date object if it's provided as a string.
    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()
    logger.info(f"VIIRS GEOS-5 FP target date: {cl.time(target_date)}")

    # Define the local solar time for processing (13:30 local solar time is common for remote sensing)
    time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
    logger.info(f"VIIRS GEOS-5 FP target time solar: {cl.time(time_solar)}")

    # Convert the local solar time to UTC based on the longitude of the geometry's centroid.
    # This is crucial for correctly querying global datasets that are often indexed by UTC.
    time_UTC = solar_to_UTC(time_solar, geometry.centroid.latlon.x)
    logger.info(f"VIIRS GEOS-5 FP target time UTC: {cl.time(time_UTC)}")

    # --- Section: Directory Setup ---
    # Set up the main working directory. If not provided, use the current directory.
    if working_directory is None:
        working_directory = "."
    # Resolve to an absolute path and expand user directory (e.g., '~' to '/home/user').
    working_directory = abspath(expanduser(working_directory))
    logger.info(f"VIIRS GEOS-5 FP working directory: {cl.dir(working_directory)}")

    # --- Section: SRTM (Elevation Data) Handling ---
    # Initialize a NASADEMConnection object if one is not already provided.
    # This connection is used to retrieve elevation and water mask data.
    if SRTM_connection is None:
        SRTM_connection = NASADEMConnection(
            working_directory=static_directory, # Directory for static data like DEMs
            download_directory=SRTM_download,   # Specific download directory for SRTM data
            offline_ok=True                     # Allow using local data without checking for online updates
        )

    # Retrieve the water mask if not provided as input.
    # The water mask (Surface Water Body) is derived from SRTM data.
    if water is None:
        logger.info(f"retrieving {cl.name('water mask')} from SRTM")
        water = SRTM_connection.swb(geometry)
    # Add the water mask to the results, whether it was provided or retrieved.
    results["water"] = water

    # Retrieve elevation data if not provided as input.
    # Elevation in kilometers is also derived from SRTM data.
    if elevation_km is None:
        logger.info(f"retrieving {cl.name('elevation')} from SRTM")
        elevation_km = SRTM_connection.elevation_km(geometry)

    # --- Section: VIIRS Data Connection Setup ---
    # Set up the download directory for VIIRS VNP09GA (surface reflectance) data.
    if VNP09GA_download_directory is None:
        VNP09GA_download_directory = VNP09GA_DOWNLOAD_DIRECTORY
    logger.info(f"VNP09GA download directory: {cl.dir(VNP09GA_download_directory)}")
    # Initialize VNP09GA connection.
    VNP09GA_connection = VNP09GA(
        download_directory=VNP09GA_download_directory,
    )

    # Set up the download directory for VIIRS VNP21A1D (land surface temperature) data.
    if VNP21A1D_download_directory is None:
        VNP21A1D_download_directory = VNP21A1D_DOWNLOAD_DIRECTORY
    logger.info(f"VNP21A1D download directory: {cl.dir(VNP21A1D_download_directory)}")
    # Initialize VNP21A1D connection.
    VNP21A1D_connection = VNP21A1D(
        download_directory=VNP21A1D_download_directory,
    )

    # --- Section: Output Management ---
    # Determine the final output directory for processed VIIRS-GEOS5FP products.
    if VIIRS_GEOS5FP_output_directory is None:
        VIIRS_GEOS5FP_output_directory = join(working_directory, VIIRS_GEOS5FP_OUTPUT_DIRECTORY)
    logger.info(f"VIIRS GEOS-5 FP output directory: {cl.dir(VIIRS_GEOS5FP_output_directory)}")

    # Check if the data for the target date and products has already been processed and saved.
    VIIRS_GEOS5FP_already_processed = check_VIIRS_GEOS5FP_already_processed(
        VIIRS_GEOS5FP_output_directory=VIIRS_GEOS5FP_output_directory,
        target_date=target_date,
        time_UTC=time_UTC,
        target=target,
        products=target_variables # Check for the specific variables that will be targeted for output
    )

    # If data is already processed and 'load_previous' is True, load and return it.
    if VIIRS_GEOS5FP_already_processed:
        if load_previous:
            logger.info("loading previously generated VIIRS GEOS-5 FP output")
            return load_VIIRS_GEOS5FP(
                VIIRS_GEOS5FP_output_directory=VIIRS_GEOS5FP_output_directory,
                target_date=target_date,
                target=target
            )
        else:
            # If data is found but not configured to be loaded, log and return empty.
            logger.info("Previous VIIRS GEOS-5 FP output found, but 'load_previous' is False. Skipping reprocessing.")
            return {} # Return empty dictionary to indicate no new data was processed

    # --- Section: GEOS-5 FP Data Handling ---
    # Initialize a GEOS5FP connection object if one is not already provided.
    if GEOS5FP_connection is None:
        try:
            logger.info(f"connecting to GEOS-5 FP")
            GEOS5FP_connection = GEOS5FP(
                working_directory=working_directory,
                download_directory=GEOS5FP_download
            )
        except Exception as e:
            # If connection fails, log the error and raise a custom exception.
            logger.exception(e)
            raise GEOS5FPNotAvailableError("unable to connect to GEOS-5 FP")

    # Check GEOS-5 FP data availability if not in offline processing mode.
    # This ensures that we don't try to process data for a future time or a time
    # for which GEOS-5 FP data hasn't been released yet.
    if not GEOS5FP_offline_processing:
        latest_GEOS5FP_time = GEOS5FP_connection.latest_time_available
        logger.info(f"latest GEOS-5 FP time available: {latest_GEOS5FP_time}")
        logger.info(f"processing time: {time_UTC}")

        # Compare the target UTC time with the latest available GEOS-5 FP time.
        # Use string formatting for robust comparison of datetime objects at specific precision.
        if time_UTC.strftime("%Y-%m-%d %H:%M:%S") > latest_GEOS5FP_time.strftime("%Y-%m-%d %H:%M:%S"):
            raise GEOS5FPNotAvailableError(
                f"VIIRS GEOS-5 FP target time {time_UTC} is past latest available GEOS-5 FP time {latest_GEOS5FP_time}")

    # --- Section: Land Surface Temperature (ST_C) Processing ---
    # Process Land Surface Temperature (ST_C) if not provided as input.
    if ST_C is None:
        logger.info(
            f"retrieving {cl.name('VNP21A1D')} {cl.name('ST_C')} from VIIRS on {cl.time(target_date)}")
        # Retrieve ST_C from VNP21A1D, resampling to the target geometry.
        ST_C = VNP21A1D_connection.ST_C(date_UTC=target_date, geometry=geometry, resampling="cubic")

        # Gap-filling ST_C using VIIRS composite data from previous days if enabled.
        # This helps to fill missing data (e.g., due to clouds) by looking at nearby dates.
        if use_VIIRS_composite:
            for days_back in range(1, VIIRS_composite_days + 1): # Iterate up to VIIRS_composite_days
                fill_date = target_date - timedelta(days=days_back)
                logger.info(
                    f"gap-filling {cl.name('VNP21A1D')} {cl.name('ST_C')} from VIIRS on {cl.time(fill_date)} for {cl.time(target_date)}")
                # Retrieve ST_C from a previous day.
                ST_C_fill = VNP21A1D_connection.ST_C(date_UTC=fill_date, geometry=geometry, resampling="cubic")
                # Use rt.where to replace NaN values in ST_C with values from ST_C_fill.
                ST_C = rt.where(np.isnan(ST_C), ST_C_fill, ST_C)

        # Gap-filling any remaining NaN values in ST_C with resampled GEOS-5 FP surface temperature.
        # This provides a broader fill if VIIRS composites are insufficient.
        logger.info(f"gap-filling remaining {cl.name('ST_C')} with GEOS-5 FP surface temperature")
        # Retrieve GEOS-5 FP surface temperature (Ts_K) and convert from Kelvin to Celsius.
        ST_C_smooth = GEOS5FP_connection.Ts_K(time_UTC=time_UTC, geometry=geometry, resampling="cubic") - 273.15
        # Fill remaining NaN values in ST_C with the smoothed GEOS-5 FP data.
        ST_C = rt.where(np.isnan(ST_C), ST_C_smooth, ST_C)

    # Store the processed Land Surface Temperature in the results dictionary.
    results["ST"] = ST_C

    # --- Section: Normalized Difference Vegetation Index (NDVI) Processing ---
    # Process Normalized Difference Vegetation Index (NDVI) if not provided as input.
    if NDVI is None:
        logger.info(
            f"retrieving {cl.name('VNP09GA')} {cl.name('NDVI')} on {cl.time(target_date)}")
        # Retrieve NDVI from VNP09GA, resampling to the target geometry.
        NDVI = VNP09GA_connection.NDVI(date_UTC=target_date, geometry=geometry, resampling="cubic")

        # Gap-filling NDVI using VIIRS composite data from previous days if enabled.
        if use_VIIRS_composite:
            for days_back in range(1, VIIRS_composite_days + 1):
                fill_date = target_date - timedelta(days=days_back)
                logger.info(
                    f"gap-filling {cl.name('VNP09GA')} {cl.name('NDVI')} on {cl.time(fill_date)} for {cl.time(target_date)}")
                # Retrieve NDVI from a previous day.
                NDVI_fill = VNP09GA_connection.NDVI(date_UTC=fill_date, geometry=geometry, resampling="cubic")
                # Corrected: Use NDVI_fill to fill NaN values in NDVI.
                NDVI = rt.where(np.isnan(NDVI), NDVI_fill, NDVI)

    # Store the processed NDVI in the results dictionary.
    results["NDVI"] = NDVI

    # --- Section: Emissivity Derivation ---
    # Derive Emissivity if not provided as input.
    # A common empirical relationship is used to estimate emissivity from NDVI.
    if emissivity is None:
        logger.info(f"deriving {cl.name('emissivity')} from NDVI")
        emissivity = 1.0094 + 0.047 * np.log(NDVI) # Empirical formula for emissivity
        # Set emissivity to a typical water value (0.96) for water bodies, as the formula is for land.
        emissivity = rt.where(water, 0.96, emissivity)

    # Store the derived emissivity in the results dictionary.
    results["emissivity"] = emissivity

    # --- Section: Albedo Processing ---
    # Process Albedo if not provided as input.
    if albedo is None:
        logger.info(
            f"retrieving {cl.name('VNP09GA')} {cl.name('albedo')} on {cl.time(target_date)}")
        # Retrieve albedo from VNP09GA, resampling to the target geometry.
        albedo = VNP09GA_connection.albedo(date_UTC=target_date, geometry=geometry, resampling="cubic")

        # Gap-filling albedo using VIIRS composite data from previous days if enabled.
        if use_VIIRS_composite:
            for days_back in range(1, VIIRS_composite_days + 1):
                fill_date = target_date - timedelta(days=days_back)
                logger.info(
                    f"gap-filling {cl.name('VNP09GA')} {cl.name('albedo')} from VIIRS on {cl.time(fill_date)} for {cl.time(target_date)}")
                # Retrieve albedo from a previous day.
                albedo_fill = VNP09GA_connection.albedo(date_UTC=fill_date, geometry=geometry, resampling="cubic")
                # Use rt.where to replace NaN values in albedo with values from albedo_fill.
                albedo = rt.where(np.isnan(albedo), albedo_fill, albedo)

    # Store the processed albedo in the results dictionary.
    results["albedo"] = albedo

    # --- Section: Downscaling Preparations ---
    # Define a coarse geometry for downscaling operations.
    # This geometry will have a larger cell size (coarse_cell_size) than the target geometry,
    # representing the resolution of the original GEOS-5 FP data.
    coarse_geometry = geometry.rescale(coarse_cell_size)

    # --- Section: Air Temperature (Ta_C) Processing ---
    # Process Air Temperature (Ta_C) if not provided as input.
    if Ta_C is None:
        if downscale_air:
            logger.info(f"downscaling {cl.name('air temperature')} using GEOS-5 FP and VIIRS ST")
            ST_K = ST_C + 273.15 # Convert surface temperature from Celsius to Kelvin for calculations.
            # Retrieve coarse resolution air temperature from GEOS-5 FP.
            Ta_K_coarse = GEOS5FP_connection.Ta_K(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic")
            # Apply downscaling to air temperature using the fine-resolution ST_K.
            Ta_K = downscale_air_temperature(
                time_UTC=time_UTC,
                Ta_K_coarse=Ta_K_coarse,
                ST_K=ST_K, # Fine-resolution surface temperature is used to guide downscaling
                fine_geometry=geometry,
                coarse_geometry=coarse_geometry
            )
            Ta_C = Ta_K - 273.15 # Convert downscaled air temperature back to Celsius.
        else:
            logger.info(f"retrieving {cl.name('air temperature')} from GEOS-5 FP")
            # If downscaling is not enabled, retrieve Ta_C directly from GEOS-5 FP at the target resolution.
            Ta_C = GEOS5FP_connection.Ta_C(time_UTC=time_UTC, geometry=geometry, resampling="cubic")

    # Store the processed Air Temperature in the results dictionary.
    results["Ta"] = Ta_C

    # --- Section: Soil Moisture (SM) Processing ---
    # Process Soil Moisture (SM) if not provided as input.
    if SM is None:
        if downscale_moisture:
            logger.info(f"downscaling {cl.name('soil moisture')} using GEOS-5 FP, VIIRS ST and NDVI")
            ST_K = ST_C + 273.15 # Convert ST_C to Kelvin.
            # Retrieve coarse resolution soil moisture from GEOS-5 FP.
            SM_coarse = GEOS5FP_connection.SFMC(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic")
            # Retrieve resampled GEOS-5 FP soil moisture at fine resolution for comparison/guidance.
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
            logger.info(f"retrieving {cl.name('soil moisture')} from GEOS-5 FP")
            # If downscaling is not enabled, retrieve SM directly from GEOS-5 FP at the target resolution.
            SM = GEOS5FP_connection.SFMC(time_UTC=time_UTC, geometry=geometry, resampling="cubic")

    # Store the processed Soil Moisture in the results dictionary.
    results["SM"] = SM

    # --- Section: Relative Humidity (RH) Processing ---
    # Process Relative Humidity (RH) if not provided as input.
    if RH is None:
        if downscale_humidity:
            logger.info(f"downscaling {cl.name('relative humidity')} using GEOS-5 FP, VIIRS ST and downscaled SM")
            ST_K = ST_C + 273.15 # Convert ST_C to Kelvin.
            # Retrieve coarse resolution Vapor Pressure Deficit (VPD) from GEOS-5 FP.
            VPD_Pa_coarse = GEOS5FP_connection.VPD_Pa(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic")

            # Downscale Vapor Pressure Deficit (VPD).
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
                SM=SM, # Downscaled soil moisture is used here
                ST_K=ST_K,
                VPD_kPa=VPD_kPa,
                water=water,
                fine_geometry=geometry,
                coarse_geometry=coarse_geometry
            )
        else:
            logger.info(f"retrieving {cl.name('relative humidity')} from GEOS-5 FP")
            # If downscaling is not enabled, retrieve RH directly from GEOS-5 FP at the target resolution.
            RH = GEOS5FP_connection.RH(time_UTC=time_UTC, geometry=geometry, resampling="cubic")

    # Store the processed Relative Humidity in the results dictionary.
    results["RH"] = RH

    # --- Section: Incoming Shortwave Radiation (SWin) and Net Radiation (Rn) ---
    # Retrieve or generate Incoming Shortwave Radiation (SWin).
    # If SWin is None or specifically requested from 'GEOS5FP', retrieve it.
    if SWin is None or isinstance(SWin, str):
        logger.info("generating solar radiation using GEOS-5 FP")
        SWin = GEOS5FP_connection.SWin(time_UTC=time_UTC, geometry=geometry, resampling="cubic")

    # Calculate Net Radiation (Rn) if not provided as input.
    # The Verma net radiation model is used for this calculation.
    if Rn is None:
        logger.info(f"calculating {cl.name('net radiation')} using Verma model")
        verma_results = process_verma_net_radiation(
            SWin=SWin,
            albedo=albedo,
            ST_C=ST_C,
            emissivity=emissivity,
            Ta_C=Ta_C,
            RH=RH
        )
        Rn = verma_results["Rn"] # Extract the net radiation component from the results.

    # Store the processed Net Radiation in the results dictionary.
    results["Rn"] = Rn

    # --- Section: Evapotranspiration (ET) Modeling (PT-JPL) ---
    # Run the PT-JPL (Priestley-Taylor Jet Propulsion Laboratory) Evapotranspiration model.
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

    # Add all calculated PT-JPL results (e.g., various ET components) to the main results dictionary.
    for k, v in PTJPL_results.items():
        results[k] = v

    # --- Section: Saving Processed Products ---
    # Save all processed raster products (from the 'results' dictionary) to GeoTIFF files.
    for product, image in results.items():
        # Generate the output filename for each specific product.
        filename = generate_VIIRS_GEOS5FP_output_filename(
            VIIRS_GEOS5FP_output_directory=VIIRS_GEOS5FP_output_directory,
            target_date=target_date,
            time_UTC=time_UTC,
            target=target,
            product=product # The specific product name (e.g., "ST", "NDVI", "ET_total")
        )

        # Skip saving if the image raster object is None (i.e., data wasn't successfully generated).
        if image is None:
            logger.warning(f"no image result for {cl.name(product)}, skipping saving for this product.")
            continue

        # Log the saving action.
        logger.info(
            f"writing VIIRS GEOS-5 FP {cl.name(product)} at {cl.place(target)} at {cl.time(time_UTC)} to file: {cl.file(filename)}")
        # Save the raster object to a GeoTIFF file.
        image.to_geotiff(filename)

    # Return the dictionary containing all processed raster objects.
    return results