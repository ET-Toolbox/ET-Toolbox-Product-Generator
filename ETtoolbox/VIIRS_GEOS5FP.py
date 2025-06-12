from typing import Union, List, Dict, Callable
from datetime import date, datetime, timedelta
from dateutil import parser
from os.path import join, abspath, expanduser, basename, exists, splitext
import numpy as np
import rasters as rt
from rasters import RasterGrid

from glob import glob

import logging
import colored_logging as cl

import VNP09GA_002
import VNP21A1D_002
from gedi_canopy_height import GEDICanopyHeight
from GEOS5FP import GEOS5FP
from MODISCI import MODISCI
from PTJPL import PTJPL
from verma_net_radiation import process_verma_net_radiation
import NASADEM
from soil_capacity_wilting import SoilGrids

from GEOS5FP.downscaling import downscale_air_temperature, downscale_soil_moisture, \
    downscale_vapor_pressure_deficit, downscale_relative_humidity, bias_correct
from solar_apparent_time import solar_to_UTC

from .constants import *
from .generate_VIIRS_output_directory import generate_VIIRS_output_directory
from .generate_VIIRS_output_filename import generate_VIIRS_output_filename
from .check_VIIRS_already_processed import check_VIIRS_already_processed
from .load_VIIRS import load_VIIRS

logger = logging.getLogger(__name__)

class GEOS5FPNotAvailableError(Exception):
    """
    Custom exception raised when GEOS-5 FP data cannot be accessed or if the target time
    is beyond the latest available data, particularly critical in Near Real-Time (NRT) processing.
    """
    pass

def VIIRS_GEOS5FP(
        target_date: Union[date, str],
        geometry: RasterGrid,
        target: str,
        nrt_mode: bool = False, # FLAG: True for Near Real-Time processing, False for standard/historical.
        ST_C: rt.Raster = None,
        emissivity: rt.Raster = None,
        NDVI: rt.Raster = None,
        albedo: rt.Raster = None,
        SWin: Union[rt.Raster, str] = None,
        Rn: Union[rt.Raster, str] = None,
        SM: rt.Raster = None,
        wind_speed: rt.Raster = None, # Not currently used, reserved for future extensions.
        Ta_C: rt.Raster = None,
        RH: rt.Raster = None,
        water: rt.Raster = None,
        elevation_km: rt.Raster = None,
        model: PTJPL = None, # An instance of the PTJPL model.
        model_name: str = ET_MODEL_NAME, # Not actively used for model selection.
        working_directory: str = None,
        static_directory: str = None,
        VIIRS_download_directory: str = None,
        output_directory: str = None, # Renamed for clarity and common use.
        output_bucket_name: str = None, # S3 bucket name for cloud storage outputs. Not directly used here.
        SRTM_connection: NASADEM = None,
        SRTM_download: str = None,
        GEOS5FP_connection: GEOS5FP = None,
        GEOS5FP_download: str = None,
        GEOS5FP_products: str = None, # Not actively used here.
        GEOS5FP_offline_processing: bool = True, # Only relevant when nrt_mode is False.
        GEDI_connection: GEDICanopyHeight = None, # Not currently used.
        GEDI_download: str = None, # Not currently used.
        ORNL_connection: MODISCI = None, # Not currently used.
        CI_directory: str = None, # Not currently used.
        soil_grids_connection: SoilGrids = None, # Not currently used.
        soil_grids_download: str = None, # Not currently used.
        intermediate_directory: str = None, # Not currently used.
        spacetrack_credentials_filename: str = None, # Not directly used in this function.
        ERS_credentials_filename: str = None, # Not directly used in this function.
        preview_quality: int = PREVIEW_QUALITY, # Not currently used.
        ANN_model: Callable = None, # Not currently used.
        ANN_model_filename: str = None, # Not currently used.
        resampling: str = RESAMPLING, # Default resampling method.
        coarse_cell_size: float = COARSE_CELL_SIZE,
        downscale_air: bool = DOWNSCALE_AIR,
        downscale_humidity: bool = DOWNSCALE_HUMIDITY,
        downscale_moisture: bool = DOWNSCALE_MOISTURE,
        floor_Topt: bool = FLOOR_TOPT,
        use_VIIRS_composite: bool = USE_VIIRS_COMPOSITE, # Only relevant when nrt_mode is False.
        VIIRS_composite_days: int = VIIRS_COMPOSITE_DAYS, # Only relevant when nrt_mode is False.
        save_intermediate: bool = False, # Not currently used.
        include_preview: bool = True, # Not currently used.
        show_distribution: bool = True, # Not currently used.
        load_previous: bool = True,
        target_variables: List[str] = TARGET_VARIABLES) -> Dict[str, rt.Raster]:
    """
    Processes VIIRS and GEOS-5 FP data to generate various environmental products,
    including land surface temperature, NDVI, emissivity, albedo, and
    evapotranspiration components using the PT-JPL model. It handles data
    retrieval, downscaling of atmospheric variables, and saving of results.

    This comprehensive function adapts its behavior based on the `nrt_mode` flag,
    allowing for both standard (historical/research) and Near Real-Time processing.

    Args:
        target_date (Union[date, str]): The target date for processing.
        geometry (RasterGrid): The spatial extent and resolution for the output rasters.
        target (str): A string identifier for the target area or region.
        nrt_mode (bool, optional): If True, the function operates in Near Real-Time mode,
            which implies stricter GEOS-5 FP availability checks, and typically avoids
            multi-day VIIRS composites for gap-filling to prioritize speed.
            Defaults to False (standard processing).
        ST_C (rt.Raster, optional): Pre-computed surface temperature in Celsius. Defaults to None.
        emissivity (rt.Raster, optional): Pre-computed emissivity. Defaults to None.
        NDVI (rt.Raster, optional): Pre-computed Normalized Difference Vegetation Index. Defaults to None.
        albedo (rt.Raster, optional): Pre-computed albedo. Defaults to None.
        SWin (Union[rt.Raster, str], optional): Pre-computed incoming shortwave radiation. Defaults to None.
        Rn (Union[rt.Raster, str], optional): Pre-computed net radiation. Defaults to None.
        SM (rt.Raster, optional): Pre-computed soil moisture. Defaults to None.
        wind_speed (rt.Raster, optional): Pre-computed wind speed. This parameter is not
            currently used in the processing, but is kept for potential future extensions. Defaults to None.
        Ta_C (rt.Raster, optional): Pre-computed air temperature in Celsius. Defaults to None.
        RH (rt.Raster, optional): Pre-computed relative humidity. Defaults to None.
        water (rt.Raster, optional): Pre-computed water mask. Defaults to None.
        elevation_km (rt.Raster, optional): Pre-computed elevation in kilometers. Defaults to None.
        model (PTJPL, optional): An instance of the PTJPL model. Defaults to None.
        model_name (str, optional): Name of the ET model. This parameter is not
            currently used to select models but is reserved for future expansion. Defaults to ET_MODEL_NAME.
        working_directory (str, optional): Base directory for data processing.
            Defaults to "~/data/ETtoolbox" in NRT mode, or "." otherwise.
        static_directory (str, optional): Directory for static data. Defaults to None.
        VIIRS_download_directory (str, optional): Directory for downloading VIIRS data. Defaults to None.
        output_directory (str, optional): Directory for saving processed outputs.
            Defaults to a subdirectory within the `working_directory`.
        output_bucket_name (str, optional): S3 bucket name for cloud storage outputs. Not directly used. Defaults to None.
        SRTM_connection (NASADEM, optional): Connection object for NASADEM. Defaults to None.
        SRTM_download (str, optional): Directory for downloading SRTM data. Defaults to None.
        GEOS5FP_connection (GEOS5FP, optional): Connection object for GEOS5FP. Defaults to None.
        GEOS5FP_download (str, optional): Directory for downloading GEOS5FP data. Defaults to None.
        GEOS5FP_products (str, optional): Specific GEOS5FP products to retrieve. Not currently used. Defaults to None.
        GEOS5FP_offline_processing (bool, optional): Whether to allow offline
            processing for GEOS-5 FP (i.e., use local data without checking
            for latest availability online). Only applicable when `nrt_mode` is False. Defaults to True.
        GEDI_connection (GEDICanopyHeight, optional): GEDI connection object. Not currently used. Defaults to None.
        GEDI_download (str, optional): GEDI download directory. Not currently used. Defaults to None.
        ORNL_connection (MODISCI, optional): ORNL MODIS CI connection object. Not currently used. Defaults to None.
        CI_directory (str, optional): CI data directory. Not currently used. Defaults to None.
        soil_grids_connection (SoilGrids, optional): SoilGrids connection object. Not currently used. Defaults to None.
        soil_grids_download (str, optional): SoilGrids download directory. Not currently used. Defaults to None.
        intermediate_directory (str, optional): Directory for saving intermediate
            products. Not currently used. Defaults to None.
        spacetrack_credentials_filename (str, optional): Filename for Space-Track credentials. Not directly used. Defaults to None.
        ERS_credentials_filename (str, optional): Filename for ERS credentials. Not directly used. Defaults to None.
        preview_quality (int, optional): Quality setting for previews. Not currently used. Defaults to PREVIEW_QUALITY.
        ANN_model (Callable, optional): Artificial Neural Network model. Not currently used. Defaults to None.
        ANN_model_filename (str, optional): Filename for ANN model. Not currently used. Defaults to None.
        resampling (str, optional): Resampling method. Defaults to RESAMPLING.
        coarse_cell_size (float, optional): Cell size for coarse resolution data. Defaults to COARSE_CELL_SIZE.
        downscale_air (bool, optional): Whether to downscale air temperature. Defaults to DOWNSCALE_AIR.
        downscale_humidity (bool, optional): Whether to downscale humidity. Defaults to DOWNSCALE_HUMIDITY.
        downscale_moisture (bool, optional): Whether to downscale soil moisture. Defaults to DOWNSCALE_MOISTURE.
        floor_Topt (bool, optional): Whether to floor Topt in PT-JPL. Defaults to FLOOR_TOPT.
        use_VIIRS_composite (bool, optional): Whether to use VIIRS composite
            data for gap-filling. Only applicable when `nrt_mode` is False. Defaults to USE_VIIRS_COMPOSITE constant.
        VIIRS_composite_days (int, optional): Number of days to look back for
            VIIRS compositing. Only applicable when `nrt_mode` is False. Defaults to VIIRS_COMPOSITE_DAYS constant.
        save_intermediate (bool, optional): Whether to save intermediate products. Not currently used. Defaults to False.
        include_preview (bool, optional): Whether to include preview images. Not currently used. Defaults to True.
        show_distribution (bool, optional): Whether to show distribution plots. Not currently used. Defaults to True.
        load_previous (bool, optional): Whether to load previously processed
            results if available. Defaults to True.
        target_variables (List[str], optional): A list of target variables to
            process and save. Defaults to TARGET_VARIABLES constant.

    Returns:
        Dict[str, rt.Raster]: A dictionary containing the processed raster outputs.

    Raises:
        GEOS5FPNotAvailableError: If GEOS-5 FP data is not available for the target time,
            especially in NRT mode or if `GEOS5FP_offline_processing` is False.
    """
    results = {}

    # --- Section: Date and Time Setup ---
    # Convert the target_date to a datetime.date object if it's provided as a string.
    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()
    logger.info(f"Processing target date: {cl.time(target_date)}{' (NRT mode)' if nrt_mode else ''}")

    # Define the local solar time for processing (13:30 local solar time is common for remote sensing products)
    time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
    logger.info(f"Target time solar: {cl.time(time_solar)}")

    # Convert the local solar time to UTC based on the longitude of the geometry's centroid.
    # This is crucial for correctly querying global datasets that are often indexed by UTC.
    time_UTC = solar_to_UTC(time_solar, geometry.centroid.latlon.x)
    logger.info(f"Target time UTC: {cl.time(time_UTC)}")

    # --- Section: Directory Setup ---
    # Set up the main working directory. Default depends on NRT mode.
    if working_directory is None:
        if nrt_mode:
            working_directory = "~/data/ETtoolbox" # Common default for NRT systems
        else:
            working_directory = "." # Current directory for standard processing
    # Resolve to an absolute path and expand user directory (e.g., '~' to '/home/user').
    working_directory = abspath(expanduser(working_directory))
    logger.info(f"Working directory: {cl.dir(working_directory)}")

    # --- Section: SRTM (Elevation Data) Handling ---
    # Initialize a NASADEMConnection object if one is not already provided.
    # This connection is used to retrieve elevation and water mask data from SRTM.
    if SRTM_connection is None:
        SRTM_connection = NASADEM.NASADEMConnection(
            download_directory=SRTM_download,
            offline_ok=not nrt_mode # Allow offline if not in NRT mode (as NRT demands timeliness)
        )

    # Retrieve the water mask if it was not provided as input.
    if water is None:
        logger.info(f"Retrieving {cl.name('water mask')} from SRTM.")
        water = SRTM_connection.swb(geometry)
    # Add the water mask to the results dictionary.
    results["water"] = water

    # Retrieve elevation data in kilometers if not provided as input.
    if elevation_km is None:
        logger.info(f"Retrieving {cl.name('elevation')} from SRTM.")
        elevation_km = SRTM_connection.elevation_km(geometry)

    # --- Section: VIIRS Data Connection Setup ---
    # Set up the base download directory for VIIRS data (both VNP09GA and VNP21A1D).
    if VIIRS_download_directory is None:
        VIIRS_download_directory = join(working_directory, VIIRS_DOWNLOAD_DIRECTORY)
    logger.info(f"VIIRS download directory: {cl.dir(VIIRS_download_directory)}")

    # Initialize VIIRS data connections.
    VNP21_connection = VNP21A1D_002.VNP21A1D(download_directory=VIIRS_download_directory)
    VNP09_connection = VNP09GA_002.VNP09GA(download_directory=VIIRS_download_directory)

    # --- Section: Output Directory Setup and Previous Data Check ---
    # Determine the final output directory for processed products.
    if output_directory is None:
        # Default output directory name might depend on NRT or not
        if nrt_mode:
            output_directory = join(working_directory, VIIRS_OUTPUT_DIRECTORY) # e.g., 'VIIRS_NRT_Output'
        else:
            output_directory = join(working_directory, VIIRS_GEOS5FP_OUTPUT_DIRECTORY) # e.g., 'VIIRS_GEOS5FP_Output'
    logger.info(f"Output directory: {cl.dir(output_directory)}")

    # Check if the data for the target date and desired products has already been processed and saved.
    VIIRS_already_processed = check_VIIRS_already_processed(
        VIIRS_output_directory=output_directory, # Use the consolidated output_directory
        target_date=target_date,
        time_UTC=time_UTC,
        target=target,
        products=target_variables
    )

    # If data is already processed and 'load_previous' is True, load and return it.
    if VIIRS_already_processed:
        if load_previous:
            logger.info("Loading previously generated VIIRS GEOS-5 FP output.")
            return load_VIIRS( # Use the consolidated load_VIIRS helper
                VIIRS_output_directory=output_directory,
                target_date=target_date,
                target=target
            )
        else:
            logger.info("Output already exists and 'load_previous' is False. Skipping processing.")
            return {}

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
            logger.exception(e)
            raise GEOS5FPNotAvailableError("Unable to connect to GEOS-5 FP.")

    # FLAG: GEOS-5 FP Availability Check Logic
    # In NRT mode, a strict availability check is always performed.
    # In standard mode, the check depends on `GEOS5FP_offline_processing` flag.
    if nrt_mode or not GEOS5FP_offline_processing:
        latest_GEOS5FP_time = GEOS5FP_connection.latest_time_available
        logger.info(f"Latest GEOS-5 FP time available: {latest_GEOS5FP_time}")
        logger.info(f"Processing time: {time_UTC}")

        # If the target time is after the latest available GEOS-5 FP data, raise an error.
        # This is crucial for NRT to ensure timely data.
        if time_UTC.strftime("%Y-%m-%d %H:%M:%S") > latest_GEOS5FP_time.strftime("%Y-%m-%d %H:%M:%S"):
            raise GEOS5FPNotAvailableError(
                f"Target time {time_UTC} is past latest available GEOS-5 FP time {latest_GEOS5FP_time}. "
                f"Cannot proceed with {'NRT' if nrt_mode else 'current'} processing."
            )

    # Define the actual processing date and time variables, which are the same as target_date/time_UTC here.
    VIIRS_processing_date = target_date
    VIIRS_processing_time = time_UTC

    # --- Section: Land Surface Temperature (ST_C) Processing ---
    # Retrieve or calculate Land Surface Temperature (ST_C) if not provided.
    if ST_C is None:
        logger.info(f"Retrieving {cl.name('VNP21A1D')} {cl.name('ST_C')} from VIIRS on {cl.time(VIIRS_processing_date)}.")
        ST_K = VNP21_connection.ST_K(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )
        ST_C = ST_K - 273.15 # Convert to Celsius.

        # FLAG: VIIRS Composite Gap-filling for ST_C
        # Only perform multi-day composite gap-filling if not in NRT mode and `use_VIIRS_composite` is True.
        if not nrt_mode and use_VIIRS_composite:
            for days_back in range(1, VIIRS_composite_days + 1):
                fill_date = target_date - timedelta(days=days_back)
                logger.info(
                    f"Gap-filling {cl.name('VNP21A1D')} {cl.name('ST_C')} from VIIRS on {cl.time(fill_date)} for {cl.time(target_date)}."
                )
                ST_C_fill = VNP21_connection.ST_K(date_UTC=fill_date, geometry=geometry, resampling="cubic") - 273.15
                ST_C = rt.where(np.isnan(ST_C), ST_C_fill, ST_C)

        # Gap-filling any remaining NaN values in ST_C with spatially smoothed GEOS-5 FP surface temperature.
        # This is common to both NRT and standard processing.
        logger.info(f"Gap-filling remaining {cl.name('ST_C')} with GEOS-5 FP surface temperature.")
        ST_C_smooth = GEOS5FP_connection.Ts_K(time_UTC=time_UTC, geometry=geometry, resampling="cubic") - 273.15
        ST_C = rt.where(np.isnan(ST_C), ST_C_smooth, ST_C)
    results["ST"] = ST_C

    # --- Section: Normalized Difference Vegetation Index (NDVI) Processing ---
    # Retrieve or calculate NDVI if not provided.
    if NDVI is None:
        logger.info(f"Retrieving {cl.name('VNP09GA')} {cl.name('NDVI')} from VIIRS on {cl.time(VIIRS_processing_date)}.")
        NDVI = VNP09_connection.NDVI(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )

        # FLAG: VIIRS Composite Gap-filling for NDVI
        # Only perform multi-day composite gap-filling if not in NRT mode and `use_VIIRS_composite` is True.
        if not nrt_mode and use_VIIRS_composite:
            for days_back in range(1, VIIRS_composite_days + 1):
                fill_date = target_date - timedelta(days=days_back)
                logger.info(
                    f"Gap-filling {cl.name('VNP09GA')} {cl.name('NDVI')} on {cl.time(fill_date)} for {cl.time(target_date)}."
                )
                NDVI_fill = VNP09_connection.NDVI(date_UTC=fill_date, geometry=geometry, resampling="cubic")
                NDVI = rt.where(np.isnan(NDVI), NDVI_fill, NDVI)

    results["NDVI"] = NDVI

    # --- Section: Emissivity Processing ---
    # Retrieve or calculate Emissivity if not provided.
    if emissivity is None:
        # FLAG: Emissivity Source Preference
        # In NRT mode, try to retrieve Emis_14 directly from VIIRS first.
        # In standard mode, or if Emis_14 retrieval fails/is not preferred, default to NDVI-based empirical.
        if nrt_mode:
            logger.info(f"Attempting to retrieve {cl.name('VNP21A1D')} {cl.name('emissivity')} from VIIRS on {cl.time(VIIRS_processing_date)} (NRT preference).")
            emissivity = VNP21_connection.Emis_14(
                date_UTC=VIIRS_processing_date,
                geometry=geometry,
                resampling="cubic"
            )
        else:
            logger.info(f"Deriving {cl.name('emissivity')} from NDVI.")
            emissivity = 1.0094 + 0.047 * np.log(NDVI)

    # Adjust emissivity for water bodies and fill any remaining missing values using NDVI-based empirical relationship.
    emissivity = rt.where(water, 0.96, emissivity)
    logger.info(f"Final gap-filling/derivation of {cl.name('emissivity')} with NDVI-based empirical relationship.")
    emissivity = rt.where(np.isnan(emissivity), 1.0094 + 0.047 * np.log(NDVI), emissivity)
    results["emissivity"] = emissivity

    # --- Section: Albedo Processing ---
    # Retrieve or calculate Albedo if not provided.
    if albedo is None:
        logger.info(f"Retrieving {cl.name('VNP09GA')} {cl.name('albedo')} from VIIRS on {cl.time(VIIRS_processing_date)}.")
        albedo = VNP09_connection.albedo(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )

        # FLAG: VIIRS Composite Gap-filling for Albedo
        # Only perform multi-day composite gap-filling if not in NRT mode and `use_VIIRS_composite` is True.
        if not nrt_mode and use_VIIRS_composite:
            for days_back in range(1, VIIRS_composite_days + 1):
                fill_date = target_date - timedelta(days=days_back)
                logger.info(
                    f"Gap-filling {cl.name('VNP09GA')} {cl.name('albedo')} from VIIRS on {cl.time(fill_date)} for {cl.time(target_date)}."
                )
                albedo_fill = VNP09_connection.albedo(date_UTC=fill_date, geometry=geometry, resampling="cubic")
                albedo = rt.where(np.isnan(albedo), albedo_fill, albedo)
    results["albedo"] = albedo

    # --- Section: Downscaling Preparations ---
    # Define a coarse geometry for downscaling operations, approximating the resolution of GEOS-5 FP.
    coarse_geometry = geometry.rescale(coarse_cell_size)

    # --- Section: Air Temperature (Ta_C) Processing with Downscaling ---
    # Retrieve or calculate Air Temperature (Ta_C) if not provided.
    if Ta_C is None:
        if downscale_air:
            logger.info(f"Downscaling {cl.name('air temperature')} using GEOS-5 FP and VIIRS ST.")
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
            logger.info(f"Retrieving {cl.name('air temperature')} directly from GEOS-5 FP.")
            Ta_C = GEOS5FP_connection.Ta_C(time_UTC=time_UTC, geometry=geometry, resampling="cubic")
    results["Ta"] = Ta_C

    # --- Section: Soil Moisture (SM) Processing with Downscaling ---
    # Retrieve or calculate Soil Moisture (SM) if not provided.
    if SM is None:
        if downscale_moisture:
            logger.info(f"Downscaling {cl.name('soil moisture')} using GEOS-5 FP, VIIRS ST and NDVI.")
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
            logger.info(f"Retrieving {cl.name('soil moisture')} directly from GEOS-5 FP.")
            SM = GEOS5FP_connection.SFMC(time_UTC=time_UTC, geometry=geometry, resampling="cubic")
    results["SM"] = SM

    # --- Section: Relative Humidity (RH) Processing with Downscaling ---
    # Retrieve or calculate Relative Humidity (RH) if not provided.
    if RH is None:
        if downscale_humidity:
            logger.info(f"Downscaling {cl.name('relative humidity')} using GEOS-5 FP, VIIRS ST and downscaled SM.")
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
            logger.info(f"Retrieving {cl.name('relative humidity')} directly from GEOS-5 FP.")
            RH = GEOS5FP_connection.RH(time_UTC=time_UTC, geometry=geometry, resampling="cubic")
    results["RH"] = RH

    # --- Section: Incoming Shortwave Radiation (SWin) and Net Radiation (Rn) ---
    # Retrieve or generate Incoming Shortwave Radiation (SWin).
    if SWin is None:
        logger.info("Generating solar radiation using GEOS-5 FP.")
        SWin = GEOS5FP_connection.SWin(time_UTC=time_UTC, geometry=geometry, resampling="cubic")
    results["SWin"] = SWin

    # Calculate Net Radiation (Rn) using Verma's model if not provided.
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

    # --- Section: Evapotranspiration (ET) Modeling (PT-JPL) ---
    # Run the PT-JPL Evapotranspiration model.
    logger.info(f"Running PT-JPL ET model hindcast at {cl.time(time_UTC)}.")
    PTJPL_results = PTJPL(
        NDVI=NDVI,
        ST_C=ST_C,
        emissivity=emissivity,
        albedo=albedo,
        Rn=Rn,
        Ta_C=Ta_C,
        RH=RH,
        floor_Topt=floor_Topt
    )

    # Add all calculated PT-JPL results to the main results dictionary.
    for k, v in PTJPL_results.items():
        results[k] = v

    # --- Section: Saving Processed Outputs ---
    # Save all processed raster products to GeoTIFF files.
    for product, image in results.items():
        # Generate the standardized output filename for each product.
        filename = generate_VIIRS_output_filename(
            VIIRS_output_directory=output_directory,
            target_date=target_date,
            time_UTC=time_UTC,
            target=target,
            product=product
        )

        # If a specific image result is None, log a warning and skip saving.
        if image is None:
            logger.warning(f"No image result for {cl.name(product)}. Skipping save operation.")
            continue

        logger.info(
            f"Writing VIIRS GEOS-5 FP {cl.name(product)} for {cl.place(target)} at {cl.time(time_UTC)} to file: {cl.file(filename)}."
        )
        image.to_geotiff(filename)

    return results

# Assuming all necessary imports from above are available in the module scope
# from typing import Union, List, Dict # etc.
# from datetime import date, datetime # etc.
# from rasters import RasterGrid # etc.

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
        working_directory: str = None, # Allow override for NRT
        static_directory: str = None,
        VIIRS_download_directory: str = None,
        output_directory: str = None, # Allow override for NRT
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
    A wrapper function for `VIIRS_GEOS5FP` specifically configured for Near Real-Time (NRT) processing.

    This function sets `nrt_mode=True` and adjusts other default parameters to suit NRT requirements,
    such as using the standard NRT working directory and prioritizing direct emissivity retrieval.

    All arguments are passed directly to the underlying `VIIRS_GEOS5FP` function.
    Refer to `VIIRS_GEOS5FP`'s documentation for detailed parameter descriptions.

    Returns:
        Dict[str, rt.Raster]: A dictionary containing the processed raster outputs.
    Raises:
        GEOS5FPNotAvailableError: If GEOS-5 FP data is not available for the target time,
            as required for NRT processing.
    """
    logger.info(f"Initiating VIIRS GEOS-5 FP NRT processing for {cl.time(target_date)}.")

    # Call the main VIIRS_GEOS5FP function with nrt_mode set to True
    # and potentially override some defaults to ensure NRT-specific behavior.
    return VIIRS_GEOS5FP(
        target_date=target_date,
        geometry=geometry,
        target=target,
        nrt_mode=True,  # Crucial flag for NRT behavior
        ST_C=ST_C,
        emissivity=emissivity,
        NDVI=NDVI,
        albedo=albedo,
        SWin=SWin,
        Rn=Rn,
        SM=SM,
        wind_speed=wind_speed,
        Ta_C=Ta_C,
        RH=RH,
        water=water,
        elevation_km=elevation_km,
        model=model,
        model_name=model_name,
        working_directory=working_directory, # Pass through, allows override or uses default in VIIRS_GEOS5FP
        static_directory=static_directory,
        VIIRS_download_directory=VIIRS_download_directory,
        output_directory=output_directory, # Pass through, allows override or uses default in VIIRS_GEOS5FP
        output_bucket_name=output_bucket_name,
        SRTM_connection=SRTM_connection,
        SRTM_download=SRTM_download,
        GEOS5FP_connection=GEOS5FP_connection,
        GEOS5FP_download=GEOS5FP_download,
        GEOS5FP_products=GEOS5FP_products,
        GEOS5FP_offline_processing=False, # NRT implies online check is mandatory
        GEDI_connection=GEDI_connection,
        GEDI_download=GEDI_download,
        ORNL_connection=ORNL_connection,
        CI_directory=CI_directory,
        soil_grids_connection=soil_grids_connection,
        soil_grids_download=soil_grids_download,
        intermediate_directory=intermediate_directory,
        spacetrack_credentials_filename=spacetrack_credentials_filename,
        ERS_credentials_filename=ERS_credentials_filename,
        preview_quality=preview_quality,
        ANN_model=ANN_model,
        ANN_model_filename=ANN_model_filename,
        resampling=resampling,
        coarse_cell_size=coarse_cell_size,
        downscale_air=downscale_air,
        downscale_humidity=downscale_humidity,
        downscale_moisture=downscale_moisture,
        floor_Topt=floor_Topt,
        use_VIIRS_composite=False, # NRT generally avoids multi-day composites for speed
        VIIRS_composite_days=0,     # Explicitly set to 0 as composites are avoided in NRT
        save_intermediate=save_intermediate,
        include_preview=include_preview,
        show_distribution=show_distribution,
        load_previous=load_previous,
        target_variables=target_variables
    )