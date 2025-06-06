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
from MODISCI import MODISCI
from PTJPL import PTJPL
from soil_capacity_wilting import SoilGrids
from GEOS5FP.downscaling import downscale_air_temperature, downscale_soil_moisture, bias_correct
from sentinel_tiles import sentinel_tiles
from solar_apparent_time import solar_to_UTC
import colored_logging as cl
from verma_net_radiation import process_verma_net_radiation
from NASADEM import NASADEMConnection
from check_distribution import check_distribution
import pandas as pd

from .constants import *
from .generate_GFS_output_directory import generate_GFS_output_directory
from .generate_GFS_output_filename import generate_GFS_output_filename
from .check_GFS_already_processed import check_GFS_already_processed
from .load_GFS import load_GFS

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
        model: PTJPL = None,
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
        GEOS5FP_products: str = None,
        GEDI_connection: GEDICanopyHeight = None,
        GEDI_download: str = None,
        ORNL_connection: MODISCI = None,
        CI_directory: str = None,
        soil_grids_connection: SoilGrids = None,
        soil_grids_download: str = None,
        intermediate_directory=None,
        model_name: str = "PTJPL",
        preview_quality: int = PREVIEW_QUALITY,
        ANN_model: Callable = None,
        ANN_model_filename: str = None,
        spacetrack_credentials_filename: str = None,
        ERS_credentials_filename: str = None,
        resampling: str = RESAMPLING,
        downscale_air: bool = DOWNSCALE_AIR,
        downscale_humidity: bool = DOWNSCALE_HUMIDITY,
        downscale_moisture: bool = DOWNSCALE_MOISTURE,
        apply_GEOS5FP_GFS_bias_correction: bool = True,
        VIIRS_processing_date: Union[date, str] = None,
        GFS_listing: pd.DataFrame = None,
        save_intermediate: bool = False,
        include_preview: bool = True,
        show_distribution: bool = True,
        load_previous: bool = True,
        target_variables: List[str] = TARGET_VARIABLES) -> Dict[str, Raster]:
    """
    Generates a VIIRS GFS (Global Forecast System) forecast for a specified target date and geometry.

    This function integrates data from VIIRS (Visible Infrared Imaging Radiometer Suite) and GFS
    to produce various biophysical parameters, including land surface temperature, NDVI,
    emissivity, albedo, and meteorological variables (air temperature, relative humidity,
    wind speed, shortwave incoming radiation, and soil moisture). It can also perform
    downscaling and bias correction using GEOS-5 FP data. The core biophysical model
    used is PT-JPL.

    Args:
        target_date (Union[date, str]): The target date for the forecast. Can be a datetime.date object or a string.
        geometry (RasterGrid): The desired spatial extent and resolution for the output rasters.
        target (str): A string identifying the target area or region.
        coarse_geometry (RasterGrid, optional): A coarser resolution RasterGrid for downscaling.
            Defaults to None, in which case it's derived from the target and GFS_CELL_SIZE.
        coarse_cell_size (float, optional): The cell size for the coarse geometry if `coarse_geometry` is None.
            Defaults to GFS_CELL_SIZE.
        ST_C (Raster, optional): Land surface temperature in Celsius. If None, it will be retrieved from VNP21A1D.
        emissivity (Raster, optional): Surface emissivity. If None, it will be retrieved from VNP21A1D.
        NDVI (Raster, optional): Normalized Difference Vegetation Index. If None, it will be retrieved from VNP09GA.
        albedo (Raster, optional): Surface albedo. If None, it will be retrieved from VNP09GA.
        SWin (Raster, optional): Shortwave incoming radiation. If None, it will be retrieved from GFS.
        Rn (Raster, optional): Net radiation. If None, it will be calculated using `process_verma_net_radiation`.
        SM (Raster, optional): Soil moisture. If None, it will be retrieved from GFS.
        wind_speed (Raster, optional): Wind speed. If None, it will be retrieved from GFS.
        Ta_C (Raster, optional): Air temperature in Celsius. If None, it will be retrieved from GFS.
        RH (Raster, optional): Relative humidity. If None, it will be retrieved from GFS.
        water (Raster, optional): Water mask. If None, it will be retrieved from SRTM.
        model (PTJPL, optional): An instance of the PTJPL model. Not directly used in the current implementation,
            as PTJPL is called as a function.
        working_directory (str, optional): The main working directory for temporary files and outputs.
            Defaults to the current directory.
        static_directory (str, optional): Directory for static datasets like SRTM.
        GFS_download (str, optional): Directory to download GFS data. Defaults to GFS_DOWNLOAD_DIRECTORY.
        GFS_output_directory (str, optional): Directory to save GFS processed outputs.
            Defaults to a subdirectory within `working_directory`.
        VNP21A1D_download_directory (str, optional): Directory to download VNP21A1D data.
            Defaults to VNP21A1D_DOWNLOAD_DIRECTORY.
        VNP09GA_download_directory (str, optional): Directory to download VNP09GA data.
            Defaults to VNP09GA_DOWNLOAD_DIRECTORY.
        SRTM_connection (NASADEMConnection, optional): An existing NASADEMConnection object. If None, a new one is created.
        SRTM_download (str, optional): Directory to download SRTM data.
        GEOS5FP_connection (GEOS5FP, optional): An existing GEOS5FP connection object. If None, a new one is created.
        GEOS5FP_download (str, optional): Directory to download GEOS-5 FP data.
        GEOS5FP_products (str, optional): Not used in the current implementation.
        GEDI_connection (GEDICanopyHeight, optional): An existing GEDICanopyHeight connection object. Not used.
        GEDI_download (str, optional): Directory to download GEDI data. Not used.
        ORNL_connection (MODISCI, optional): An existing MODISCI connection object. Not used.
        CI_directory (str, optional): Not used in the current implementation.
        soil_grids_connection (SoilGrids, optional): An existing SoilGrids connection object. Not used.
        soil_grids_download (str, optional): Directory to download SoilGrids data. Not used.
        intermediate_directory (str, optional): Not used in the current implementation.
        model_name (str, optional): The name of the ET model to use. Currently only "PTJPL" is supported.
        preview_quality (int, optional): Quality setting for previews. Not directly used for saving.
        ANN_model (Callable, optional): Artificial Neural Network model. Not used.
        ANN_model_filename (str, optional): Filename for ANN model. Not used.
        spacetrack_credentials_filename (str, optional): Credentials for SpaceTrack. Not used.
        ERS_credentials_filename (str, optional): Credentials for ERS. Not used.
        resampling (str, optional): Resampling method to use. Defaults to RESAMPLING.
        downscale_air (bool, optional): Whether to downscale air temperature. Defaults to DOWNSCALE_AIR.
        downscale_humidity (bool, optional): Whether to downscale humidity. Defaults to DOWNSCALE_HUMIDITY.
        downscale_moisture (bool, optional): Whether to downscale soil moisture. Defaults to DOWNSCALE_MOISTURE.
        apply_GEOS5FP_GFS_bias_correction (bool, optional): Whether to apply bias correction using GEOS-5 FP.
            Defaults to True.
        VIIRS_processing_date (Union[date, str], optional): The date to process VIIRS data. If None, it
            defaults to `target_date`.
        GFS_listing (pd.DataFrame, optional): A pre-loaded GFS listing. If None, it will be pulled.
        save_intermediate (bool, optional): Whether to save intermediate products. Not fully implemented for all.
        include_preview (bool, optional): Whether to include previews. Not fully implemented for all.
        show_distribution (bool, optional): Whether to show distribution plots. Defaults to True.
        load_previous (bool, optional): Whether to load previously processed GFS output if available.
            Defaults to True.
        target_variables (List[str], optional): A list of target variables to process.
            Defaults to TARGET_VARIABLES.

    Returns:
        Dict[str, Raster]: A dictionary where keys are variable names and values are the
        corresponding Raster objects.

    Raises:
        ValueError: If the `target_date` is before the earliest available GFS date.
        Exception: If unable to connect to GEOS-5 FP.
    """
    results = {}

    # Parse target_date if it's a string
    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()

    logger.info(f"GFS-VIIRS target date: {cl.time(target_date)}")
    # Define solar time for the forecast
    time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
    logger.info(f"GFS-VIIRS target time solar: {cl.time(time_solar)}")
    # Convert solar time to UTC based on geometry centroid longitude
    time_UTC = solar_to_UTC(time_solar, geometry.centroid.latlon.x)
    logger.info(f"GFS-VIIRS target time UTC: {cl.time(time_UTC)}")
    date_UTC = time_UTC.date()

    # Parse VIIRS_processing_date if it's a string
    if isinstance(VIIRS_processing_date, str):
        VIIRS_processing_date = parser.parse(VIIRS_processing_date).date()

    # Set working directory
    if working_directory is None:
        working_directory = "."
    working_directory = abspath(expanduser(working_directory))
    logger.info(f"GFS-VIIRS working directory: {cl.dir(working_directory)}")

    # Set GFS download directory
    if GFS_download is None:
        GFS_download = GFS_DOWNLOAD_DIRECTORY
    logger.info(f"GFS download directory: {cl.dir(GFS_download)}")

    # Set GFS output directory
    if GFS_output_directory is None:
        GFS_output_directory = join(working_directory, GFS_OUTPUT_DIRECTORY)
    logger.info(f"GFS output directory: {cl.dir(GFS_output_directory)}")

    # Initialize NASADEMConnection for SRTM data if not provided
    if SRTM_connection is None:
        SRTM_connection = NASADEMConnection(
            download_directory=SRTM_download,
        )

    # Retrieve water mask if not provided
    if water is None:
        water = SRTM_connection.swb(geometry)

    # Initialize GEOS5FP connection if not provided
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

    # Check if GFS data for the target date and products has already been processed
    GFS_already_processed = check_GFS_already_processed(
        GFS_output_directory=GFS_output_directory,
        target_date=target_date,
        time_UTC=time_UTC,
        target=target,
        products=target_variables
    )

    # If already processed and loading is enabled, load and return the results
    if GFS_already_processed:
        if load_previous:
            logger.info("loading previously generated VIIRS GEOS-5 FP output")
            return load_GFS(
                GFS_output_directory=GFS_output_directory,
                target_date=target_date,
                target=target
            )
        else:
            return {}  # Return empty dictionary if not loading previous

    # Pull GFS listing to determine available dates
    logger.info("pulling GFS listing")
    if GFS_listing is None:
        GFS_listing = get_GFS_listing()
    earliest_GFS_date = global_forecasting_system.earliest_date_UTC(listing=GFS_listing)

    # Validate target date against earliest GFS date
    if target_date < earliest_GFS_date:
        raise ValueError(
            f"target date {cl.time(target_date)} is before earliest GFS date {cl.time(earliest_GFS_date)}")

    # Set VIIRS processing date and time
    if VIIRS_processing_date is None:
        VIIRS_processing_date = target_date
    VIIRS_processing_time = time_UTC

    VIIRS_processing_datetime_solar = datetime(VIIRS_processing_date.year, VIIRS_processing_date.month,
                                               VIIRS_processing_date.day, 13, 30)
    logger.info(f"VIIRS processing date/time solar: {cl.time(VIIRS_processing_datetime_solar)}")
    VIIRS_processing_datetime_UTC = solar_to_UTC(VIIRS_processing_datetime_solar, geometry.centroid.latlon.x)
    logger.info(f"VIIRS processing date/time UTC: {cl.time(VIIRS_processing_datetime_UTC)}")

    # Calculate forecast distance in days
    forecast_distance_days = (target_date - VIIRS_processing_date).days
    if forecast_distance_days > 0:
        logger.info(
            f"target date {cl.time(target_date)} is {cl.val(forecast_distance_days)} days past VIIRS processing date {cl.time(VIIRS_processing_date)}")

    # Initialize VNP21A1D and VNP09GA connections
    VNP21_connection = VNP21A1D_002.VNP21A1D(download_directory=VNP21A1D_download_directory)
    VNP09_connection = VNP09GA_002.VNP09GA(download_directory=VNP09GA_download_directory)

    # Retrieve or use provided Land Surface Temperature (ST_C)
    if ST_C is None:
        logger.info(
            f"retrieving {cl.name('VNP21A1D')} {cl.name('ST_C')} from VIIRS on {cl.time(VIIRS_processing_date)}")
        ST_K = VNP21_connection.ST_K(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )
        ST_C = ST_K - 273.15

        # Fill NaNs in ST_C with GEOS-5 FP surface temperature
        ST_C_smooth = GEOS5FP_connection.Ts_K(time_UTC=time_UTC, geometry=geometry, resampling="cubic") - 273.15
        ST_C = rt.where(np.isnan(ST_C), ST_C_smooth, ST_C)
    results["ST"] = ST_C

    # Retrieve or use provided NDVI
    if NDVI is None:
        logger.info(
            f"retrieving {cl.name('VNP09GA')} {cl.name('NDVI')} from VIIRS on {cl.time(VIIRS_processing_date)}")
        NDVI = VNP09_connection.NDVI(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )
    results["NDVI"] = NDVI

    # Retrieve or use provided Emissivity
    if emissivity is None:
        logger.info(
            f"retrieving {cl.name('VNP21A1D')} {cl.name('emissivity')} from VIIRS on {cl.time(VIIRS_processing_date)}")
        emissivity = VNP21_connection.Emis_14(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )
    # Adjust emissivity for water bodies and fill NaNs based on NDVI
    emissivity = rt.where(water, 0.96, emissivity)
    emissivity = rt.where(np.isnan(emissivity), 1.0094 + 0.047 * np.log(NDVI), emissivity)
    results["emissivity"] = emissivity

    # Retrieve or use provided Albedo
    if albedo is None:
        logger.info(
            f"retrieving {cl.name('VNP09GA')} {cl.name('albedo')} from VIIRS on {cl.time(VIIRS_processing_date)}")
        albedo = VNP09_connection.albedo(
            date_UTC=VIIRS_processing_date,
            geometry=geometry,
            resampling="cubic"
        )
    results["albedo"] = albedo

    # Re-initialize SRTM connection if not provided (redundant check, already done above, but kept for safety)
    if SRTM_connection is None:
        logger.info("connecting to SRTM")
        SRTM_connection = NASADEMConnection( # Assuming SRTM is part of NASADEM, otherwise use a specific SRTM class
            working_directory=static_directory,
            download_directory=SRTM_download,
            offline_ok=True
        )

    logger.info("retrieving water mask from SRTM")
    # Retrieve water mask from SRTM (redundant call if already retrieved, but ensures it's available)
    water = SRTM_connection.swb(geometry)
    logger.info(f"running PT-JPL-SM ET model forecast at {cl.time(time_UTC)}")

    # Define coarse geometry if not provided
    if coarse_geometry is None:
        coarse_geometry = sentinel_tiles.grid(target, coarse_cell_size)

    # Retrieve or use provided Air Temperature (Ta_C)
    if Ta_C is None:
        logger.info(f"retrieving GFS {cl.name('Ta')} forecast at {cl.time(time_UTC)}")
        if downscale_air:
            Ta_K_coarse = forecast_Ta_K(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic",
                                        listing=GFS_listing)

            # Apply GEOS-5 FP GFS bias correction for air temperature
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

            # Downscale air temperature
            Ta_K = downscale_air_temperature(
                time_UTC=time_UTC,
                Ta_K_coarse=Ta_K_coarse,
                ST_K=ST_K,
                fine_geometry=geometry,
                coarse_geometry=coarse_geometry
            )
            Ta_C = Ta_K - 273.15
        else:
            # Apply GEOS-5 FP GFS bias correction for air temperature without downscaling
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
                # Retrieve Ta_C directly from GFS without downscaling or bias correction
                Ta_C = forecast_Ta_C(
                    time_UTC=time_UTC,
                    geometry=geometry,
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )
    results["Ta"] = Ta_C

    # Retrieve or use provided Soil Moisture (SM) for PTJPL model
    if SM is None and model_name == "PTJPL":
        logger.info(f"retrieving GFS {cl.name('SM')} forecast at {cl.time(time_UTC)}")

        if downscale_moisture:
            SM_coarse = forecast_SM(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic")

            # Apply GEOS-5 FP GFS bias correction for soil moisture
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

            ST_K = ST_C + 273.15
            # Downscale soil moisture
            SM = downscale_soil_moisture(
                time_UTC=time_UTC,
                fine_geometry=geometry,
                coarse_geometry=coarse_geometry,
                SM_coarse=SM_coarse,
                SM_resampled=forecast_SM(time_UTC=time_UTC, geometry=geometry, resampling="cubic", listing=GFS_listing), # SM_smooth
                ST_fine=ST_K,
                NDVI_fine=NDVI,
                water=water
            )
        else:
            # Apply GEOS-5 FP GFS bias correction for soil moisture without downscaling
            if apply_GEOS5FP_GFS_bias_correction:
                SM_coarse = forecast_SM(
                    time_UTC=time_UTC,
                    geometry=coarse_geometry,
                    resampling="cubic",
                    listing=GFS_listing
                )

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
                # Retrieve SM directly from GFS without downscaling or bias correction
                SM = forecast_SM(
                    time_UTC=time_UTC,
                    geometry=geometry,
                    directory=GFS_download,
                    resampling="cubic",
                    listing=GFS_listing
                )
    results["SM"] = SM

    # Retrieve or use provided Relative Humidity (RH)
    if RH is None:
        logger.info(f"retrieving GFS {cl.name('RH')} forecast at {cl.time(time_UTC)}")

        if downscale_humidity:
            # Calculate saturation vapor pressure (SVP)
            SVP_Pa = 0.6108 * np.exp((17.27 * Ta_C) / (Ta_C + 237.3)) * 1000  # [Pa]
            RH_smooth = forecast_RH(time_UTC=time_UTC, geometry=geometry, resampling="cubic", listing=GFS_listing)
            Ea_Pa_estimate = RH_smooth * SVP_Pa
            VPD_Pa_estimate = SVP_Pa - Ea_Pa_estimate
            VPD_kPa_estimate = VPD_Pa_estimate / 1000
            RH_estimate = SM ** (1 / VPD_kPa_estimate) # This line seems like a custom estimation for RH
            RH_coarse = forecast_RH(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic",
                                    listing=GFS_listing)

            # Apply GEOS-5 FP GFS bias correction for humidity
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

            # Bias correct humidity
            RH = bias_correct(
                coarse_image=RH_coarse,
                fine_image=RH_estimate,
                upsampling="average",
                downsampling="linear",
                return_bias=False
            )
        else:
            # Apply GEOS-5 FP GFS bias correction for humidity without downscaling
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
                # Retrieve RH directly from GFS without downscaling or bias correction
                RH = forecast_RH(time_UTC=time_UTC, geometry=geometry, directory=GFS_download, listing=GFS_listing)

    # Check and display distribution of RH
    check_distribution(RH, "RH", date_UTC=date_UTC, target=target)
    results["RH"] = RH

    # Retrieve or use provided Wind Speed
    if wind_speed is None:
        logger.info(f"retrieving GFS {cl.name('wind_speed')} forecast at {cl.time(time_UTC)}")

        # Apply GEOS-5 FP GFS bias correction for wind speed
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
            # Retrieve wind speed directly from GFS without bias correction
            wind_speed = forecast_wind(
                time_UTC=time_UTC,
                geometry=geometry,
                directory=GFS_download,
                resampling="cubic",
                listing=GFS_listing
            )
    results["wind_speed"] = wind_speed

    # Retrieve or use provided Shortwave Incoming Radiation (SWin)
    if SWin is None:
        logger.info(f"retrieving GFS {cl.name('SWin')} forecast at {cl.time(time_UTC)}")

        # Apply GEOS-5 FP GFS bias correction for SWin
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
            # Retrieve SWin directly from GFS without bias correction
            SWin = forecast_SWin(time_UTC=time_UTC, geometry=geometry, directory=GFS_download, listing=GFS_listing)
    results["SWin"] = SWin

    # Process Verma Net Radiation
    verma_results = process_verma_net_radiation(
        SWin=SWin,
        albedo=albedo,
        ST_C=ST_C,
        emissivity=emissivity,
        Ta_C=Ta_C,
        RH=RH
    )
    Rn = verma_results["Rn"] # Assign Rn from Verma results

    logger.info(f"running PT-JPL ET model hindcast at {cl.time(time_UTC)}")

    # Run PT-JPL ET model
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

    # Save all processed results to GeoTIFF files
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
            f"writing VIIRS GFS {cl.name(product)} at {cl.place(target)} at {cl.time(time_UTC)} to file: {cl.file(filename)}")
        image.to_geotiff(filename)

    return results
