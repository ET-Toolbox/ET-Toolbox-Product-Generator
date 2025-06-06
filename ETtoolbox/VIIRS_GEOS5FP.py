from typing import Union, List
from datetime import date, datetime, timedelta
from dateutil import parser
from os.path import join, exists, basename, abspath, expanduser
import rasters as rt
import numpy as np

from glob import glob
from os.path import splitext
from typing import Dict, Callable

from rasters import RasterGrid

# Import GEDI Canopy Height and GEOS-5 FP modules
from gedi_canopy_height import GEDICanopyHeight
from GEOS5FP import GEOS5FP

# Import MODIS CI, PTJPL, and SoilGrids modules
from MODISCI import MODISCI
from PTJPL import PTJPL
from soil_capacity_wilting import SoilGrids

# Import downscaling and bias correction functions from GEOS5FP
from GEOS5FP.downscaling import downscale_air_temperature, downscale_soil_moisture, downscale_vapor_pressure_deficit, \
    downscale_relative_humidity, bias_correct
from PTJPL import FLOOR_TOPT

import logging
import colored_logging as cl

from solar_apparent_time import solar_to_UTC

from verma_net_radiation import process_verma_net_radiation

# Import VIIRS VNP09GA and VNP21A1D modules
from VNP09GA_002 import VNP09GA
from VNP21A1D_002 import VNP21A1D

# Import NASADEM connection module
from NASADEM import NASADEMConnection

# Import constants and helper functions from the current package
from .constants import *
from .generate_VIIRS_GEOS5FP_output_directory import generate_VIIRS_GEOS5FP_output_directory
from .generate_VIIRS_GEOS5FP_output_filename import generate_VIIRS_GEOS5FP_output_filename
from .check_VIIRS_GEOS5FP_already_processed import check_VIIRS_GEOS5FP_already_processed
from .load_VIIRS_GEOS5FP import load_VIIRS_GEOS5FP

# Set up logging for the module
logger = logging.getLogger(__name__)

class GEOS5FPNotAvailableError(Exception):
    """Custom exception raised when GEOS-5 FP data is not available."""
    pass

def VIIRS_GEOS5FP(
        target_date: Union[date, str],
        geometry: RasterGrid,
        target: str,
        ST_C: rt.Raster = None,
        emissivity: rt.Raster = None,
        NDVI: rt.Raster = None,
        albedo: rt.Raster = None,
        SWin: Union[rt.Raster, str] = None,
        Rn: Union[rt.Raster, str] = None,
        SM: Union[rt.Raster, str] = None,
        wind_speed: rt.Raster = None, # This parameter is not used in the current code
        Ta_C: Union[rt.Raster, str] = None,
        RH: Union[rt.Raster, str] = None,
        water: rt.Raster = None,
        elevation_km: rt.Raster = None,
        ET_model_name: str = ET_MODEL_NAME, # This parameter is not used in the current code
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
        GEOS5FP_products: str = None, # This parameter is not used in the current code
        GEOS5FP_offline_processing: bool = True,
        GEDI_connection: GEDICanopyHeight = None, # This parameter is not used in the current code
        GEDI_download: str = None, # This parameter is not used in the current code
        ORNL_connection: MODISCI = None, # This parameter is not used in the current code
        CI_directory: str = None, # This parameter is not used in the current code
        soil_grids_connection: SoilGrids = None, # This parameter is not used in the current code
        soil_grids_download: str = None, # This parameter is not used in the current code
        intermediate_directory: str = None, # This parameter is not used in the current code
        preview_quality: int = PREVIEW_QUALITY, # This parameter is not used in the current code
        ANN_model: Callable = None, # This parameter is not used in the current code
        ANN_model_filename: str = None, # This parameter is not used in the current code
        resampling: str = RESAMPLING, # This parameter is not used in the current code
        coarse_cell_size: float = COARSE_CELL_SIZE,
        downscale_air: bool = DOWNSCALE_AIR,
        downscale_humidity: bool = DOWNSCALE_HUMIDITY,
        downscale_moisture: bool = DOWNSCALE_MOISTURE,
        floor_Topt: bool = FLOOR_TOPT, # This parameter is not used in the current code
        save_intermediate: bool = False, # This parameter is not used in the current code
        include_preview: bool = True, # This parameter is not used in the current code
        show_distribution: bool = True, # This parameter is not used in the current code
        load_previous: bool = True,
        target_variables: List[str] = TARGET_VARIABLES) -> Dict[str, rt.Raster]:
    """
    Processes VIIRS and GEOS-5 FP data to derive various biophysical parameters,
    including land surface temperature, NDVI, emissivity, albedo, air temperature,
    soil moisture, relative humidity, and evapotranspiration components.

    The function handles data retrieval, gap-filling, downscaling, and output
    generation, providing a comprehensive workflow for integrating VIIRS and
    GEOS-5 FP datasets.

    Args:
        target_date (Union[date, str]): The target date for processing. Can be a
            datetime.date object or a string parseable by dateutil.parser.
        geometry (RasterGrid): The spatial extent and resolution for the output
            rasters.
        target (str): A string identifying the target area or site, used for
            naming output files.
        ST_C (rt.Raster, optional): Pre-computed land surface temperature in
            Celsius. If None, it will be retrieved from VNP21A1D. Defaults to None.
        emissivity (rt.Raster, optional): Pre-computed surface emissivity. If None,
            it will be derived from NDVI. Defaults to None.
        NDVI (rt.Raster, optional): Pre-computed Normalized Difference Vegetation
            Index. If None, it will be retrieved from VNP09GA. Defaults to None.
        albedo (rt.Raster, optional): Pre-computed surface albedo. If None, it
            will be retrieved from VNP09GA. Defaults to None.
        SWin (Union[rt.Raster, str], optional): Pre-computed incoming shortwave
            radiation. If None or 'GEOS5FP', it will be retrieved from GEOS-5 FP.
            Defaults to None.
        Rn (Union[rt.Raster, str], optional): Pre-computed net radiation. If None,
            it will be calculated using the Verma net radiation model. Defaults to None.
        SM (Union[rt.Raster, str], optional): Pre-computed soil moisture. If None,
            it will be retrieved and potentially downscaled from GEOS-5 FP.
            Defaults to None.
        wind_speed (rt.Raster, optional): Wind speed data. Not currently used in
            the processing, but kept for potential future extensions. Defaults to None.
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
            Currently, only PT-JPL is implemented. Defaults to ET_MODEL_NAME.
        working_directory (str, optional): The main working directory for data
            processing and output. Defaults to current directory.
        static_directory (str, optional): Directory for static datasets like SRTM.
            Defaults to None.
        VNP09GA_download_directory (str, optional): Directory for downloading
            VNP09GA data. Defaults to VNP09GA_download_directory constant.
        VNP21A1D_download_directory (str, optional): Directory for downloading
            VNP21A1D data. Defaults to VNP21A1D_DOWNLOAD_DIRECTORY constant.
        use_VIIRS_composite (bool, optional): Whether to use VIIRS composite
            data for gap-filling. Defaults to USE_VIIRS_COMPOSITE constant.
        VIIRS_composite_days (int, optional): Number of days to look back for
            VIIRS compositing. Defaults to VIIRS_COMPOSITE_DAYS constant.
        VIIRS_GEOS5FP_output_directory (str, optional): Directory to save final
            VIIRS-GEOS5FP products. Defaults to a subdirectory within the
            working_directory.
        SRTM_connection (NASADEMConnection, optional): An existing NASADEMConnection
            object. If None, a new one will be initialized. Defaults to None.
        SRTM_download (str, optional): Directory for downloading SRTM data.
            Defaults to None.
        GEOS5FP_connection (GEOS5FP, optional): An existing GEOS5FP connection
            object. If None, a new one will be initialized. Defaults to None.
        GEOS5FP_download (str, optional): Directory for downloading GEOS-5 FP data.
            Defaults to None.
        GEOS5FP_products (str, optional): GEOS-5 FP products to download. Not
            currently used directly for product selection. Defaults to None.
        GEOS5FP_offline_processing (bool, optional): Whether to allow offline
            processing for GEOS-5 FP (i.e., use local data without checking
            for latest availability). Defaults to True.
        GEDI_connection (GEDICanopyHeight, optional): GEDI connection object. Not
            currently used. Defaults to None.
        GEDI_download (str, optional): GEDI download directory. Not currently used.
            Defaults to None.
        ORNL_connection (MODISCI, optional): ORNL MODIS CI connection object. Not
            currently used. Defaults to None.
        CI_directory (str, optional): CI data directory. Not currently used.
            Defaults to None.
        soil_grids_connection (SoilGrids, optional): SoilGrids connection object. Not
            currently used. Defaults to None.
        soil_grids_download (str, optional): SoilGrids download directory. Not
            currently used. Defaults to None.
        intermediate_directory (str, optional): Directory for saving intermediate
            products. Not currently used. Defaults to None.
        preview_quality (int, optional): Quality setting for previews. Not
            currently used. Defaults to PREVIEW_QUALITY constant.
        ANN_model (Callable, optional): Artificial Neural Network model. Not
            currently used. Defaults to None.
        ANN_model_filename (str, optional): Filename for ANN model. Not currently
            used. Defaults to None.
        resampling (str, optional): Resampling method. Not currently used for
            all resampling operations. Defaults to RESAMPLING constant.
        coarse_cell_size (float, optional): Cell size for coarse resolution data
            used in downscaling. Defaults to COARSE_CELL_SIZE constant.
        downscale_air (bool, optional): Whether to downscale air temperature.
            Defaults to DOWNSCALE_AIR constant.
        downscale_humidity (bool, optional): Whether to downscale relative humidity.
            Defaults to DOWNSCALE_HUMIDITY constant.
        downscale_moisture (bool, optional): Whether to downscale soil moisture.
            Defaults to DOWNSCALE_MOISTURE constant.
        floor_Topt (bool, optional): Whether to floor Topt in PT-JPL. Not
            currently used directly here. Defaults to FLOOR_TOPT constant.
        save_intermediate (bool, optional): Whether to save intermediate products.
            Not currently used. Defaults to False.
        include_preview (bool, optional): Whether to include preview images. Not
            currently used. Defaults to True.
        show_distribution (bool, optional): Whether to show distribution plots.
            Not currently used. Defaults to True.
        load_previous (bool, optional): Whether to load previously processed
            results if available. Defaults to True.
        target_variables (List[str], optional): A list of target variables to
            process and save. Defaults to TARGET_VARIABLES constant.

    Returns:
        Dict[str, rt.Raster]: A dictionary where keys are variable names (e.g.,
            "ST", "NDVI", "ET_total") and values are the corresponding
            raster objects.

    Raises:
        GEOS5FPNotAvailableError: If unable to connect to GEOS-5 FP or if the
            target time is beyond the latest available GEOS-5 FP data.
    """
    results = {}

    # Convert target_date to a date object if it's a string
    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()

    logger.info(f"VIIRS GEOS-5 FP target date: {cl.time(target_date)}")
    # Define solar time for the processing (13:30 local solar time)
    time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
    logger.info(f"VIIRS GEOS-5 FP target time solar: {cl.time(time_solar)}")
    # Convert solar time to UTC based on the geometry's centroid longitude
    time_UTC = solar_to_UTC(time_solar, geometry.centroid.latlon.x)
    logger.info(f"VIIRS GEOS-5 FP target time UTC: {cl.time(time_UTC)}")

    # Set up working directory
    if working_directory is None:
        working_directory = "."
    working_directory = abspath(expanduser(working_directory))
    logger.info(f"VIIRS GEOS-5 FP working directory: {cl.dir(working_directory)}")

    # Initialize SRTM connection if not provided
    if SRTM_connection is None:
        SRTM_connection = NASADEMConnection(
            working_directory=static_directory,
            download_directory=SRTM_download,
            offline_ok=True
        )

    # Retrieve water mask if not provided
    if water is None:
        logger.info(f"retrieving {cl.name('water mask')} from SRTM")
        water = SRTM_connection.swb(geometry)
    results["water"] = water # Add water to results even if provided initially

    # Retrieve elevation if not provided
    if elevation_km is None:
        logger.info(f"retrieving {cl.name('elevation')} from SRTM")
        elevation_km = SRTM_connection.elevation_km(geometry)

    # Set up VNP09GA download directory and connection
    if VNP09GA_download_directory is None:
        VNP09GA_download_directory = VNP09GA_DOWNLOAD_DIRECTORY
    logger.info(f"VNP09GA download directory: {cl.dir(VNP09GA_download_directory)}")
    VNP09GA_connection = VNP09GA(
        download_directory=VNP09GA_download_directory,
    )

    # Set up VNP21A1D download directory and connection
    if VNP21A1D_download_directory is None:
        VNP21A1D_download_directory = VNP21A1D_DOWNLOAD_DIRECTORY
    logger.info(f"VNP21A1D download directory: {cl.dir(VNP21A1D_download_directory)}")
    VNP21A1D_connection = VNP21A1D(
        download_directory=VNP21A1D_download_directory,
    )

    # Set up output directory
    if VIIRS_GEOS5FP_output_directory is None:
        VIIRS_GEOS5FP_output_directory = join(working_directory, VIIRS_GEOS5FP_OUTPUT_DIRECTORY)
    logger.info(f"VIIRS GEOS-5 FP output directory: {cl.dir(VIIRS_GEOS5FP_output_directory)}")

    # Check if the data for the target date and products has already been processed
    VIIRS_GEOS5FP_already_processed = check_VIIRS_GEOS5FP_already_processed(
        VIIRS_GEOS5FP_output_directory=VIIRS_GEOS5FP_output_directory,
        target_date=target_date,
        time_UTC=time_UTC,
        target=target,
        products=target_variables
    )

    if VIIRS_GEOS5FP_already_processed:
        if load_previous:
            logger.info("loading previously generated VIIRS GEOS-5 FP output")
            return load_VIIRS_GEOS5FP(
                VIIRS_GEOS5FP_output_directory=VIIRS_GEOS5FP_output_directory,
                target_date=target_date,
                target=target
            )
        else:
            logger.info("Previous VIIRS GEOS-5 FP output found, but 'load_previous' is False. Skipping reprocessing.")
            return {} # Return empty dictionary as per current behavior if not loading

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
            raise GEOS5FPNotAvailableError("unable to connect to GEOS-5 FP")

    # Check GEOS-5 FP data availability if not in offline processing mode
    if not GEOS5FP_offline_processing:
        latest_GEOS5FP_time = GEOS5FP_connection.latest_time_available
        logger.info(f"latest GEOS-5 FP time available: {latest_GEOS5FP_time}")
        logger.info(f"processing time: {time_UTC}")

        if time_UTC.strftime("%Y-%m-%d %H:%M:%S") > latest_GEOS5FP_time.strftime("%Y-%m-%d %H:%M:%S"):
            raise GEOS5FPNotAvailableError(
                f"VIIRS GEOS-5 FP target time {time_UTC} is past latest available GEOS-5 FP time {latest_GEOS5FP_time}")

    # Retrieve and process Land Surface Temperature (ST_C)
    if ST_C is None:
        logger.info(
            f"retrieving {cl.name('VNP21A1D')} {cl.name('ST_C')} from VIIRS on {cl.time(target_date)}")
        ST_C = VNP21A1D_connection.ST_C(date_UTC=target_date, geometry=geometry, resampling="cubic")

        # Gap-filling ST_C using VIIRS composite if enabled
        if use_VIIRS_composite:
            for days_back in range(1, VIIRS_composite_days):
                fill_date = target_date - timedelta(days=days_back)
                logger.info(
                    f"gap-filling {cl.name('VNP21A1D')} {cl.name('ST_C')} from VIIRS on {cl.time(fill_date)} for {cl.time(target_date)}")
                ST_C_fill = VNP21A1D_connection.ST_C(date_UTC=fill_date, geometry=geometry, resampling="cubic")
                # Use np.isnan for more robust NaN checking
                ST_C = rt.where(np.isnan(ST_C), ST_C_fill, ST_C)

        # Gap-filling remaining NaN values with resampled GEOS-5 FP surface temperature
        logger.info(f"gap-filling remaining {cl.name('ST_C')} with GEOS-5 FP surface temperature")
        ST_C_smooth = GEOS5FP_connection.Ts_K(time_UTC=time_UTC, geometry=geometry, resampling="cubic") - 273.15
        ST_C = rt.where(np.isnan(ST_C), ST_C_smooth, ST_C)

    results["ST"] = ST_C

    # Retrieve and process Normalized Difference Vegetation Index (NDVI)
    if NDVI is None:
        logger.info(
            f"retrieving {cl.name('VNP09GA')} {cl.name('NDVI')} on {cl.time(target_date)}")

        NDVI = VNP09GA_connection.NDVI(date_UTC=target_date, geometry=geometry, resampling="cubic")

        # Gap-filling NDVI using VIIRS composite if enabled
        if use_VIIRS_composite:
            for days_back in range(1, VIIRS_composite_days):
                fill_date = target_date - timedelta(days=days_back)
                logger.info(
                    f"gap-filling {cl.name('VNP09GA')} {cl.name('NDVI')} on {cl.time(fill_date)} for {cl.time(target_date)}")
                NDVI_fill = VNP09GA_connection.NDVI(date_UTC=fill_date, geometry=geometry, resampling="cubic")
                ST_C = rt.where(np.isnan(ST_C), ST_C_fill, ST_C) # Typo: This line should be NDVI = rt.where(np.isnan(NDVI), NDVI_fill, NDVI)
                NDVI = rt.where(np.isnan(NDVI), NDVI_fill, NDVI) # Corrected line

    results["NDVI"] = NDVI

    # Derive Emissivity if not provided
    if emissivity is None:
        logger.info(f"deriving {cl.name('emissivity')} from NDVI")
        emissivity = 1.0094 + 0.047 * np.log(NDVI)
        emissivity = rt.where(water, 0.96, emissivity) # Set emissivity to a typical water value for water bodies

    results["emissivity"] = emissivity

    # Retrieve and process Albedo
    if albedo is None:
        logger.info(
            f"retrieving {cl.name('VNP09GA')} {cl.name('albedo')} on {cl.time(target_date)}")

        albedo = VNP09GA_connection.albedo(date_UTC=target_date, geometry=geometry, resampling="cubic")

        # Gap-filling albedo using VIIRS composite if enabled
        if use_VIIRS_composite:
            for days_back in range(1, VIIRS_composite_days):
                fill_date = target_date - timedelta(days=days_back)
                logger.info(
                    f"gap-filling {cl.name('VNP09GA')} {cl.name('albedo')} from VIIRS on {cl.time(fill_date)} for {cl.time(target_date)}")
                albedo_fill = VNP09GA_connection.albedo(date_UTC=fill_date, geometry=geometry, resampling="cubic")
                albedo = rt.where(np.isnan(albedo), albedo_fill, albedo)

    results["albedo"] = albedo

    # Define coarse geometry for downscaling
    coarse_geometry = geometry.rescale(coarse_cell_size)

    # Retrieve and process Air Temperature (Ta_C)
    if Ta_C is None:
        if downscale_air:
            logger.info(f"downscaling {cl.name('air temperature')} using GEOS-5 FP and VIIRS ST")
            ST_K = ST_C + 273.15 # Convert surface temperature to Kelvin
            Ta_K_coarse = GEOS5FP_connection.Ta_K(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic")
            Ta_K = downscale_air_temperature(
                time_UTC=time_UTC,
                Ta_K_coarse=Ta_K_coarse,
                ST_K=ST_K,
                fine_geometry=geometry,
                coarse_geometry=coarse_geometry
            )
            Ta_C = Ta_K - 273.15 # Convert downscaled air temperature back to Celsius
        else:
            logger.info(f"retrieving {cl.name('air temperature')} from GEOS-5 FP")
            Ta_C = GEOS5FP_connection.Ta_C(time_UTC=time_UTC, geometry=geometry, resampling="cubic")

    results["Ta"] = Ta_C

    # Retrieve and process Soil Moisture (SM)
    if SM is None:
        if downscale_moisture:
            logger.info(f"downscaling {cl.name('soil moisture')} using GEOS-5 FP, VIIRS ST and NDVI")
            ST_K = ST_C + 273.15
            SM_coarse = GEOS5FP_connection.SFMC(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic")
            SM_smooth = GEOS5FP_connection.SFMC(time_UTC=time_UTC, geometry=geometry, resampling="cubic") # Resampled GEOS-5 FP for comparison

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
            logger.info(f"retrieving {cl.name('soil moisture')} from GEOS-5 FP")
            SM = GEOS5FP_connection.SFMC(time_UTC=time_UTC, geometry=geometry, resampling="cubic")

    results["SM"] = SM

    # Retrieve and process Relative Humidity (RH)
    if RH is None:
        if downscale_humidity:
            logger.info(f"downscaling {cl.name('relative humidity')} using GEOS-5 FP, VIIRS ST and downscaled SM")
            ST_K = ST_C + 273.15
            VPD_Pa_coarse = GEOS5FP_connection.VPD_Pa(time_UTC=time_UTC, geometry=coarse_geometry, resampling="cubic")

            VPD_Pa = downscale_vapor_pressure_deficit(
                time_UTC=time_UTC,
                VPD_Pa_coarse=VPD_Pa_coarse,
                ST_K=ST_K,
                fine_geometry=geometry,
                coarse_geometry=coarse_geometry
            )

            VPD_kPa = VPD_Pa / 1000 # Convert Vapor Pressure Deficit from Pa to kPa

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
            logger.info(f"retrieving {cl.name('relative humidity')} from GEOS-5 FP")
            RH = GEOS5FP_connection.RH(time_UTC=time_UTC, geometry=geometry, resampling="cubic")

    results["RH"] = RH

    # Retrieve or generate Incoming Shortwave Radiation (SWin)
    if SWin is None or isinstance(SWin, str):
        logger.info("generating solar radiation using GEOS-5 FP")
        SWin = GEOS5FP_connection.SWin(time_UTC=time_UTC, geometry=geometry, resampling="cubic")

    # Calculate Net Radiation (Rn) if not provided
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
        Rn = verma_results["Rn"]

    results["Rn"] = Rn

    # Run the PT-JPL Evapotranspiration model
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

    # Add PT-JPL results to the main results dictionary
    for k, v in PTJPL_results.items():
        results[k] = v

    # Save all processed raster products to files
    for product, image in results.items():
        # Generate the output filename for each product
        filename = generate_VIIRS_GEOS5FP_output_filename(
            VIIRS_GEOS5FP_output_directory=VIIRS_GEOS5FP_output_directory,
            target_date=target_date,
            time_UTC=time_UTC,
            target=target,
            product=product
        )

        if image is None:
            logger.warning(f"no image result for {product}, skipping saving for this product.")
            continue

        logger.info(
            f"writing VIIRS GEOS-5 FP {cl.name(product)} at {cl.place(target)} at {cl.time(time_UTC)} to file: {cl.file(filename)}")
        image.to_geotiff(filename)

    return results
